# target_context.py
# ARCHITECTURE_FINGERPRINT: unchanged (normalization, retrieval params, top-K, ranking, K=15)
# IMPLEMENTATION_VERSION: v2_target_context
# Target-side structures built ONCE per (source, country) — zero semantic changes.

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


logger = logging.getLogger(__name__)

IMPLEMENTATION_VERSION = "v2_target_context"


@dataclass
class ChannelTargetCache:
    """TF-IDF target representation for one (source, country, channel). Built once."""
    vec: object                   # fitted TfidfVectorizer
    X_target_cpu: sp.csr_matrix   # CPU scipy CSR, shape (n_targets, vocab)
    X_target_gpu: object          # CuPy CSR of X_target_cpu.T, or None
    cand_ids: np.ndarray          # entity_id ordering matching rows
    build_seconds: float = 0.0


@dataclass
class SourceCountryContext:
    """All pre-built target-side structures for one (source_name, country) pair."""
    source_name: str
    country: str
    target_df: pd.DataFrame
    target_unique_name_df: pd.DataFrame  # R4 unique-name rows, pre-computed
    rare_postings: dict                  # token -> [entity_id, ...]
    rare_df_map: dict                    # token -> count
    numeric_postings: dict
    numeric_df_map: dict
    name_cache: Optional[ChannelTargetCache] = None
    addr_cache: Optional[ChannelTargetCache] = None
    build_seconds_total: float = 0.0


class TargetContext:
    """
    Container for all pre-built target structures across sources and countries.

    Usage:
        ctx = TargetContext()
        ctx.build(source_dfs={'S2': s2_df, 'S3': s3_df}, config=config, countries=['US'])
        for shard in shards:
            generator.generate_with_context(s1_chunk, ctx)
        ctx.release()
    """

    def __init__(self) -> None:
        self._store: Dict[tuple, SourceCountryContext] = {}

    def build(
        self,
        source_dfs: Dict[str, pd.DataFrame],
        config: dict,
        countries: List[str],
    ) -> None:
        """Build all target contexts. Call ONCE before the shard loop."""
        try:
            import cupy as _cp  # noqa: F401
            import cupyx.scipy.sparse as cpx_sparse  # noqa: F401
            HAS_CUPY = True
        except ImportError:
            HAS_CUPY = False

        try:
            from cuml.feature_extraction.text import TfidfVectorizer as TFIDF
            HAS_CUML = True
        except ImportError:
            if os.environ.get("ALLOW_CPU_TFIDF") != "1":
                raise RuntimeError(
                    "GPU_REQUIRED=TRUE. cuML not available. "
                    "Set ALLOW_CPU_TFIDF=1 for unit-test CPU fallback."
                )
            from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
            HAS_CUML = False

        name_cfg = config.get("name_word", {})
        addr_cfg = config.get("address_word", {})
        rare_cfg = config.get("rare_token", {})
        num_cfg  = config.get("numeric", {})
        name_enabled    = name_cfg.get("enabled", False)
        addr_enabled    = addr_cfg.get("enabled", False)
        rare_enabled    = rare_cfg.get("enabled", False)
        numeric_enabled = num_cfg.get("enabled", False)

        def _vec_kwargs(ch_cfg: dict) -> dict:
            kw = {
                "analyzer":     ch_cfg.get("analyzer", "word"),
                "ngram_range":  (ch_cfg.get("ngram_min", 1), ch_cfg.get("ngram_max", 2)),
                "sublinear_tf": ch_cfg.get("sublinear_tf", True),
                "min_df":       ch_cfg.get("min_df", 1),
                "max_df":       ch_cfg.get("max_df", 1.0),
                "max_features": ch_cfg.get("max_features", None),
            }
            if not HAS_CUML:
                kw["dtype"] = np.float32
            return kw

        for source_name, target_df_full in source_dfs.items():
            for country in countries:
                t0 = time.time()
                key = (source_name, country)

                c_target = target_df_full[
                    target_df_full["country"] == country
                ].reset_index(drop=True)

                if len(c_target) == 0:
                    logger.info("[TargetContext] BUILD %s/%s: 0 rows — skipping.", source_name, country)
                    continue

                logger.info("[TargetContext] BUILD %s/%s: %d rows.", source_name, country, len(c_target))

                # R4: unique-name pre-computation
                t_exact = time.time()
                target_valid_name = c_target[c_target["name_norm_clean"] != ""]
                counts = (
                    target_valid_name
                    .groupby(["name_norm_clean", "country"])
                    .size()
                    .reset_index(name="count")
                )
                unique_keys = counts[counts["count"] == 1].drop(columns=["count"])
                target_unique_name_df = target_valid_name.merge(unique_keys, on=["name_norm_clean", "country"])
                logger.info(
                    "[TargetContext][%s][%s] Exact unique-name: %d rows (%.2fs)",
                    source_name, country, len(target_unique_name_df), time.time() - t_exact,
                )

                # Rare-token inverted index
                rare_postings: dict = {}
                rare_df_map: dict = {}
                if rare_enabled:
                    t_rare = time.time()
                    max_df_rare = rare_cfg.get("max_df_threshold", 10)
                    valid_rare = c_target[
                        ~(c_target["name_norm_clean"].isna() | (c_target["name_norm_clean"] == ""))
                    ]
                    raw: dict = defaultdict(list)
                    for e_id, name in zip(valid_rare["entity_id"].values, valid_rare["name_norm_clean"].values):
                        for tok in set(name.split()):
                            raw[tok].append(e_id)
                    rare_postings = {k: v for k, v in raw.items() if 0 < len(v) <= max_df_rare}
                    rare_df_map   = {k: len(v) for k, v in rare_postings.items()}
                    logger.info(
                        "[TargetContext][%s][%s] Rare index: %d tokens (max_df=%d, %.2fs)",
                        source_name, country, len(rare_postings), max_df_rare, time.time() - t_rare,
                    )

                # Numeric inverted index
                numeric_postings: dict = {}
                numeric_df_map: dict = {}
                if numeric_enabled:
                    t_num = time.time()
                    max_df_num = num_cfg.get("max_df_threshold", 50)
                    valid_num = c_target[
                        ~(c_target["addr_numeric_tokens"].isna() | (c_target["addr_numeric_tokens"] == ""))
                    ]
                    raw_n: dict = defaultdict(list)
                    for e_id, nums in zip(valid_num["entity_id"].values, valid_num["addr_numeric_tokens"].values):
                        for tok in set(nums.split()):
                            raw_n[tok].append(e_id)
                    numeric_postings = {k: v for k, v in raw_n.items() if 0 < len(v) <= max_df_num}
                    numeric_df_map   = {k: len(v) for k, v in numeric_postings.items()}
                    logger.info(
                        "[TargetContext][%s][%s] Numeric index: %d tokens (max_df=%d, %.2fs)",
                        source_name, country, len(numeric_postings), max_df_num, time.time() - t_num,
                    )

                # Name TF-IDF target matrix
                name_cache: Optional[ChannelTargetCache] = None
                if name_enabled:
                    t_name = time.time()
                    valid_name = c_target[
                        ~(c_target["name_norm_clean"].isna() | (c_target["name_norm_clean"] == ""))
                    ]
                    if len(valid_name) > 0:
                        vec = TFIDF(**_vec_kwargs(name_cfg))
                        X_raw = vec.fit_transform(valid_name["name_norm_clean"])
                        X_cpu: sp.csr_matrix = X_raw.get() if (HAS_CUML and hasattr(X_raw, "get")) else X_raw
                        X_gpu = None
                        if HAS_CUPY:
                            import cupyx.scipy.sparse as _cpx
                            X_gpu = _cpx.csr_matrix(X_cpu.T)  # uploaded ONCE
                        name_cache = ChannelTargetCache(
                            vec=vec, X_target_cpu=X_cpu, X_target_gpu=X_gpu,
                            cand_ids=valid_name["entity_id"].values,
                            build_seconds=time.time() - t_name,
                        )
                        logger.info(
                            "[TargetContext][%s][%s] Name TF-IDF: %s GPU=%s (%.2fs)",
                            source_name, country, X_cpu.shape,
                            "YES" if X_gpu is not None else "NO", name_cache.build_seconds,
                        )

                # Address TF-IDF target matrix
                addr_cache: Optional[ChannelTargetCache] = None
                if addr_enabled:
                    t_addr = time.time()
                    valid_addr = c_target[
                        ~(c_target["addr_norm_clean"].isna() | (c_target["addr_norm_clean"] == ""))
                    ]
                    if len(valid_addr) > 0:
                        vec = TFIDF(**_vec_kwargs(addr_cfg))
                        X_raw = vec.fit_transform(valid_addr["addr_norm_clean"])
                        X_cpu = X_raw.get() if (HAS_CUML and hasattr(X_raw, "get")) else X_raw
                        X_gpu = None
                        if HAS_CUPY:
                            import cupyx.scipy.sparse as _cpx
                            X_gpu = _cpx.csr_matrix(X_cpu.T)
                        addr_cache = ChannelTargetCache(
                            vec=vec, X_target_cpu=X_cpu, X_target_gpu=X_gpu,
                            cand_ids=valid_addr["entity_id"].values,
                            build_seconds=time.time() - t_addr,
                        )
                        logger.info(
                            "[TargetContext][%s][%s] Addr TF-IDF: %s GPU=%s (%.2fs)",
                            source_name, country, X_cpu.shape,
                            "YES" if X_gpu is not None else "NO", addr_cache.build_seconds,
                        )

                elapsed = time.time() - t0
                self._store[key] = SourceCountryContext(
                    source_name=source_name, country=country,
                    target_df=c_target,
                    target_unique_name_df=target_unique_name_df,
                    rare_postings=rare_postings, rare_df_map=rare_df_map,
                    numeric_postings=numeric_postings, numeric_df_map=numeric_df_map,
                    name_cache=name_cache, addr_cache=addr_cache,
                    build_seconds_total=elapsed,
                )
                logger.info(
                    "[TargetContext] BUILT %s/%s in %.1fs. REUSE=YES for all shards.",
                    source_name, country, elapsed,
                )

    def get(self, source_name: str, country: str) -> Optional[SourceCountryContext]:
        return self._store.get((source_name, country))

    def has(self, source_name: str, country: str) -> bool:
        return (source_name, country) in self._store

    def countries(self) -> List[str]:
        return sorted(set(k[1] for k in self._store))

    def sources(self) -> List[str]:
        return sorted(set(k[0] for k in self._store))

    def summary(self) -> dict:
        out: dict = {}
        for (src, cty), ctx in self._store.items():
            out[f"{src}/{cty}"] = {
                "target_rows": len(ctx.target_df),
                "rare_tokens": len(ctx.rare_postings),
                "numeric_tokens": len(ctx.numeric_postings),
                "name_tfidf_shape": list(ctx.name_cache.X_target_cpu.shape) if ctx.name_cache else None,
                "addr_tfidf_shape": list(ctx.addr_cache.X_target_cpu.shape) if ctx.addr_cache else None,
                "name_gpu_uploaded": ctx.name_cache is not None and ctx.name_cache.X_target_gpu is not None,
                "addr_gpu_uploaded": ctx.addr_cache is not None and ctx.addr_cache.X_target_gpu is not None,
                "build_seconds_total": round(ctx.build_seconds_total, 2),
            }
        return out

    def release(self) -> None:
        """Free all GPU + CPU memory. Call at country-worker exit."""
        try:
            import cupy as cp
            for ctx in self._store.values():
                for ch in [ctx.name_cache, ctx.addr_cache]:
                    if ch is not None and ch.X_target_gpu is not None:
                        del ch.X_target_gpu
                        ch.X_target_gpu = None
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        self._store.clear()
        import gc
        gc.collect()
        logger.info("[TargetContext] Released all target context memory.")
