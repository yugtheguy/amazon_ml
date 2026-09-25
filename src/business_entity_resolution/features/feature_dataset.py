"""
C003 Explicit Feature Dataset Builder

Processes canonical C002 pair shards.
Loads minimal required text from S1/S2/S3.
Computes C003A pair-local features.
Computes C003B relative features.
"""

import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.business_entity_resolution.features.schema import (
    C003_FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    write_c003_schema_contract,
    validate_leakage_firewall
)
from src.business_entity_resolution.features.source_lookup import SourceTextLookup
from src.business_entity_resolution.features.pair_features import compute_pair_local_features
from src.business_entity_resolution.features.relative_features import compute_relative_features
from src.business_entity_resolution.features.validation import (
    check_shard_resumable,
    write_shard_done,
    validate_shard_s1_isolation,
    validate_feature_schema,
    sha256_file
)
from src.business_entity_resolution.utils.logging import get_logger

logger = get_logger(__name__)

def process_shard(
    c002_shard_path: Path,
    shard_name: str,
    text_lookup: SourceTextLookup,
    global_s1_seen: Set[str]
) -> pd.DataFrame:
    """Processes a single C002 shard into a C003 feature shard."""
    
    # 1. Read C002 Base Pairs
    df = pd.read_parquet(c002_shard_path)
    if df.empty:
        return df
        
    # Isolation check
    s1_ids_in_shard = set(df["entity_id_s1"].unique())
    validate_shard_s1_isolation(s1_ids_in_shard, global_s1_seen, shard_name)
    
    # Preserve C001 columns that we might need to carry over (provenance)
    c001_cols = list(df.columns)
    
    # 2. Join Text (Vectorized, batched via memory lookup)
    # We do NOT use df.merge over raw files, we use reindex from the lookup table
    # This prevents expanding memory footprint and avoids file I/O
    
    s1_text = text_lookup.get_s1_text(df["entity_id_s1"])
    cand_text = text_lookup.get_cand_text(df["entity_id_cand"], df["candidate_source"])
    
    # Temporarily append text columns for C003A computation
    df["name_s1"] = s1_text["name_norm_clean"].values
    df["address_s1"] = s1_text["addr_norm_clean"].values
    df["numeric_tokens_s1"] = s1_text["numeric_tokens"].values
    
    df["name_cand"] = cand_text["name_norm_clean"].values
    df["address_cand"] = cand_text["addr_norm_clean"].values
    df["numeric_tokens_cand"] = cand_text["numeric_tokens"].values
    
    # Clean up intermediate arrays
    del s1_text
    del cand_text
    gc.collect()
    
    # 3. Compute C003A (Pair-Local)
    pair_feat = compute_pair_local_features(df, c001_cols)
    
    # 4. Compute C003B (Relative)
    rel_feat = compute_relative_features(df, pair_feat)
    
    # 5. Drop Raw Text
    cols_to_drop = [
        "name_s1", "address_s1", "numeric_tokens_s1",
        "name_cand", "address_cand", "numeric_tokens_cand"
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # 6. Assemble Output
    # Exclude provenance columns from df since they are now in pair_feat
    # (actually we can just select the preserved non-feature columns from df)
    # Target and metadata
    preserve_cols = ["entity_id_s1", "candidate_source", "entity_id_cand", "fold", "label"]
    if "country" in df.columns:
        preserve_cols.append("country")
        
    out_df = pd.concat([df[preserve_cols], pair_feat, rel_feat], axis=1)
    
    # 7. Validation
    validate_feature_schema(out_df, shard_name)
    
    return out_df


def build_c003_feature_dataset(cfg: dict) -> None:
    t_start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("C003 FEATURE DATASET BUILDER — START")
    logger.info("=" * 60)
    
    # Validate firewall immediately
    validate_leakage_firewall()
    
    # Paths
    c002_dir = Path(cfg["c002_pair_dataset_dir"])
    s1_path = cfg["processed_s1_path"]
    s2_path = cfg["processed_s2_path"]
    s3_path = cfg["processed_s3_path"]
    out_dir = Path(cfg["output_dir"])
    
    schema_dir = out_dir / "schema"
    shard_dir = out_dir / "shards"
    manifest_dir = out_dir / "manifest"
    diag_dir = out_dir / "diagnostics"
    package_dir = out_dir / "package"
    
    for d in [schema_dir, shard_dir, manifest_dir, diag_dir, package_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Write Schema Contract
    schema_path = schema_dir / "c003_feature_schema_v1.json"
    write_c003_schema_contract(schema_path)
    
    config_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    
    # Load Source Text mapping
    lookup = SourceTextLookup()
    lookup.load_sources(s1_path, s2_path, s3_path)
    
    global_s1_seen: Set[str] = set()
    shard_files = sorted(c002_dir.glob("*.parquet"))
    
    total_in = 0
    total_out = 0
    total_pos = 0
    
    for shard_file in shard_files:
        shard_name = shard_file.stem
        if shard_name.startswith("_"):
            continue
            
        c002_fingerprint = sha256_file(shard_file)
        
        if check_shard_resumable(shard_dir, shard_name, config_hash, c002_fingerprint, C003_FEATURE_SCHEMA_VERSION):
            logger.info(f"[C003] Shard {shard_name} already DONE. Skipping.")
            # Accumulate counts
            meta = json.load(open(shard_dir / f"{shard_name}.metadata.json"))
            total_in += meta["input_rows"]
            total_out += meta["output_rows"]
            total_pos += meta["positive_rows"]
            
            # Important: still need to update global_s1_seen for subsequent checks
            df_s1 = pd.read_parquet(shard_dir / f"{shard_name}.parquet", columns=["entity_id_s1"])
            validate_shard_s1_isolation(set(df_s1["entity_id_s1"].unique()), global_s1_seen, shard_name)
            continue
            
        t0 = time.time()
        try:
            out_df = process_shard(shard_file, shard_name, lookup, global_s1_seen)
        except Exception as e:
            logger.error(f"[C003] Shard {shard_name} FAILED: {e}")
            raise
            
        n_in = len(pd.read_parquet(shard_file, columns=["entity_id_s1"]))
        n_out = len(out_df)
        n_pos = int(out_df["label"].sum()) if not out_df.empty else 0
        
        if n_in != n_out:
            raise ValueError(f"CRITICAL: Row count mismatch! In: {n_in}, Out: {n_out}. C003 must preserve all rows.")
            
        # Atomic Write
        tmp_path = shard_dir / f"_{shard_name}.tmp.parquet"
        out_path = shard_dir / f"{shard_name}.parquet"
        
        out_df.to_parquet(tmp_path, index=False, compression="snappy")
        os.replace(tmp_path, out_path)
        
        elapsed = time.time() - t0
        meta = {
            "shard": shard_name,
            "schema_version": C003_FEATURE_SCHEMA_VERSION,
            "config_hash": config_hash,
            "c002_fingerprint": c002_fingerprint,
            "input_rows": n_in,
            "output_rows": n_out,
            "positive_rows": n_pos,
            "elapsed_s": round(elapsed, 2)
        }
        write_shard_done(shard_dir, shard_name, meta)
        
        total_in += n_in
        total_out += n_out
        total_pos += n_pos
        
        logger.info(f"[C003] Shard {shard_name} DONE in {elapsed:.1f}s | in={n_in:,} out={n_out:,} pos={n_pos:,}")
        
        del out_df
        gc.collect()
        
    lookup.clear()
    
    # Consolidate package
    package_path = package_dir / "c003_features_v1.parquet"
    writer = None
    for shard_file in sorted(shard_dir.glob("*.parquet")):
        if shard_file.stem.startswith("_"):
            continue
        df = pd.read_parquet(shard_file)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(package_path, table.schema, compression="snappy")
        writer.write_table(table)
        del df, table
        gc.collect()
    if writer:
        writer.close()
        
    # Diagnostics
    diag = {
        "total_input_rows": total_in,
        "total_output_rows": total_out,
        "total_positive_rows": total_pos,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "row_preservation": total_in == total_out
    }
    with open(diag_dir / "c003_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
        
    logger.info("=" * 60)
    logger.info("C003 FEATURE DATASET BUILDER — COMPLETE")
    logger.info(f"  Rows:      {total_out:,}")
    logger.info(f"  Positives: {total_pos:,}")
    logger.info(f"  Features:  {len(MODEL_FEATURE_COLUMNS)}")
    logger.info("=" * 60)
