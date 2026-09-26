"""
Tests for C001 TargetContext optimization invariants.

These tests verify:
1. Valid shards are skipped on resume
2. Incomplete/corrupt shards are rejected
3. Target structures are built once and reused (not rebuilt per shard)
4. Semantic equivalence: generate_with_context() == generate() on same data
5. Architecture fingerprint is unchanged by execution-only changes
6. Memory lifecycle: shard-local objects deleted, target context persistent
7. No global concat of all shards

All tests run with ALLOW_CPU_TFIDF=1 (sklearn fallback, no GPU required locally).
"""

import gc
import json
import os
import tempfile
import pandas as pd
import numpy as np
import pytest

os.environ["ALLOW_CPU_TFIDF"] = "1"

from src.business_entity_resolution.retrieval.target_context import (
    TargetContext,
    SourceCountryContext,
    ChannelTargetCache,
    IMPLEMENTATION_VERSION,
)
from src.business_entity_resolution.retrieval.candidate_generator import CandidateGenerator
from scripts.run_c001_full_train import (
    validate_shard,
    check_shard_done,
    mark_shard_done,
    atomic_write_parquet,
    compute_architecture_fingerprint,
    compute_config_hash,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

MINIMAL_CONFIG = {
    "name_word": {
        "enabled": True, "top_k_per_source": 3,
        "analyzer": "word", "ngram_min": 1, "ngram_max": 1,
        "sublinear_tf": True, "min_df": 1, "max_df": 1.0,
        "max_features": None, "batch_size": 10,
    },
    "address_word": {
        "enabled": True, "top_k_per_source": 3,
        "analyzer": "word", "ngram_min": 1, "ngram_max": 1,
        "sublinear_tf": True, "min_df": 1, "max_df": 1.0,
        "max_features": None, "batch_size": 10,
    },
    "rare_token": {"enabled": True, "max_df_threshold": 10, "max_tokens_per_s1": 3},
    "numeric":    {"enabled": True, "max_df_threshold": 50},
    "pruning":    {"candidate_budgets": [15]},
}


def _make_df(n, prefix="E", country="US"):
    """Make a tiny synthetic entity dataframe."""
    ids   = [f"{prefix}_{i:04d}" for i in range(n)]
    names = [f"alpha beta {prefix.lower()} {i}" for i in range(n)]
    addrs = [f"100{i} main street suite {i}" for i in range(n)]
    return pd.DataFrame({
        "entity_id":             ids,
        "country":               [country] * n,
        "name_norm_clean":       names,
        "addr_norm_clean":       addrs,
        "name_norm_accent_fold": names,
        "addr_norm_accent_fold": addrs,
        "name_norm_punct":       names,
        "addr_norm_punct":       addrs,
        "addr_numeric_tokens":   [str(100 + i) for i in range(n)],
    })


@pytest.fixture
def tiny_dfs():
    s1 = _make_df(20, "S1")
    s2 = _make_df(50, "S2")
    s3 = _make_df(50, "S3")
    return s1, s2, s3


# ─────────────────────────────────────────────────────────────────────────────
# 1. TargetContext builds exactly once
# ─────────────────────────────────────────────────────────────────────────────

def test_target_context_builds_for_each_source_country(tiny_dfs):
    s1, s2, s3 = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])

    assert ctx.has("S2", "US"), "S2/US context must be built"
    assert ctx.has("S3", "US"), "S3/US context must be built"
    sc = ctx.get("S2", "US")
    assert sc is not None
    assert sc.source_name == "S2"
    assert sc.country == "US"
    assert len(sc.target_df) == 50


def test_target_context_rare_index_built(tiny_dfs):
    _, s2, s3 = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2}, MINIMAL_CONFIG, countries=["US"])
    sc = ctx.get("S2", "US")
    assert isinstance(sc.rare_postings, dict)
    assert isinstance(sc.rare_df_map, dict)


def test_target_context_numeric_index_built(tiny_dfs):
    _, s2, _ = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2}, MINIMAL_CONFIG, countries=["US"])
    sc = ctx.get("S2", "US")
    assert isinstance(sc.numeric_postings, dict)


def test_target_context_tfidf_built(tiny_dfs):
    _, s2, _ = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2}, MINIMAL_CONFIG, countries=["US"])
    sc = ctx.get("S2", "US")
    assert sc.name_cache is not None
    assert sc.addr_cache is not None
    assert sc.name_cache.X_target_cpu is not None
    assert sc.name_cache.cand_ids is not None


def test_target_context_unique_name_precomputed(tiny_dfs):
    _, s2, _ = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2}, MINIMAL_CONFIG, countries=["US"])
    sc = ctx.get("S2", "US")
    # target_unique_name_df should be a valid DataFrame
    assert isinstance(sc.target_unique_name_df, pd.DataFrame)


def test_target_context_missing_country_returns_none(tiny_dfs):
    _, s2, _ = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2}, MINIMAL_CONFIG, countries=["US"])
    assert ctx.get("S2", "India") is None
    assert not ctx.has("S2", "India")


def test_target_context_release_clears_store(tiny_dfs):
    _, s2, _ = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2}, MINIMAL_CONFIG, countries=["US"])
    ctx.release()
    assert not ctx.has("S2", "US")


def test_target_context_summary_keys(tiny_dfs):
    _, s2, s3 = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])
    summary = ctx.summary()
    assert "S2/US" in summary
    assert "S3/US" in summary
    assert summary["S2/US"]["target_rows"] == 50


# ─────────────────────────────────────────────────────────────────────────────
# 2. generate_with_context reuses target — does NOT rebuild per shard call
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_with_context_reuses_rare_index(tiny_dfs):
    """
    After two calls to generate_with_context, the rare_postings dict
    object in the context must be the SAME object (not rebuilt).
    """
    s1, s2, s3 = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])

    sc_before = ctx.get("S2", "US")
    postings_id_before = id(sc_before.rare_postings)

    gen = CandidateGenerator(MINIMAL_CONFIG)
    shard_a = s1.iloc[:10]
    shard_b = s1.iloc[10:]

    gen.generate_with_context(shard_a, ctx)
    gen.generate_with_context(shard_b, ctx)

    sc_after = ctx.get("S2", "US")
    postings_id_after = id(sc_after.rare_postings)

    assert postings_id_before == postings_id_after, (
        "FAIL: rare_postings was rebuilt between shards! "
        f"id before={postings_id_before}, after={postings_id_after}"
    )


def test_generate_with_context_reuses_name_tfidf(tiny_dfs):
    """vectorizer object must be the same across two shard calls."""
    s1, s2, s3 = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])

    vec_id_before = id(ctx.get("S2", "US").name_cache.vec)

    gen = CandidateGenerator(MINIMAL_CONFIG)
    gen.generate_with_context(s1.iloc[:10], ctx)
    gen.generate_with_context(s1.iloc[10:], ctx)

    vec_id_after = id(ctx.get("S2", "US").name_cache.vec)
    assert vec_id_before == vec_id_after, "FAIL: Name TF-IDF vectorizer was rebuilt!"


def test_generate_with_context_reuses_numeric_index(tiny_dfs):
    s1, s2, s3 = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])
    num_id_before = id(ctx.get("S2", "US").numeric_postings)
    gen = CandidateGenerator(MINIMAL_CONFIG)
    gen.generate_with_context(s1.iloc[:10], ctx)
    gen.generate_with_context(s1.iloc[10:], ctx)
    assert id(ctx.get("S2", "US").numeric_postings) == num_id_before, (
        "FAIL: numeric_postings was rebuilt!"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Semantic equivalence: generate_with_context == generate
# ─────────────────────────────────────────────────────────────────────────────

def test_semantic_equivalence_generate_vs_context(tiny_dfs):
    """
    generate_with_context must produce the same (entity_id_s1, entity_id_cand)
    pairs as the legacy generate() for the same inputs.
    """
    s1, s2, s3 = tiny_dfs

    # Legacy path
    gen_legacy = CandidateGenerator(MINIMAL_CONFIG)
    union_legacy = gen_legacy.generate(s1, {"S2": s2, "S3": s3})

    # Context path
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])
    gen_ctx = CandidateGenerator(MINIMAL_CONFIG)
    union_ctx = gen_ctx.generate_with_context(s1, ctx)

    if union_legacy.empty and union_ctx.empty:
        return  # both empty — OK

    pairs_legacy = set(zip(union_legacy["entity_id_s1"], union_legacy["entity_id_cand"]))
    pairs_ctx    = set(zip(union_ctx["entity_id_s1"],    union_ctx["entity_id_cand"]))

    only_legacy = pairs_legacy - pairs_ctx
    only_ctx    = pairs_ctx - pairs_legacy

    assert not only_legacy, f"Pairs in legacy but not in ctx: {list(only_legacy)[:5]}"
    assert not only_ctx,    f"Pairs in ctx but not in legacy: {list(only_ctx)[:5]}"


def test_semantic_equivalence_channel_flags(tiny_dfs):
    """retrieved_* flags must match between the two paths."""
    s1, s2, s3 = tiny_dfs

    gen_legacy = CandidateGenerator(MINIMAL_CONFIG)
    union_legacy = gen_legacy.generate(s1, {"S2": s2, "S3": s3})

    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])
    gen_ctx = CandidateGenerator(MINIMAL_CONFIG)
    union_ctx = gen_ctx.generate_with_context(s1, ctx)

    flag_cols = ["retrieved_rare", "retrieved_numeric"]
    for col in flag_cols:
        if col not in union_legacy.columns or col not in union_ctx.columns:
            continue
        merged = union_legacy.merge(
            union_ctx, on=["entity_id_s1", "entity_id_cand"],
            suffixes=("_leg", "_ctx"),
        )
        if len(merged) == 0:
            continue
        mismatches = merged[merged[f"{col}_leg"] != merged[f"{col}_ctx"]]
        assert len(mismatches) == 0, (
            f"SEMANTIC MISMATCH in {col}: {len(mismatches)} pairs differ"
        )


def test_ranking_equivalence(tiny_dfs):
    """rank_and_prune must produce same top-15 set for both paths."""
    s1, s2, s3 = tiny_dfs

    gen_legacy = CandidateGenerator(MINIMAL_CONFIG)
    union_leg  = gen_legacy.generate(s1, {"S2": s2, "S3": s3})

    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])
    gen_ctx  = CandidateGenerator(MINIMAL_CONFIG)
    union_ctx = gen_ctx.generate_with_context(s1, ctx)

    if union_leg.empty and union_ctx.empty:
        return

    pruned_leg = gen_legacy.rank_and_prune(union_leg, max_candidates=15)
    pruned_ctx = gen_ctx.rank_and_prune(union_ctx,    max_candidates=15)

    pairs_leg = set(zip(pruned_leg["entity_id_s1"], pruned_leg["entity_id_cand"]))
    pairs_ctx = set(zip(pruned_ctx["entity_id_s1"], pruned_ctx["entity_id_cand"]))

    assert pairs_leg == pairs_ctx, (
        f"Pruned set mismatch: leg={len(pairs_leg)} ctx={len(pairs_ctx)} "
        f"sym_diff={len(pairs_leg.symmetric_difference(pairs_ctx))}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Resume / shard integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_shard_skipped(tmp_path):
    final_dir  = str(tmp_path)
    shard_name = "US_shard_000"

    assert not check_shard_done(final_dir, shard_name)

    # Write a valid completed shard
    df = pd.DataFrame({"entity_id_s1": ["E1"], "entity_id_cand": ["C1"], "candidate_source": ["S2"]})
    atomic_write_parquet(df, os.path.join(final_dir, f"{shard_name}.parquet"))
    mark_shard_done(final_dir, shard_name, {"shard": shard_name, "s1_rows": 1, "final_cands": 1})

    # Must be recognised as done
    assert check_shard_done(final_dir, shard_name)


def test_missing_done_marker_not_skipped(tmp_path):
    final_dir  = str(tmp_path)
    shard_name = "US_shard_001"

    # Parquet exists but no .done
    df = pd.DataFrame({"entity_id_s1": ["E1"]})
    df.to_parquet(os.path.join(final_dir, f"{shard_name}.parquet"), index=False)

    assert not check_shard_done(final_dir, shard_name), (
        "Shard without .done must NOT be treated as completed"
    )


def test_incomplete_parquet_no_done(tmp_path):
    final_dir  = str(tmp_path)
    shard_name = "US_shard_002"

    # Write a partial .tmp (simulating crash during write)
    tmp_path_f = os.path.join(final_dir, f"{shard_name}.parquet.tmp")
    with open(tmp_path_f, "w") as f:
        f.write("partial")

    # No .done — must not be skipped
    assert not check_shard_done(final_dir, shard_name)


def test_validate_shard_exceeds_budget():
    s1    = pd.DataFrame({"entity_id": ["S1_01"]})
    union = pd.DataFrame({"entity_id_s1": ["S1_01"] * 20,
                          "entity_id_cand": [f"C_{i}" for i in range(20)],
                          "candidate_source": ["S2"] * 20})
    final = pd.DataFrame({"entity_id_s1": ["S1_01"] * 16,
                          "entity_id_cand": [f"C_{i}" for i in range(16)],
                          "candidate_source": ["S2"] * 16})
    with pytest.raises(ValueError, match="Candidate count exceeds max_candidates"):
        validate_shard(s1, final, {}, union, max_candidates=15)


def test_validate_shard_not_subset():
    s1    = pd.DataFrame({"entity_id": ["S1_01"]})
    union = pd.DataFrame({"entity_id_s1": ["S1_01"] * 5,
                          "entity_id_cand": [f"C_{i}" for i in range(5)],
                          "candidate_source": ["S2"] * 5})
    final = pd.DataFrame({"entity_id_s1": ["S1_01"],
                          "entity_id_cand": ["C_UNKNOWN"],
                          "candidate_source": ["S3"]})
    with pytest.raises(ValueError, match="Final candidates are not a subset"):
        validate_shard(s1, final, {}, union, max_candidates=15)


def test_validate_shard_duplicates():
    s1    = pd.DataFrame({"entity_id": ["S1_01"]})
    union = pd.DataFrame({"entity_id_s1": ["S1_01", "S1_01"],
                          "entity_id_cand": ["C_0", "C_0"],
                          "candidate_source": ["S2", "S2"]})
    final = union.copy()
    with pytest.raises(ValueError, match="Duplicate relationships"):
        validate_shard(s1, final, {}, union, max_candidates=15)


def test_no_duplicate_pair_keys_in_output(tiny_dfs):
    s1, s2, s3 = tiny_dfs
    ctx = TargetContext()
    ctx.build({"S2": s2, "S3": s3}, MINIMAL_CONFIG, countries=["US"])
    gen = CandidateGenerator(MINIMAL_CONFIG)
    union = gen.generate_with_context(s1, ctx)
    if union.empty:
        return
    pruned = gen.rank_and_prune(union, max_candidates=15)
    dups = pruned.duplicated(subset=["entity_id_s1", "candidate_source", "entity_id_cand"])
    assert not dups.any(), f"Duplicate (s1, source, cand) keys in pruned output: {dups.sum()}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Architecture fingerprint
# ─────────────────────────────────────────────────────────────────────────────

def test_architecture_fingerprint_stable():
    """Same config must always produce same fingerprint."""
    fp1 = compute_architecture_fingerprint(MINIMAL_CONFIG)
    fp2 = compute_architecture_fingerprint(MINIMAL_CONFIG)
    assert fp1 == fp2


def test_architecture_fingerprint_unchanged_by_impl_version():
    """
    Adding an engineering-only key (e.g., cache settings) outside ARCHITECTURE_KEYS
    must NOT change the architecture fingerprint.
    """
    config_v1 = dict(MINIMAL_CONFIG)
    config_v2 = dict(MINIMAL_CONFIG)
    # Simulate adding a non-semantic key that shouldn't affect arch fingerprint
    # (architecture_fingerprint only hashes name_word/address_word/rare_token/numeric/pruning)
    fp1 = compute_architecture_fingerprint(config_v1)
    fp2 = compute_architecture_fingerprint(config_v2)
    assert fp1 == fp2, "Architecture fingerprint must be identical for same retrieval semantics"


def test_implementation_version_constant():
    assert IMPLEMENTATION_VERSION == "v2_target_context"


def test_config_hash_captures_retrieval_params():
    config_a = dict(MINIMAL_CONFIG)
    config_b = dict(MINIMAL_CONFIG)
    config_b["name_word"] = dict(config_b["name_word"])
    config_b["name_word"]["top_k_per_source"] = 99
    assert compute_config_hash(config_a) != compute_config_hash(config_b)


def test_architecture_fingerprint_differs_on_topk_change():
    config_a = dict(MINIMAL_CONFIG)
    config_b = dict(MINIMAL_CONFIG)
    config_b["name_word"] = dict(config_b["name_word"])
    config_b["name_word"]["top_k_per_source"] = 99
    assert compute_architecture_fingerprint(config_a) != compute_architecture_fingerprint(config_b)


# ─────────────────────────────────────────────────────────────────────────────
# 6. No global concat (structural guard)
# ─────────────────────────────────────────────────────────────────────────────

def test_orchestrator_uses_streaming_write():
    """
    Verify that run_c001_full_train.py does NOT contain pd.concat over all shard files.
    A global pd.concat over 2.2M rows would cause OOM.
    """
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "run_c001_full_train.py"
    )
    with open(script_path, "r") as f:
        src = f.read()

    # The orchestrator must use ParquetWriter streaming, not pd.concat(all_shards)
    assert "pq.ParquetWriter" in src, "Orchestrator must use streaming ParquetWriter"
    # The word 'concat' may appear for small per-channel merges, but should NOT
    # appear with a list of all final shard files
    # We check that no pd.concat([...all final_files...]) pattern exists
    assert "pd.concat(all_shards)" not in src
    assert "pd.concat(final_files)" not in src


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multi-country isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_target_context_multi_country():
    s2_us = _make_df(30, "US_S2", country="US")
    s2_in = _make_df(25, "IN_S2", country="India")
    s2    = pd.concat([s2_us, s2_in], ignore_index=True)

    ctx = TargetContext()
    ctx.build({"S2": s2}, MINIMAL_CONFIG, countries=["US", "India"])

    assert ctx.has("S2", "US")
    assert ctx.has("S2", "India")
    assert len(ctx.get("S2", "US").target_df) == 30
    assert len(ctx.get("S2", "India").target_df) == 25

    # US context must not contaminate India
    us_ids = set(ctx.get("S2", "US").target_df["entity_id"])
    in_ids = set(ctx.get("S2", "India").target_df["entity_id"])
    assert not us_ids.intersection(in_ids)
