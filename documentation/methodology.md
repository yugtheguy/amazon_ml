"""Methodology description for the Business Entity Resolution system."""

# Architecture (Conceptual — Locked)

## Pipeline Overview

```
RAW DATA
→ MULTI-VIEW NORMALIZATION
→ HYBRID RETRIEVAL (candidate generation)
→ CANDIDATE UNION
→ HIGH-RECALL CANDIDATE POOL
→ MULTILINGUAL CROSS-ENCODER (matching/scoring)
→ OOF CALIBRATION
→ ENTITY-LEVEL NO-MATCH / ANY-MATCH GATE
→ F0.5-OPTIMIZED SET DECISION
→ SUBMISSION
```

## Current Implementation Status

### P0 Foundation (IMPLEMENTED, TESTED)
- Permanent 5-fold Source-1-level fold manifest with stratification
- Multi-view text normalization (Unicode, case-fold, punctuation, accent-fold, numeric tokens)
- Versioned processed dataset layer (Parquet)
- Submission building and validation infrastructure
- Entity-level macro F0.5 metric implementation
- All components unit-tested

### Future Components (NOT YET IMPLEMENTED)
- E000: Exact / deterministic retrieval baseline
- E001+: Fuzzy similarity baseline
- Blocking / candidate generation
- Feature engineering
- Model training
- Threshold optimization
- Error analysis
