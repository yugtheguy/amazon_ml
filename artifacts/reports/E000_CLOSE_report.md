# E000-CLOSE — RETRIEVAL METRIC AND ARTIFACT AUDIT

## A. Integrity
- test S1 count: 1,732,544
- candidate file row count: 1,732,544
- matching file row count: 1,732,544
- validator result: `PASS — no blocking issues found. Safe to submit.`
- test-suite result: `pytest tests/ -v` passed (78/78).

## B. Candidate Volume
- total candidate pairs: 985,928
- S2: 531,539
- S3: 454,389
- mean: 0.447
- median: 0.0
- P90: 1.0
- P95: 2.0
- P99: 2.0
- max: 5.0
- zero-candidate %: 61.80%

## C. Retrieval Ceiling
*(Zero-match S1 entities do not have GT pairs and are excluded from recall calculation.)*
- pair candidate recall overall: 9.32%
- S2 / S3: 10.29% / 8.41%
- US / India: 10.76% / 7.17%
- single / multi: 11.86% / 9.28%

- full GT coverage overall: 0.95%
- US / India: 1.16% / 0.64%
- single / multi: 11.86% / 0.29%
- S2-only / S3-only / BOTH: 5.19% / 4.20% / 0.31%

## D. ZERO_MATCH Behavior
- zero candidate %: 86.18%
- >=1 candidate %: 13.82%
- mean candidates: 0.159
- P95/P99: 1.0 / 2.0

## E. Channel Contribution

| Channel | Candidate pairs | Unique candidates | GT recovered | Unique GT rescued |
|---|---|---|---|---|
| exact_name_address_clean | 62,952 | 0 | 62,952 | 0 |
| exact_name_address_accent | 77,092 | 14,139 | 77,092 | 14,139 |
| exact_name_address_punct | 89,371 | 20,838 | 89,371 | 20,838 |
| unique_exact_name | 921,716 | 882,417 | 647,730 | 608,431 |

## F. Channel Overlap
- exactly 1 channel: 917,394
- exactly 2 channels: 5,582
- exactly 3 channels: 29,235
- all applicable channels: 33,717

## G. Exact-Name Collision Statistics

**Source 2 (name_norm_clean):**
- mapping to 1 target: 3,868,742
- mapping to >1 target: 339,262
- P50 multiplicity: 1.0
- P95 multiplicity: 2.0
- P99 multiplicity: 5.0
- max multiplicity: 464

**Source 3 (name_norm_clean):**
- mapping to 1 target: 4,152,916
- mapping to >1 target: 327,428
- P50 multiplicity: 1.0
- P95 multiplicity: 2.0
- P99 multiplicity: 5.0
- max multiplicity: 462

*(Note: Generic country equality is used as part of the lookup key for these exact-name counts, restricting collisions to within the same country.)*

## H. D0 vs D1 Detailed

### Overall
- D0 macro F0.5: 0.0839
- D1 macro F0.5: 0.0944

### Per fold (D1)
- fold 0: 0.0946
- fold 1: 0.0948
- fold 2: 0.0943
- fold 3: 0.0944
- fold 4: 0.0941

### Country (D1)
- US: 0.1039
- India: 0.0803

### Match cardinality (D1)
- ZERO_MATCH: 0.9924
- SINGLE_MATCH: 0.0204
- MULTI_MATCH: 0.0426

**Status of D1**: `E000_BASELINE_DECISION`

## I. Country Audit
> Generic country equality is empirically fully supported by the observed training positive pairs and is used as a retrieval partition.

*(Measured GT pair count: 7,638,365 | Country mismatch count: 0. Note: Country handling is simple string equality without a hardcoded US/India whitelist.)*

## J. Missed GT
- count: 6,926,423
- artifact path: `artifacts/retrieval/E000/missed_gt_pairs.parquet`

## K. Experiment Log
Confirmed exactly one E000 entry is present in `experiments/experiment_log.csv`.

## L. Interpretation
1. **What exact retrieval captures well:** Identical text formatting and highly regular, completely normalized entity inputs.
2. **What it misses:** Any entity with minor typographical variations (swapped words, missing keywords, misspellings, differing conventions like "Ltd" vs "Limited"), which explains why ~6.9 million pairs (90.7%) are missed.
3. **Whether unique-name missing-address rescue helped:** Yes, rescuing entities that had a globally unique name but missed the address marginally improved overall F0.5 from 0.0839 to 0.0944, proving that sparse addresses are a significant bottleneck when using strict matching.
4. **Where candidate collision occurs:** Common entity names (even when restricted by country) map to massive numbers of true candidates across regions, with some keys pointing to over 450 distinct records. This explains why unrestricted exact-name retrieval would cause an explosion of false candidates.
5. **Why fuzzy name retrieval is the logical next experiment:** The low overall pair recall (9.32%) indicates strict exact matching is far too restrictive. A statistical, token-level matching mechanism like TF-IDF is required to retrieve variations.

## M. NEXT STEP
`E001 — NAME CHARACTER TF-IDF RETRIEVAL`
