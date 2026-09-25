# E000 - Exact / Deterministic Retrieval + Conservative Rule Decision Baseline

## 1. P0 Architecture & Compliance Checklist
✅ **Processed data generation complete** - all `v001` Parquet files (`train_source1`, `train_source2`, `train_source3`, `test_source1`, `test_source2`, `test_source3`) and manifest are successfully built.
✅ **Country Consistency Audit** - Measured ground-truth pairs across sources. Result: `country_mismatch_gt_pairs: 0`. The generic country equality heuristic is mathematically sound and is enforced.
✅ **E000 Scope Validation** - Enforced EXACT retrieval using clean, accent-folded, and punctuation-removed exact matching, as well as unique name exact matching. NO fuzzy matching or ML components were included, strictly complying with the baseline experiment definition.

## 2. Evaluation Results (Overall)
*The results from E000 provide our first true performance benchmark.*
- **D0 Policy (Strict Exact Match):** Macro F0.5 = 0.0839
- **D1 Policy (D0 + Address Missing Rescue):** Macro F0.5 = 0.0944
- **Selected Policy:** **D1**

## 3. Submission Artifacts
The official submission validator (`validate_submission.py`) was executed on the generated outputs with ID-checking enabled.
- **`candidate_pairs.tsv`:** Generated and validated (1,732,544 rows).
- **`matching_results.tsv`:** Generated and validated (1,732,544 rows).
- **Validator Result:** `PASS — no blocking issues found. Safe to submit.`

## 4. Testing & Code Quality
- **Unit Tests (`tests/test_e000.py`)**: Authored tests verifying exact deduplication, collision rules, channel flagging, and missing-address logic. All passed.
- **Global Test Suite**: 78/78 tests passed.
- **Experiment Log**: Appended E000 baseline results to `experiments/experiment_log.csv`.

## 5. Next Recommended Experiment
- **E001 - Name Character TF-IDF Retrieval**: The next sequential step in the Hybrid Retrieval strategy is introducing a TF-IDF matching approach to capture approximate name similarity and drastically boost pair candidate recall over the exact-match baseline.
