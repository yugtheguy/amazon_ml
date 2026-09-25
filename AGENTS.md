# Amazon ML Challenge 2026 - Agent Instructions

This document preserves the comprehensive instructions, constraints, and requirements provided for the primary ML engineering and research coding agent.

The agent must operate under strict scientific, engineering, reproducibility, competition-integrity, and repository-maintenance rules to build a competition-grade, reproducible, auditable, mathematically correct Business Entity Resolution system.

## 1. COMPETITION CONTEXT
The task is Business Entity Resolution across 3 independent sources.
Source 1 is the deduplicated reference source. For every Source 1 entity, we must identify zero, one, or multiple matching entity_ids from Source 2 and Source 3 that refer to the same real-world business.
(See full context in the root rules.)

## 2. OFFICIAL OUTPUT REQUIREMENTS
The system must produce: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
`candidate_pairs.tsv` must contain the EXACT FINAL CANDIDATE SET.

## 3. OFFICIAL MODEL AND FAIR-PLAY CONSTRAINTS
- Max 8 billion parameters.
- MIT or Apache 2.0 license compatible.

## 4. STRICTLY PROHIBITED EXTERNAL DATA USE
No Google Search, Maps, geocoding APIs, or external business databases.

## 5. SCIENTIFIC TRUTHFULNESS RULE
Never fabricate results. Separate FACT, HYPOTHESIS, INFERENCE, ASSUMPTION.

## 6. MATHEMATICAL CORRECTNESS RULE
Exact macro F_beta (beta = 0.5) at the entity level. Singletons heavily influence the score.

## 7. DATA LEAKAGE RULES
Split validation at Source 1 entity level. No tuning on the test set.

## 8. VALIDATION DESIGN RULES
Every serious model experiment must be evaluated through the full pipeline. Track blocking recall, pair metrics, and entity-level macro F0.5.

## 9. ZERO-SHOT FRANCE RULE
France appears only in test. Treat country as an open-set string. Do not hard-code logic for only US/India.

## 10. CANDIDATE GENERATION RULES
Candidate generation is a first-class ML component. Measure its recall, reduction ratio, and speed.

## 11. FEATURE ENGINEERING RULES
Deterministic and reproducible features with descriptive names.

## 12. NORMALIZATION RULES
Conservative normalization. Never overwrite the raw data.

## 13. HARD NEGATIVE MINING RULES
Create informative negatives from the blocking stage. Track negative sampling method.

## 14. SINGLETON-AWARE DECISION RULE
A no-match prediction is a legitimate prediction. Thresholds must be tuned for macro F0.5.

## 15. MODEL DEVELOPMENT PHILOSOPHY
Start simple (baseline exact/rule, TF-IDF, simple ML models) before jumping to large transformers.

## 16. REPOSITORY STRUCTURE
Adhere strictly to the provided folder structure. Do not change without explanation.

## 17. RAW DATA IMMUTABILITY
Files under `data/raw/` must NEVER be modified.

## 18. EXPERIMENT TRACKING
Log all experiments in `experiments/experiment_log.csv`.

## 19. RANDOMNESS AND REPRODUCIBILITY
Explicit seeds everywhere.

## 20. CONFIGURATION RULE
No magic numbers in notebooks. Use `configs/`.

## 21. NOTEBOOK RULE
Notebooks for EDA/analysis. Production logic goes to `src/`.

## 22. CODE QUALITY RULE
Production-quality Python. Type hints, docstrings, modular code.

## 23. TESTING RULE
Mandatory tests for metric correctness, loading, normalization, and validations.

## 24. PERFORMANCE RULE
Avoid O(N*M) loops. Use sparse matrices, FAISS, vectorized operations.

## 25. MEMORY RULE
Never materialize full Cartesian products unless safe.

## 26. KAGGLE / AWS ENVIRONMENT RULE
Portable code, project-relative paths.

## 27. SUBMISSION VERSIONING
Never overwrite submissions. Keep full submission history in `submissions/`.

## 28. PUBLIC LEADERBOARD RULE
Trust local CV over public leaderboard.

## 29. ERROR ANALYSIS RULE
Classify false positives and negatives to guide the next experiment.

## 30. DOCUMENTATION RULE
Maintain `documentation/` continuously.

## 31. GIT RULES
Meaningful commits. Do not commit large files or secrets.

## 32. AGENT BEHAVIOR RULES
State intentions, validate, report actual results. Do not silently delete working code.

## 33. NO FAKE COMPLETENESS
Use clear status updates: IMPLEMENTED, TESTED, VALIDATED, NOT YET VALIDATED.

## 34. NO SILENT FALLBACKS
Fail clearly if missing dependencies.

## 35. DATA ASSERTIONS
Add assertions for critical assumptions.

## 36. BASELINE PRESERVATION
Keep a simple, reproducible baseline working.

## 37. ABLATION DISCIPLINE
Change one major thing at a time.

## 38. CLAIM DISCIPLINE
Tie technical conclusions to evidence.

## 39. RESEARCH IMPLEMENTATION RULE
Document deviations from original methods.

## 40. FINAL PIPELINE REQUIREMENT
Support one clear end-to-end execution path.

## 41. FINAL PACKAGE COMPATIBILITY
Trivial production of `<team_name>_submission.zip`.

## 42. RESPONSE FORMAT WHEN WORKING
PLAN, FILES, IMPLEMENTATION, VALIDATION, RESULTS, RISKS/LIMITATIONS, NEXT RECOMMENDED EXPERIMENT.

## 43. FINAL PRINCIPLE
Correctness > leaderboard chasing. Evidence > intuition. Reproducibility > shortcuts. Competition integrity > score.
