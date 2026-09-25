# HARD RULES

This file is a compact reminder of the non-negotiable competition and engineering rules for the Amazon ML Challenge 2026. Review this before every major architectural change.

1. **NO EXTERNAL DATA LOOKUP**
   - no Google/Maps/geocoding
   - no government/company databases
   - no external business enrichment
   - no commercial entity-resolution APIs

2. **MODEL CONSTRAINTS**
   - final model <= 8B parameters
   - license must be MIT or Apache 2.0 compatible
   - verify license before use

3. **OUTPUT INTEGRITY**
   - every test S1 entity exactly once
   - only valid S2/S3 IDs
   - no duplicate IDs
   - singleton = empty match list
   - final matches must be a subset of candidate_pairs.tsv

4. **METRIC**
   - official metric = macro entity-level F0.5
   - precision is weighted more than recall
   - singletons matter heavily
   - never optimize only pairwise accuracy/AUC

5. **BLOCKING**
   - candidate generation is part of the system
   - final recall cannot exceed blocking recall
   - candidate_pairs.tsv must be the exact candidate set used by the matcher

6. **VALIDATION**
   - split by Source 1 entity
   - prevent leakage
   - never tune on hidden/test labels
   - trust local validation over leaderboard chasing

7. **FRANCE / ZERO-SHOT**
   - train countries = US, India
   - test additionally includes France
   - never hard-code pipeline only for US/India
   - country is open-set

8. **SCIENTIFIC TRUTHFULNESS**
   - never fabricate scores/results
   - separate FACT / HYPOTHESIS / ASSUMPTION / INFERENCE
   - no improvement claims without measured evidence

9. **REPRODUCIBILITY**
   - raw data immutable
   - seed experiments
   - log configs
   - log git commit
   - preserve experiment history
   - never overwrite submissions

10. **CODE QUALITY**
    - production logic belongs in src/
    - notebooks only for EDA/analysis
    - critical logic must have tests
    - no silent fallbacks
    - no hidden magic constants

11. **SUBMISSIONS**
    - max 5 submissions per day
    - validate locally first
    - preserve every submitted version
    - do not spend submissions on trivial threshold guesses

12. **FINAL PRINCIPLE**
    - Correctness > leaderboard chasing
    - Evidence > intuition
    - Reproducibility > shortcuts
    - Competition integrity > score
