# Architecture

## Data Flow

```
data/raw/train/*.tsv    ──┐
data/raw/test/*.tsv     ──┤
                          │
                          ▼
        src/business_entity_resolution/normalization/text.py
        (Multi-view normalization: unicode, casefold, punct, accent-fold, numeric)
                          │
                          ▼
        data/processed/v001/*.parquet
        (Versioned processed datasets with manifest.json)
                          │
                          ▼
        [FUTURE: blocking / candidate generation]
                          │
                          ▼
        [FUTURE: feature engineering]
                          │
                          ▼
        [FUTURE: model scoring / thresholding]
                          │
                          ▼
        output/matching_results.tsv
        output/candidate_pairs.tsv
```

## Fold System

```
data/raw/train/train_source1.tsv  ──┐
data/raw/train/train_ground_truth.tsv──┤
                                      │
                                      ▼
        src/business_entity_resolution/data/folds.py
        (Stratified 5-fold split at Source-1 entity level)
                                      │
                                      ▼
        artifacts/folds/folds_v1.parquet
        artifacts/folds/folds_v1_metadata.json
```

## Module Map

| Module | Purpose |
|--------|---------|
| `data/load.py` | Raw TSV loading |
| `data/validation.py` | Schema & ID validation |
| `data/schema.py` | Match bucket & source pattern classification |
| `data/folds.py` | Permanent fold manifest generation |
| `data/split.py` | Simple train/val split (legacy, still functional) |
| `data/processing.py` | Processed dataset builder |
| `normalization/text.py` | Multi-view text normalization |
| `evaluation/f05.py` | Official macro F0.5 metric |
| `submission/__init__.py` | Submission builder & validator |
| `utils/seed.py` | Deterministic seeding |
| `utils/logging.py` | Logger configuration |
