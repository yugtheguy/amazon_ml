# Amazon ML Challenge 2026 - Business Entity Resolution

This repository contains our team's submission for the Amazon ML Challenge 2026. The objective is to build a mathematically correct, reproducible, and competition-compliant Business Entity Resolution system across 3 independent sources.

## Core Rules
- **No External Data**: Geocoding, Google searches, and commercial APIs are strictly forbidden.
- **Model Constraints**: <= 8B parameters, MIT or Apache 2.0 license compatible.
- **Metric**: Macro Entity-Level F0.5.
- **Output Integrity**: Every test Source 1 entity must appear exactly once in the final output. Matches must be a subset of candidate pairs.

Please see [HARD_RULES.md](HARD_RULES.md) for a compact reminder of the non-negotiable competition and engineering rules, and [AGENTS.md](AGENTS.md) for the full agent operational instructions.

## Repository Structure

```
├── configs/               # Hyperparameter and blocking configurations
├── data/                  # raw (immutable), interim, and processed data
├── notebooks/             # EDA and diagnostics
├── src/                   # Production-grade Python modules
├── scripts/               # Entrypoint execution scripts
├── tests/                 # Unit tests ensuring mathematical correctness
├── experiments/           # Logs and notes for reproducible experiments
├── artifacts/             # Serialized models, vectorizers, and embeddings
├── output/                # Final TSV outputs
└── documentation/         # Methodology, architecture, assumptions
```

## Running the Pipeline

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Placement**:
   Place the competition raw data into `data/raw/train/` and `data/raw/test/`.

3. **Full Pipeline Execution**:
   *(Instructions will be filled as the pipeline is developed. Stay tuned!)*
