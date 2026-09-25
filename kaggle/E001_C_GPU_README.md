# Amazon ML Challenge 2026 - E001-C Kaggle Execution

This directory provides instructions to execute the E001-C GPU Character TF-IDF viability probe on a Kaggle notebook.

## Objective
To determine if global Name Character TF-IDF (E001) is operationally viable using Kaggle GPU compute (RAPIDS / CuPy / cuML), replacing the severely bottlenecked CPU execution paths (E001-A and E001-B).

## Kaggle Notebook Setup

1. **Create a new Kaggle Notebook**.
2. **Accelerator**: Select GPU (P100 or T4x2).
3. **Environment**: Use the default Kaggle environment (which pre-installs RAPIDS components like `cuml` and `cupy`). Do **NOT** install custom RAPIDS versions unless required, as it may break other notebook dependencies.
4. **Data Mounts**: Ensure the competition dataset is attached.

## Execution Code

Instead of copy-pasting code, a ready-to-run Jupyter Notebook is provided in this repository.

**Instructions:**
1. Upload or open the `kaggle/E001_C_GPU_Probe.ipynb` notebook in your Kaggle workspace.
2. The notebook acts as a thin execution layer. It will:
   - Discover your environment and Kaggle dataset automatically.
   - Run a smoke test before committing to the full 50k sample.
   - Handle cloning the repository.
   - Output artifacts and bundle them into a zip file.
3. Run the notebook cells sequentially top-to-bottom.

## Collecting Results
The script will output artifacts into `/kaggle/working/artifacts/retrieval/E001-C/`.
These include:
- `environment.json`
- `config.yaml`
- `probe_ids.parquet`
- `probe_metrics.json`

Download this directory from Kaggle and commit the lightweight JSON/YAML reports back to the repository on your local machine to record the experiment outcome. Do not commit large generated Parquet files (like candidate predictions) to Git.
