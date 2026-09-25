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

In a single notebook cell, execute:

```python
import os
import subprocess

# 1. Clone the repository (Replace with your actual GitHub repository URL/token if private)
REPO_URL = "https://github.com/yugtheguy/amazon_ml.git"
if not os.path.exists("amazon_ml"):
    subprocess.run(["git", "clone", REPO_URL], check=True)
    
os.chdir("amazon_ml")

# 2. Set Environment Paths
# Update KAGGLE_DATA_ROOT to match the attached dataset path (e.g. "/kaggle/input/amazon-ml-challenge-2026")
os.environ["KAGGLE_DATA_ROOT"] = "/kaggle/input/amazon-ml-challenge-2026"
os.environ["KAGGLE_ARTIFACT_DIR"] = "/kaggle/working/artifacts/retrieval/E001-C"

# 3. Verify GPU Environment
print("--- Verifying GPU Environment ---")
subprocess.run(["python", "scripts/check_gpu_environment.py"])

# 4. Run E001-C GPU Probe
print("\n--- Running E001-C GPU Probe ---")
# Make sure your Python path includes the current directory so modules are resolved
subprocess.run(
    ["python", "-u", "scripts/run_e001_gpu_probe.py", "--config", "configs/e001_gpu.yaml"],
    env=dict(os.environ, PYTHONPATH=".")
)
```

## Collecting Results
The script will output artifacts into `/kaggle/working/artifacts/retrieval/E001-C/`.
These include:
- `environment.json`
- `config.yaml`
- `probe_ids.parquet`
- `probe_metrics.json`

Download this directory from Kaggle and commit the lightweight JSON/YAML reports back to the repository on your local machine to record the experiment outcome. Do not commit large generated Parquet files (like candidate predictions) to Git.
