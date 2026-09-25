import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# C001: Full Train Candidate Pool V1\n",
    "\n",
    "**Objective**: Generate the final, frozen Candidate Pool V1 for the entire 2.2M Source 1 dataset using the proven R001 architecture.\n",
    "\n",
    "**Rules**:\n",
    "- Uses GPU TF-IDF with `batch_size=100`.\n",
    "- Checkpoints every 50K rows per country to prevent data loss on OOM/Timeout.\n",
    "- Generates a final packaged tarball."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Environment & GPU Verification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import subprocess\n",
    "\n",
    "print(f\"Python Version: {sys.version}\")\n",
    "print(f\"CWD: {os.getcwd()}\")\n",
    "!nvidia-smi\n",
    "\n",
    "try:\n",
    "    import cuml\n",
    "    import cupy as cp\n",
    "    print(\"cuML/CuPy is available. GPU_BACKEND_ACTIVE = TRUE\")\n",
    "except ImportError:\n",
    "    print(\"WARNING: cuML/CuPy not found. The script will halt loudly.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Clone Repository"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "REPO_URL = \"https://github.com/yugtheguy/amazon_ml.git\"\n",
    "REPO_DIR = \"/kaggle/working/amazon_ml\"\n",
    "if not os.path.exists(REPO_DIR):\n",
    "    print(f\"Cloning {REPO_URL} ...\")\n",
    "    subprocess.run([\"git\", \"clone\", REPO_URL, REPO_DIR], check=True)\n",
    "else:\n",
    "    print(\"Repo exists, pulling latest...\")\n",
    "    subprocess.run([\"git\", \"-C\", REPO_DIR, \"pull\"], check=True)\n",
    "os.chdir(REPO_DIR)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Link Processed Data & Folds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "PROCESSED_DATA_ROOT = None\n",
    "known_path = \"/kaggle/input/datasets/yugdeshmukh/amazon-ml-processed-v001\"\n",
    "if os.path.exists(os.path.join(known_path, \"train_source1.parquet\")):\n",
    "    PROCESSED_DATA_ROOT = known_path\n",
    "else:\n",
    "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
    "        if \"train_source1.parquet\" in files:\n",
    "            PROCESSED_DATA_ROOT = root\n",
    "            break\n",
    "if PROCESSED_DATA_ROOT:\n",
    "    print(f\"Found processed dataset at {PROCESSED_DATA_ROOT}\")\n",
    "else:\n",
    "    raise FileNotFoundError(\"CRITICAL: Processed data (train_source1.parquet) not found in /kaggle/input.\")\n",
    "\n",
    "RAW_DATA_ROOT = None\n",
    "for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
    "    if \"train_source1.tsv\" in files and \"train\" in root:\n",
    "        RAW_DATA_ROOT = os.path.dirname(root)\n",
    "        break\n",
    "if not RAW_DATA_ROOT:\n",
    "    raise FileNotFoundError(\"CRITICAL: Raw data (train_source1.tsv) not found in /kaggle/input.\")\n",
    "os.makedirs(\"data/raw\", exist_ok=True)\n",
    "if not os.path.exists(\"data/raw/train\") and os.path.exists(os.path.join(RAW_DATA_ROOT, \"train\")):\n",
    "    os.symlink(os.path.join(RAW_DATA_ROOT, \"train\"), \"data/raw/train\")\n",
    "if not os.path.exists(\"data/raw/test\") and os.path.exists(os.path.join(RAW_DATA_ROOT, \"test\")):\n",
    "    os.symlink(os.path.join(RAW_DATA_ROOT, \"test\"), \"data/raw/test\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Install Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -r requirements.txt -q\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Engineering Smoke Check (Small Pre-flight)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Running smoke test (1000 rows)...\")\n",
    "!PYTHONPATH=. python -u scripts/run_c001_full_train.py --data-dir data --processed-dir {PROCESSED_DATA_ROOT} --out-dir /kaggle/working/artifacts/candidate_pool --smoke-size 1000\n",
    "\n",
    "# Clear smoke artifacts so we run clean for full\n",
    "import shutil\n",
    "shutil.rmtree(\"/kaggle/working/artifacts/candidate_pool/C001\")\n",
    "print(\"Smoke test passed and cleared.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. C001 FULL TRAINING RUN (2.2M Rows)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!PYTHONPATH=. python -u scripts/run_c001_full_train.py --data-dir data --processed-dir {PROCESSED_DATA_ROOT} --out-dir /kaggle/working/artifacts/candidate_pool\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Metrics Review"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import os\n",
    "metrics_file = \"/kaggle/working/artifacts/candidate_pool/C001/candidate_pool_v1/metrics/evaluation_metrics.json\"\n",
    "if os.path.exists(metrics_file):\n",
    "    with open(metrics_file) as f:\n",
    "        print(json.dumps(json.load(f), indent=2))\n",
    "else:\n",
    "    print(\"Metrics file not found. Did the run complete?\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Extract Final Artifacts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "import os\n",
    "bundle_src = \"/kaggle/working/artifacts/candidate_pool/C001/candidate_pool_v1/candidate_pool_v1_bundle.tar.gz\"\n",
    "bundle_dst = \"/kaggle/working/candidate_pool_v1_bundle.tar.gz\"\n",
    "if os.path.exists(bundle_src):\n",
    "    shutil.copy2(bundle_src, bundle_dst)\n",
    "    print(f\"Bundle ready at {bundle_dst}\")\n",
    "else:\n",
    "    print(\"Bundle not found.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

os.makedirs("kaggle", exist_ok=True)
with open("kaggle/c001_full_train.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
