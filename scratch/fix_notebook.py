import json
import os

with open("kaggle/E001_C_GPU_Probe.ipynb", "r") as f:
    nb = json.load(f)

new_markdown_cell = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.5 Data Symlinking & Preprocessing\n",
    "The repository expects data under `data/raw/...` and builds processed parquet files in `data/processed/...`.\n",
    "Since Kaggle's `/kaggle/input` is read-only, we symlink the raw data into the repo and run the preprocessing scripts."
   ]
}

new_code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess\n",
    "if KAGGLE_DATA_ROOT and KAGGLE_DATA_ROOT != os.path.abspath(\"data/raw\"):\n",
    "    print(\"\\n--- Creating Local Data Symlinks ---\")\n",
    "    os.makedirs(\"data/raw\", exist_ok=True)\n",
    "    if not os.path.exists(\"data/raw/train\"):\n",
    "        os.symlink(os.path.join(KAGGLE_DATA_ROOT, \"train\"), \"data/raw/train\")\n",
    "    if not os.path.exists(\"data/raw/test\"):\n",
    "        os.symlink(os.path.join(KAGGLE_DATA_ROOT, \"test\"), \"data/raw/test\")\n",
    "    print(\"Symlinks created.\")\n",
    "\n",
    "    print(\"\\n--- Building Processed Data ---\")\n",
    "    !PYTHONPATH=. python scripts/build_processed_data.py\n",
    "\n",
    "    print(\"\\n--- Building Folds ---\")\n",
    "    !PYTHONPATH=. python scripts/build_folds.py\n",
    "\n",
    "    # Override to local data directory which now contains raw/ and processed/\n",
    "    KAGGLE_DATA_ROOT = os.path.abspath(\"data\")\n"
   ]
}

insert_idx = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and any("## 5. Artifact Path" in line for line in cell["source"]):
        insert_idx = i
        break

if insert_idx > 0:
    nb["cells"].insert(insert_idx, new_code_cell)
    nb["cells"].insert(insert_idx, new_markdown_cell)

with open("kaggle/E001_C_GPU_Probe.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
