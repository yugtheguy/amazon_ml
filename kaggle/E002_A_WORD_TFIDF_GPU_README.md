# E002-A Word TF-IDF GPU Probe

This experiment runs Word TF-IDF similarity on Kaggle T4 GPUs using cuML.

**Goal:** Determine if Word-level TF-IDF can retrieve unique ground truth pairs over E000 while remaining computationally viable (no OOM, unlike E001).

## Instructions for Kaggle
1. Open Kaggle.
2. Create a new notebook and set the accelerator to **T4 x2**.
3. Attach the Amazon ML Challenge dataset.
4. Upload `E002_A_WORD_TFIDF_GPU_Probe.ipynb` to the notebook.
5. Run the cells in order.
6. The notebook will download this GitHub repository, setup the symlinks, and run the probe!
