import json

with open("kaggle/E002_A_WORD_TFIDF_GPU_Probe.ipynb", "r") as f:
    nb_str = f.read()

nb_str = nb_str.replace("run_e002_gpu_probe.py", "run_e002_word_probe.py")

nb = json.loads(nb_str)
with open("kaggle/E002_A_WORD_TFIDF_GPU_Probe.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
