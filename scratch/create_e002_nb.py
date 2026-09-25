import json

with open("kaggle/E001_C_GPU_Probe.ipynb", "r") as f:
    nb_str = f.read()

# Replacements
nb_str = nb_str.replace("E001-C", "E002-A")
nb_str = nb_str.replace("E001", "E002")
nb_str = nb_str.replace("e001", "e002")
nb_str = nb_str.replace("gpu_char_tfidf", "gpu_word_tfidf")
nb_str = nb_str.replace("e002_gpu.yaml", "e002_word_gpu.yaml") # Fix config name
nb_str = nb_str.replace("Character TF-IDF", "Word TF-IDF")
nb_str = nb_str.replace("global Name Character TF-IDF retrieval", "global Name Word TF-IDF retrieval")

nb = json.loads(nb_str)
with open("kaggle/E002_A_WORD_TFIDF_GPU_Probe.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
