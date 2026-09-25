"""
GPU Word TF-IDF Retrieval implementation.
Reuses the robust CuPy sparse top-K matmul kernel from gpu_char_tfidf.
"""

from .gpu_char_tfidf import sp_matmul_topn_cupy

# The core Word TF-IDF logic uses cuML's TfidfVectorizer,
# which is orchestrated directly in the E002 probe script.
# This file serves as the architectural placeholder for Word-specific
# GPU retrieval utilities if needed in the future.
