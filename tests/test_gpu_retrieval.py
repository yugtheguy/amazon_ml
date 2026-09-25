import pytest
import yaml
import numpy as np
import scipy.sparse
from src.business_entity_resolution.retrieval.gpu_char_tfidf import HAS_CUPY, sp_matmul_topn_cupy

def test_e001_gpu_config():
    with open("configs/e001_gpu.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    assert "retrieval" in config
    assert config["retrieval"]["method"] == "gpu_char_tfidf"
    assert config["retrieval"]["dtype"] == "float32"

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed. Skipping GPU sparse tests locally.")
def test_sp_matmul_topn_cupy_correctness():
    # Construct a tiny brute force test
    import cupy as cp
    
    A_data = np.array([1.0, 1.0, 0.5], dtype=np.float32)
    A_indices = np.array([0, 2, 1], dtype=np.int32)
    A_indptr = np.array([0, 2, 3], dtype=np.int32)
    A_cpu = scipy.sparse.csr_matrix((A_data, A_indices, A_indptr), shape=(2, 3))
    
    B_T_data = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    B_T_indices = np.array([0, 1, 0, 1], dtype=np.int32)
    B_T_indptr = np.array([0, 1, 2, 4], dtype=np.int32)
    B_T_cpu = scipy.sparse.csr_matrix((B_T_data, B_T_indices, B_T_indptr), shape=(3, 2))
    
    # Brute force CPU dense:
    # A = [[1, 0, 1], [0, 0.5, 0]]
    # B = [[1, 0], [0, 1], [1, 1]] (since B_T is given, B is its transpose)
    # A * B_T = [[1, 0, 1], [0, 0.5, 0]] * [[1, 0, 1], [0, 1, 1]].T
    # = [[1, 0, 1], [0, 0.5, 0]] * [[1, 0], [0, 1], [1, 1]]
    # = [[2, 1], [0, 0.5]]
    
    dense_expected = A_cpu.toarray().dot(B_T_cpu.toarray())
    
    out_sparse = sp_matmul_topn_cupy(A_cpu, B_T_cpu, top_k=1, batch_size=2)
    
    # row 0 top 1 should be val 2.0 at index 0
    # row 1 top 1 should be val 0.5 at index 1
    
    out_dense = out_sparse.toarray()
    
    assert np.isclose(out_dense[0, 0], 2.0)
    assert np.isclose(out_dense[0, 1], 0.0) # top 1 means the second element is dropped
    assert np.isclose(out_dense[1, 0], 0.0)
    assert np.isclose(out_dense[1, 1], 0.5)

