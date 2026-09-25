import time
import numpy as np
import scipy.sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    
try:
    from cuml.neighbors import NearestNeighbors
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


def extract_topk_sparse_cupy_kernel(csr_mat, k):
    """
    Extract top K elements per row from a CuPy CSR matrix.
    Brings data to CPU per batch to perform efficient np.argpartition.
    """
    indptr = csr_mat.indptr.get()
    n_rows = len(indptr) - 1
    
    new_data_list = []
    new_indices_list = []
    new_indptr = np.zeros(n_rows + 1, dtype=np.int32)
    
    current_ptr = 0
    
    data_cpu = csr_mat.data.get()
    indices_cpu = csr_mat.indices.get()
    
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i+1]
        row_nnz = end - start
        
        if row_nnz == 0:
            new_indptr[i+1] = current_ptr
            continue
            
        row_data = data_cpu[start:end]
        row_indices = indices_cpu[start:end]
        
        if row_nnz <= k:
            sort_idx = np.argsort(-row_data)
            new_data_list.append(row_data[sort_idx])
            new_indices_list.append(row_indices[sort_idx])
            current_ptr += row_nnz
        else:
            idx = np.argpartition(-row_data, k)[:k]
            top_k_data = row_data[idx]
            sort_idx = np.argsort(-top_k_data)
            idx = idx[sort_idx]
            
            new_data_list.append(row_data[idx])
            new_indices_list.append(row_indices[idx])
            current_ptr += k
            
        new_indptr[i+1] = current_ptr
        
    if new_data_list:
        final_data = np.concatenate(new_data_list)
        final_indices = np.concatenate(new_indices_list)
    else:
        final_data = np.array([], dtype=np.float32)
        final_indices = np.array([], dtype=np.int32)
        
    return final_data, final_indices, new_indptr


def sp_matmul_topn_cupy(A, B_T, top_k, batch_size=1000):
    """
    Perform sparse dot product A * B_T on GPU via CuPy, preserving only top_k per row.
    A: (n_queries, vocab) scipy CSR
    B_T: (vocab, n_targets) scipy CSR
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is required for cupy-backend GPU sparse operations.")
        
    B_gpu = cpx_sparse.csr_matrix(B_T)
    n_queries = A.shape[0]
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        A_batch_gpu = cpx_sparse.csr_matrix(A[i:end])
        
        # GPU cuSPARSE SpGEMM
        sim_gpu = A_batch_gpu.dot(B_gpu)
        
        # Top-K extraction
        batch_data, batch_indices, batch_indptr = extract_topk_sparse_cupy_kernel(sim_gpu, top_k)
        
        out_data.append(batch_data)
        out_indices.append(batch_indices)
        
        if len(out_indptr) > 1:
            last_ptr = out_indptr[-1]
            out_indptr.extend((batch_indptr[1:] + last_ptr).tolist())
        else:
            out_indptr.extend(batch_indptr[1:].tolist())
            
    final_data = np.concatenate(out_data) if out_data else np.array([], dtype=np.float32)
    final_indices = np.concatenate(out_indices) if out_indices else np.array([], dtype=np.int32)
    final_indptr = np.array(out_indptr, dtype=np.int32)
    
    return scipy.sparse.csr_matrix((final_data, final_indices, final_indptr), shape=(n_queries, B_T.shape[1]))

