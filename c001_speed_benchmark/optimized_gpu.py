import cupy as cp
import numpy as np
import scipy.sparse
import cupyx.scipy.sparse as cpx_sparse

def extract_topk_sparse_cupy_kernel_fast(csr_mat, k):
    """
    Optimized GPU Top-K extraction.
    Keeps data on GPU as long as possible.
    """
    indptr = csr_mat.indptr
    n_rows = len(indptr) - 1
    
    # We can't easily vectorize CSR row-wise argpartition in raw CuPy without a custom kernel
    # But we can iterate over rows on GPU, or just do dense if the batch is small.
    # A CuPy raw kernel is the fastest way to extract top K per row from CSR.
    
    # Alternatively, if we just pull the indptr to CPU but leave data/indices on GPU:
    indptr_cpu = indptr.get()
    
    data_gpu = csr_mat.data
    indices_gpu = csr_mat.indices
    
    new_data_list = []
    new_indices_list = []
    new_indptr = np.zeros(n_rows + 1, dtype=np.int32)
    
    current_ptr = 0
    
    for i in range(n_rows):
        start = indptr_cpu[i]
        end = indptr_cpu[i+1]
        row_nnz = end - start
        
        if row_nnz == 0:
            new_indptr[i+1] = current_ptr
            continue
            
        row_data = data_gpu[start:end]
        row_indices = indices_gpu[start:end]
        
        if row_nnz <= k:
            sort_idx = cp.argsort(-row_data)
            new_data_list.append(row_data[sort_idx])
            new_indices_list.append(row_indices[sort_idx])
            current_ptr += row_nnz
        else:
            # CuPy argpartition
            idx = cp.argpartition(-row_data, k)[:k]
            top_k_data = row_data[idx]
            sort_idx = cp.argsort(-top_k_data)
            idx = idx[sort_idx]
            
            new_data_list.append(row_data[idx])
            new_indices_list.append(row_indices[idx])
            current_ptr += k
            
        new_indptr[i+1] = current_ptr
        
    if new_data_list:
        final_data = cp.concatenate(new_data_list).get()
        final_indices = cp.concatenate(new_indices_list).get()
    else:
        final_data = np.array([], dtype=np.float32)
        final_indices = np.array([], dtype=np.int32)
        
    return final_data, final_indices, new_indptr


def sp_matmul_topn_cupy_optimized(A, B_T, top_k, batch_size=1000):
    B_gpu = cpx_sparse.csr_matrix(B_T)
    n_queries = A.shape[0]
    
    out_data = []
    out_indices = []
    out_indptr = [0]
    
    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        A_batch_gpu = cpx_sparse.csr_matrix(A[i:end])
        
        sim_gpu = A_batch_gpu.dot(B_gpu)
        
        batch_data, batch_indices, batch_indptr = extract_topk_sparse_cupy_kernel_fast(sim_gpu, top_k)
        
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
