import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
import logging

logger = logging.getLogger(__name__)

def sparse_top_k(query_matrix: sp.csr_matrix, index_matrix_t: sp.csr_matrix, k: int, chunk_size: int = 1000):
    """
    Computes top K highest similarity scores using sparse_dot_topn.
    
    Args:
        query_matrix: CSR matrix of queries (num_queries x vocab)
        index_matrix_t: CSR matrix of target corpus TRANSPOSED (vocab x num_docs)
                        sparse_dot_topn wants A to be N x V and B to be V x M. 
                        It expects both to be CSR. So index_matrix_t should ideally be CSR.
        k: Number of top candidates to retrieve
        chunk_size: Ignored. Maintained for API compatibility.
        
    Returns:
        top_indices: (num_queries, k) array of column indices in index_matrix_t (target doc indices).
                     Unfilled slots (when a query matches < k docs) will be -1.
        top_scores: (num_queries, k) array of float scores.
                    Unfilled slots will be 0.0.
    """
    num_queries = query_matrix.shape[0]
    num_docs = index_matrix_t.shape[1]
    
    actual_k = min(k, num_docs)
    
    assert sp.issparse(query_matrix), "Query matrix must be sparse."
    assert sp.issparse(index_matrix_t), "Index matrix must be sparse."

    # Perform the top-N multiplication in C/C++ directly
    # query_matrix is (num_queries x vocab)
    # index_matrix_t is (vocab x num_docs)
    # the output C will be (num_queries x num_docs) containing at most actual_k elements per row
    C = sp_matmul_topn(
        query_matrix.tocsr(),
        index_matrix_t.tocsr(),
        top_n=actual_k,
        sort=True,
        n_threads=-1 # use all available cores minus one
    )
    
    top_indices = np.full((num_queries, actual_k), -1, dtype=np.int32)
    top_scores = np.zeros((num_queries, actual_k), dtype=np.float32)
    
    # Efficient extraction from CSR
    for i in range(num_queries):
        start_ptr = C.indptr[i]
        end_ptr = C.indptr[i+1]
        length = end_ptr - start_ptr
        
        if length > 0:
            take_len = min(actual_k, length)
            top_indices[i, :take_len] = C.indices[start_ptr:start_ptr+take_len]
            top_scores[i, :take_len] = C.data[start_ptr:start_ptr+take_len]
            
    return top_indices, top_scores
