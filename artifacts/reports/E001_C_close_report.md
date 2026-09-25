# E001-C Close Report: GPU Character TF-IDF Viability

## Conclusion
**Status:** REJECTED

## Reasoning
The E001-C probe was designed to test if offloading character n-gram TF-IDF sparse matrix multiplication to the GPU using CuPy would solve the performance issues encountered in E001-A and E001-B.

During the smoke test (sample size = 1000), the `cusparse.spgemm` operation failed with a CUDA `MemoryError: std::bad_alloc: out_of_memory (failed to allocate 44.6 GB)`. 

### Why did this happen?
Character n-grams (e.g., 3 to 5 characters) have very high collision rates. Many common character sequences appear in almost every business name. Consequently, when multiplying a batch of query TF-IDF vectors (size 616 x 1,689,975) by the transposed target matrix (size 1,689,975 x 3,016,815), the resulting matrix is overwhelmingly dense. 
Even though it's technically a "sparse" multiplication, the output contains billions of non-zero similarity scores (before Top-K is applied), causing a massive 44 GB intermediate buffer allocation that instantly exceeds the 15 GB limit of Kaggle's T4 GPUs.

## Next Steps
Global character TF-IDF without prior blocking is **computationally infeasible** for 3 million records, both on CPU (too slow) and GPU (OOM).

We must abandon global character n-gram retrieval and proceed to **E002 (Word-level TF-IDF / BM25)** or introduce a **Blocking Strategy (E003)** to partition the dataset before applying expensive string distance metrics.
