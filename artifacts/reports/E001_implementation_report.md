# E001 Implementation Report

## A. E001-A Status
`E001-A_SCIPY_DOT = OPERATIONALLY REJECTED`
Reason: The naive SciPy sparse matrix multiplication evaluated the full `M x N` cross-product into memory, resulting in excessive intermediate sparse-product memory allocation (>3.5 GiB per 2000 queries) and unacceptable measured CPU throughput (estimated 20+ hours full run).

## B. E001-B Implementation
- **Package/Version**: `sparse_dot_topn` version 1.2.0
- **API**: `sparse_dot_topn.sp_matmul_topn` (integrated C++ kernel for Top-N sparse multiplication)
- **Matrix Representation**: `scipy.sparse.csr_matrix` for both queries (A) and transposed corpus (B).
- **Dtype**: `float32` natively supported by `sp_matmul_topn`.
- **Source Separation**: Queries against S2 and S3 executed completely independently in `scripts/run_e001_probe_b.py`.

## C. Exact Configuration
Loaded explicitly from `configs/e001.yaml`:
- **analyzer**: `char_wb`
- **ngram_min**: 3
- **ngram_max**: 5
- **min_df**: 5
- **max_df**: 0.05 (pruned to mitigate extreme density of operations)
- **max_features**: null
- **dtype**: `float32`
- **query_chunk_size**: null (using integrated C++ top-K thread loop)
- **top_k**: 20
- **n_threads**: -1 (utilizing 10+ cores)
- **threshold**: 0.0

## D. Probe Dataset
- **S1 Queries Sampled**: 50,000 entities (stratified deterministic sample from `folds_v1.parquet`).
- **Target Counts**: Evaluated against the full real S2 and S3 corpora (e.g., S2 US contains 3,016,815 targets).
- **S1 Total for Projection**: 2,206,821 queries.

## E. Matrix Statistics (S2 US Partition with max_df=0.05)
- **Target Matrix Shape**: (3016815, 424858)
- **Vocabulary Size**: 424,858 char n-grams
- **Target NNZ**: 141,006,252 non-zeros
- **Query Matrix Shape**: (30225, 424858)
- **Query NNZ**: 1,333,138 non-zeros
- **Approximate Memory Peak**: Python process peaked over 4.2 GB during matrix fitting and CSR transformation.

## F. Performance
- **Fit Time**: ~2.5 - 3 minutes for S2 US Target corpus.
- **Transform Time**: ~1.5 seconds for S1 US queries.
- **Retrieval Time**: Despite avoiding intermediate memory allocations, the `sp_matmul_topn` multi-threaded C++ kernel took **> 4 minutes** and was still actively processing the 30k query batch when manually terminated. 
- **Queries/Sec**: < 125 queries/sec (extrapolated).
- **Peak Memory**: ~4.2 GB.

## G. K Frontier
- **N/A**. Operationally aborted prior to complete execution due to dense float-ops bottleneck.

## H. E000 + E001-B
- **N/A**.

## I. Positive Rank Distribution
- **N/A**.

## J. Full-Scale Projection
- Measured `sparse_dot_topn` throughput was < 125 queries/sec on a modern CPU utilizing all threads.
- For 2,206,821 full queries, the projected retrieval time for S2 alone is:
  `2.2M / 125 = 17,600 seconds = ~4.9 hours`.
- Adding S3 processing brings the full retrieval time to **~10 hours on CPU**.
- **Assumption**: This assumes linear scaling and does not account for the unpruned config (`max_df=1.0`), which resulted in >8 minutes of processing for 30k queries (projected >20 hours).

## K. Acceptance Decision
`GLOBAL NAME CHAR-TFIDF OPERATIONALLY REJECTED`
**Evidence**: Even utilizing an optimal, highly multi-threaded C++ sparse-topN matrix multiplication engine (`sparse_dot_topn`), the collision density of Character TF-IDF (141M NNZ target against 1.3M NNZ query) produces an operationally unviable amount of scalar multiplications. It bottlenecks the CPU, violating the practical O(N*M) performance limits for this competition's CPU constraint.

## L. Full-Scale Results
- (Skipped due to rejection).

## M. Tests
- Tested `sparse_top_k` behavior manually using the probe script, verifying it prevented OOM but hit compute limits.

## N. Architecture Compliance
- Strict compliance maintained. The experiment evaluated purely the retrieval efficiency of Character TF-IDF. No later retrieval or matcher stages (like cross-encoders) were implemented.

## O. Next Step
**STOP E001 GLOBAL RETRIEVAL.**
Character TF-IDF generates an intractable intersection density for 5 million target records.
**Recommendation**: Proceed to evaluate **E002 — Word TF-IDF / BM25**. Word-level tokenization produces a substantially sparser vocabulary structure, drastically reducing collisions in the inverted index and enabling the `sparse_dot_topn` multiplication to execute in seconds instead of hours. Character retrieval should only be used as a targeted reranker on top of a highly constrained candidate pool, not as a global cross-retrieval blocker.
