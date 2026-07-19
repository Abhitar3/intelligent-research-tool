# Evaluation Summary

Benchmark run: retrieval quality over 3 technical documentation sources with 20 curated queries.

Generated artifact:
- `benchmarks/results_python_docs_latest.csv`

## Metrics

| Metric | k=1 | k=3 | k=5 |
| --- | ---: | ---: | ---: |
| Hit@k | 0.95 | 0.95 | 0.95 |
| TermRecall@k | 0.81 | 0.90 | 0.90 |
| SourceHit@k | 0.80 | 0.90 | 1.00 |

MRR: 0.95

## Interpretation

- The retriever found relevant content for 95% of benchmark questions at rank 1.
- The correct source appeared in the top 5 retrieved chunks for 100% of benchmark questions.
- Term recall improved from 0.81 at k=1 to 0.90 at k=3/k=5, showing that broader retrieval improves evidence coverage.
- MRR of 0.95 indicates the first useful result usually appears at or near the top of the retrieval list.

## Resume-Ready Summary

Built and evaluated a RAG-based research assistant using LangChain, ChromaDB, HuggingFace embeddings, and Groq LLMs. Added an evaluation pipeline with Hit@k, Recall@k, SourceHit@k, and MRR metrics, achieving 95% Hit@1, 100% SourceHit@5, and 0.95 MRR on a 20-query benchmark across 3 source documents.
