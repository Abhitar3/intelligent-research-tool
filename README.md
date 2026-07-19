# Research Paper Assistant

Research Paper Assistant is a Streamlit-based RAG application that helps users read, understand, compare, and question 1-3 research papers or technical source URLs. It ingests the provided links, builds a Chroma vector store, retrieves relevant evidence, and uses an LLM to generate concise research-focused outputs.

The goal is to move beyond simple document Q&A and behave more like a practical research assistant: summarize papers, extract methods and results, compare sources, and support evidence-backed answers.

## Features

- Ingest up to 3 paper/source URLs from a Streamlit sidebar
- Chunk documents with overlap for better context preservation
- Generate embeddings with `Alibaba-NLP/gte-base-en-v1.5`
- Store and retrieve context using ChromaDB
- Answer questions using Groq `llama-3.3-70b-versatile`
- Return evidence snippets from retrieved source chunks
- Generate structured research briefs
- Compare multiple papers against a user-specified goal
- Extract methods, datasets, metrics, baselines, results, and limitations
- Evaluate retrieval quality with Hit@k, TermRecall@k, SourceHit@k, and MRR

## Research Workflows

The app provides four main workflows:

1. **Research Brief**
   - Quick summary
   - Research problem
   - Methods or techniques
   - Datasets, metrics, and experimental setup
   - Key findings
   - Strengths, limitations, and practical takeaway

2. **Compare Papers**
   - Compare problem, method, dataset, metrics, results, and limitations
   - Identify similarities and differences
   - Recommend the most useful paper or technique for a user goal

3. **Methods & Results**
   - Extract model/algorithm/architecture details
   - Identify datasets and experimental setup
   - Pull out metrics, baselines, numerical results, and ablation findings

4. **Ask Question**
   - Ask targeted questions about the processed papers
   - Receive concise answers with supporting evidence snippets

## Tech Stack

- Python
- Streamlit
- LangChain (`langchain-classic`, `langchain-community`, `langchain-chroma`, `langchain-groq`, `langchain-huggingface`, `langchain-text-splitters`)
- ChromaDB
- HuggingFace embeddings: `Alibaba-NLP/gte-base-en-v1.5`
- Groq LLM: `llama-3.3-70b-versatile`

## Project Structure

- `main.py`: Streamlit UI and research workflow tabs
- `rag.py`: URL ingestion, chunking, embeddings, vector store, retrieval, and research assistant functions
- `prompt.py`: Research-paper-focused prompt template
- `evaluate.py`: Retrieval evaluation pipeline
- `benchmarks/python_docs_benchmark.json`: Curated benchmark dataset
- `benchmarks/results_python_docs_latest.csv`: Latest per-query evaluation results
- `benchmarks/evaluation_summary.md`: Human-readable evaluation summary
- `resources/vectorstore/`: Local persisted Chroma vector database, ignored by git

## Evaluation

The project includes a retrieval evaluation pipeline with 20 curated benchmark queries over 3 source documents.

Latest benchmark results:

| Metric | k=1 | k=3 | k=5 |
| --- | ---: | ---: | ---: |
| Hit@k | 0.95 | 0.95 | 0.95 |
| TermRecall@k | 0.81 | 0.90 | 0.90 |
| SourceHit@k | 0.80 | 0.90 | 1.00 |

MRR: `0.95`

Interpretation:

- Relevant content was retrieved for 95% of queries at rank 1.
- Correct source coverage reached 100% by top 5 retrieved chunks.
- MRR of 0.95 shows that useful evidence usually appears near the top of retrieval results.

Run evaluation:

```powershell
.\venv\Scripts\python.exe evaluate.py --benchmark-file benchmarks\python_docs_benchmark.json --output-csv benchmarks\results_python_docs_latest.csv
```

## Setup

1. Clone the repo and move into the project folder.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Add your Groq API key to `.env`.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U pip
pip install streamlit python-dotenv langchain-classic langchain-community langchain-chroma langchain-groq langchain-huggingface langchain-text-splitters sentence-transformers transformers tokenizers huggingface-hub==0.36.2 chromadb unstructured beautifulsoup4 lxml nltk
```

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run

Run the Streamlit app from the project virtual environment:

```powershell
.\venv\Scripts\python.exe -m streamlit run main.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How To Use

1. Paste 1 to 3 paper or source URLs in the sidebar.
2. Click **Process Papers**.
3. Choose a workflow tab:
   - **Research Brief**
   - **Compare Papers**
   - **Methods & Results**
   - **Ask Question**
4. Review the generated answer and evidence snippets.

## Notes

- The app is source-grounded and should say it does not know when the retrieved context is insufficient.
- The current pipeline resets the vector store each time new papers are processed.
- Internet access is required for URL loading, embedding model access, and Groq model calls.
- Current evaluation uses technical documentation as a retrieval benchmark. A research-paper-specific benchmark is the recommended next improvement.

## Roadmap

- Add PDF and arXiv-specific ingestion
- Extract paper metadata such as title, authors, year, venue, and abstract
- Improve table/result extraction from PDFs
- Add project/session memory instead of resetting the vector store
- Add research-paper benchmark cases for methods, datasets, metrics, and limitations
- Add answer-quality evaluation for groundedness and citation correctness

## Author

Built by [Abhitar3](https://github.com/Abhitar3)
