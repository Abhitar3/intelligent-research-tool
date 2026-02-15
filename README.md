# Intelligent Research Tool

AI-powered research assistant that ingests up to 3 URLs, builds a vector database from their content, and answers questions using a Retrieval-Augmented Generation (RAG) pipeline.

## Features
- URL ingestion from Streamlit sidebar inputs
- Automatic chunking and embedding of scraped content
- Persistent Chroma vector store
- Question answering with `ChatGroq` + custom prompt template
- Simple Streamlit UI for non-technical usage

## Tech Stack
- Python
- Streamlit
- LangChain ecosystem (`langchain-classic`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`, `langchain-text-splitters`)
- ChromaDB
- HuggingFace embeddings (`Alibaba-NLP/gte-base-en-v1.5`)
- Groq LLM (`llama-3.3-70b-versatile`)

## Project Structure
- `main.py`: Streamlit UI and app flow
- `rag.py`: URL processing, vector DB creation, retrieval + answer generation
- `prompt.py`: Custom QA prompt template
- `resources/vectorstore/`: Local persisted vector database

## Setup
1. Clone repo and move into project folder.
2. Create and activate virtual environment.
3. Install dependencies.
4. Add API key to `.env`.

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
Run Streamlit app:

```powershell
streamlit run main.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## How To Use
1. Paste 1 to 3 URLs in sidebar.
2. Click **Process Urls**.
3. Ask questions in the text box.
4. Review generated answer and sources.

## Notes
- If no relevant context is found, the app returns "I don't know."
- Current pipeline resets the vector store each time URLs are processed.
- Ensure internet access for URL loading and model usage.

## Roadmap
- Better source attribution formatting
- Multi-query memory/session support
- Validation for invalid URLs
- Docker support
- Unit/integration tests

## Author
Built by [Abhitar3](https://github.com/Abhitar3)
