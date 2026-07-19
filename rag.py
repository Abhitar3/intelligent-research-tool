#Imports
from uuid   import uuid4
from dotenv import load_dotenv
from pathlib import Path
#from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains.qa_with_sources import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from prompt import PROMPT


load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

llm=None
vector_store=None

def initialize_components():
    global llm,vector_store
    if llm is None:
        llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9,max_tokens=1200)
    if vector_store is None:

        ef=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store=Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls):
    """
    This Function scraps the data from a url and stores it in a vector database
    :param urls:input urls
    :return:
    """
    yield "Initialize components"
    initialize_components()

    yield "Resetting Vector Store"
    vector_store.reset_collection()

    yield "Load papers"
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()

    yield "Split text"
    textsplitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = textsplitter.split_documents(data)

    yield "Add docs to vectordb"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)
    yield "Done adding docs to vector database"

def format_sources(docs: list[Document]) -> list[dict[str, str]]:
    """
    Convert retrieved documents into simple source snippets for the UI.
    """
    sources = []

    for index, doc in enumerate(docs, start=1):
        source_url = doc.metadata.get("source", "Unknown source")
        snippet = doc.page_content.strip().replace("\n", " ")

        sources.append(
            {
                "source_number": str(index),
                "source": source_url,
                "snippet": snippet[:500],
            }
        )

    return sources

def generate_answer(query, k=5):
    if not vector_store:
        raise RuntimeError("vectordb is not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    result = chain.invoke({"query": query}, return_only_outputs=True)

    answer = result["result"]
    source_documents = result.get("source_documents", [])
    sources = format_sources(source_documents)

    return answer, sources

def generate_research_brief():
    research_brief_query = """
    Create a concise research brief from the provided papers.

    Include:
    1. Quick Summary
    2. Main Research Problem
    3. Methods or Techniques Used
    4. Datasets, Metrics, or Experimental Setup
    5. Key Results or Findings
    6. Strengths
    7. Limitations
    8. Best Practical Takeaway

    If multiple papers are present, compare them briefly.
    If any section is not supported by the provided context, write "Not found in the provided papers."
    """

    return generate_answer(research_brief_query, k=10)

def compare_papers(user_goal=""):
    comparison_query = f"""
    Compare the provided research papers.

    User goal:
    {user_goal if user_goal else "No specific goal provided."}

    Include:
    1. Comparison Table:
       - Paper or source
       - Research problem
       - Method or technique
       - Dataset or experimental setup
       - Metrics used
       - Key results
       - Main limitation

    2. Similarities:
       What do the papers agree on or share?

    3. Differences:
       How do the papers differ in method, evidence, results, or assumptions?

    4. Best Option:
       If the user goal is provided, recommend the most useful paper or technique for that goal.
       Explain the recommendation using evidence from the papers.

    If the provided context does not contain enough evidence for any section,
    write "Not found in the provided papers."
    """

    return generate_answer(comparison_query, k=12)

def extract_methods_and_results():
    extraction_query = """
    Extract the methods and results from the provided research papers.

    For each paper or source, include:
    1. Method or Technique Used
    2. Model, Algorithm, or Architecture
    3. Dataset or Data Source
    4. Experimental Setup
    5. Evaluation Metrics
    6. Baselines Compared Against
    7. Key Numerical Results
    8. Ablation Study Findings, if available
    9. Main Limitation

    Keep the answer concise and structured.
    If a detail is not available in the provided context, write "Not found in the provided papers."
    """

    return generate_answer(extraction_query, k=12)