#Imports
from uuid   import uuid4
from dotenv import load_dotenv
from pathlib import Path
#from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains.qa_with_sources import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from prompt import PROMPT


load_dotenv()

#Constants
CHUNK_SIZE=1000
Collection_Name="real_estate"
Embedding_Model="Alibaba-NLP/gte-base-en-v1.5"
Vectorstore_Dir=Path(__file__).parent/"resources/vectorstore"

llm=None
vector_store=None

def initialize_components():
    global llm,vector_store
    if llm is None:
        llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9,max_tokens=500)
    if vector_store is None:

        ef=HuggingFaceEmbeddings(
            model_name=Embedding_Model,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store=Chroma(
            collection_name=Collection_Name,
            embedding_function=ef,
            persist_directory= str(Vectorstore_Dir)
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

    yield 'Load data'
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()

    yield 'split text'
    textsplitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',' '],
        chunk_size=CHUNK_SIZE
    )
    docs=textsplitter.split_documents(data)

    print('Add docs to vectordb')
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)
    yield "Done adding docs to vector database"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError('vectordb is not initialized')
    # chain=retrieval_qa.from_chain_type(llm=llm,retriever=vector_store.as_retriever())
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vector_store.as_retriever(),chain_type_kwargs={"prompt": PROMPT})


    result = chain.invoke({'query': query}, return_only_outputs=True)  
    sources=result.get("sources","")
    return result['result'], sources


if __name__ == "__main__":
    urls = [
        "https://www.forbes.com/companies/google/",
        "https://www.forbes.com/companies/openai/"
    ]

    # Force full execution of generator to actually initialize
    for _ in process_urls(urls):
        pass

    # OR you can just call initialize_components() directly before using generate_answer
    # initialize_components()

    answer, sources = generate_answer("which one is the leading company in AI research sector")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")

    # results=vector_store.similarity_search(
    #     "Leading company in AI research sector",
    #     k=2
    # )
    #print(results)
