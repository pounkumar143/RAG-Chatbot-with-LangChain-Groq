import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

def load_docs(file_path):
    """Load documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()

def create_vector_store(docs):
    """Create a vector store from the documents."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceHubEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def get_LLM():
    """Initialize the Groq LLM."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-8b-8192"
    )

def get_rag_chain(vector_store):
    """Set up the RetrievalQA chain."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = get_LLM()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
