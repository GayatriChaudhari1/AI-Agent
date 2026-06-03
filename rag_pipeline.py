# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq

# def create_qa_chain(pdf_path):
#     # Load PDF
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()

#     # Use FREE embeddings (no OpenAI)
#     embeddings = HuggingFaceEmbeddings()

#     # Store in FAISS
#     db = FAISS.from_documents(documents, embeddings)

#     # Use Groq LLM (FREE)
#     llm = ChatGroq(
#         groq_api_key="your_groq_api_key_here",  # OR use env
#         model_name="llama3-70b-8192"
#     )

#     # Create QA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=db.as_retriever()
#     )

#     return qa_chain


# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
# import os

# def create_qa_chain(pdf_path):
#     # Load PDF
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()

#     # Use FREE embeddings
#     embeddings = HuggingFaceEmbeddings()

#     # Store in FAISS
#     db = FAISS.from_documents(documents, embeddings)

#     # Groq LLM
#     # llm = ChatGroq(
#     #     groq_api_key=os.getenv("GROQ_API_KEY"),
#     #     model_name="llama3-8b-8192"
#     # )


#     llm = ChatGroq(
#         model_name="llama3-8b-8192",
#         groq_api_key=os.getenv("GROQ_API_KEY"),
#     )

#     # QA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=db.as_retriever()
#     )

#     return qa_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os


def create_qa_chain(pdf_path):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create vector DB
    db = FAISS.from_documents(docs, embeddings)

    # Create LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    return qa_chain