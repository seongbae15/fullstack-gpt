import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS  # FAISS vs Chroma?

import time


MESSAGES = "messages"
ROLE = "role"


st.set_page_config(page_title="DocumentGPT", page_icon="📑")
st.title("Document GPT")


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitters = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitters)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


st.markdown(
    """
    Welcome! Use this chatbot to ask questions about your documents.
"""
)

file = st.file_uploader(
    "Upload a .txt, .pdf or .docx file",
    type=["txt", "pdf", "docx"],
)

if file:
    retriever = embed_file(file)
    s = retriever.invoke("Winston")
    st.write(s)
