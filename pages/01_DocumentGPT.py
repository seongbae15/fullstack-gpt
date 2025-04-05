import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS  # FAISS vs Chroma?
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


MESSAGES = "messages"
ROLE = "role"


st.set_page_config(page_title="DocumentGPT", page_icon="📑")
st.title("Document GPT")


@st.cache_data(show_spinner="Embedding file...")
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


def save_message(message, role):
    st.session_state[MESSAGES].append({MESSAGES: message, ROLE: role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state[MESSAGES]:
        send_message(
            message[MESSAGES],
            message[ROLE],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the question using ONLY the following context.
         If you don't know the answer, just say you don't know.
         DON'T make anything up. 
         
         Context: {context}""",
        ),
        ("human", "{question}"),
    ]
)

st.markdown(
    """
    Welcome! Use this chatbot to ask questions about your documents.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt, .pdf or .docx file",
        type=["txt", "pdf", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anythin about your file...")
    if message:
        send_message(message=message, role="human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state[MESSAGES] = []
