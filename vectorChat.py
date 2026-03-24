# vectorChat.py
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# ========= SPLITTER PYTHON PURO =========
def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks
# =======================================

PDF_DIR = "./docs"
DB_DIR = "./chroma_langchain_db"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

add_documents = not os.path.exists(DB_DIR)

vector_store = Chroma(
    collection_name="score_rag",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

if add_documents:
    documents = []

    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, file))
            pages = loader.load()

            for page in pages:
                chunks = split_text(page.page_content)

                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file,
                                "page": page.metadata.get("page", None)
                            }
                        )
                    )

    vector_store.add_documents(documents)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
