# ============================================================
# vectorChat.py - Sprint 4
# ------------------------------------------------------------
# Responsável pela camada RAG da aplicação.
#
# Funções principais:
# 1. Ler os PDFs da pasta ./docs
# 2. Quebrar os textos em chunks menores
# 3. Gerar embeddings locais com Ollama
# 4. Persistir os vetores no Chroma
# 5. Expor um retriever para o mainChat.py
# ============================================================

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PDF_DIR = "./docs"
DB_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "score_rag"
EMBEDDING_MODEL = "mxbai-embed-large"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 1


# ============================================================
# 1. SPLITTER DE TEXTO
# ============================================================
# Divide textos longos em blocos menores com sobreposição.
# Isso ajuda a melhorar a recuperação semântica no RAG.
def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks


# ============================================================
# 2. EMBEDDINGS
# ============================================================
# Usa o modelo local do Ollama para gerar embeddings.
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


# ============================================================
# 3. VECTOR STORE
# ============================================================
# Se o diretório já existir, o Chroma reaproveita a base persistida.
# Se não existir, os documentos serão processados e adicionados.
add_documents = not os.path.exists(DB_DIR)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_DIR
)


# ============================================================
# 4. INDEXAÇÃO DOS PDFs
# ============================================================
# Só roda a indexação quando a base ainda não existe.
if add_documents:
    documents = []

    if os.path.exists(PDF_DIR):
        for file in os.listdir(PDF_DIR):
            if file.endswith(".pdf"):
                file_path = os.path.join(PDF_DIR, file)

                # Carrega PDF por páginas
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                for page in pages:
                    # Quebra cada página em chunks menores
                    chunks = split_text(page.page_content)

                    for chunk in chunks:
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": file,
                                    "page": page.metadata.get("page", None),
                                }
                            )
                        )

    # Adiciona ao banco vetorial somente se houver documentos
    if documents:
        vector_store.add_documents(documents)
    else:
        print("Aviso: nenhum PDF encontrado em ./docs para indexação.")


# ============================================================
# 5. RETRIEVER
# ============================================================
# Recupera os top-k chunks mais similares à pergunta.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)