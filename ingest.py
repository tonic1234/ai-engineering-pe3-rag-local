"""ingest.py — Ingesta: leer documentos, partirlos en chunks y guardarlos en ChromaDB.

APUNTE DE CLASE (unidad de chunking):
La idea del chunking es partir el texto en pedazos del tamaño justo. Si los hago muy
chicos pierdo el contexto de la frase; si los hago muy grandes, el embedding "diluye" el
significado y la búsqueda se vuelve imprecisa. La consigna pide ~500 tokens con 50 de
solape (overlap).

Detalle que me quedó de la clase: el chunk_size es un TECHO, no un piso. Si un archivo
es corto y entra entero en un chunk, el splitter lo deja entero y está perfecto.

Otra cosa importante: el script chequea si la colección ya existe. Si ya está, no
reindexa (así no gasto tiempo ni cómputo cada vez que lo corro).

Sobre los embeddings: usé un modelo LOCAL de HuggingFace (all-MiniLM-L6-v2). La ventaja
es que no necesita API key ni cuesta nada, y corre en la máquina. Regla de oro: usar el
MISMO modelo para indexar y para consultar, si no las distancias no significan nada.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PERSIST_DIR = "./vectorstore"
COLLECTION_NAME = "dendra_politicas"
DATA_DIR = Path("./data")

# Modelo de embeddings local y gratuito (no requiere API key).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Consigna: mínimo 500 tokens con 50 de overlap.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Carga todos los .txt/.md de la carpeta /data como Documents."""

    archivos = sorted([*data_dir.glob("*.txt"), *data_dir.glob("*.md")])
    if not archivos:
        raise FileNotFoundError(f"No encontré documentos en {data_dir.resolve()}")

    documentos: list[Document] = []
    for archivo in archivos:
        docs = TextLoader(str(archivo), encoding="utf-8").load()
        # Guardo la fuente en la metadata: la uso después para las referencias (citas).
        for doc in docs:
            doc.metadata["source"] = archivo.name
        documentos.extend(docs)
    logger.info("Cargué %d documento(s) desde %s", len(documentos), data_dir.resolve())
    return documentos


def get_embeddings() -> HuggingFaceEmbeddings:
    """Un único lugar donde se define el modelo de embeddings."""

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def split_documents(documentos: list[Document]) -> list[Document]:
    """Aplica el chunking medido en TOKENS (no en caracteres)."""

    # from_tiktoken_encoder mide en tokens, que es como lo pide la consigna.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documentos)
    logger.info("Generé %d chunks (size=%d tokens, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def build_vectorstore(reset: bool = False) -> Chroma:
    """Crea (o reutiliza) la base vectorial persistente y la devuelve."""

    embeddings = get_embeddings()
    ya_existe = Path(PERSIST_DIR).exists() and any(Path(PERSIST_DIR).iterdir())

    if ya_existe and not reset:
        logger.info("♻️  Índice existente en %s — lo cargo sin reindexar", PERSIST_DIR)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    documentos = split_documents(load_documents())
    store = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    logger.info("🆕 Indexé %d chunks en la colección '%s'", len(documentos), COLLECTION_NAME)
    return store


if __name__ == "__main__":
    store = build_vectorstore()
    print("Documentos en la colección:", store._collection.count())
