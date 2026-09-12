"""ingest.py — Ingesta: leer documentos, partirlos en chunks y guardarlos en ChromaDB.

APUNTE DE CLASE (unidad de chunking):
La idea del chunking es partir el texto en pedazos del tamaño justo. Si los hago muy
chicos pierdo el contexto de la frase; si los hago muy grandes, el embedding "diluye"
el significado y la búsqueda se vuelve imprecisa. La consigna pide ~500 tokens con 50
de solape (overlap), que es el valor que venimos usando.

El overlap sirve para que una idea que queda cortada entre dos chunks no se pierda del
todo: el final del chunk N se repite al principio del N+1.

Otra cosa importante: el script chequea si la colección ya existe. Si ya está, no
reindexa (así no gasto tiempo ni plata cada vez que lo corro).
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PERSIST_DIR = "./vectorstore"
COLLECTION = "apuntes"
DATA_DIR = Path("./data")

CHUNK_SIZE = 1000       # en caracteres (aprox. 500 tokens en español)
CHUNK_OVERLAP = 100     # 50 tokens de solape aprox. -> 100 caracteres


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


def split_documents(documentos: list[Document]) -> list[Document]:
    """Aplica el chunking con solape."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Orden de corte: primero por párrafo, y si no alcanza baja a oraciones.
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documentos)
    logger.info("Generé %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def get_embeddings() -> OpenAIEmbeddings:
    """Un único lugar donde se define el modelo de embeddings.

    Regla de oro: indexar y consultar SIEMPRE con el mismo modelo. Si mezclo modelos,
    las distancias no significan nada y la búsqueda devuelve cualquier cosa.
    """

    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vectorstore(reset: bool = False) -> Chroma:
    """Crea (o reutiliza) la base vectorial persistente y la devuelve."""

    embeddings = get_embeddings()

    if not reset and Path(PERSIST_DIR).exists():
        logger.info("La base ya existe en %s — no reindexo (usá reset=True para forzar)", PERSIST_DIR)
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    documentos = split_documents(load_documents())
    store = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )
    logger.info("Indexé %d chunks en la colección '%s'", len(documentos), COLLECTION)
    return store


if __name__ == "__main__":
    build_vectorstore()
