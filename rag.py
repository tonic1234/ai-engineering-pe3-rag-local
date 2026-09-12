"""rag.py — Cadena de recuperación + generación "anclada" al contexto (grounded).

APUNTE DE CLASE:
El flujo es: pregunta -> embedding -> búsqueda por similitud en ChromaDB -> armo un
prompt con los fragmentos encontrados -> el LLM responde SOLO con eso.

La parte clave es el prompt de "filtro de veracidad": si la respuesta no está en el
contexto, el sistema tiene que decir que no lo sabe, no inventar. Eso es lo que se
prueba con la "pregunta trampa".

El otro detalle es el top_k: la consigna dice entre 3 y 5. Si mando 50 fragmentos, el
modelo se pierde (efecto "lost in the middle") y además me como el límite de tokens.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ingest import get_embeddings, PERSIST_DIR, COLLECTION
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TOP_K = 4  # dentro del rango 3-5 que pide la consigna

SYSTEM_PROMPT = (
    "Sos un asistente técnico que responde EXCLUSIVAMENTE con la información del "
    "CONTEXTO entregado. Si la respuesta no está en el contexto, respondé que no tenés "
    "acceso a esa información. No uses conocimiento propio. Citá la fuente entre "
    "corchetes cuando uses un fragmento, por ejemplo [apuntes-asyncio.txt]."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
)


class RespuestaRAG(BaseModel):
    """Salida estructurada: el texto y las fuentes que usó."""

    respuesta: str = Field(description="Respuesta basada solo en el contexto")
    fuentes: list[str] = Field(default_factory=list, description="Documentos citados")
    encontrado: bool = Field(description="False si la respuesta no estaba en el contexto")


def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def format_docs(docs: list[Document]) -> str:
    """Arma el bloque de contexto, etiquetando cada fragmento con su fuente."""

    return "\n\n".join(f"[{d.metadata.get('source', 'doc')}] {d.page_content}" for d in docs)


def build_rag_chain(model: str = "gpt-4o-mini"):
    """Devuelve la cadena LCEL: prompt -> modelo con salida estructurada -> parser."""

    llm = ChatOpenAI(model=model, temperature=0)
    parser = PydanticOutputParser(pydantic_object=RespuestaRAG)
    return PROMPT | llm.with_structured_output(RespuestaRAG), parser


_chain, _parser = build_rag_chain()


async def get_rag_response(query: str, k: int = TOP_K) -> RespuestaRAG:
    """Busca en la base vectorial y genera una respuesta anclada al contexto."""

    store = load_vectorstore()
    # Búsqueda por similitud: acá se convierte la pregunta en embedding y se comparan
    # las distancias contra los chunks guardados.
    docs = store.similarity_search(query, k=k)
    logger.info("Recuperé %d fragmentos para: %r", len(docs), query)

    if not docs:
        return RespuestaRAG(respuesta="No tengo acceso a esa información.", fuentes=[], encontrado=False)

    resultado: RespuestaRAG = await _chain.ainvoke(
        {"contexto": format_docs(docs), "pregunta": query}
    )

    # Si el modelo marcó "no encontrado", igual devuelvo las fuentes que miró (útil
    # para depurar por qué no encontró la respuesta).
    if not resultado.fuentes:
        resultado.fuentes = sorted({d.metadata.get("source", "doc") for d in docs})
    return resultado


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        for pregunta in (
            "¿Qué pasa si uso la versión síncrona del cliente dentro de una función async?",
            "¿Cómo cocino un risotto de hongos?",  # pregunta trampa
        ):
            print(f"\n### {pregunta}")
            r = await get_rag_response(pregunta)
            print(r.model_dump_json(indent=2))

    asyncio.run(_demo())
