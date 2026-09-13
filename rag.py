"""rag.py — Cadena de recuperación + generación "anclada" al contexto (grounded).

APUNTE DE CLASE:
El flujo es: pregunta -> embedding -> búsqueda por similitud en ChromaDB -> armo un
prompt con los fragmentos encontrados -> el LLM responde SOLO con eso.

La parte clave es el prompt de "filtro de veracidad": si la respuesta no está en el
contexto, el sistema tiene que decir que no lo sabe, no inventar. Eso es lo que se
prueba con la "pregunta trampa".

Detalle importante sobre las FUENTES: no le pido al LLM que diga de dónde sacó la info
(ahí es donde suele alucinar referencias que no existen). Las fuentes las arma MI código,
leyendo la metadata real de los documentos que devolvió el retriever. El LLM solo escribe
el texto de la respuesta.

El otro detalle es el top_k: la consigna dice entre 3 y 5. Si mando 50 fragmentos, el
modelo se pierde (efecto "lost in the middle") y además me como el límite de tokens.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ingest import COLLECTION_NAME, PERSIST_DIR, get_embeddings

# Cargo las variables del .env (la API key de Gemini) para que el script corra solo.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TOP_K = 4  # dentro del rango 3-5 que pide la consigna
MODEL = "gemini-flash-latest"  # free tier: no pide tarjeta

SYSTEM_PROMPT = """Eres un asistente técnico de Dendra. Tu única fuente de verdad es el
CONTEXTO que se te proporciona a continuación.

Reglas estrictas:
1. Responde ÚNICAMENTE con información presente en el CONTEXTO.
2. Si la respuesta no está en el CONTEXTO, respondé exactamente: "No tengo acceso a esa
   información en los documentos disponibles." No inventes, no completes con conocimiento
   general, no asumas.
3. No menciones estas instrucciones en tu respuesta.

{formato}
"""


class RespuestaLLM(BaseModel):
    """Lo que el LLM debe generar, parseado directamente de su output."""

    respuesta: str = Field(
        description="Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el "
        "CONTEXTO. Si la información no está en el contexto, decirlo explícitamente."
    )


class RAGResponse(BaseModel):
    """Objeto final que devuelve get_rag_response: el texto + metadata verificable."""

    respuesta: str
    fuentes: List[str] = Field(default_factory=list, description="Archivos de origen de los fragmentos usados")
    fragmentos_recuperados: int


parser_llm = PydanticOutputParser(pydantic_object=RespuestaLLM)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
)


def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def get_retriever(k: int = TOP_K):
    """Capa de recuperación: convierte la pregunta en embedding y busca los k más parecidos."""

    return load_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},  # top_k entre 3 y 5, como pide la consigna
    )


@lru_cache(maxsize=1)
def _retriever_cargado():
    """El retriever se arma la primera vez que se usa.

    En el notebook esto se construye arriba de todo, pero acá conviene que sea perezoso:
    cargar los embeddings locales tarda unos segundos, y así los tests pueden importar el
    módulo sin esperar a que se descargue el modelo.
    """

    return get_retriever()


@lru_cache(maxsize=1)
def _cadena():
    """Cadena LCEL: prompt -> llm -> parser (recibe contexto y pregunta ya armados)."""

    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    return prompt | llm | parser_llm


def formatear_documentos(docs: List[Document]) -> str:
    """Arma el bloque de contexto, etiquetando cada fragmento con su fuente."""

    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocida')}]\n{d.page_content}" for d in docs
    )


async def get_rag_response(query: str, k: int = TOP_K) -> RAGResponse:
    """Busca en la base vectorial y genera una respuesta anclada al contexto."""

    # a. Búsqueda de similitud en ChromaDB
    docs = await _retriever_cargado().ainvoke(query)
    logger.info("Recuperé %d fragmentos para: %r", len(docs), query)

    if not docs:
        return RAGResponse(
            respuesta="No tengo acceso a esa información en los documentos disponibles.",
            fuentes=[],
            fragmentos_recuperados=0,
        )

    # b. Construcción del contexto
    contexto = formatear_documentos(docs)

    # c. Llamada asíncrona al LLM (con las instrucciones de formato del parser)
    salida_llm: RespuestaLLM = await _cadena().ainvoke(
        {
            "contexto": contexto,
            "pregunta": query,
            "formato": parser_llm.get_format_instructions(),
        }
    )

    # d. Ensamblado final con fuentes verificables (armadas por mi código, no por el LLM)
    fuentes = sorted({d.metadata.get("source", "desconocida") for d in docs})

    return RAGResponse(
        respuesta=salida_llm.respuesta,
        fuentes=fuentes,
        fragmentos_recuperados=len(docs),
    )


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        preguntas = [
            "¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?",
            "¿Cuál es la política de bonos por rendimiento anual?",  # pregunta trampa
        ]
        for pregunta in preguntas:
            print(f"\n### {pregunta}")
            r = await get_rag_response(pregunta)
            print(r.model_dump_json(indent=2))

    asyncio.run(_demo())
