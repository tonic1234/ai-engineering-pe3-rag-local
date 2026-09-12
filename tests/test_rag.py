"""tests/test_rag.py — Pruebas de la ingesta y de la construcción del pipeline.

APUNTE: el chunking no necesita llamar a ningún modelo, así que se puede testear
completo sin API key. Verifico tres cosas concretas:
  1. Que levante los documentos de /data y les ponga la metadata de la fuente.
  2. Que los chunks respeten el tamaño y TENGAN solape (si overlap=0 el test falla).
  3. Que el prompt de sistema exija responder solo con el contexto (el "filtro de
     veracidad" que pide la consigna).

Correr:  pytest -q
"""

from __future__ import annotations

from ingest import CHUNK_OVERLAP, CHUNK_SIZE, load_documents, split_documents
from rag import PROMPT, RespuestaRAG, TOP_K, format_docs


def test_carga_documentos_con_fuente():
    docs = load_documents()
    assert len(docs) >= 3
    assert all(d.metadata.get("source") for d in docs)


def test_chunking_respeta_tamano():
    chunks = split_documents(load_documents())
    assert chunks, "No se generó ningún chunk"
    assert all(len(c.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP for c in chunks)


def test_chunking_tiene_solape():
    # Si el splitter quedara con overlap=0, las ideas cortadas al medio se perderían.
    assert CHUNK_OVERLAP > 0
    documentos = load_documents()
    chunks = split_documents(documentos)
    assert len(chunks) > len(documentos), "Debería haber más chunks que documentos"


def test_top_k_en_rango_recomendado():
    # La consigna pide entre 3 y 5 fragmentos para no diluir el contexto.
    assert 3 <= TOP_K <= 5


def test_prompt_exige_respuesta_anclada():
    system = PROMPT.messages[0].prompt.template.lower()
    assert "contexto" in system and "no" in system


def test_format_docs_etiqueta_fuentes():
    class D:
        def __init__(self, txt, src):
            self.page_content = txt
            self.metadata = {"source": src}

    bloque = format_docs([D("uno", "a.txt"), D("dos", "b.txt")])
    assert "[a.txt]" in bloque and "[b.txt]" in bloque


def test_respuesta_no_encontrada_se_modela():
    r = RespuestaRAG(respuesta="No tengo acceso a esa información.", fuentes=[], encontrado=False)
    assert r.encontrado is False
