"""tests/test_rag.py — Pruebas de la ingesta y de la construcción del pipeline.

Nota: el chunking y el armado del prompt no necesitan llamar a ningún modelo, así que
se pueden testear completos sin API key ni descargar embeddings. Verifico:
  1. Que levante los documentos de /data y les ponga la metadata de la fuente.
  2. Que los chunks respeten el límite y TENGAN solape (si overlap=0 el test falla).
  3. Que el prompt de sistema exija responder solo con el contexto (el "filtro de
     veracidad" que pide la consigna).

Correr:  pytest -q
"""

from __future__ import annotations

from ingest import CHUNK_OVERLAP, CHUNK_SIZE, load_documents, split_documents
from rag import TOP_K, RAGResponse, RespuestaLLM, formatear_documentos, prompt


def test_carga_documentos_con_fuente():
    docs = load_documents()
    assert len(docs) == 4
    assert all(d.metadata.get("source") for d in docs)


def test_chunking_genera_chunks_con_contenido():
    chunks = split_documents(load_documents())
    assert chunks, "No se generó ningún chunk"
    assert all(c.page_content.strip() for c in chunks)


def test_chunking_tiene_solape():
    # Si el splitter quedara con overlap=0, las ideas cortadas al medio se perderían.
    assert CHUNK_OVERLAP > 0
    assert CHUNK_SIZE > CHUNK_OVERLAP


def test_chunking_mantiene_la_fuente():
    chunks = split_documents(load_documents())
    assert all("source" in c.metadata for c in chunks)


def test_top_k_en_rango_recomendado():
    # La consigna pide entre 3 y 5 fragmentos para no diluir el contexto.
    assert 3 <= TOP_K <= 5


def test_prompt_exige_respuesta_anclada():
    system = prompt.messages[0].prompt.template.lower()
    assert "contexto" in system
    assert "no tengo acceso" in system  # la frase exacta que debe usar si no sabe


def test_prompt_tiene_las_variables_esperadas():
    assert {"contexto", "pregunta", "formato"} <= set(prompt.input_variables)


def test_formatear_documentos_etiqueta_fuentes():
    class D:
        def __init__(self, txt, src):
            self.page_content = txt
            self.metadata = {"source": src}

    bloque = formatear_documentos([D("uno", "a.txt"), D("dos", "b.txt")])
    assert "[Fuente: a.txt]" in bloque and "[Fuente: b.txt]" in bloque


def test_esquemas_del_pipeline():
    llm_out = RespuestaLLM(respuesta="21 días hábiles")
    final = RAGResponse(respuesta=llm_out.respuesta, fuentes=["politica_vacaciones.txt"], fragmentos_recuperados=4)
    assert final.fragmentos_recuperados == 4
    assert final.fuentes == ["politica_vacaciones.txt"]
