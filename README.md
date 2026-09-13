# Sistema de recuperación semántica local (RAG)

Pre-entrega 3 del curso **AI Engineering** (Coderhouse).
Flujo RAG end-to-end **local**: ingesta de documentos → chunking → **ChromaDB** →
recuperación por similitud → respuesta generada **solo** con el contexto recuperado.

## Qué hay adentro

| Archivo | Qué hace |
|---|---|
| `ingest.py` | Carga los documentos de `data/`, hace el chunking y los persiste en ChromaDB. |
| `rag.py` | `get_rag_response(query)`: recupera fragmentos y genera una respuesta anclada. |
| `data/` | Dataset de ejemplo (apuntes en `.txt` sobre asyncio, LCEL y RAG). |
| `tests/` | Pruebas de chunking, del `top_k` y del prompt "filtro de veracidad". |

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # completar OPENAI_API_KEY

python ingest.py            # indexa /data en ./vectorstore (la primera vez)
python rag.py               # hace una pregunta real y una "pregunta trampa"
```

La base queda persistida en `./vectorstore`. El script de ingesta **no reindexa** si la
colección ya existe (para no gastar tiempo ni tokens); para forzarlo, usar
`build_vectorstore(reset=True)`.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` | Requerida: se usa para los embeddings y para la generación. |

## Ejemplo de salida

```
### ¿Qué pasa si uso la versión síncrona del cliente dentro de una función async?
{
  "respuesta": "Bloquea el event loop: mientras esa función sincrónica corre, el programa
                no atiende a ningún otro usuario. Se soluciona con await asyncio.sleep o
                con asyncio.to_thread para librerías que no son async. [apuntes-asyncio.txt]",
  "fuentes": ["apuntes-asyncio.txt"],
  "encontrado": true
}

### ¿Cómo cocino un risotto de hongos?
{
  "respuesta": "No tengo acceso a esa información.",
  "fuentes": [],
  "encontrado": false
}
```

## Decisiones de diseño

- **Chunking**: `RecursiveCharacterTextSplitter` con `chunk_size=1000` caracteres y
  `chunk_overlap=100` (≈500 y 50 tokens). Partir por párrafo y bajar a oración si hace falta.
- **Mismo modelo de embeddings** para indexar y para consultar (`text-embedding-3-small`).
  Si se mezclan modelos, la distancia vectorial no significa nada.
- **`top_k = 4`**: dentro del rango 3-5 que recomienda la consigna. Pasar 50 fragmentos
  degrada la atención del modelo y puede superar el límite de tokens.
- **Prompt de filtro de veracidad**: el system prompt obliga a responder solo con el
  CONTEXTO y a decir que no lo sabe si la respuesta no está ahí. Se prueba con una
  pregunta trampa.
- **Salida estructurada**: `RespuestaRAG` (respuesta + fuentes + `encontrado`) permite
  verificar la no-alucinación por código, no a ojo.

## Tests

```bash
pytest -q
```

El chunking y el armado del prompt se prueban sin API key.
