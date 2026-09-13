# Sistema de recuperación semántica local (RAG)

Pre-entrega 3 del curso **AI Engineering** (Coderhouse).
Flujo RAG end-to-end **local**: ingesta de documentos → chunking por tokens → **ChromaDB**
→ recuperación por similitud → respuesta generada **solo** con el contexto recuperado.

## Qué hay adentro

| Archivo | Qué hace |
|---|---|
| `ingest.py` | Carga los documentos de `data/`, hace el chunking y los persiste en ChromaDB. |
| `rag.py` | `get_rag_response(query)`: recupera fragmentos y genera una respuesta anclada. |
| `data/` | 4 políticas internas de ejemplo (`.txt`): vacaciones, teletrabajo, seguridad, onboarding. |
| `tests/` | Pruebas del chunking, del `top_k` y del prompt "filtro de veracidad". |

## Contexto del dataset

Uso como dominio de ejemplo el de **Dendra**, la agencia donde trabajo: son documentos del
tipo que existen en cualquier empresa (vacaciones, teletrabajo, seguridad, onboarding), lo
que hace que el RAG corra sobre un caso realista y no un texto genérico de prueba.

**Importante**: el contenido de los `.txt` es **inventado** — solo imita la lógica de una
política interna (rangos de antigüedad, plazos, requisitos). No son las políticas reales de
la empresa ni contienen datos reales de nadie.

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # completar GOOGLE_API_KEY (Gemini, gratis)

python ingest.py            # indexa /data en ./vectorstore (la primera vez)
python rag.py               # hace una pregunta real y una "pregunta trampa"
```

La base queda persistida en `./vectorstore`. El script de ingesta **no reindexa** si la
colección ya existe (para no gastar tiempo ni cómputo); para forzarlo, usar
`build_vectorstore(reset=True)`.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `GOOGLE_API_KEY` | Requerida para el LLM de Gemini (free tier, sin tarjeta). |

> Los **embeddings son locales** (`sentence-transformers/all-MiniLM-L6-v2`): no necesitan
> API key ni cuestan nada. Solo se descarga el modelo la primera vez.

## Ejemplo de salida

```
### ¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?
{
  "respuesta": "A un empleado con 5 años de antigüedad le corresponden 20 días hábiles
                de vacaciones por año calendario.",
  "fuentes": ["politica_vacaciones.txt"],
  "fragmentos_recuperados": 4
}

### ¿Cuál es la política de bonos por rendimiento anual?
{
  "respuesta": "No tengo acceso a esa información en los documentos disponibles.",
  "fuentes": ["onboarding_nuevos_empleados.txt", "politica_seguridad_informatica.txt",
              "politica_teletrabajo.txt", "politica_vacaciones.txt"],
  "fragmentos_recuperados": 4
}
```

## Decisiones de diseño

- **Chunking**: `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `chunk_size=500`
  y `chunk_overlap=50` **tokens** (no caracteres). El chunk_size es un techo, no un piso:
  un archivo corto entra entero en un chunk y está bien.
- **Embeddings locales** (`all-MiniLM-L6-v2`): mismo modelo para indexar y para consultar.
  Si se mezclan modelos, la distancia vectorial no significa nada.
- **`top_k = 4`**: dentro del rango 3-5 que recomienda la consigna. Pasar 50 fragmentos
  degrada la atención del modelo y puede superar el límite de tokens.
- **Prompt de filtro de veracidad**: el system prompt obliga a responder solo con el
  CONTEXTO y a decir exactamente *"No tengo acceso a esa información en los documentos
  disponibles."* si la respuesta no está ahí. Se prueba con una pregunta trampa.
- **Fuentes sin alucinar**: las fuentes NO las escribe el LLM. Las arma el código a partir
  de la metadata real de los documentos recuperados; el LLM solo genera el texto.
- **Salida estructurada**: `RespuestaLLM` (lo que genera el modelo, vía
  `PydanticOutputParser`) + `RAGResponse` (respuesta + fuentes + `fragmentos_recuperados`).

## Tests

```bash
pytest -q
```

El chunking y el armado del prompt se prueban sin API key ni descarga de modelos.
