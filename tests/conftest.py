"""conftest.py — Configuración común para los tests.

Los tests no llaman a ningún modelo real, pero las clases de LangChain verifican que
exista una API key al construirse. Le pongo una clave dummy para que no falle el import.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
