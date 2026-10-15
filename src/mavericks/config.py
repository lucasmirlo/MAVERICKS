"""Toda la configuración en un sitio.

Cada valor se puede cambiar sin tocar código con una variable de entorno,
para que quien tenga menos VRAM use otro modelo:

    Windows (PowerShell):  $env:MAVERICKS_LLM = "mistral-small3.2:24b"
    Mac / Linux:           export MAVERICKS_LLM=mistral-small3.2:24b
"""

import os
from pathlib import Path

# PydanticAI imprime un banner publicitario en cada ejecución; fuera
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# src/mavericks/config.py -> la raíz del repo está dos niveles por encima del paquete
ROOT = Path(__file__).resolve().parents[2]

# --- Ollama ---------------------------------------------------------------
OLLAMA_URL = os.getenv("MAVERICKS_OLLAMA_URL", "http://localhost:11434")
# llama3.1:8b (5 GB) cabe entero en una GPU de 8 GB. mistral-small3.2:24b (15 GB)
# es mejor, pero sin 16 GB de VRAM se reparte con la CPU y tarda minutos por respuesta.
LLM_MODEL = os.getenv("MAVERICKS_LLM", "llama3.1:8b")
EMBED_MODEL = os.getenv("MAVERICKS_EMBED", "nomic-embed-text")

# --- Rutas ------------------------------------------------------------------
DATA_DIR = ROOT / "data"          # los documentos del reto
INDEX_DIR = ROOT / "index"        # el índice; se regenera con `python -m mavericks.ingest`
EVAL_CSV = ROOT / "eval" / "eval.csv"

# --- Troceado -----------------------------------------------------------------
CHUNK_WORDS = 300     # palabras por trozo
CHUNK_OVERLAP = 50    # palabras que se repiten entre trozos seguidos

# --- Recuperación ----------------------------------------------------------
TOP_K = 5
# Distancia coseno: 0 = idéntico, 1 = sin relación. Por encima de este umbral
# un trozo no cuenta como contexto. Ajústalo mirando distancias reales en la app.
MAX_DISTANCE = 0.55
