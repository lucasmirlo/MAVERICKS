"""La única función que convierte texto en vectores.

ingest.py y retrieve.py la importan de aquí. Si cada uno calculara los
embeddings por su cuenta, un cambio de modelo en un solo lado haría que las
búsquedas devolvieran basura sin dar ningún error.

Usa el endpoint compatible con OpenAI de Ollama (/v1/embeddings) con el
cliente `openai`, que ya viene instalado como dependencia de pydantic-ai.
"""

from functools import lru_cache

from openai import OpenAI

from mavericks import config


@lru_cache
def _client() -> OpenAI:
    # Ollama no pide clave, pero el cliente exige que haya una
    return OpenAI(base_url=f"{config.OLLAMA_URL}/v1", api_key="ollama")


# Algunos modelos de embeddings se entrenaron con un prefijo que dice si el
# texto es un documento o una pregunta. Sin él, todas las distancias se
# parecen y la búsqueda apenas distingue un trozo de otro.
PREFIJOS = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


def embed(textos: list[str], consulta: bool = False) -> list[list[float]]:
    """Devuelve un vector por cada texto, en el mismo orden.

    consulta=True para la pregunta del usuario; False (por defecto) para los
    trozos de los documentos. Cambiar esto obliga a reindexar.
    """
    if not textos:
        return []
    doc, query = PREFIJOS.get(config.EMBED_MODEL.split(":")[0], ("", ""))
    prefijo = query if consulta else doc
    r = _client().embeddings.create(model=config.EMBED_MODEL, input=[prefijo + t for t in textos])
    return [d.embedding for d in r.data]
