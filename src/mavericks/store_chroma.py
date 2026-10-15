"""Índice vectorial en Chroma.

Misma interfaz que store_numpy.py: guardar(), contar() y buscar(). El resto
del proyecto no sabe cuál de los dos está usando.

En disco: index/chroma/.
"""

from functools import lru_cache

import chromadb
from chromadb.config import Settings

from mavericks import config

DIR = config.INDEX_DIR / "chroma"
COLECCION = "docs"
LOTE = 500  # Chroma limita cuántos elementos acepta por llamada a add()


@lru_cache
def _cliente():
    return chromadb.PersistentClient(path=str(DIR), settings=Settings(anonymized_telemetry=False))


def _coleccion():
    return _cliente().get_or_create_collection(
        COLECCION,
        configuration={"hnsw": {"space": "cosine"}},  # distancia = 1 - similitud coseno
        embedding_function=None,  # los embeddings los ponemos nosotros, siempre
    )


def guardar(vectores: list[list[float]], trozos: list[dict]) -> None:
    """Sobrescribe el índice entero. `trozos[i]` describe la fila `vectores[i]`."""
    if len(vectores) != len(trozos):
        raise ValueError(f"{len(vectores)} vectores para {len(trozos)} trozos")
    try:
        _cliente().delete_collection(COLECCION)
    except Exception:
        pass  # no existía todavía
    col = _coleccion()
    for i in range(0, len(trozos), LOTE):
        lote = trozos[i : i + LOTE]
        col.add(
            ids=[t["id"] for t in lote],
            documents=[t["texto"] for t in lote],
            metadatas=[{"fuente": t["fuente"], "pagina": t["pagina"]} for t in lote],
            embeddings=vectores[i : i + LOTE],
        )


def contar() -> int:
    return _coleccion().count()


def buscar(vector: list[float], k: int) -> list[tuple[dict, float]]:
    """Los k trozos más cercanos como [(trozo, distancia coseno)], del más al menos parecido."""
    col = _coleccion()
    n = col.count()
    if n == 0:
        return []
    r = col.query(
        query_embeddings=[vector],
        n_results=min(k, n),
        include=["documents", "metadatas", "distances"],
    )
    return [
        ({"id": id_, "texto": doc, "fuente": meta["fuente"], "pagina": meta["pagina"]}, dist)
        for id_, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]
