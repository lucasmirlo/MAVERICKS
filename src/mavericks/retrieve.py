"""Fase 2, primer paso: pregunta -> los k trozos más parecidos.

Cuando una respuesta sea mala, mira PRIMERO lo que devuelve esta función.
Si el trozo correcto no está aquí, el problema es la ingesta o la
recuperación, no el prompt.

    uv run python -m mavericks.retrieve "tu pregunta"
"""

import sys

from mavericks import config, store
from mavericks.embeddings import embed
from mavericks.schemas import Fragmento


def buscar(pregunta: str, k: int = config.TOP_K) -> list[Fragmento]:
    if store.contar() == 0:
        raise RuntimeError("El índice está vacío. Ejecuta: uv run python -m mavericks.ingest")
    return [
        Fragmento(texto=t["texto"], fuente=t["fuente"], pagina=t["pagina"], distancia=d)
        for t, d in store.buscar(embed([pregunta])[0], k)
    ]


def relevantes(fragmentos: list[Fragmento]) -> list[Fragmento]:
    """Solo los que pasan el umbral. Si no queda ninguno, el sistema dice que no sabe."""
    return [f for f in fragmentos if f.distancia <= config.MAX_DISTANCE]


if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) or "¿De qué tratan los documentos?"
    for f in buscar(pregunta):
        marca = "OK " if f.distancia <= config.MAX_DISTANCE else "-- "
        print(f"{marca}{f.distancia:.3f}  {f.fuente} p.{f.pagina}  {f.texto[:90]}...")
