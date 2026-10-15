"""Índice vectorial en NumPy puro (plan B si Chroma falla).

Con los documentos de un reto (cientos o pocos miles de trozos) buscar por
fuerza bruta es instantáneo: un producto matriz-vector. Y es exacto, no
aproximado como el HNSW de Chroma.

En disco son dos archivos en index/numpy/:
    vectores.npy   una fila por trozo, normalizada
    trozos.json    texto, fuente y página de cada fila, en el mismo orden
"""

import json

import numpy as np

from mavericks import config

DIR = config.INDEX_DIR / "numpy"
VECTORES = DIR / "vectores.npy"
TROZOS = DIR / "trozos.json"

_cache: dict = {"mtime": None, "vec": None, "trozos": None}


def _normalizar(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norma = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norma == 0, 1, norma)


def guardar(vectores: list[list[float]], trozos: list[dict]) -> None:
    """Sobrescribe el índice entero. `trozos[i]` describe la fila `vectores[i]`."""
    if len(vectores) != len(trozos):
        raise ValueError(f"{len(vectores)} vectores para {len(trozos)} trozos")
    DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORES, _normalizar(vectores))
    TROZOS.write_text(json.dumps(trozos, ensure_ascii=False), encoding="utf-8")
    _cache["mtime"] = None  # fuerza a recargar en la siguiente búsqueda


def _cargar() -> tuple[np.ndarray, list[dict]]:
    if not VECTORES.exists() or not TROZOS.exists():
        return np.zeros((0, 0), dtype=np.float32), []
    mtime = VECTORES.stat().st_mtime
    if _cache["mtime"] != mtime:  # solo relee del disco si el índice ha cambiado
        _cache.update(
            mtime=mtime,
            vec=np.load(VECTORES),
            trozos=json.loads(TROZOS.read_text(encoding="utf-8")),
        )
    return _cache["vec"], _cache["trozos"]


def contar() -> int:
    return len(_cargar()[1])


def buscar(vector: list[float], k: int) -> list[tuple[dict, float]]:
    """Los k trozos más cercanos como [(trozo, distancia coseno)], del más al menos parecido.

    Distancia coseno = 1 - similitud: 0 es idéntico, 1 es sin relación.
    """
    vec, trozos = _cargar()
    if not trozos:
        return []
    similitudes = vec @ _normalizar(vector)
    k = min(k, len(trozos))
    mejores = np.argpartition(-similitudes, k - 1)[:k]
    mejores = mejores[np.argsort(-similitudes[mejores])]
    return [(trozos[i], float(1 - similitudes[i])) for i in mejores]
