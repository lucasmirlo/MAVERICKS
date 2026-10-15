"""El índice vectorial. Elige la implementación según config.VECTOR_STORE.

    chroma  (por defecto)  store_chroma.py
    numpy   (plan B)       store_numpy.py

Las dos tienen la misma interfaz, así que ingest.py y retrieve.py no
cambian. Para pasar a NumPy si Chroma se cae:

    Windows (PowerShell):  $env:MAVERICKS_STORE = "numpy"
    Mac / Linux:           export MAVERICKS_STORE=numpy

Después hay que reindexar: cada uno guarda su índice en su propia carpeta.
"""

from mavericks import config

if config.VECTOR_STORE == "numpy":
    from mavericks.store_numpy import buscar, contar, guardar
elif config.VECTOR_STORE == "chroma":
    from mavericks.store_chroma import buscar, contar, guardar
else:
    raise ValueError(f"MAVERICKS_STORE debe ser 'chroma' o 'numpy', no {config.VECTOR_STORE!r}")

__all__ = ["buscar", "contar", "guardar"]
