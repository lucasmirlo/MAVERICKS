"""Comprueba que todo está listo. Ejecútalo antes de salir de casa y al llegar:

    uv run python -m mavericks.doctor

Mensajes en ASCII a propósito: la consola de Windows a veces no imprime
símbolos como ✔ y el script se rompería justo cuando lo necesitas.
"""

import importlib
import sys
import time

import httpx

from mavericks import config

ok = True


def check(nombre: str, bien: bool, arreglo: str = "") -> None:
    global ok
    print(f"[{'OK' if bien else 'FALLA'}] {nombre}")
    if not bien:
        ok = False
        if arreglo:
            print(f"        -> {arreglo}")


def main() -> None:
    check(f"Python {sys.version.split()[0]}", sys.version_info >= (3, 12), "Necesitas Python 3.12 o superior")

    for mod in ("pydantic", "pydantic_ai", "numpy", "streamlit", "pypdf", "openai"):
        try:
            m = importlib.import_module(mod)
            check(f"{mod} {getattr(m, '__version__', '')}".strip(), True)
        except ImportError:
            check(mod, False, "Ejecuta: uv sync")

    try:
        v = httpx.get(f"{config.OLLAMA_URL}/api/version", timeout=3).json()["version"]
        check(f"Ollama {v} en {config.OLLAMA_URL}", True)
    except Exception:
        check("Ollama responde", False, "Abre la app de Ollama o ejecuta: ollama serve")
        return

    tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json().get("models", [])
    instalados = {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}
    for modelo in (config.LLM_MODEL, config.EMBED_MODEL):
        check(f"Modelo {modelo} descargado", modelo in instalados, f"Ejecuta: ollama pull {modelo}")
    if not ok:
        return

    # Calentamiento: la primera llamada carga el modelo en memoria y tarda mucho.
    from mavericks.embeddings import embed

    t = time.time()
    dim = len(embed(["prueba"])[0])
    check(f"Embeddings: vector de {dim} dimensiones en {time.time() - t:.1f}s", dim > 0)

    t = time.time()
    r = httpx.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": config.LLM_MODEL, "prompt": "Responde solo: ok", "stream": False},
        timeout=300,
    )
    check(f"LLM responde en {time.time() - t:.1f}s (la 1ª vez incluye cargarlo)", r.status_code == 200)

    from mavericks import store

    n = store.contar()
    check(f"Índice con {n} trozos", n > 0, "Ejecuta: uv run python -m mavericks.ingest")


if __name__ == "__main__":
    main()
    print("\nTodo listo." if ok else "\nHay cosas que arreglar arriba.")
    sys.exit(0 if ok else 1)
