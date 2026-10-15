"""Fase 1: documentos -> trozos -> embeddings -> índice.

Se ejecuta una vez, antes de la demo, y cada vez que cambien los documentos:

    uv run python -m mavericks.ingest

Lee PDF, TXT y MD de data/ y rehace el índice entero, así nunca quedan
trozos duplicados.
"""

from pathlib import Path

from pypdf import PdfReader

from mavericks import config, store
from mavericks.embeddings import embed

BATCH = 32  # trozos por llamada de embeddings


def leer_paginas(path: Path) -> list[tuple[int, str]]:
    """[(número de página, texto), ...]. Un TXT o MD cuenta como página 1."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8"))]


def trocear(texto: str) -> list[str]:
    palabras = texto.split()
    paso = config.CHUNK_WORDS - config.CHUNK_OVERLAP
    trozos = []
    for i in range(0, len(palabras), paso):
        trozo = " ".join(palabras[i : i + config.CHUNK_WORDS])
        if trozo:
            trozos.append(trozo)
        if i + config.CHUNK_WORDS >= len(palabras):
            break
    return trozos


def main() -> None:
    archivos = sorted(
        p for p in config.DATA_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}
    )
    if not archivos:
        print(f"[AVISO] No hay documentos en {config.DATA_DIR}")
        return

    trozos: list[dict] = []
    for path in archivos:
        n_antes = len(trozos)
        for pagina, texto in leer_paginas(path):
            for j, t in enumerate(trocear(texto)):
                trozos.append(
                    {"id": f"{path.name}-p{pagina}-c{j}", "texto": t, "fuente": path.name, "pagina": pagina}
                )
        n = len(trozos) - n_antes
        aviso = "   <- 0 trozos: ¿PDF escaneado? Necesita OCR" if n == 0 else ""
        print(f"  {path.name}: {n} trozos{aviso}")

    vectores: list[list[float]] = []
    for i in range(0, len(trozos), BATCH):
        vectores += embed([t["texto"] for t in trozos[i : i + BATCH]])

    store.guardar(vectores, trozos)
    print(f"[OK] {store.contar()} trozos indexados en {config.INDEX_DIR.name}/")


if __name__ == "__main__":
    main()
