"""El número que llevas a la presentación.

    uv run python eval/run_eval.py              # solo recuperación: rápido, sin LLM
    uv run python eval/run_eval.py --generar    # además, ¿la respuesta cita la fuente correcta?

eval/eval.csv tiene dos columnas: pregunta, fuente_esperada.
Escribid 20-30 filas la primera tarde, con preguntas reales del reto.
"""

import csv
import sys

from mavericks import config
from mavericks.retrieve import buscar


def main(generar: bool = False) -> None:
    with open(config.EVAL_CSV, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))

    hits = citas_ok = 0
    for fila in filas:
        fuentes = [fr.fuente for fr in buscar(fila["pregunta"])]
        hit = fila["fuente_esperada"] in fuentes
        hits += hit
        linea = f"[{'OK' if hit else '--'}] {fila['pregunta'][:60]}"

        if generar:
            from mavericks.agent import responder

            respuesta, _ = responder(fila["pregunta"])
            cita = any(c.fuente == fila["fuente_esperada"] for c in respuesta.citas)
            citas_ok += cita
            linea += f"   cita correcta: {'sí' if cita else 'no'}"
        print(linea)

    n = len(filas)
    print(f"\nHit@{config.TOP_K} (recuperación): {hits}/{n} = {hits / n:.0%}")
    if generar:
        print(f"Respuestas que citan la fuente correcta: {citas_ok}/{n} = {citas_ok / n:.0%}")


if __name__ == "__main__":
    main(generar="--generar" in sys.argv)
