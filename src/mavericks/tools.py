"""Mind + Tools: todo lo que tenga un resultado correcto lo calcula Python.

El LLM interpreta y redacta; estas funciones calculan. Mismo resultado
siempre, cero alucinación en los números.

Para que el agente las pueda llamar, pásalas en agent.py:

    Agent(..., tools=[cuota_mensual])

El docstring y los type hints son lo que el modelo lee para decidir cuándo
y cómo llamarlas: escríbelos claros.
"""


def cuota_mensual(capital: float, interes_anual_pct: float, anios: int) -> float:
    """Cuota mensual de un préstamo con sistema francés.

    Args:
        capital: importe prestado en euros.
        interes_anual_pct: tipo de interés anual en porcentaje, p. ej. 3.5
        anios: plazo en años.
    """
    n = anios * 12
    i = interes_anual_pct / 100 / 12
    if i == 0:
        return round(capital / n, 2)
    return round(capital * i / (1 - (1 + i) ** -n), 2)


def puntuar_encaje(cumplidos_must: int, total_must: int, cumplidos_nice: int, total_nice: int) -> float:
    """Puntuación de 0 a 100 de un candidato frente a una oferta.

    Los must-have pesan el 80 % y los nice-to-have el 20 %. Si falta algún
    must-have, la puntuación se limita a 50.
    """
    must = cumplidos_must / total_must if total_must else 1.0
    nice = cumplidos_nice / total_nice if total_nice else 1.0
    score = 100 * (0.8 * must + 0.2 * nice)
    return round(min(score, 50.0) if cumplidos_must < total_must else score, 1)
