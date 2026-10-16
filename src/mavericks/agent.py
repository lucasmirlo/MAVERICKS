"""Fase 2, segundo paso: pregunta + trozos -> Respuesta validada.

    uv run python -m mavericks.agent "tu pregunta"
"""

import sys
from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider

from mavericks import config
from mavericks.retrieve import buscar, relevantes
from mavericks.schemas import Fragmento, Respuesta

INSTRUCCIONES = """\
Eres un asistente que responde en español usando SOLO el contexto que se te da.
Cada fragmento del contexto empieza con [fuente, página].
Primero razona en `razonamiento`: qué fragmentos sirven y, si la respuesta exige
contar, sumar o aplicar condiciones, hazlo paso a paso.
Si el contexto responde aunque sea en parte, responde con lo que hay y di qué falta.
Pon suficiente=false solo si el contexto no tiene nada que ver con la pregunta.
No inventes datos. Cita las fuentes de las que sale cada afirmación.
"""


@lru_cache
def get_agent() -> Agent:
    model = OllamaModel(
        config.LLM_MODEL,
        provider=OllamaProvider(base_url=f"{config.OLLAMA_URL}/v1"),
    )
    return Agent(
        model,
        output_type=NativeOutput(Respuesta),  # Ollama >= 0.5 respeta el esquema al generar
        instructions=INSTRUCCIONES,
        retries=2,
        # Tope de longitud: si el modelo entra en bucle, se corta aquí en vez de
        # llenar el contexto y empezar a decir sinsentidos
        model_settings={"max_tokens": 800},
        # tools=[cuota_mensual],  # Mind + Tools: ver tools.py
    )


def construir_prompt(pregunta: str, fragmentos: list[Fragmento]) -> str:
    contexto = "\n\n".join(f"[{f.fuente}, p. {f.pagina}]\n{f.texto}" for f in fragmentos)
    return f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"


def responder(pregunta: str) -> tuple[Respuesta, list[Fragmento]]:
    """Devuelve la respuesta y TODOS los fragmentos recuperados, para depurar."""
    recuperados = buscar(pregunta)
    contexto = relevantes(recuperados)
    if not contexto:
        # Ningún trozo pasa el umbral: decimos que no sabemos sin llamar al modelo
        sin_info = Respuesta(
            razonamiento="Ningún fragmento supera el umbral de distancia; no se ha llamado al modelo.",
            respuesta="No tengo información suficiente en los documentos para responder a eso.",
            citas=[],
            suficiente=False,
        )
        return sin_info, recuperados
    resultado = get_agent().run_sync(construir_prompt(pregunta, contexto))
    return resultado.output, recuperados


if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) or "¿De qué tratan los documentos?"
    respuesta, _ = responder(pregunta)
    print(respuesta.model_dump_json(indent=2))
