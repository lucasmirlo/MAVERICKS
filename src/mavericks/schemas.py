"""El contrato del sistema. Lo primero que se escribe y lo último que se cambia.

Los `description` de cada campo se le envían al modelo como parte del
esquema: son prompt. Escríbelos con cuidado.

Esto es un RAG con citas genérico. Cuando sepáis el reto, cambiad `Respuesta`
por lo que de verdad devuelva el producto (un ranking, un cálculo, una
decisión...). Hay un ejemplo de ranking comentado al final.
"""

from pydantic import BaseModel, Field


class Fragmento(BaseModel):
    """Un trozo recuperado del índice. No lo genera el modelo."""

    texto: str
    fuente: str
    pagina: int
    distancia: float


class Cita(BaseModel):
    fuente: str = Field(description="Nombre del archivo del que sale la información, tal cual aparece en el contexto")
    pagina: int = Field(description="Número de página indicado en el contexto")


class Respuesta(BaseModel):
    # Va PRIMERO a propósito: el modelo escribe los campos en orden, así que
    # razona antes de comprometerse con una respuesta.
    razonamiento: str = Field(
        description="En 2 a 4 frases cortas, antes de responder: qué fragmentos sirven y, "
        "si hay que contar o aplicar condiciones, la cuenta. Sin divagar"
    )
    respuesta: str = Field(description="Respuesta a la pregunta, apoyada solo en el contexto proporcionado")
    citas: list[Cita] = Field(description="Fragmentos del contexto en los que se apoya la respuesta")
    suficiente: bool = Field(description="False solo si el contexto no tiene nada que ver con la pregunta")


# --- Ejemplo para un reto de ranking (p. ej. CVs contra ofertas) ---------------
#
# class Encaje(BaseModel):
#     oferta: str
#     must_have_cumplidos: list[str]
#     must_have_faltantes: list[str]
#     motivo: str = Field(description="Una frase que explica el encaje")
#
# El modelo extrae `Encaje`; la PUNTUACIÓN y el orden los calcula tools.py.
