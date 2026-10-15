def main() -> None:
    print(
        "Mavericks\n"
        "  uv run python -m mavericks.doctor     comprueba que todo está listo\n"
        "  uv run python -m mavericks.ingest     indexa los documentos de data/\n"
        "  uv run python -m mavericks.retrieve   prueba solo la búsqueda\n"
        "  uv run python -m mavericks.agent      prueba búsqueda + respuesta\n"
        "  uv run python -m streamlit run app.py abre la interfaz\n"
        "  uv run python eval/run_eval.py        mide la recuperación"
    )
