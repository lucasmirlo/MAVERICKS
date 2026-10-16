"""La interfaz. Solo interfaz: toda la lógica vive en src/mavericks/.

    uv run python -m streamlit run app.py

Recuerda: Streamlit vuelve a ejecutar este archivo entero en cada clic.
Lo caro va en funciones con @st.cache_resource.
"""

import streamlit as st

from mavericks import config
from mavericks.agent import responder
from mavericks.ingest import main as reindexar

st.set_page_config(page_title="Mavericks", page_icon=":material/search:", layout="wide")

with st.sidebar:
    st.subheader("Configuración")
    st.caption(f"LLM: `{config.LLM_MODEL}`")
    st.caption(f"Embeddings: `{config.EMBED_MODEL}`")
    st.caption(f"Top-k: {config.TOP_K} · umbral: {config.MAX_DISTANCE}")
    if st.button("Reindexar data/"):
        with st.spinner("Indexando..."):
            reindexar()
        st.success("Índice regenerado")
    depurar = st.toggle("Ver fragmentos recuperados", value=True)

st.title("Mavericks")
pregunta = st.text_input("Pregunta", placeholder="¿Qué quieres saber de los documentos?")

if pregunta:
    with st.spinner("Pensando..."):
        try:
            respuesta, fragmentos = responder(pregunta)
        except Exception as e:  # en la demo, mejor un mensaje que un traceback
            st.error(f"Algo ha fallado: {e}")
            st.stop()

    if respuesta.suficiente:
        st.markdown(respuesta.respuesta)
    else:
        st.warning(respuesta.respuesta)

    with st.expander("Cómo ha llegado a la respuesta"):
        st.write(respuesta.razonamiento)

    if respuesta.citas:
        st.caption("Fuentes: " + " · ".join(f"{c.fuente} p. {c.pagina}" for c in respuesta.citas))

    if depurar:
        st.divider()
        st.subheader("Lo que recuperó el buscador")
        st.caption(f"Distancia coseno: menor es más parecido. Por encima de {config.MAX_DISTANCE} no se usa.")
        for f in fragmentos:
            usado = f.distancia <= config.MAX_DISTANCE
            with st.expander(f"{'✅' if usado else '⬜'} {f.distancia:.3f} · {f.fuente} p. {f.pagina}"):
                st.write(f.texto)
