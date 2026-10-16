# Mavericks

RAG con citas sobre modelos locales: Ollama + Chroma + PydanticAI + Streamlit.
Ningún dato sale de la máquina.

## Instalar (una vez por ordenador)

**1. uv.** Gestiona Python y las dependencias. Si no tienes Python 3.12, lo descarga solo.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                      # Mac / Linux
```

Cierra y vuelve a abrir la terminal para que reconozca `uv`. Si usas Anaconda, haz antes `conda deactivate`.

**2. Ollama.** Descárgalo de [ollama.com](https://ollama.com/download) y ábrelo.

**3. El repo.**

```bash
git clone https://github.com/lucasmirlo/mavericks.git
cd mavericks
uv sync --python-preference only-managed   # usa el Python de uv, no el de conda
```

<details>
<summary>Plan B: no puedo instalar uv</summary>

Con Python 3.12 o superior ya instalado:

```powershell
python -m venv .venv
.venv\Scripts\activate          # Mac / Linux: source .venv/bin/activate
pip install -e .
```

pip no lee `uv.lock`, así que las versiones pueden variar algo. A partir de aquí,
usa `python -m ...` donde abajo pone `uv run python -m ...`.
</details>

## Arrancar (en este orden)

```bash
uv sync                                   # instala dependencias
ollama pull llama3.1:8b                   # el modelo que genera (5 GB, cabe en 8 GB de VRAM)
ollama pull nomic-embed-text              # el de embeddings (274 MB)
uv run python -m mavericks.doctor         # comprueba que todo está listo
uv run python -m mavericks.ingest         # indexa data/
uv run python -m streamlit run app.py     # abre la interfaz
```

¿Una gráfica de 16 GB o más? Usa un modelo mayor sin tocar código (`ollama ps` debe decir `100% GPU`):

```powershell
$env:MAVERICKS_LLM = "mistral-small3.2:24b"   # Windows (PowerShell)
```
```bash
export MAVERICKS_LLM=mistral-small3.2:24b     # Mac / Linux
```

## Estructura

| Archivo | Qué hace |
| --- | --- |
| `app.py` | Interfaz Streamlit. Solo interfaz, nada de lógica |
| `src/mavericks/config.py` | Modelos, rutas, tamaño de trozo, top-k, umbral |
| `src/mavericks/schemas.py` | El contrato: qué devuelve el sistema (Pydantic) |
| `src/mavericks/embeddings.py` | La ÚNICA función texto → vector |
| `src/mavericks/ingest.py` | documentos → trozos → embeddings → índice |
| `src/mavericks/store.py` | El índice vectorial: Chroma, o NumPy como plan B (`MAVERICKS_STORE`) |
| `src/mavericks/retrieve.py` | pregunta → los k trozos más cercanos |
| `src/mavericks/agent.py` | pregunta + trozos → respuesta validada (PydanticAI) |
| `src/mavericks/tools.py` | Cálculos deterministas que el agente puede llamar |
| `src/mavericks/doctor.py` | Comprueba Python, librerías, Ollama, modelos e índice |
| `eval/run_eval.py` | Hit@k sobre `eval/eval.csv`: el número de la presentación |
| `data/` | Documentos del reto. `ejemplo_politica.md` es solo de prueba |

## Cuando una respuesta sea mala

1. Mira los fragmentos recuperados (en la app, o `uv run python -m mavericks.retrieve "pregunta"`).
2. Si el fragmento correcto **no** está: el fallo es de ingesta o recuperación (troceado, umbral, top-k).
3. Si **sí** está: el fallo es de generación (instrucciones en `agent.py`, descripciones en `schemas.py`).
