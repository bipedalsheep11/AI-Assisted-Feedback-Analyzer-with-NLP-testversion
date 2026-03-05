# ◉ Feedback Analysis System

A Streamlit web application that processes post-program evaluation surveys — both numerical Likert ratings and free-text responses — and produces structured, actionable insights for training managers and program coordinators.

---

## What It Does

The system runs a six-stage pipeline on your survey data:

1. **Data Ingestion** — Upload a CSV containing Likert rating columns and/or open-ended text responses. Optionally upload a program brief (PDF, DOCX, or TXT) to ground the AI analysis in real context.

2. **Automatic Clustering** — Respondents are grouped into behaviorally similar clusters using sentence embeddings, Min-Max normalization, PCA dimensionality reduction, and K-Means with automatic k-selection via silhouette scoring.

3. **Cluster Labelling** — Each cluster is named and profiled by an LLM, producing a short label, a respondent profile, key drivers, and a distinguishing feature — all grounded in the actual ratings and responses.

4. **Sentiment Analysis** — Every respondent's text is classified as positive, negative, neutral, or mixed, with a confidence rating and optional urgent-flag for responses that need immediate attention.

5. **Thematic Clustering** — Responses are grouped into recurring themes. You can supply predefined theme labels or let the model discover them automatically.

6. **Actionable Insight Extraction** — The system surfaces specific, concrete improvement suggestions participants embedded in their comments, ranked by priority and breadth (isolated → widespread).

---

## Screenshots

| Page | Description |
|---|---|
| Upload & Config | Upload CSV + optional brief, configure clustering, run pipeline |
| Cluster Profiles | Per-cluster labels, profiles, key drivers, and rating bars |
| Dashboard | KPI row, PCA scatter, sentiment chart, themes, action items |
| Respondent Table | Filterable table with sentiment, urgency, and detail drilldown |
| Ask AI | Chat interface with full analysis context loaded |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML clustering | scikit-learn (K-Means, PCA, silhouette) |
| Text embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM inference | Groq Cloud (primary) / Ollama (local fallback) |
| LLM model | `llama-3.3-70b-versatile` (Groq) / `qwen3:8b` (Ollama) |
| Document parsing | PyMuPDF, python-docx |
| Visualization | Plotly |
| Data | pandas, numpy |

---

## Project Structure

```
feedback-analysis-system/
├── app.py                          ← Main Streamlit application (entry point)
├── requirements.txt                ← All Python dependencies
├── .env.example                    ← Environment variable template
├── .gitignore
├── README.md
├── Documentation.txt               ← Full technical documentation
│
└── backend/
    ├── nlp/
    │   ├── llm_client.py           ← Groq + Ollama wrapper with retry logic
    │   ├── auto_clustering.py      ← Full ML clustering pipeline
    │   ├── format_responses.py     ← DataFrame → LLM-readable text formatters
    │   └── analysis_modules.py     ← Labelling, sentiment, themes, insights
    │
    └── utils/
        ├── get_system_prompt.py    ← System prompt builder (document-aware)
        ├── document_reader.py      ← PDF / DOCX / TXT / CSV / XLSX loader
        └── data_cleaning.py        ← Likert value normalization utilities
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/feedback-analysis-system.git
cd feedback-analysis-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The sentence-transformers package will download the `all-MiniLM-L6-v2` model (~80 MB) on first use. Subsequent runs use the cached version.

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

**No key?** Install [Ollama](https://ollama.com) and pull a model locally:
```bash
ollama pull qwen3:8b
```
The app automatically falls back to Ollama if Groq is unavailable.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## CSV Format

Your survey CSV should have:

- **Likert columns** — numeric ratings (e.g. 1–5 scale). Any column containing only numbers with 10 or fewer unique values is automatically detected as Likert.
- **Text columns** — open-ended responses. Any column containing alphabetic text is automatically detected as a text column.
- An optional leading index column (named `""`, `"Unnamed: 0"`, or `"index"`) is automatically dropped.

**Example CSV structure:**

```
Respondent,Q1_Facilitation,Q2_Content,Q3_Pacing,Q4_Overall,What_Worked,What_Could_Improve
1,4,5,3,4,"The facilitator was excellent","Sessions felt rushed in the afternoon"
2,5,4,4,5,"Very practical examples","More time for Q&A would help"
...
```

---

## LLM Backends

### Groq (default)

Groq provides fast, free-tier cloud inference using Meta's Llama models.

- Sign up at [console.groq.com](https://console.groq.com)
- Add `GROQ_API_KEY=...` to your `.env` file
- Model used: `llama-3.3-70b-versatile`

### Ollama (local fallback)

Ollama runs models locally with no internet required after the initial model download.

- Install from [ollama.com](https://ollama.com)
- Run `ollama pull qwen3:8b` to download the default model
- Start the Ollama app; the system auto-detects it

The active backend is shown in the bottom-left sidebar.

---

## Configuration Options (Upload & Config page)

| Setting | Description |
|---|---|
| Auto-discover clusters | Let the algorithm choose the optimal k (recommended) |
| Number of clusters (k) | Override automatic selection with a fixed k |
| Predefined themes | Supply theme names; the LLM assigns responses to them |
| Programme name | Labels all outputs with your program's name |
| Flag urgent responses | Enable/disable the urgent-response flagging pass |
| Extract actionable insights | Enable/disable the insight extraction stage |

---

## Privacy Note

Survey data is sent to the Groq API for LLM analysis. If your data contains personally identifiable information (PII), either:

- Anonymize the data before uploading, or
- Use the Ollama local backend, which keeps all data on your machine

Neither backend stores conversation history after the session ends.

---

## Development

To contribute or extend the system:

```bash
# Install in editable mode (no reinstall needed after edits)
pip install -e .

# Run with auto-reload on file changes
streamlit run app.py --server.runOnSave true
```

### Adding a new analysis module

1. Write your function in `backend/nlp/analysis_modules.py`
2. Import it in `app.py`
3. Add a new pipeline stage in the Upload & Config page handler

### Switching LLM models

Edit the constants at the top of `backend/nlp/llm_client.py`:

```python
GROQ_MODEL   = "llama-3.3-70b-versatile"   # Change to any Groq-supported model
OLLAMA_MODEL = "qwen3:8b"                   # Change to any locally pulled model
```

---

## License

MIT License. See `LICENSE` for details.
