# Text Summarization Web App

A full-stack web application that summarizes long text using extractive or abstractive models. Built with **Flask**, **React**, and **Hugging Face Transformers**.

## Features

- **Summarization models:** PEGASUS and BART abstractive transformers
- **Relative word-budget scaling:** Summary length scales relative to source text word count (% of source words):
  - `short`: 25% - 45% (paragraph cap: 250 words)
  - `medium` (default): 45% - 60% (paragraph cap: 450 words)
  - `long`: 75% - 80% (paragraph cap: 800 words)
  - Edge cases: $\le 20$ words (proportional scaling), $\le 60$ words ($\min \ge 5$, $\max \ge 10$), $> 60$ words (sentence-boundary bounded)
- Large text input with drag-and-drop file upload
- Post-summary analytics: keywords, compression, reading time, content accuracy score
- Heuristic AI-writing signal estimate
- **Hallucination Detection:** Deep NLI verification against source documents using DeBERTa
- **Optimized Inference:** Uses `torch.inference_mode()` and mixed precision `autocast` for fast GPU acceleration
- Voice input via browser speech recognition
- Summary history (browser localStorage)
- Copy to clipboard and PDF export
- Vercel-inspired React UI

## Project Structure

```text
Text_summarization/
├── backend/
│   ├── app.py                     # Flask entry point
│   ├── hierarchical_summarizer.py # Relative word-budget rules & narrative cleaner
│   ├── bart_model.py              # BART abstractive model
│   ├── pegasus_model.py           # PEGASUS abstractive model
│   ├── chunking.py                # Token-aware chunking & map-reduce
│   ├── analysis.py                # Analytics & heuristics
│   ├── file_utils.py              # File upload parsing
│   ├── hallucination_detector.py  # NLI factual verification engine
│   └── summarizer.py              # Text cleaning & sentence segmentation
├── frontend/
│   ├── src/App.jsx                # Main React app
│   └── vite.config.js             # Builds to ../static/dist/
├── static/dist/                   # Compiled frontend assets
├── templates/index.html           # Flask template loading React build
└── requirements.txt
```

## Run Locally

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Start the server

```bash
python -m backend.app
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Frontend development (optional)

```bash
cd frontend
npm run dev
```

Note: the dev server does not proxy to Flask by default; rebuild or run Flask alongside for API calls.

## Models

| `model_type` | Engine | Description |
|--------------|--------|-------------|
| `extractive` | TextRank or supervised SGD | Default. Uses trained artifact if present, otherwise TextRank PageRank on sentences. |
| `pegasus` | `google/pegasus-cnn_dailymail` | Abstractive transformer (downloads on first use). |
| `bart` | `facebook/bart-large-cnn` | Abstractive transformer (downloads on first use). |

BART and PEGASUS require `torch` and `transformers`.

### Hardware Acceleration (GPU & CPU Support)

The application automatically adapts to your system's hardware configuration:
- **CUDA GPU Available:** Transformer models (BART, PEGASUS, DeBERTa-v3 NLI, and Sentence-Transformers) automatically execute on GPU using `torch.inference_mode()` and mixed-precision `torch.autocast(device_type="cuda")` for accelerated inference.
- **No GPU (CPU-Only):** If no CUDA-capable GPU is installed or PyTorch is running a CPU build, the system automatically runs on CPU with multi-threading optimization (`torch.set_num_threads`).
- **Resilient Fallback:** If a GPU runs out of VRAM (OutOfMemory error) during long-document chunking or heavy batch inference, the system automatically catches the exception, clears the GPU memory cache, and gracefully falls back to CPU execution without failing the request.
- **Force CPU Mode (Optional):** To force CPU execution even on systems with GPUs (e.g., for benchmarking or testing), set the environment variable:
  ```bash
  # Windows PowerShell
  $env:FORCE_CPU="1"

  # Linux / macOS
  export FORCE_CPU=1
  ```

### 100% Offline Capability (Air-Gapped / Presentation Ready)

The application is engineered to work completely offline without requiring an active internet connection:
- **Zero External Pings:** The system automatically configures `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, preventing Hugging Face from sending background network requests or checking remote repositories.
- **Local Cache Priority:** All transformer models (`facebook/bart-large-cnn`, `google/pegasus-cnn_dailymail`, `cross-encoder/nli-deberta-v3-base`, and `sentence-transformers/all-MiniLM-L6-v2`) are loaded with `local_files_only=True` via [`load_pretrained_safe`](file:///c:/Users/sulta/Desktop/AI_Hallucination-transformer-hallucination-v2/AI_Hallucination-transformer-hallucination-v2/backend/device_utils.py#L110).
- **Self-Contained Frontend:** Compiled assets are served locally from `static/dist/`, with zero reliance on external CDNs or online tracking scripts. System font fallbacks ensure crisp typography even when disconnected from Google Fonts.

## Train The Supervised Extractive Model

Train on article/abstract CSV pairs (columns: `article`, plus `abstract`, `highlights`, or `summary`):

```bash
python -m backend.train_model --train "path/to/train.csv" --validation "path/to/validation.csv" --test "path/to/test.csv"
```

Artifacts are saved to:

- `model_artifacts/supervised_summarizer.joblib`
- `model_artifacts/supervised_metrics.json`

When this artifact exists, the `extractive` mode prefers the trained model over TextRank.

## Benchmark Models

Compare supervised extractive, PEGASUS, and BART on a test CSV:

```bash
python -m backend.benchmark_models --test "path/to/test.csv" --samples 25 --summary-length 3
```

Requires a trained supervised artifact. Outputs ROUGE-1, ROUGE-2, ROUGE-L, and average runtime per model.

## API

### `GET /`

Serves the React application.

### `POST /summarize`

Accepts **JSON** or **multipart form-data**.

**JSON body:**

```json
{
  "text": "Your long paragraph goes here.",
  "summary_length": 3,
  "model_type": "extractive"
}
```

**Form fields:** `text`, `summary_length`, `model_type`, optional `file` (upload).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | — | Source text to summarize |
| `summary_length` | int | `3` | Target length (1–10 sentences) |
| `model_type` | string | `extractive` | `extractive`, `pegasus`, or `bart` |
| `file` | file | — | Optional upload (PDF, DOCX, etc.) |

**Response:**

```json
{
  "model_type": "bart",
  "overview": "One-line overview of the content.",
  "summary": "Generated summary text.",
  "source_file": "article.pdf",
  "analysis": {
    "input_word_count": 120,
    "summary_word_count": 42,
    "input_character_count": 800,
    "summary_character_count": 280,
    "input_sentence_count": 8,
    "summary_sentence_count": 3,
    "estimated_reading_time_minutes": 0.6,
    "summary_reading_time_minutes": 0.21,
    "compression_ratio": 35.0,
    "lexical_diversity": 0.72,
    "content_accuracy_score": 83.3,
    "top_keywords": ["summarization", "text", "ranking"],
    "ai_writing_signals": {
      "score": 48,
      "label": "Mixed signals",
      "details": "Heuristic estimate based on sentence uniformity, repetition, transitions, and structure."
    }
  },
  "hallucination_analysis": null
}
```

### `POST /analyze_hallucination`

Takes the source text and generated summary to perform a deep NLI (Natural Language Inference) check, detecting hallucinated or ungrounded claims.

**JSON body:**
```json
{
  "text": "The full original text...",
  "summary": "The generated summary..."
}
```

**Response:**
```json
{
  "overall_label": "POTENTIAL_HALLUCINATION",
  "overall_hallucinated": true,
  "overall_confidence": 0.94,
  "total_claims": 3,
  "supported_claims": 2,
  "contradicted_claims": 1,
  "unverifiable_claims": 0,
  "hallucination_ratio": 0.33,
  "processing_time_seconds": 1.45,
  "sentence_analysis": [
    {
      "claim": "The summary claim",
      "label": "CONTRADICTION",
      "is_hallucinated": true,
      "confidence": 0.91,
      "evidence": "The retrieved sentence from the source."
    }
  ]
}
```

## Upload Notes

Supported formats include:

- `TXT`, `MD`, `CSV`, `JSON`, `HTML`, `XML`, `LOG`
- `PDF`, `DOCX`, `PPTX`
- Common code and config files

Legacy `.ppt` files are not supported; convert to `.pptx` first. Empty or unreadable files return a clear error.

## Roadmap

Planned features (not yet implemented):

- Explainable AI: cross-attention heatmaps on source text
- Hallucination detection via NLI/entailment models
- Server-side history and optional authentication

See `anchor.md/TODO.md` for the full task list (local AI context, gitignored).
