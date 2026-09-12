# Text Summarization Web App

A simple full-stack web application that summarizes long text using an extractive TextRank-style algorithm built with Python and Flask.

## Features

- Large text input area
- File upload support for common text-bearing formats
- Summary generation through `/summarize`
- Adjustable summary length
- Word count, reading-time, compression, and keyword analytics
- Heuristic AI-writing signal check
- Loading indicator and clear button
- Empty-input and invalid-input handling

## Project Structure

```text
backend/                  Flask API and summarization models
frontend/                 React + Vite source
static/dist/              Built frontend served by Flask
hallucination_detection/  Verification pipeline and training code
templates/index.html      Flask page shell
tests/                    API regression tests
requirements.txt          Runtime dependencies
```

## Run Locally

The checked-in `venv` may have been created on a different computer. Create a
fresh virtual environment instead of relying on it.

```bash
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m backend.app
```

Then open `http://127.0.0.1:5000`.

For frontend development, start Flask first and then run `npm run dev` from
`frontend`. Vite proxies `/summarize` and `/health` to Flask.

### Operational limits

- Requests and uploads are limited to 5 MB by default. Set `MAX_REQUEST_BYTES`
  and `MAX_UPLOAD_BYTES` to change the limit deliberately.
- BART and PEGASUS download their model weights on first use. Long inputs are
  truncated to each model's 1,024-token context window.
- The verification and writing-pattern values are heuristics, not factual or AI
  authorship guarantees. The UI labels the overlap metric as “Topic Overlap.”
- Python 3.14 is supported on Windows x86-64. The pinned PyTorch release ships
  a CPython 3.14 wheel.

### Local-only models

BART and PEGASUS run from the local Hugging Face cache only. The application
does not contact Hugging Face during startup or summarization. Their cached
files are stored under `%USERPROFILE%\.cache\huggingface\hub` and must be kept
on the machine. If a cache is missing, the API reports a clear error instead of
retrying network requests. The extractive summarizer and hallucination detector
are also local-only.

## Train The Supervised Model

You can train the classical supervised extractive summarizer on article/abstract CSV pairs:

```bash
python -m backend.train_model --train "C:\path\to\train.csv" --validation "C:\path\to\validation.csv" --test "C:\path\to\test.csv"
```

The training script saves the model to `model_artifacts/supervised_summarizer.joblib` and metrics to `model_artifacts/supervised_metrics.json`.

## API

### `POST /summarize`

Request body:

```json
{
  "text": "Your long paragraph goes here.",
  "summary_length": 3
}
```

Response body:

```json
{
  "summary": "Generated summary text.",
  "source_file": "article.pdf",
  "analysis": {
    "input_word_count": 120,
    "summary_word_count": 42,
    "estimated_reading_time_minutes": 0.6,
    "compression_ratio": 35.0,
    "topic_overlap_score": 83.3,
    "top_keywords": ["summarization", "text", "ranking"],
    "ai_writing_signals": {
      "score": 48,
      "label": "Mixed signals",
      "details": "Heuristic estimate based on sentence uniformity, repetition, transitions, and structure. This is not a definitive AI detector."
    }
  }
}
```

## Upload Notes

The app accepts pasted text and uploaded text-bearing files. Best-supported formats include:

- `TXT`, `MD`, `CSV`, `JSON`, `HTML`, `XML`, `LOG`
- `PDF`
- `DOCX`
- `PPTX`
- Common code and config files

Legacy `.ppt` files are not parsed yet; convert them to `.pptx` first. Binary formats without readable text content are rejected with a clear error message.
