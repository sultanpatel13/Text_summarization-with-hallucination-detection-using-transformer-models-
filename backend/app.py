import os
from pathlib import Path
try:
    import torch
    if hasattr(torch, "set_num_threads"):
        optimal_threads = min(8, max(1, (os.cpu_count() or 4) - 1))
        torch.set_num_threads(optimal_threads)
except Exception:
    pass

try:
    from backend.hallucination_detector import HallucinationDetector
except ModuleNotFoundError:
    from hallucination_detector import HallucinationDetector
from flask import Flask, jsonify, render_template, request
import time
try:
    from backend.analysis import analyze_text
    from backend.bart_model import BART_SUMMARIZER
    from backend.file_utils import extract_text_from_upload
    from backend.pegasus_model import PEGASUS_SUMMARIZER
    from backend.summarizer import generate_overview, normalize_input_text, split_sentences
except ModuleNotFoundError:
    from analysis import analyze_text
    from bart_model import BART_SUMMARIZER
    from file_utils import extract_text_from_upload
    from pegasus_model import PEGASUS_SUMMARIZER
    from summarizer import generate_overview, normalize_input_text, split_sentences


BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

print("Initializing Hallucination Detector (this might take a moment)...")
hallucination_detector = HallucinationDetector()

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/summarize")
def summarize():
    uploaded_file = request.files.get("file")
    is_form_request = request.content_type and (
        "multipart/form-data" in request.content_type
        or "application/x-www-form-urlencoded" in request.content_type
    )

    if is_form_request:
        text = (request.form.get("text") or "").strip()
        summary_length = request.form.get("summary_length")
        summary_preset = request.form.get("summary_preset")
        model_type = (request.form.get("model_type") or "bart").strip().lower()
    else:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        summary_length = payload.get("summary_length")
        summary_preset = payload.get("summary_preset")
        model_type = (payload.get("model_type") or "bart").strip().lower()

    file_name = None

    if uploaded_file and uploaded_file.filename:
        try:
            extracted_text, file_name = extract_text_from_upload(uploaded_file)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        if extracted_text and extracted_text.strip():
            text = extracted_text
        elif not text:
            return jsonify({"error": f"Could not extract readable text from '{uploaded_file.filename}'. The file may be empty or contain unsupported content."}), 400

    if not text:
        return jsonify({"error": "Please enter text or upload a supported file to summarize."}), 400

    text = normalize_input_text(text)
    words = text.split()
    sentences = split_sentences(text)
    if len(words) < 5 or not sentences:
        return jsonify({"error": "Input text is too short to summarize. Please provide at least a few sentences."}), 400

    # Resolve preset: explicit summary_preset takes priority, or map legacy summary_length
    resolved_preset = "medium"
    if summary_preset and str(summary_preset).lower().strip() in {"short", "medium", "long"}:
        resolved_preset = str(summary_preset).lower().strip()
    elif summary_length is not None:
        try:
            val = int(summary_length)
            if val < 1 or val > 20:
                return jsonify({"error": "Summary length must be between 1 and 20 sentences."}), 400
            if val == 6:
                resolved_preset = "short"
            elif val == 10:
                resolved_preset = "medium"
            elif val == 15:
                resolved_preset = "long"
            elif val <= 2:
                resolved_preset = "short"
            elif val >= 5:
                resolved_preset = "long"
            else:
                resolved_preset = "medium"
        except (TypeError, ValueError):
            return jsonify({"error": "Summary length must be a valid number or preset."}), 400

    valid_models = {"bart", "pegasus"}
    if model_type not in valid_models:
        return jsonify({"error": f"Unsupported model type '{model_type}'. Supported models: bart, pegasus."}), 400

    try:
        if model_type == "pegasus":
            summary = PEGASUS_SUMMARIZER.summarize(text, preset=resolved_preset, summary_length=summary_length or 3)
        elif model_type == "bart":
            summary = BART_SUMMARIZER.summarize(text, preset=resolved_preset, summary_length=summary_length or 3)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        labels = {"pegasus": "PEGASUS", "bart": "BART"}
        label = labels.get(model_type, "Summarization")
        return jsonify({"error": f"{label} failed: {error}"}), 500

    if not summary or not summary.strip():
        labels = {"pegasus": "PEGASUS", "bart": "BART"}
        label = labels.get(model_type, "Selected model")
        return jsonify({"error": f"{label} was unable to generate a summary for the provided text."}), 500

    summary = summary.strip()
    overview = generate_overview(summary or text, source_text=text)
    analysis = analyze_text(text, summary)

    return jsonify(
        {
            "model_type": model_type,
            "preset": resolved_preset,
            "overview": overview,
            "summary": summary,
            "source_file": file_name,
            "source_text": text,
            "analysis": analysis,
            "hallucination_analysis": None,
        }
    )


@app.post("/analyze_hallucination")
def analyze_hallucination():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    summary = (payload.get("summary") or "").strip()

    if not text or not summary:
        return jsonify({"error": "Missing text or summary"}), 400

    print("Running Hallucination Analysis...")
    hallucination_start = time.perf_counter()

    try:
        hallucination_analysis = (
            hallucination_detector.analyze_summary(
                text,
                summary
            )
        )
        hallucination_time = (
            time.perf_counter()
            - hallucination_start
        )
        hallucination_analysis[
            "processing_time_seconds"
        ] = round(
            hallucination_time,
            3
        )
    except Exception as error:
        print(
            f"Error during hallucination detection: {error}"
        )
        hallucination_analysis = {
            "error": "Failed to analyze hallucination",
            "details": str(error),
        }

    return jsonify(hallucination_analysis)


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.get("/privacy")
def privacy():
    template_path = BASE_DIR / "templates" / "privacy.html"
    if template_path.is_file():
        return render_template("privacy.html")
    return jsonify({"title": "Privacy Policy", "message": "All text processing is performed strictly locally on your machine."}), 200


@app.errorhandler(404)
def page_not_found(e):
    template_path = BASE_DIR / "templates" / "404.html"
    if template_path.is_file():
        return render_template("404.html"), 404
    return jsonify({"error": "Page not found", "status": 404}), 404


if __name__ == "__main__":
     app.run(debug=True, use_reloader=False)
