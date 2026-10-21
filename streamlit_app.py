"""Streamlit Application for AI Text Summarizer & Factual Hallucination Detector.

Provides an executive-grade web dashboard running BART / PEGASUS abstractive
summarization and DeBERTa-v3 NLI factual verification.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root and backend to python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "backend"))

import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="AI Summarizer & Hallucination Detector",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-contrast executive appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .badge-supported {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #bbf7d0;
    }
    .badge-contradiction {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #fecaca;
    }
    .badge-unverifiable {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #fde68a;
    }
    .claim-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left-width: 4px;
    }
    .claim-supported { border-left-color: #22c55e; }
    .claim-contradiction { border-left-color: #ef4444; }
    .claim-unverifiable { border-left-color: #f59e0b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cached resource loaders to prevent reloading weights on every interaction
@st.cache_resource(show_spinner=False)
def get_bart_summarizer():
    try:
        from backend.bart_model import BART_SUMMARIZER
        return BART_SUMMARIZER
    except Exception as err:
        st.error(f"Failed to load BART model: {err}")
        return None

@st.cache_resource(show_spinner=False)
def get_pegasus_summarizer():
    try:
        from backend.pegasus_model import PEGASUS_SUMMARIZER
        return PEGASUS_SUMMARIZER
    except Exception as err:
        st.error(f"Failed to load PEGASUS model: {err}")
        return None

@st.cache_resource(show_spinner=False)
def get_hallucination_detector():
    try:
        from backend.hallucination_detector import HallucinationDetector
        return HallucinationDetector()
    except Exception as err:
        st.error(f"Failed to load Hallucination Detector: {err}")
        return None

def extract_text_from_file(uploaded_file) -> tuple[str, str]:
    from backend.file_utils import extract_text_from_upload
    return extract_text_from_upload(uploaded_file)

def main():
    try:
        from backend.device_utils import get_device, get_device_info
        dev_info = get_device_info()
    except Exception:
        dev_info = {"active_device": "cpu", "is_gpu": False, "cuda_available": False}

    # Header section
    st.markdown('<div class="main-title">AI Summary & Factual Verification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Abstractive Synthesis with Deep Natural Language Inference (NLI) Audit</div>',
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_choice = st.selectbox(
            "Summarization Model",
            options=["BART (facebook/bart-large-cnn)", "PEGASUS (google/pegasus-cnn_dailymail)"],
            index=0,
            help="BART is fast and balanced; PEGASUS is optimized for abstractive news/article summaries.",
        )
        selected_model = "bart" if "BART" in model_choice else "pegasus"

        preset_choice = st.select_slider(
            "Summary Length Preset",
            options=["Short (6 sentences)", "Medium (10 sentences)", "Long (15 sentences)"],
            value="Medium (10 sentences)",
            help="Select the target length for abstractive generation.",
        )
        preset_map = {
            "Short (6 sentences)": "short",
            "Medium (10 sentences)": "medium",
            "Long (15 sentences)": "long",
        }
        selected_preset = preset_map[preset_choice]

        enable_verification = st.checkbox(
            "Audit Hallucinations (NLI DeBERTa-v3)",
            value=True,
            help="Verifies factual consistency of each generated sentence against source evidence.",
        )

        st.divider()

        # Hardware Diagnostic Box
        st.subheader("🖥️ Hardware Acceleration")
        if dev_info.get("is_gpu"):
            st.success(f"GPU Active: {dev_info.get('gpu_name', 'CUDA')}")
            st.caption(f"VRAM: {dev_info.get('gpu_memory_total_mb', 0):.0f} MB | PyTorch: {dev_info.get('torch_version')}")
        else:
            st.info("Compute Device: CPU (Multi-threaded)")
            st.caption(f"Logical Cores: {dev_info.get('cpu_count', 4)} | PyTorch: {dev_info.get('torch_version', 'N/A')}")

        st.divider()
        st.markdown(
            """
            **Supported Formats:**
            PDF, DOCX, PPTX, TXT, MD, HTML, CSV, JSON
            """
        )

    # Document Input Section
    input_method = st.radio("Input Source", ["Upload File", "Paste Text"], horizontal=True)
    input_text = ""
    file_name = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a document to summarize",
            type=["pdf", "docx", "pptx", "txt", "md", "html", "htm", "xml", "csv", "json"],
            help="Upload your document. Text will be automatically extracted.",
        )
        if uploaded_file is not None:
            try:
                with st.spinner("Extracting text from file..."):
                    input_text, file_name = extract_text_from_file(uploaded_file)
                st.success(f"Loaded '{file_name}' ({len(input_text.split()):,} words)")
                with st.expander("Preview Extracted Text"):
                    st.text_area("Extracted Document Content", value=input_text, height=200, disabled=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                input_text = ""
    else:
        sample_placeholder = (
            "Paste reference text here (at least 2-3 sentences to summarize)..."
        )
        input_text = st.text_area("Source Text", placeholder=sample_placeholder, height=250)

    # Summarize & Verify Trigger
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_button = st.button("⚡ Generate Summary & Audit", type="primary", use_container_width=True)

    with col_info:
        if input_text:
            word_count = len(input_text.split())
            st.caption(f"Input Document: {word_count:,} words | Ready to process with {selected_model.upper()}")

    if run_button:
        if not input_text or len(input_text.split()) < 5:
            st.warning("Please provide at least a few sentences of text or upload a document to summarize.")
            return

        from backend.summarizer import generate_overview, normalize_input_text
        from backend.analysis import analyze_text

        cleaned_text = normalize_input_text(input_text)

        # 1. Summarization
        summarizer = get_bart_summarizer() if selected_model == "bart" else get_pegasus_summarizer()
        if summarizer is None:
            st.error("Summarizer model could not be initialized.")
            return

        summary_text = ""
        with st.spinner(f"Generating abstractive summary using {selected_model.upper()} ({selected_preset} preset)..."):
            t0 = time.perf_counter()
            try:
                summary_text = summarizer.summarize(cleaned_text, preset=selected_preset)
            except Exception as e:
                st.error(f"Summarization error: {e}")
                return
            gen_time = time.perf_counter() - t0

        if not summary_text:
            st.error("The model did not return a valid summary. Try a longer source text.")
            return

        # 2. Analytics & Overview
        overview_text = generate_overview(summary_text, source_text=cleaned_text)
        analysis_data = analyze_text(cleaned_text, summary_text)

        # 3. Hallucination Detection (Optional)
        hallucination_result = None
        if enable_verification:
            detector = get_hallucination_detector()
            if detector:
                with st.spinner("Auditing factual consistency against source evidence (NLI DeBERTa-v3)..."):
                    try:
                        hallucination_result = detector.analyze_summary(cleaned_text, summary_text)
                    except Exception as e:
                        st.warning(f"Factual verification encountered an issue: {e}")

        # Render Results
        st.divider()
        st.subheader("📋 Executive Summary Results")

        # Overview Box
        if overview_text:
            st.info(f"**Executive Overview:** {overview_text}")

        # Summary text
        st.text_area("Generated Abstractive Summary", value=summary_text, height=180)

        # Copy / Export action bar
        col_c1, col_c2, _ = st.columns([1, 1, 2])
        with col_c1:
            st.download_button(
                "📥 Download Summary (.txt)",
                data=summary_text,
                file_name=f"Summary_{selected_model}_{selected_preset}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_c2:
            export_payload = {
                "source_file": file_name,
                "model": selected_model,
                "preset": selected_preset,
                "overview": overview_text,
                "summary": summary_text,
                "analysis": analysis_data,
                "hallucination_analysis": hallucination_result,
            }
            st.download_button(
                "📊 Export Report (.json)",
                data=json.dumps(export_payload, indent=2),
                file_name="Verification_Audit_Report.json",
                mime="application/json",
                use_container_width=True,
            )

        # Analytics KPI Grid
        st.markdown("### 📊 Document Analytics")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Summary Words</div>
                    <div class="metric-value">{analysis_data.get('summary_word_count', 0)}</div>
                    <small>from {analysis_data.get('input_word_count', 0)} source words</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Compression</div>
                    <div class="metric-value">{analysis_data.get('compression_ratio', 0)}%</div>
                    <small>content reduction</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Est. Read Time</div>
                    <div class="metric-value">{analysis_data.get('summary_reading_time_minutes', 0)} min</div>
                    <small>at 200 WPM</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Accuracy Match</div>
                    <div class="metric-value">{analysis_data.get('content_accuracy_score', 0)}%</div>
                    <small>heuristic alignment</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Hallucination & Factuality Section
        if hallucination_result and not hallucination_result.get("error"):
            st.markdown("### 🛡️ Factual Consistency & NLI Verification Audit")

            overall_label = hallucination_result.get("overall_label")
            ratio = (hallucination_result.get("hallucination_ratio", 0) * 100)

            if overall_label == "VERIFIED_FAITHFUL":
                st.success(
                    f"**VERIFIED FAITHFUL:** All {hallucination_result.get('total_claims', 0)} claims are supported by source evidence with 0 detected contradictions."
                )
            elif overall_label == "POTENTIAL_HALLUCINATION":
                st.warning(
                    f"**POTENTIAL UNVERIFIED CLAIMS:** Some claims could not be decisively verified against the document text (Hallucination ratio: {ratio:.1f}%)."
                )
            else:
                st.error(
                    f"**CONTRADICTORY CLAIMS DETECTED:** One or more claims contradict statements in the source document (Hallucination ratio: {ratio:.1f}%)."
                )

            # Verification Stats Bar
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Statements", hallucination_result.get("total_claims", 0))
            s2.metric("Supported", hallucination_result.get("supported_claims", 0))
            s3.metric("Contradicted", hallucination_result.get("contradicted_claims", 0))
            s4.metric("Unverifiable", hallucination_result.get("unverifiable_claims", 0))

            # Itemized breakdown
            claims = hallucination_result.get("sentence_analysis", [])
            if claims:
                st.markdown("#### Itemized Claim Verification")
                for idx, item in enumerate(claims, 1):
                    lbl = item.get("label", "UNVERIFIABLE")
                    claim_text = item.get("claim", "")
                    scores = item.get("scores", {})
                    s_ent = scores.get("entailment", 0) * 100
                    s_con = scores.get("contradiction", 0) * 100
                    s_unv = scores.get("unverifiable", 0) * 100

                    badge_class = "badge-supported" if lbl == "ENTAILMENT" else ("badge-contradiction" if lbl == "CONTRADICTION" else "badge-unverifiable")
                    card_class = "claim-supported" if lbl == "ENTAILMENT" else ("claim-contradiction" if lbl == "CONTRADICTION" else "claim-unverifiable")
                    display_lbl = "SUPPORTED" if lbl == "ENTAILMENT" else ("CONTRADICTION" if lbl == "CONTRADICTION" else "UNVERIFIABLE")

                    st.markdown(
                        f"""
                        <div class="claim-card {card_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                                <strong style="font-size: 0.85rem; color: #64748b;">STATEMENT {idx} OF {len(claims)}</strong>
                                <span class="{badge_class}">{display_lbl}</span>
                            </div>
                            <div style="font-weight: 600; font-size: 0.95rem; color: #0f172a; margin-bottom: 6px;">
                                {claim_text}
                            </div>
                            <div style="font-size: 0.78rem; color: #64748b;">
                                Entailment: {s_ent:.1f}% &nbsp;•&nbsp; Contradiction: {s_con:.1f}% &nbsp;•&nbsp; Unverifiable: {s_unv:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    evidence = item.get("evidence")
                    if evidence:
                        sim = (item.get("evidence_similarity", 0) * 100)
                        with st.expander(f"View Retrieved Source Evidence ({sim:.1f}% match)"):
                            st.markdown(f"*{evidence}*")

if __name__ == "__main__":
    main()
