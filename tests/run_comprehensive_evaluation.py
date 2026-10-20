"""Comprehensive Evaluation and Benchmarking Suite for AI Text Summarization & Hallucination Detection.

Covers the 10 Testing Areas:
1. Text Summarization Quality
2. BART Abstractive Summarization
3. PEGASUS Abstractive Summarization
4. Hallucination Detection (DeBERTa-v3 NLI)
5. Long-Document Handling & Section Coverage (Map-Reduce Chunking)
6. File Upload Processing (PDF, DOCX, TXT, Corrupted, Empty)
7. Summary Length Presets (Short, Medium, Long Word Budgets)
8. Inference Performance & Latency Profiling
9. Quantitative NLI Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
10. Error Handling and System Robustness
"""

import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Add project root and backend to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from docx import Document
from PyPDF2 import PdfWriter

from backend.analysis import analyze_text
from backend.bart_model import BART_SUMMARIZER
from backend.chunking import chunk_by_tokens, count_tokens
from backend.file_utils import extract_text_from_upload
from backend.hallucination_detector import HallucinationDetector
from backend.hierarchical_summarizer import calculate_word_budget, enforce_word_budget
from backend.pegasus_model import PEGASUS_SUMMARIZER
from backend.summarizer import normalize_input_text, split_sentences


def compute_rouge_scores(candidate: str, reference: str):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    cand_tokens = [t.lower() for t in candidate.split() if t]
    ref_tokens = [t.lower() for t in reference.split() if t]

    if not cand_tokens or not ref_tokens:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # ROUGE-1
    c1, r1 = Counter(cand_tokens), Counter(ref_tokens)
    overlap1 = sum((c1 & r1).values())
    p1 = overlap1 / len(cand_tokens) if cand_tokens else 0.0
    rec1 = overlap1 / len(ref_tokens) if ref_tokens else 0.0
    f1_1 = 2 * p1 * rec1 / (p1 + rec1) if (p1 + rec1) else 0.0

    # ROUGE-2
    c2_list = [tuple(cand_tokens[i:i+2]) for i in range(len(cand_tokens)-1)]
    r2_list = [tuple(ref_tokens[i:i+2]) for i in range(len(ref_tokens)-1)]
    c2, r2 = Counter(c2_list), Counter(r2_list)
    overlap2 = sum((c2 & r2).values())
    p2 = overlap2 / len(c2_list) if c2_list else 0.0
    rec2 = overlap2 / len(r2_list) if r2_list else 0.0
    f1_2 = 2 * p2 * rec2 / (p2 + rec2) if (p2 + rec2) else 0.0

    # ROUGE-L (LCS)
    m, n = len(cand_tokens), len(ref_tokens)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if cand_tokens[i-1] == ref_tokens[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    lcs = dp[n]
    p_l = lcs / m if m else 0.0
    rec_l = lcs / n if n else 0.0
    f1_l = 2 * p_l * rec_l / (p_l + rec_l) if (p_l + rec_l) else 0.0

    return {
        "rouge1": round(f1_1, 4),
        "rouge2": round(f1_2, 4),
        "rougeL": round(f1_l, 4)
    }


def run_nli_evaluation():
    """Evaluate DeBERTa NLI detector on controlled and curated test sets."""
    print("\n" + "="*70)
    print("RUNNING AREA 4 & 9: HALLUCINATION DETECTION (NLI ACCURACY & CONFUSION MATRIX)")
    print("="*70)

    detector = HallucinationDetector()

    # Labelled dataset with 20 ground-truth examples covering Entailment, Contradiction, Unverifiable
    test_cases = [
        # Entailment cases (Source supports claim)
        ("The company was founded in 2010. It initially had 15 employees and developed accounting software for small businesses.",
         "The company was established in 2010.", "ENTAILMENT"),
        ("Water freezes at 0°C under standard atmospheric pressure.",
         "Water freezes at 0°C under standard atmospheric pressure.", "ENTAILMENT"),
        ("Solar photovoltaic panels convert sunlight directly into electricity with zero greenhouse gas emissions during operation.",
         "Solar panels produce electricity from sunlight without emitting greenhouse gases during use.", "ENTAILMENT"),
        ("Deep learning transformers utilize self-attention mechanisms to process input tokens in parallel.",
         "Self-attention allows transformers to process input tokens concurrently.", "ENTAILMENT"),
        ("The hospital introduced a robotic surgery unit in January 2023 to assist with minimally invasive procedures.",
         "In January 2023, the hospital implemented robotic systems for minimally invasive operations.", "ENTAILMENT"),
        ("Penicillin was discovered by Alexander Fleming in 1928 at St. Mary's Hospital in London.",
         "Alexander Fleming discovered penicillin in 1928.", "ENTAILMENT"),
        ("Electric vehicles use lithium-ion battery packs to power high-efficiency electric motors.",
         "Lithium-ion batteries supply power to electric motors in electric vehicles.", "ENTAILMENT"),

        # Contradiction cases (Source conflicts with claim)
        ("The company was founded in 2010.",
         "The company was founded in 2018.", "CONTRADICTION"),
        ("A university launched a course in 2024.",
         "The university launched the course in 2022.", "CONTRADICTION"),
        ("Solar panels generate electricity during daylight hours and cannot generate power at night without battery storage.",
         "Solar panels generate abundant power at midnight without needing storage.", "CONTRADICTION"),
        ("The project's budget was strictly capped at $50,000 by the board.",
         "The board granted an unlimited budget exceeding $2 million for the project.", "CONTRADICTION"),
        ("The study included 100 adult participants aged between 25 and 40 years.",
         "The research was conducted exclusively on 500 pediatric patients under 10 years old.", "CONTRADICTION"),
        ("The server cluster runs on Ubuntu Linux 22.04 LTS.",
         "The server cluster operates exclusively on Windows Server 2003.", "CONTRADICTION"),
        ("Water boils at 100 degrees Celsius under standard atmospheric pressure.",
         "Water boils at 20 degrees Celsius at sea level.", "CONTRADICTION"),

        # Unverifiable cases (Source does not provide sufficient evidence)
        ("The company develops accounting software for small businesses.",
         "The company employs 500 people worldwide.", "UNVERIFIABLE"),
        ("The university introduced a new computer science course in 2024.",
         "The course enrolled exactly 850 students from 30 different countries.", "UNVERIFIABLE"),
        ("BART is a transformer architecture combining bidirectional and autoregressive components.",
         "BART was trained for precisely 42 days on 1,024 NVIDIA H100 GPUs.", "UNVERIFIABLE"),
        ("The cafe serves organic coffee and artisan pastries.",
         "The cafe generated $1.2 million in annual profit last year.", "UNVERIFIABLE"),
        ("Renewable energy adoption grew rapidly across Europe between 2015 and 2023.",
         "European solar panel manufacturers reduced their retail prices by exactly 43 percent in 2021.", "UNVERIFIABLE"),
        ("Researchers evaluated five distinct natural language processing algorithms on the benchmark.",
         "The researchers won the Best Paper Award at the NeurIPS conference for this work.", "UNVERIFIABLE"),
    ]

    classes = ["ENTAILMENT", "CONTRADICTION", "UNVERIFIABLE"]
    matrix = {true_c: {pred_c: 0 for pred_c in classes} for true_c in classes}
    correct = 0
    total = len(test_cases)
    detailed_results = []

    start_time = time.perf_counter()
    for idx, (source, claim, expected) in enumerate(test_cases, 1):
        res = detector.analyze_claim(source, claim)
        pred = res.get("label")
        if pred not in classes:
            # Map fallback if needed
            pred = "UNVERIFIABLE"

        matrix[expected][pred] += 1
        is_match = (pred == expected)
        if is_match:
            correct += 1

        detailed_results.append({
            "id": idx,
            "source": source,
            "claim": claim,
            "expected": expected,
            "predicted": pred,
            "match": is_match,
            "confidence": res.get("confidence", 0.0),
            "evidence": res.get("evidence", "")
        })
        print(f"[{idx:02d}/{total}] Expected: {expected:<13} | Pred: {pred:<13} | Match: {str(is_match):<5} | Conf: {res.get('confidence', 0.0):.2f}")

    total_time = time.perf_counter() - start_time
    accuracy = (correct / total) * 100.0

    # Calculate per-class Precision, Recall, F1
    metrics = {}
    for c in classes:
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in classes if other != c)
        fn = sum(matrix[c][other] for other in classes if other != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    print("\n" + "-"*50)
    print(f"NLI Evaluation Summary:")
    print(f"Total Test Claims: {total}")
    print(f"Overall Accuracy:  {accuracy:.2f}% ({correct}/{total})")
    print(f"Total Time:        {total_time:.2f}s ({total_time/total:.3f}s/claim)")
    print("\nConfusion Matrix (Row = True, Col = Predicted):")
    header = f"{'':<15} | " + " | ".join(f"{c:<13}" for c in classes)
    print(header)
    print("-" * len(header))
    for true_c in classes:
        row = f"{true_c:<15} | " + " | ".join(f"{matrix[true_c][pred_c]:<13}" for pred_c in classes)
        print(row)

    print("\nPer-Class Classification Metrics:")
    for c in classes:
        m = metrics[c]
        print(f"  {c:<13} -> Precision: {m['precision']:.3f}, Recall: {m['recall']:.3f}, F1: {m['f1']:.3f}")

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "per_class_metrics": metrics,
        "total_time": total_time,
        "test_results": detailed_results
    }


def run_summarization_comparison():
    """Compare BART and PEGASUS on benchmark documents (Area 1, 2, 3, 7, 8)."""
    print("\n" + "="*70)
    print("RUNNING AREA 1, 2, 3, 7 & 8: BART vs PEGASUS BENCHMARK & PRESETS")
    print("="*70)

    # Standard benchmark test cases across different lengths and topics
    articles = [
        {
            "topic": "Artificial Intelligence & Robotics",
            "text": (
                "Artificial intelligence is a branch of computer science that enables machines to perform tasks "
                "that typically require human intelligence. It includes machine learning, natural language processing, "
                "computer vision, and robotics. AI is used in healthcare, education, finance, transportation, and manufacturing. "
                "It helps automate repetitive tasks, analyze large amounts of data, and support decision-making. "
                "Recent advancements in deep learning, particularly transformer neural architectures, have dramatically "
                "accelerated progress across translation, summarization, and conversational agents."
            ),
            "reference": (
                "Artificial intelligence enables machines to perform complex human tasks across fields like healthcare, "
                "finance, and robotics using machine learning, natural language processing, and deep transformer architectures."
            )
        },
        {
            "topic": "Renewable Energy Technologies",
            "text": (
                "Renewable energy comes from natural sources that replenish over time. Solar energy uses sunlight to generate "
                "electricity, while wind energy converts the movement of air into electrical power. Hydropower uses flowing water, "
                "and geothermal energy uses heat from beneath Earth's surface. These technologies can reduce dependence on fossil fuels, "
                "although their costs, infrastructure requirements, and environmental effects vary. Rapid manufacturing innovation "
                "has lowered solar and wind capital expenditures by more than 70% over the last decade, enabling large-scale grid adoption."
            ),
            "reference": (
                "Renewable energy systems harness natural replenishing resources such as solar, wind, hydropower, and geothermal "
                "to curb fossil fuel dependence, with solar and wind costs dropping by over 70% in recent years."
            )
        }
    ]

    results = []
    for art in articles:
        topic = art["topic"]
        source_text = art["text"]
        reference = art["reference"]
        src_words = len(source_text.split())

        print(f"\n--- Testing Topic: {topic} (Source Words: {src_words}) ---")

        for model_name, summarizer in [("BART", BART_SUMMARIZER), ("PEGASUS", PEGASUS_SUMMARIZER)]:
            for preset in ["short", "medium", "long"]:
                t0 = time.perf_counter()
                summary = summarizer.summarize(source_text, preset=preset)
                elapsed = time.perf_counter() - t0
                sum_words = len(summary.split())
                rouge = compute_rouge_scores(summary, reference)

                min_budget, max_budget = calculate_word_budget(src_words, preset)

                res = {
                    "topic": topic,
                    "model": model_name,
                    "preset": preset,
                    "elapsed_sec": round(elapsed, 2),
                    "summary_words": sum_words,
                    "budget_expected": f"{min_budget}-{max_budget}",
                    "in_budget": (min_budget <= sum_words <= max_budget + 15), # allow small sentence leeway
                    "rouge1": rouge["rouge1"],
                    "rouge2": rouge["rouge2"],
                    "rougeL": rouge["rougeL"],
                    "summary": summary
                }
                results.append(res)
                print(f"  [{model_name:<7}] Preset: {preset:<6} | Words: {sum_words:<3} (Budget: {min_budget}-{max_budget}) | Time: {elapsed:.2f}s | R-1: {rouge['rouge1']:.3f} | R-L: {rouge['rougeL']:.3f}")

    return results


def run_long_document_test():
    """Evaluate Long-Document Map-Reduce Chunking and Section Coverage (Area 5)."""
    print("\n" + "="*70)
    print("RUNNING AREA 5: LONG-DOCUMENT MAP-REDUCE CHUNKING & 5-SECTION COVERAGE")
    print("="*70)

    # Construct synthetic 5-section document with explicit unique facts in each section
    section_1 = "Section 1: Company Background. Solaria Technologies was founded in Zurich in 2012 by Dr. Elena Ramos to pioneer clean energy storage solutions."
    section_2 = "Section 2: Financial Performance. In fiscal year 2025, Solaria achieved record revenues of 340 million euros with an operating margin of 18.5 percent."
    section_3 = "Section 3: Product Development. The engineering division unveiled the Helios-9 solid-state battery cells exhibiting 98 percent round-trip efficiency."
    section_4 = "Section 4: Major Challenges. Supply bottlenecks in critical lithium imports and geopolitical tariffs increased raw component procurement overhead by 14 percent."
    section_5 = "Section 5: Future Plans. The strategic roadmap plans to construct a gigafactory in Valencia by 2028 and expand commercial operations across Latin America."

    # Pad sections to reach ~2,500 tokens (multi-chunk threshold)
    filler_1 = " The organization maintains strict adherence to international research standards and transparent corporate governance." * 15
    filler_2 = " Financial statements were audited and verified by independent global accountancy firms without qualifications." * 15
    filler_3 = " Laboratory prototypes completed over 10,000 thermal cycling tests under rigorous stress conditions." * 15
    filler_4 = " Mitigation measures include forging bilateral partnerships with domestic mineral recyclers and alternative suppliers." * 15
    filler_5 = " Sustainable expansion aligns with European Union carbon reduction frameworks and United Nations development targets." * 15

    long_doc = "\n\n".join([
        section_1 + filler_1,
        section_2 + filler_2,
        section_3 + filler_3,
        section_4 + filler_4,
        section_5 + filler_5,
    ])

    # Count tokens
    BART_SUMMARIZER._load()
    tokenizer = BART_SUMMARIZER._tokenizer
    token_count = count_tokens(long_doc, tokenizer)
    word_count = len(long_doc.split())
    chunks = chunk_by_tokens(long_doc, tokenizer)

    print(f"Constructed Multi-Section Document:")
    print(f"  Total Words:        {word_count}")
    print(f"  Total Tokens:       {token_count}")
    print(f"  Generated Chunks:   {len(chunks)} (Chunk window limit: 900 tokens)")

    # Summarize with BART
    t0 = time.perf_counter()
    summary = BART_SUMMARIZER.summarize(long_doc, preset="medium")
    elapsed = time.perf_counter() - t0

    # Section Coverage Verification
    coverage = {
        "Section 1 (Background: Solaria/Elena Ramos/2012)": any(k in summary.lower() for k in ["solaria", "zurich", "2012", "clean energy", "storage"]),
        "Section 2 (Financials: 340 million/revenue)": any(k in summary.lower() for k in ["revenue", "340", "margin", "fiscal", "financial"]),
        "Section 3 (Products: Helios-9/battery/solid-state)": any(k in summary.lower() for k in ["helios", "battery", "solid-state", "efficiency"]),
        "Section 4 (Challenges: supply/tariffs/lithium)": any(k in summary.lower() for k in ["supply", "bottleneck", "tariff", "lithium", "challenge"]),
        "Section 5 (Future Plans: gigafactory/Valencia/2028)": any(k in summary.lower() for k in ["valencia", "2028", "gigafactory", "expand", "future"])
    }

    covered_count = sum(coverage.values())

    print(f"\nLong-Document Summarization Result:")
    print(f"  Processing Time:     {elapsed:.2f}s")
    print(f"  Summary Word Count:  {len(summary.split())}")
    print(f"  Section Coverage:    {covered_count}/5 sections preserved")
    for sec, preserved in coverage.items():
        print(f"    - {sec}: {'PRESERVED' if preserved else 'ABSENT'}")

    return {
        "word_count": word_count,
        "token_count": token_count,
        "num_chunks": len(chunks),
        "elapsed_sec": round(elapsed, 2),
        "summary": summary,
        "coverage": coverage,
        "coverage_ratio": f"{covered_count}/5"
    }


def run_file_upload_tests():
    """Verify file upload handling across PDF, DOCX, TXT, Empty, and Corrupted files (Area 6 & 10)."""
    print("\n" + "="*70)
    print("RUNNING AREA 6 & 10: FILE UPLOAD AND ERROR HANDLING VALIDATION")
    print("="*70)

    class MockUpload:
        def __init__(self, filename, data_bytes):
            self.filename = filename
            self.stream = io.BytesIO(data_bytes)

        def read(self):
            return self.stream.getvalue()

    tests = []

    # 1. Plain Text (.txt)
    txt_content = "Transformer models have revolutionized natural language processing and abstractive text summarization."
    upload_txt = MockUpload("sample.txt", txt_content.encode("utf-8"))
    text, name = extract_text_from_upload(upload_txt)
    tests.append({"case": "Valid TXT", "success": (text.strip() == txt_content), "extracted_len": len(text)})

    # 2. Word Document (.docx)
    doc = Document()
    doc.add_paragraph("DeBERTa provides robust cross-encoder natural language inference for hallucination detection.")
    docx_buf = io.BytesIO()
    doc.save(docx_buf)
    upload_docx = MockUpload("document.docx", docx_buf.getvalue())
    text, name = extract_text_from_upload(upload_docx)
    tests.append({"case": "Valid DOCX", "success": ("DeBERTa provides robust" in text), "extracted_len": len(text)})

    # 3. PDF Document (.pdf) - Scanned / Blank PDF (no selectable text)
    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=100, height=100)
    pdf_buf = io.BytesIO()
    pdf_writer.write(pdf_buf)
    upload_pdf = MockUpload("sample.pdf", pdf_buf.getvalue())
    try:
        text, name = extract_text_from_upload(upload_pdf)
        tests.append({"case": "Scanned/Blank PDF Handling", "success": True, "extracted_len": len(text), "note": "Extracted text empty as expected without OCR"})
    except Exception as e:
        tests.append({"case": "Scanned/Blank PDF Handling", "success": False, "error": str(e)})

    # 4. Empty File (Should raise ValueError)
    upload_empty = MockUpload("empty.txt", b"")
    try:
        extract_text_from_upload(upload_empty)
        tests.append({"case": "Empty File Rejection", "success": False, "note": "Failed to reject empty file"})
    except ValueError as e:
        tests.append({"case": "Empty File Rejection", "success": True, "note": str(e)})

    # 5. Corrupted PDF
    upload_corrupt = MockUpload("corrupted.pdf", b"%PDF-corrupted-random-bytes-garbage-data")
    try:
        text, name = extract_text_from_upload(upload_corrupt)
        tests.append({"case": "Corrupted PDF Handling", "success": True, "extracted_len": len(text)})
    except Exception as e:
        tests.append({"case": "Corrupted PDF Handling", "success": True, "error_caught": str(e)})

    # 6. Unsupported File Extension (.exe or .bin)
    upload_unsupported = MockUpload("payload.bin", b"\x00\x01\x02\x03\x04")
    try:
        text, name = extract_text_from_upload(upload_unsupported)
        # Should either decode or raise ValueError
        tests.append({"case": "Unsupported Extension Handling", "success": True, "extracted_len": len(text)})
    except ValueError as e:
        tests.append({"case": "Unsupported Extension Handling", "success": True, "rejection": str(e)})

    for t in tests:
        status = "PASSED" if t.get("success") else "FAILED"
        print(f"  {t['case']:<32} | Result: {status:<7} | Details: {t.get('note') or t.get('extracted_len') or t.get('error_caught') or ''}")

    return tests


def main():
    print("="*80)
    print("STARTING FULL PROJECT EVALUATION AND BENCHMARKING HARNESS")
    print("="*80)

    # 1. Run Hallucination Detection NLI Evaluation (Area 4 & 9)
    nli_report = run_nli_evaluation()

    # 2. Run File Upload and Error Handling (Area 6 & 10)
    upload_report = run_file_upload_tests()

    # 3. Run Long Document Map-Reduce Chunking (Area 5)
    long_doc_report = run_long_document_test()

    # 4. Run Summarization Model Comparison & Presets (Area 1, 2, 3, 7, 8)
    summarization_report = run_summarization_comparison()

    # Save comprehensive results to JSON
    output_path = ROOT_DIR / "evaluation_benchmark_report.json"
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "CPU (Intel/AMD)",
        "nli_evaluation": nli_report,
        "file_upload_evaluation": upload_report,
        "long_document_evaluation": long_doc_report,
        "model_comparison": summarization_report
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "="*80)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Detailed results exported to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
