from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class HallucinationDetector:
    """
    Transformer-based hallucination detector.

    Pipeline:
        Source document
            -> sentence splitting
            -> semantic evidence retrieval
            -> DeBERTa NLI
            -> entailment / contradiction / unverifiable
            -> claim-level + document-level result

    This version is designed for document-grounded
    abstractive summarization.
    """

    def __init__(
        self,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        retrieval_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.top_k = top_k

        print(
            f"[Detector] Device: {self.device}"
        )

        print(
            f"[Detector] Loading NLI model: "
            f"{nli_model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                nli_model_name
            )
        )

        self.nli_model = (
            AutoModelForSequenceClassification
            .from_pretrained(nli_model_name)
            .to(self.device)
        )

        self.nli_model.eval()

        print(
            f"[Detector] Loading retrieval model: "
            f"{retrieval_model_name}"
        )

        self.retrieval_model = (
            SentenceTransformer(
                retrieval_model_name,
                device=self.device,
            )
        )

        self.label_map = self._build_label_map()

        print(
            f"[Detector] Label map: "
            f"{self.label_map}"
        )

    # ---------------------------------------------------------
    # MODEL LABELS
    # ---------------------------------------------------------

    def _build_label_map(self) -> dict[int, str]:
        """
        Convert model labels into project terminology.

        Standard NLI:
            contradiction
            neutral
            entailment

        Project terminology:
            CONTRADICTION
            UNVERIFIABLE
            ENTAILMENT
        """

        result = {}

        for index, label in self.nli_model.config.id2label.items():

            normalized = label.lower()

            if "contradiction" in normalized:
                result[int(index)] = "CONTRADICTION"

            elif "entailment" in normalized:
                result[int(index)] = "ENTAILMENT"

            elif "neutral" in normalized:
                result[int(index)] = "UNVERIFIABLE"

            else:
                result[int(index)] = normalized.upper()

        return result

    # ---------------------------------------------------------
    # SENTENCE SPLITTING
    # ---------------------------------------------------------

    @staticmethod
    def split_sentences(text: str) -> list[str]:

        text = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        if not text:
            return []

        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+(?=[A-Z0-9])",
                text,
            )
            if sentence.strip()
        ]

    # ---------------------------------------------------------
    # EVIDENCE RETRIEVAL
    # ---------------------------------------------------------

    def retrieve_evidence(
        self,
        claim: str,
        source_sentences: list[str],
        source_embeddings: Any | None = None,
    ) -> list[dict[str, Any]]:

        if not source_sentences:
            return []

        claim_embedding = self.retrieval_model.encode(
            claim,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )

        if source_embeddings is None:
            source_embeddings = self.retrieval_model.encode(
                source_sentences,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )

        similarities = np.dot(
            source_embeddings,
            claim_embedding,
        )

        ranked_indices = np.argsort(similarities)[::-1]

        evidence = []
        for index in ranked_indices[: self.top_k]:
            evidence.append(
                {
                    "sentence": source_sentences[int(index)],
                    "similarity": float(similarities[int(index)]),
                    "index": int(index),
                }
            )

        return evidence

    # ---------------------------------------------------------
    # NLI
    # ---------------------------------------------------------

    def _predict_nli_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 16,
    ) -> list[dict[str, float]]:
        """Predict NLI scores for a batch of (premise, hypothesis) pairs."""
        if not pairs:
            return []

        all_scores: list[dict[str, float]] = []
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"

        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            premises = [p[0] for p in chunk]
            hypotheses = [p[1] for p in chunk]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.inference_mode():
                with torch.autocast(device_type=device_type):
                    outputs = self.nli_model(**inputs)

                probabilities = torch.softmax(outputs.logits, dim=-1)

            for probs in probabilities.cpu().tolist():
                scores = {}
                for index, probability in enumerate(probs):
                    label = self.label_map.get(index, f"LABEL_{index}")
                    scores[label] = float(probability)
                all_scores.append(scores)

        return all_scores

    def _predict_nli_single(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:
        res = self._predict_nli_batch([(premise, hypothesis)])
        return res[0] if res else {}

    def _predict_nli(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:
        """Paragraph-aware NLI with sliding window over 512-token limit."""
        try:
            premise_ids = self.tokenizer.encode(premise, add_special_tokens=False)
            hypothesis_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
        except Exception:
            return self._predict_nli_single(premise, hypothesis)

        # 4 reserved for [CLS]/[SEP] special tokens.
        if len(premise_ids) + len(hypothesis_ids) + 4 <= 512:
            return self._predict_nli_single(premise, hypothesis)

        max_premise = max(64, 512 - len(hypothesis_ids) - 4)
        window = min(400, max_premise)
        stride = max(64, window // 2)

        best_scores: dict[str, float] | None = None
        best_strength = -1.0
        for start in range(0, len(premise_ids), stride):
            window_ids = premise_ids[start:start + window]
            if not window_ids:
                break
            try:
                window_text = self.tokenizer.decode(window_ids, skip_special_tokens=True)
            except Exception:
                continue
            if not window_text.strip():
                continue
            scores = self._predict_nli_single(window_text, hypothesis)
            strength = max(
                scores.get("ENTAILMENT", 0.0),
                scores.get("CONTRADICTION", 0.0),
            )
            if strength > best_strength:
                best_strength = strength
                best_scores = scores
            if start + window >= len(premise_ids):
                break

        return best_scores if best_scores is not None else self._predict_nli_single(
            premise[:2000], hypothesis
        )

    # ---------------------------------------------------------
    # SINGLE CLAIM ANALYSIS
    # ---------------------------------------------------------

    def analyze_claim(
        self,
        source_text: str,
        summary_claim: str,
        source_sentences: list[str] | None = None,
        source_embeddings: Any | None = None,
    ) -> dict[str, Any]:

        if source_sentences is None:
            source_sentences = self.split_sentences(source_text)

        evidence_candidates = self.retrieve_evidence(
            summary_claim,
            source_sentences,
            source_embeddings=source_embeddings,
        )

        if not evidence_candidates:

            return {
                "claim": summary_claim,
                "label": "UNVERIFIABLE",
                "is_hallucinated": True,
                "confidence": 1.0,
                "scores": {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "unverifiable": 1.0,
                },
                "evidence": None,
                "evidence_similarity": 0.0,
            }

        # Paragraph-level evidence: concatenate top-k in original doc order
        # so multi-sentence claims verify against joint context. _predict_nli
        # handles >512 tokens via sliding window.
        ordered = sorted(evidence_candidates, key=lambda e: e.get("index", 0))
        paragraph = " ".join(e["sentence"] for e in ordered)
        top_similarity = max(float(e.get("similarity", 0.0)) for e in evidence_candidates)

        paragraph_scores = self._predict_nli(paragraph, summary_claim)

        # Batched evaluation of individual candidate evidence
        candidate_pairs = [
            (evidence_item["sentence"], summary_claim)
            for evidence_item in evidence_candidates
        ]
        candidate_scores_list = self._predict_nli_batch(candidate_pairs)

        best_single = None
        for evidence_item, scores in zip(evidence_candidates, candidate_scores_list):
            relation_strength = max(
                scores.get("ENTAILMENT", 0.0),
                scores.get("CONTRADICTION", 0.0),
            )
            current = {
                "scores": scores,
                "relation_strength": relation_strength,
                "evidence": evidence_item["sentence"],
            }
            if best_single is None or relation_strength > best_single["relation_strength"]:
                best_single = current

        assert best_single is not None
        para_strength = max(
            paragraph_scores.get("ENTAILMENT", 0.0),
            paragraph_scores.get("CONTRADICTION", 0.0),
        )
        if para_strength >= best_single["relation_strength"]:
            best_result = {
                "scores": {
                    "entailment": paragraph_scores.get("ENTAILMENT", 0.0),
                    "contradiction": paragraph_scores.get("CONTRADICTION", 0.0),
                    "unverifiable": paragraph_scores.get("UNVERIFIABLE", 0.0),
                },
                "relation_strength": para_strength,
                "evidence": paragraph,
                "similarity": top_similarity,
            }
        else:
            single_scores = best_single["scores"]
            best_result = {
                "scores": {
                    "entailment": single_scores.get("ENTAILMENT", 0.0),
                    "contradiction": single_scores.get("CONTRADICTION", 0.0),
                    "unverifiable": single_scores.get("UNVERIFIABLE", 0.0),
                },
                "relation_strength": best_single["relation_strength"],
                "evidence": best_single["evidence"],
                "similarity": top_similarity,
            }

        scores = best_result["scores"]

        entailment = scores["entailment"]
        contradiction = scores["contradiction"]
        unverifiable = scores["unverifiable"]

        # -----------------------------------------------------
        # CLAIM DECISION
        # -----------------------------------------------------

        if (
            contradiction > entailment
            and contradiction > unverifiable
        ):

            label = "CONTRADICTION"
            is_hallucinated = True
            confidence = contradiction

        elif (
            entailment > contradiction
            and entailment >= unverifiable
        ):

            label = "ENTAILMENT"
            is_hallucinated = False
            confidence = entailment

        else:

            label = "UNVERIFIABLE"
            is_hallucinated = True
            confidence = unverifiable

        return {
            "claim": summary_claim,

            "label": label,

            "is_hallucinated": is_hallucinated,

            "confidence": round(
                float(confidence),
                4,
            ),

            "scores": {
                "entailment": round(
                    entailment,
                    4,
                ),
                "contradiction": round(
                    contradiction,
                    4,
                ),
                "unverifiable": round(
                    unverifiable,
                    4,
                ),
            },

            "evidence": best_result["evidence"],

            "evidence_similarity": round(
                best_result["similarity"],
                4,
            ),
        }

    # ---------------------------------------------------------
    # FULL SUMMARY ANALYSIS
    # ---------------------------------------------------------

    def analyze_summary(
        self,
        source_text: str,
        generated_summary: str,
    ) -> dict[str, Any]:

        summary_sentences = self.split_sentences(generated_summary)
        source_sentences = self.split_sentences(source_text)

        # Pre-compute source embeddings once for the document (O(S) instead of O(M * S))
        source_embeddings = None
        if source_sentences:
            source_embeddings = self.retrieval_model.encode(
                source_sentences,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )

        analyzed_sentences = []

        for sentence in summary_sentences:
            if len(sentence.strip()) < 10:
                continue

            result = self.analyze_claim(
                source_text,
                sentence,
                source_sentences=source_sentences,
                source_embeddings=source_embeddings,
            )

            analyzed_sentences.append(result)

        total_claims = len(
            analyzed_sentences
        )

        entailed_claims = sum(
            result["label"] == "ENTAILMENT"
            for result in analyzed_sentences
        )

        contradicted_claims = sum(
            result["label"]
            == "CONTRADICTION"
            for result in analyzed_sentences
        )

        unverifiable_claims = sum(
            result["label"]
            == "UNVERIFIABLE"
            for result in analyzed_sentences
        )

        hallucinated_claims = sum(
            result["is_hallucinated"]
            for result in analyzed_sentences
        )

        if total_claims:

            hallucination_ratio = (
                hallucinated_claims
                / total_claims
            )

            average_confidence = float(
                np.mean(
                    [
                        result["confidence"]
                        for result in analyzed_sentences
                    ]
                )
            )

        else:

            hallucination_ratio = 0.0
            average_confidence = 0.0

        # -----------------------------------------------------
        # DOCUMENT DECISION
        # -----------------------------------------------------

        if contradicted_claims > 0:

            overall_label = (
                "HALLUCINATION_DETECTED"
            )

        elif hallucination_ratio >= 0.25:

            overall_label = (
                "POTENTIAL_HALLUCINATION"
            )

        else:

            overall_label = "SUPPORTED"

        return {

            "overall_label": overall_label,

            "overall_hallucinated": (
                overall_label
                != "SUPPORTED"
            ),

            "overall_confidence": round(
                average_confidence,
                4,
            ),

            "total_claims": total_claims,

            "supported_claims": (
                entailed_claims
            ),

            "contradicted_claims": (
                contradicted_claims
            ),

            "unverifiable_claims": (
                unverifiable_claims
            ),

            "hallucinated_claims": (
                hallucinated_claims
            ),

            "hallucination_ratio": round(
                hallucination_ratio,
                4,
            ),

            "source_sentence_count": (
                len(source_sentences)
            ),

            "summary_sentence_count": (
                total_claims
            ),

            "sentence_analysis": (
                analyzed_sentences
            ),
        }