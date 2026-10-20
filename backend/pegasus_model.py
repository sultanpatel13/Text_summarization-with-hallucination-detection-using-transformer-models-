from dataclasses import dataclass
from typing import Optional, Union

try:
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
except Exception:
    torch = None
    PegasusForConditionalGeneration = None
    PegasusTokenizer = None

try:
    from backend.chunking import (
        CHUNK_SIZE_TOKENS,
        MODEL_MAX_TOKENS,
        chunk_by_tokens,
        count_tokens,
        fuse_chunk_summaries,
    )
    from backend.device_utils import clear_device_cache, get_device, load_pretrained_safe
    from backend.hierarchical_summarizer import (
        calculate_word_budget,
        clean_and_filter_narrative,
        enforce_word_budget,
        words_to_token_budget,
    )
    from backend.summarizer import normalize_input_text
except ModuleNotFoundError:
    from chunking import (
        CHUNK_SIZE_TOKENS,
        MODEL_MAX_TOKENS,
        chunk_by_tokens,
        count_tokens,
        fuse_chunk_summaries,
    )
    from device_utils import clear_device_cache, get_device, load_pretrained_safe
    from hierarchical_summarizer import (
        calculate_word_budget,
        clean_and_filter_narrative,
        enforce_word_budget,
        words_to_token_budget,
    )
    from summarizer import normalize_input_text


DEFAULT_PEGASUS_MODEL = "google/pegasus-cnn_dailymail"


def resolve_preset(summary_length: Union[int, str, None] = None, preset: Optional[str] = None) -> str:
    """Resolve summary preset from name or numeric length."""
    if preset and str(preset).lower().strip() in {"short", "medium", "long"}:
        return str(preset).lower().strip()
    if summary_length is not None:
        raw_str = str(summary_length).lower().strip()
        if raw_str in {"short", "medium", "long"}:
            return raw_str
        try:
            val = int(raw_str)
            if val == 6:
                return "short"
            if val == 10:
                return "medium"
            if val == 15:
                return "long"
            if val <= 2:
                return "short"
            if val >= 5:
                return "long"
            return "medium"
        except ValueError:
            pass
    return "medium"


@dataclass
class PegasusSummarizer:
    model_name: str = DEFAULT_PEGASUS_MODEL

    def __post_init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self):
        if PegasusTokenizer is None or PegasusForConditionalGeneration is None:
            raise RuntimeError(
                "PEGASUS dependencies are not installed. Install transformers, sentencepiece, and torch to use this model."
            )

        if self._tokenizer is None or self._model is None:
            self._device = get_device()
            self._tokenizer = load_pretrained_safe(PegasusTokenizer.from_pretrained, self.model_name, use_fast=False)
            self._model = load_pretrained_safe(PegasusForConditionalGeneration.from_pretrained, self.model_name)
            if self._device == "cuda":
                try:
                    self._model = self._model.to("cuda")
                except Exception as err:
                    print(f"[PEGASUS] Failed to move model to CUDA ({err}). Falling back to CPU.")
                    self._device = "cpu"
                    self._model = self._model.to("cpu")
            else:
                self._model = self._model.to("cpu")
            self._model.eval()

    def _clean(self, decoded: str) -> str:
        # PEGASUS uses <n> as an inline newline token — replace with a space and normalise whitespace
        decoded = decoded.replace("<n>", " ")
        return " ".join(decoded.split()).strip()

    def _generate_single(self, chunk_text: str, min_length: int, max_length: int) -> str:
        if torch is None:
            raise RuntimeError("torch is required for PEGASUS inference.")

        if self._device is None:
            self._device = get_device()

        inputs = self._tokenizer(
            chunk_text,
            truncation=True,
            padding="longest",
            max_length=MODEL_MAX_TOKENS,
            return_tensors="pt",
        )

        try:
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                if self._device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        summary_ids = self._model.generate(
                            **inputs,
                            num_beams=4,
                            min_length=min_length,
                            max_length=max_length,
                            length_penalty=1.2,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            use_cache=True,
                        )
                else:
                    summary_ids = self._model.generate(
                        **inputs,
                        num_beams=4,
                        min_length=min_length,
                        max_length=max_length,
                        length_penalty=1.2,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        use_cache=True,
                    )
        except Exception as err:
            is_oom = "out of memory" in str(err).lower() or "cuda" in str(err).lower()
            if self._device == "cuda" and is_oom:
                print(f"[PEGASUS] GPU memory exceeded or CUDA error ({err}). Gracefully falling back to CPU.")
                clear_device_cache("cuda")
                self._device = "cpu"
                self._model = self._model.to("cpu")
                inputs = {key: value.to("cpu") for key, value in inputs.items()}
                with torch.inference_mode():
                    summary_ids = self._model.generate(
                        **inputs,
                        num_beams=4,
                        min_length=min_length,
                        max_length=max_length,
                        length_penalty=1.2,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        use_cache=True,
                    )
            else:
                raise

        return self._clean(self._tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    def summarize(self, text: str, summary_length: Union[int, str] = 3, preset: Optional[str] = None) -> str:
        cleaned = clean_and_filter_narrative(text)
        normalized = normalize_input_text(cleaned)
        if not normalized:
            return ""

        self._load()

        active_preset = resolve_preset(summary_length=summary_length, preset=preset)
        source_words = len(normalized.split())
        min_words, max_words = calculate_word_budget(source_words, active_preset)
        min_tokens, max_tokens = words_to_token_budget(min_words, max_words, max_token_cap=360)

        total_tokens = count_tokens(normalized, self._tokenizer)

        if total_tokens <= CHUNK_SIZE_TOKENS:
            raw_summary = self._generate_single(normalized, min_tokens, max_tokens)
            return enforce_word_budget(raw_summary, max_words)

        chunks = chunk_by_tokens(normalized, self._tokenizer)
        print(f"[PEGASUS] Long input ({total_tokens} tokens) -> {len(chunks)} chunks (map-reduce).")
        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_summaries.append(self._generate_single(chunk, 30, 220))
            print(f"[PEGASUS] Chunk {i + 1}/{len(chunks)} summarized.")

        fused = fuse_chunk_summaries(chunk_summaries)
        while count_tokens(fused, self._tokenizer) > CHUNK_SIZE_TOKENS:
            sub_chunks = chunk_by_tokens(fused, self._tokenizer)
            if len(sub_chunks) <= 1:
                break
            print(f"[PEGASUS] Reduce pass: {len(sub_chunks)} sub-chunks.")
            fused = fuse_chunk_summaries(
                [self._generate_single(c, 30, 220) for c in sub_chunks]
            )

        raw_summary = self._generate_single(fused, min_tokens, max_tokens)
        return enforce_word_budget(raw_summary, max_words)


PEGASUS_SUMMARIZER = PegasusSummarizer()
