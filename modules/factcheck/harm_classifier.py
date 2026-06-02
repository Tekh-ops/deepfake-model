"""
factcheck/harm_classifier.py
──────────────────────────────
Multi-label harmful speech and misinformation classifier.

Strategy A (default, zero-shot) — facebook/bart-large-mnli
  No training required. Works out of the box on any GPU or CPU.
  Slower (~1s/query on CPU) but no dataset needed.

Strategy B (fine-tuned) — loaded if HATE_MODEL_PATH env var is set.
  Uses cardiffnlp/twitter-roberta-base-hate or dehatebert.
  Better for colloquial Indian-English content.

Labels:
  HATE_SPEECH      — targets community/religion/caste
  INCITEMENT       — calls to violence or disorder
  RAGEBAIT         — emotionally manipulative, designed to enrage
  MISINFORMATION   — factually false or misleading claims
  HEALTH_MISINFO   — dangerous health misinformation
  ELECTION_MISINFO — false voting / candidate claims
  CLEAN            — no issues found
"""

import os
import torch

from config import HARM_THRESHOLD, DEVICE

_LABELS = [
    "hate speech targeting a community or religion",
    "incitement to violence or civil unrest",
    "emotionally manipulative ragebait content",
    "factually false political misinformation",
    "dangerous health misinformation",
    "false information about elections or voting",
    "neutral factual content",
]

_LABEL_KEYS = [
    "HATE_SPEECH",
    "INCITEMENT",
    "RAGEBAIT",
    "MISINFORMATION",
    "HEALTH_MISINFO",
    "ELECTION_MISINFO",
    "CLEAN",
]

_HARMFUL_INDICES = [0, 1, 2, 3, 4, 5]   # all except CLEAN

_classifier = None
_model_type = None


def _load_classifier():
    global _classifier, _model_type

    if _classifier is not None:
        return _classifier

    # Try fine-tuned model first (if path set in env)
    hate_model_path = os.getenv("HATE_MODEL_PATH", "")
    if hate_model_path and os.path.exists(hate_model_path):
        try:
            from transformers import pipeline
            _classifier = pipeline(
                "text-classification",
                model=hate_model_path,
                device=0 if torch.cuda.is_available() else -1,
            )
            _model_type = "finetuned"
            print(f"[HarmClassifier] Fine-tuned model loaded from {hate_model_path}")
            return _classifier
        except Exception as e:
            print(f"[HarmClassifier] Fine-tuned model failed: {e}. Trying BART.")

    # Default: zero-shot DistilBERT (Lightweight to prevent OOM/Kernel crashes)
    try:
        from transformers import pipeline
        device_id = 0 if torch.cuda.is_available() else -1
        print("[HarmClassifier] Loading typeform/distilbert-base-uncased-mnli (zero-shot) ...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=device_id,
        )
        _model_type = "zero_shot"
        print("[HarmClassifier] DistilBERT zero-shot ready.")
    except Exception as e:
        print(f"[HarmClassifier] BART load failed: {e}")
        _classifier = None

    return _classifier


def classify(text: str) -> dict:
    """
    Classify text for harmful content.

    Args:
        text: Transcript or claim text (up to 512 tokens).

    Returns:
        {
            "label":         str,    # "HARMFUL" | "CLEAN" | "UNAVAILABLE"
            "harmful_score": float,  # max score across harmful categories
            "flags":         list,   # list of triggered label keys
            "scores":        dict,   # {LABEL_KEY: score} for all labels
        }
    """
    if not text or len(text.strip()) < 10:
        return {
            "label": "CLEAN",
            "harmful_score": 0.0,
            "flags": [],
            "scores": {},
        }

    classifier = _load_classifier()
    if classifier is None:
        return {
            "label":         "UNAVAILABLE",
            "harmful_score": 0.0,
            "flags":         [],
            "scores":        {},
            "note":          "transformers library not available or model failed to load",
        }

    try:
        truncated = text[:512]

        if _model_type == "zero_shot":
            result = classifier(truncated, _LABELS, multi_label=True)
            raw_scores = dict(zip(result["labels"], result["scores"]))
            # Map human-readable labels → key labels
            scores = {}
            for i, label_text in enumerate(_LABELS):
                scores[_LABEL_KEYS[i]] = round(raw_scores.get(label_text, 0.0), 3)
        else:
            # Fine-tuned model returns single label; wrap it
            result = classifier(truncated)[0]
            scores = {k: 0.0 for k in _LABEL_KEYS}
            predicted = result.get("label", "").upper()
            if predicted in scores:
                scores[predicted] = round(result.get("score", 0.0), 3)

        harmful_score = max(
            scores.get(k, 0.0) for k in _LABEL_KEYS if k != "CLEAN"
        )

        flags = [
            k for k in _LABEL_KEYS
            if k != "CLEAN" and scores.get(k, 0.0) >= HARM_THRESHOLD
        ]

        return {
            "label":         "HARMFUL" if harmful_score >= HARM_THRESHOLD else "CLEAN",
            "harmful_score": round(harmful_score, 4),
            "flags":         flags,
            "scores":        scores,
        }

    except Exception as e:
        return {
            "label":         "ERROR",
            "harmful_score": 0.0,
            "flags":         [],
            "scores":        {},
            "note":          f"Classification error: {str(e)}",
        }


def overall_misinfo_risk(harm_result: dict, claim_results: list) -> str:
    """
    Compute an overall misinformation risk level from harm analysis
    and per-claim verification results.
    Returns: "HIGH" | "MEDIUM" | "LOW" | "MINIMAL"
    """
    harmful_score = harm_result.get("harmful_score", 0.0)

    unverified_count = sum(
        1 for c in claim_results
        if c.get("ddg_verdict", {}).get("status") in ("UNVERIFIED", "NO_RESULTS")
    )

    if harmful_score >= 0.75 or unverified_count >= 3:
        return "HIGH"
    if harmful_score >= 0.50 or unverified_count >= 1:
        return "MEDIUM"
    if harmful_score >= 0.25:
        return "LOW"
    return "MINIMAL"
