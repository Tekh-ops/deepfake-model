"""
factcheck/claim_extractor.py
─────────────────────────────
Extract falsifiable factual claims from a transcript using spaCy NLP.
No external API required — runs entirely locally.

Extracts:
  • Named entity sentences (PERSON, ORG, GPE, EVENT, LAW, NORP)
  • Numerical claims (PERCENT, MONEY, CARDINAL, QUANTITY)
  • Deduplicates and ranks by priority
"""


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
                print("[ClaimExtractor] spaCy en_core_web_sm loaded.")
            except OSError:
                print("[ClaimExtractor] Downloading en_core_web_sm ...")
                import subprocess
                import sys
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True, capture_output=True,
                )
                _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            print("[ClaimExtractor] spaCy not installed. "
                  "Run: pip install spacy && python -m spacy download en_core_web_sm")
    return _nlp


# Priority: entities most likely to be in misinformation
_HIGH_PRIORITY   = {"PERSON", "ORG", "NORP", "LAW", "EVENT"}
_MEDIUM_PRIORITY = {"GPE", "LOC", "FAC"}
_LOW_PRIORITY    = {"PERCENT", "MONEY", "CARDINAL", "QUANTITY", "DATE"}


def extract_claims(transcript_text: str, max_claims: int = 5) -> list[dict]:
    """
    Extract falsifiable claims from transcript text.

    Args:
        transcript_text: Raw transcript string.
        max_claims:      Maximum number of claims to return.

    Returns:
        List of claim dicts:
        {
            "claim":    str,    # The sentence containing the claim
            "entity":   str,    # Named entity text
            "type":     str,    # Entity label
            "priority": str,    # "HIGH" | "MEDIUM" | "LOW"
        }
    """
    if not transcript_text or len(transcript_text.strip()) < 20:
        return []

    nlp = _get_nlp()
    if nlp is None:
        # Fallback: split into sentences and return first N
        sentences = [s.strip() for s in transcript_text.split(".")
                     if len(s.strip()) > 20]
        return [
            {"claim": s, "entity": "", "type": "SENTENCE", "priority": "MEDIUM"}
            for s in sentences[:max_claims]
        ]

    import spacy
    doc    = nlp(transcript_text[:5000])   # cap at 5000 chars for speed
    claims = []
    seen   = set()

    for ent in doc.ents:
        label   = ent.label_
        sent    = ent.sent.text.strip()

        if len(sent) < 20 or len(sent) > 400:
            continue
        if sent in seen:
            continue

        if label in _HIGH_PRIORITY:
            priority = "HIGH"
        elif label in _MEDIUM_PRIORITY:
            priority = "MEDIUM"
        elif label in _LOW_PRIORITY:
            priority = "LOW"
        else:
            continue

        seen.add(sent)
        claims.append({
            "claim":    sent,
            "entity":   ent.text,
            "type":     label,
            "priority": priority,
        })

    # Sort by priority
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    claims.sort(key=lambda c: order.get(c["priority"], 3))
    return claims[:max_claims]
