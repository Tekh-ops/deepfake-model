"""
factcheck/pipeline.py
──────────────────────
Orchestrates the full fact-checking pipeline for audio/video content.

Steps:
  1. Extract audio from video (ffmpeg)
  2. Transcribe speech → text (Whisper)
  3. Extract falsifiable claims (spaCy)
  4. For each claim:
     a. Search NewsAPI
     b. Search DuckDuckGo
     c. Semantic similarity check (sentence-transformers)
  5. Classify transcript for harmful/hateful content (BART zero-shot)
  6. Compute overall misinformation risk score
  7. Return structured fact-check report

Designed for graceful degradation:
  Each step is wrapped in try/except.
  Partial results are returned rather than raising exceptions.
  A "warnings" list documents what components are missing.
"""

import os
import tempfile

from modules.factcheck.transcriber      import transcribe
from modules.factcheck.claim_extractor  import extract_claims
from modules.factcheck.newsapi_checker  import get_newsapi_checker
from modules.factcheck.ddg_checker      import search_news as ddg_search_news
from modules.factcheck.semantic_verifier import check_contradiction
from modules.factcheck.harm_classifier  import classify as classify_harm, overall_misinfo_risk


def run_factcheck_pipeline(
    media_path: str,
    media_type: str = "video",
    existing_audio_path: str | None = None,
) -> dict:
    """
    Run the complete fact-checking pipeline on a video or audio file.

    Args:
        media_path:           Path to the media file.
        media_type:           "video" | "audio" | "image"
        existing_audio_path:  If audio was already extracted, pass it here
                              to avoid re-extraction.

    Returns:
        {
            "available":        bool,
            "transcript":       str | None,
            "language":         str | None,
            "claims":           list[dict],
            "harm_analysis":    dict,
            "overall_misinfo_risk": str,
            "warnings":         list[str],
        }
    """
    warnings  = []
    result    = {
        "available":            False,
        "transcript":           None,
        "language":             None,
        "claims":               [],
        "harm_analysis":        {"label": "UNAVAILABLE", "harmful_score": 0.0,
                                 "flags": [], "scores": {}},
        "overall_misinfo_risk": "UNKNOWN",
        "warnings":             warnings,
    }

    # Fact-check only makes sense for audio/video
    if media_type == "image":
        warnings.append("Fact-checking is not available for image files.")
        return result

    # ── Step 1: Extract audio ─────────────────────────────────
    audio_path      = existing_audio_path
    tmp_audio_path  = None

    if audio_path is None:
        try:
            from modules.core.video_utils import extract_audio, has_audio_track
            if not has_audio_track(media_path):
                warnings.append("No audio track found in video.")
                return result

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_audio_path = tmp.name
            tmp.close()
            extract_audio(media_path, output_path=tmp_audio_path)
            audio_path = tmp_audio_path
        except Exception as e:
            warnings.append(f"Audio extraction failed: {e}")
            return result

    # ── Step 2: Transcribe ────────────────────────────────────
    transcript_result = None
    transcript_text   = ""
    try:
        transcript_result = transcribe(audio_path)
        if transcript_result is None:
            warnings.append(
                "Whisper not available. Install: pip install openai-whisper"
            )
        else:
            transcript_text   = transcript_result.get("text", "")
            result["transcript"] = transcript_text
            result["language"]   = transcript_result.get("language")
    except Exception as e:
        warnings.append(f"Transcription failed: {e}")
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except Exception:
                pass

    if not transcript_text.strip():
        warnings.append("Empty transcript — no speech detected or transcription failed.")
        return result

    result["available"] = True

    # ── Step 3: Extract claims ────────────────────────────────
    claims = []
    try:
        claims = extract_claims(transcript_text, max_claims=5)
    except Exception as e:
        warnings.append(f"Claim extraction failed: {e}")

    # ── Step 4: Verify each claim ─────────────────────────────
    verified_claims = []
    newsapi = get_newsapi_checker()

    for claim_dict in claims:
        claim_text = claim_dict["claim"]
        entry = {**claim_dict}

        # NewsAPI
        try:
            newsapi_result = newsapi.check_claim(claim_text)
        except Exception as e:
            newsapi_result = {"verdict": "ERROR", "articles": [],
                              "reason": str(e)}

        # DuckDuckGo news
        ddg_articles = []
        try:
            ddg_articles = ddg_search_news(claim_text, max_results=3)
            if ddg_articles and "error" in ddg_articles[0]:
                ddg_articles = []
                warnings.append("DuckDuckGo search unavailable.")
        except Exception as e:
            warnings.append(f"DDG search failed: {e}")

        # Semantic similarity
        all_articles = (
            newsapi_result.get("articles", []) + ddg_articles
        )
        try:
            sem_result = check_contradiction(claim_text, all_articles)
        except Exception as e:
            sem_result = {"status": "ERROR", "similarity": 0.0,
                          "best_match": str(e)}

        entry["newsapi_verdict"] = newsapi_result.get("verdict", "UNVERIFIABLE")
        entry["newsapi_articles"] = newsapi_result.get("articles", [])
        entry["ddg_articles"]    = ddg_articles[:2]
        entry["ddg_verdict"]     = sem_result
        entry["combined_verdict"] = (
            "CORROBORATED"  if sem_result.get("status") == "CORROBORATED" else
            "REFERENCED"    if newsapi_result.get("verdict") == "REFERENCED_IN_NEWS" else
            "UNVERIFIED"
        )

        verified_claims.append(entry)

    result["claims"] = verified_claims

    # ── Step 5: Harm classification ───────────────────────────
    try:
        harm = classify_harm(transcript_text)
        result["harm_analysis"] = harm
    except Exception as e:
        warnings.append(f"Harm classification failed: {e}")

    # ── Step 6: Overall misinformation risk ───────────────────
    try:
        result["overall_misinfo_risk"] = overall_misinfo_risk(
            result["harm_analysis"], verified_claims
        )
    except Exception as e:
        warnings.append(f"Risk scoring failed: {e}")

    result["warnings"] = warnings
    return result
