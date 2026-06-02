"""
factcheck/transcriber.py
─────────────────────────
Whisper-based speech-to-text transcription.
Uses local OpenAI Whisper — no API key, no cost.

Model size guide (from config WHISPER_MODEL env var):
  tiny    39 MB  — fast, lower accuracy (good for M1 dev)
  base    74 MB  — good baseline
  small   244 MB — recommended minimum for production
  medium  769 MB — strong accuracy
  large-v3 1.5 GB — best, fits on RTX 3070 with room

Degrades gracefully: returns None if Whisper is not installed.
"""

import os
import warnings

from config import WHISPER_MODEL_SIZE


_whisper_model = None
_whisper_size  = None


def _get_model(size: str = WHISPER_MODEL_SIZE):
    global _whisper_model, _whisper_size
    if _whisper_model is None or _whisper_size != size:
        try:
            import whisper
            print(f"[Transcriber] Loading Whisper '{size}' ...")
            cache = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 ".model_cache", "whisper")
            os.makedirs(cache, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _whisper_model = whisper.load_model(size, download_root=cache)
            _whisper_size = size
            print(f"[Transcriber] Whisper '{size}' ready.")
        except ImportError:
            print("[Transcriber] openai-whisper not installed. "
                  "Run: pip install openai-whisper")
            return None
        except Exception as e:
            print(f"[Transcriber] Failed to load Whisper: {e}")
            return None
    return _whisper_model


def transcribe(audio_path: str,
               model_size: str = WHISPER_MODEL_SIZE) -> dict | None:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path:  Path to WAV/MP3/M4A audio file.
        model_size:  Whisper model variant.

    Returns:
        {
            "text":     str,                # full transcript
            "language": str,                # detected language code
            "segments": list[dict],         # timestamped segments
            "words":    list[dict],         # word-level timestamps
        }
        or None if transcription is unavailable.
    """
    model = _get_model(model_size)
    if model is None:
        return None

    try:
        import whisper
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language=None,          # auto-detect
            word_timestamps=True,
            verbose=False,
        )
        return {
            "text":     result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "words":    [
                w for s in result.get("segments", [])
                for w in s.get("words", [])
            ],
        }
    except Exception as e:
        print(f"[Transcriber] transcription failed: {e}")
        return None
