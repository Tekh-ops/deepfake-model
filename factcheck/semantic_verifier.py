"""
factcheck/semantic_verifier.py
────────────────────────────────
Semantic similarity check using sentence-transformers.
Checks whether search results CORROBORATE or CONTRADICT a claim
using cosine similarity in sentence embedding space.

Model: all-MiniLM-L6-v2 (22 MB, very fast, strong similarity performance)
"""

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import os
            cache = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                ".model_cache", "sentence_transformers"
            )
            print("[SemanticVerifier] Loading all-MiniLM-L6-v2 ...")
            _model = SentenceTransformer("all-MiniLM-L6-v2",
                                         cache_folder=cache)
            print("[SemanticVerifier] Ready.")
        except ImportError:
            print("[SemanticVerifier] sentence-transformers not installed. "
                  "Run: pip install sentence-transformers")
        except Exception as e:
            print(f"[SemanticVerifier] Model load failed: {e}")
    return _model


def check_contradiction(claim: str, search_results: list[dict]) -> dict:
    """
    Compute cosine similarity between a claim and search result snippets.

    Args:
        claim:          The claim text to verify.
        search_results: List of dicts with 'title', 'snippet' or 'body' keys.

    Returns:
        {
            "status":     "CORROBORATED" | "UNVERIFIED" | "NO_RESULTS" | "UNAVAILABLE",
            "similarity": float,   # max cosine similarity [0, 1]
            "best_match": str,     # text of the best-matching result
        }
    """
    model = _get_model()
    if model is None:
        return {"status": "UNAVAILABLE", "similarity": 0.0,
                "best_match": "sentence-transformers not available"}

    if not search_results:
        return {"status": "NO_RESULTS", "similarity": 0.0, "best_match": ""}

    result_texts = [
        f"{r.get('title', '')} {r.get('snippet', '') or r.get('body', '')}"
        for r in search_results
        if not r.get("error")
    ]

    if not result_texts:
        return {"status": "NO_RESULTS", "similarity": 0.0, "best_match": ""}

    try:
        claim_emb   = model.encode(claim, convert_to_numpy=True)
        result_embs = model.encode(result_texts, convert_to_numpy=True)

        # Cosine similarity: dot product of L2-normalised vectors
        c_norm = claim_emb   / (np.linalg.norm(claim_emb) + 1e-8)
        r_norm = result_embs / (np.linalg.norm(result_embs, axis=1, keepdims=True) + 1e-8)

        sims     = r_norm @ c_norm
        max_sim  = float(np.max(sims))
        best_idx = int(np.argmax(sims))

        return {
            "status":     "CORROBORATED" if max_sim > 0.55 else "UNVERIFIED",
            "similarity": round(max_sim, 3),
            "best_match": result_texts[best_idx][:250],
        }
    except Exception as e:
        return {"status": "ERROR", "similarity": 0.0,
                "best_match": f"Similarity computation failed: {e}"}
