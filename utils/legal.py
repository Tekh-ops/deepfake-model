"""
utils/legal.py
──────────────
Generates IT Rules 2021-formatted legal notices and
BSA-compliant evidence summaries.
Replaces the placeholder from your friend's prototype.
"""

from datetime import datetime, timezone
from config import SYSTEM_NAME, SYSTEM_VERSION, LEGAL_AUTHORITY


# ── Risk-based notice templates ───────────────────────────────

_TEMPLATES = {
    "CRITICAL": {
        "title": "Urgent Synthetic Media Alert — Immediate Action Required",
        "it_rules_ref": "Rule 3(1)(b)(v), Rule 3(2)(b) — IT (Intermediary Guidelines "
                        "and Digital Media Ethics Code) Rules, 2021",
        "bsa_ref": "Section 63 — Bharatiya Sakshya Adhiniyam, 2023 "
                   "(digital record admissibility and hash certification)",
        "recommended_action": (
            "Content should be treated as presumptively synthetic pending "
            "further forensic review. Escalate to designated Nodal Officer "
            "under IT Rules Rule 3(2). Preserve original file and this "
            "analysis report as primary digital evidence per BSA S.63."
        ),
        "takedown_advised": True,
        "escalation_level": "IMMEDIATE — Nodal Officer + Law Enforcement",
    },
    "HIGH": {
        "title": "Synthetic Media Warning — Action Advised",
        "it_rules_ref": "Rule 3(1)(b)(v) — IT (Intermediary Guidelines "
                        "and Digital Media Ethics Code) Rules, 2021",
        "bsa_ref": "Section 63 — Bharatiya Sakshya Adhiniyam, 2023",
        "recommended_action": (
            "Submit to human analyst review within 24 hours. If confirmed "
            "synthetic, initiate takedown under IT Rules Rule 3(1)(b). "
            "Retain SHA-256-anchored evidence package for legal proceedings."
        ),
        "takedown_advised": True,
        "escalation_level": "Analyst Review — 24-hour SLA",
    },
    "MEDIUM": {
        "title": "Potential Synthetic Media — Review Recommended",
        "it_rules_ref": "Rule 3(1)(b) — IT (Intermediary Guidelines "
                        "and Digital Media Ethics Code) Rules, 2021",
        "bsa_ref": "Section 63 — Bharatiya Sakshya Adhiniyam, 2023",
        "recommended_action": (
            "Flag for analyst review. Model confidence is moderate; "
            "human verification is required before any enforcement action. "
            "Preserve evidence chain."
        ),
        "takedown_advised": False,
        "escalation_level": "Analyst Queue — Standard Priority",
    },
    "LOW": {
        "title": "Low Synthetic Media Probability",
        "it_rules_ref": "IT (Intermediary Guidelines and Digital Media "
                        "Ethics Code) Rules, 2021 — monitoring only",
        "bsa_ref": "Section 63 — Bharatiya Sakshya Adhiniyam, 2023",
        "recommended_action": (
            "No immediate action required. Detection score is below "
            "enforcement threshold. Log retained for audit purposes."
        ),
        "takedown_advised": False,
        "escalation_level": "Monitoring — No Action",
    },
    "MINIMAL": {
        "title": "Content Appears Authentic",
        "it_rules_ref": "N/A",
        "bsa_ref": "Section 63 — Bharatiya Sakshya Adhiniyam, 2023",
        "recommended_action": "No action required. Content assessed as authentic.",
        "takedown_advised": False,
        "escalation_level": "None",
    },
}


# ── Main function ─────────────────────────────────────────────

def generate_legal_notice(metadata: dict) -> dict | None:
    """
    Generate a structured legal notice from a completed metadata payload.

    Returns a dict with all legal fields, or None if detection result
    is not available in the metadata.

    The returned dict is meant to be embedded under
    metadata["legal_notice"] in the API response.
    """
    detection = metadata.get("detection", {})
    if not detection:
        return None

    result     = detection.get("label", "")
    risk_level = detection.get("risk_level", "MINIMAL")
    fake_prob  = detection.get("fake_probability", 0.0)
    confidence = detection.get("confidence", 0.0)

    # Only generate an action-oriented notice for detected fakes
    # For REAL content, still return a record for the audit trail
    template = _TEMPLATES.get(risk_level, _TEMPLATES["MINIMAL"])

    file_info      = metadata.get("file", {})
    submission_id  = metadata.get("submission_id", "N/A")
    timestamp      = metadata.get("timestamp_utc", datetime.now(timezone.utc).isoformat())

    notice = {
        # ── Header ────────────────────────────────────────────
        "notice_title":        template["title"],
        "issued_by":           SYSTEM_NAME,
        "system_version":      SYSTEM_VERSION,
        "issued_at_utc":       timestamp,
        "submission_ref":      submission_id,

        # ── Finding ───────────────────────────────────────────
        "finding": {
            "classification":   result,
            "risk_level":       risk_level,
            "fake_probability": fake_prob,
            "model_confidence": confidence,
        },

        # ── Legal references ──────────────────────────────────
        "legal_references": {
            "it_rules":    template["it_rules_ref"],
            "bsa":         template["bsa_ref"],
            "authority":   LEGAL_AUTHORITY,
        },

        # ── Evidence ──────────────────────────────────────────
        "evidence": {
            "file_name":       file_info.get("name", ""),
            "file_sha256":     file_info.get("sha256", ""),
            "file_size":       file_info.get("size_human", ""),
            "media_type":      file_info.get("media_type", ""),
            "custody_note":    (
                "SHA-256 hash computed at ingestion before any processing. "
                "Audit log entry created and HMAC-SHA256 chain appended per "
                "BSA Section 63 chain-of-custody requirements."
            ),
        },

        # ── Action ────────────────────────────────────────────
        "recommended_action":   template["recommended_action"],
        "takedown_advised":     template["takedown_advised"],
        "escalation_level":     template["escalation_level"],

        # ── Disclaimer ────────────────────────────────────────
        "disclaimer": (
            "This notice is generated by an automated AI detection system "
            "and constitutes a preliminary assessment only. It does not "
            "constitute legal advice or a final determination of authenticity. "
            "Human analyst review is required before enforcement action. "
            f"System: {SYSTEM_NAME} v{SYSTEM_VERSION}."
        ),
    }

    return notice