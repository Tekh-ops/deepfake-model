"""
utils/hashing.py
────────────────
Cryptographic hashing and BSA-compliant audit chain.

Every media submission gets:
1. A SHA-256 fingerprint computed from raw bytes
2. An audit log entry appended to a HMAC-SHA256 chain

The chain ensures tamper-evidence: any modification to a past entry
breaks all subsequent HMAC values.
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone
from config import AUDIT_LOG_FILE

# Server-side signing secret — in production load from env / Vault
# For prototype we derive it from a fixed seed
_HMAC_SECRET = os.environ.get(
    "BHARATSHIELD_HMAC_SECRET",
    "bharatshield-prototype-secret-change-in-production"
).encode("utf-8")


# ── File hashing ──────────────────────────────────────────────

def compute_sha256(file_path: str) -> str:
    """
    Compute SHA-256 hash of a file's raw bytes.
    Reads in 64KB chunks to handle large video files without
    loading the whole file into memory.

    Returns hex digest string e.g.
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 of raw bytes — used for in-memory payloads."""
    return hashlib.sha256(data).hexdigest()


# ── HMAC chain ────────────────────────────────────────────────

def _get_previous_chain_hash() -> str:
    """
    Read the last HMAC from the audit log to chain the next entry.
    Returns '0' * 64 (genesis value) if log is empty.
    """
    if not os.path.exists(AUDIT_LOG_FILE):
        return "0" * 64

    last_line = ""
    with open(AUDIT_LOG_FILE, "rb") as f:
        # Efficiently seek to last non-empty line
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode("utf-8").strip()

    if not last_line:
        return "0" * 64

    try:
        entry = json.loads(last_line)
        return entry.get("entry_hmac", "0" * 64)
    except (json.JSONDecodeError, KeyError):
        return "0" * 64


def _compute_entry_hmac(entry_data: dict, prev_hash: str) -> str:
    """
    Compute HMAC-SHA256 over:
        prev_chain_hash || JSON(entry_data sorted keys)

    This chains each entry to the previous, making the log tamper-evident.
    """
    payload = prev_hash + json.dumps(entry_data, sort_keys=True)
    return hmac.new(
        _HMAC_SECRET,
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def append_audit_entry(
    submission_id: str,
    file_hash: str,
    action: str,
    actor: str,
    result: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Append a BSA-compliant audit log entry to the chain.

    Args:
        submission_id:  Unique ID for this media submission
        file_hash:      SHA-256 of the submitted file
        action:         e.g. "MEDIA_RECEIVED", "DETECTION_COMPLETE", "REPORT_GENERATED"
        actor:          e.g. "api_gateway", "analyst_user_id"
        result:         Detection result dict (optional)
        extra:          Any additional key/value pairs to include

    Returns:
        The complete audit entry dict (also written to disk).
    """
    prev_hash = _get_previous_chain_hash()

    entry_data = {
        "submission_id": submission_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_unix": int(time.time()),
        "file_sha256": file_hash,
        "action": action,
        "actor": actor,
    }
    if result:
        entry_data["result"] = result
    if extra:
        entry_data["extra"] = extra

    entry_hmac = _compute_entry_hmac(entry_data, prev_hash)

    full_entry = {
        **entry_data,
        "prev_entry_hmac": prev_hash,
        "entry_hmac": entry_hmac,
    }

    # Append to JSONL file (one JSON object per line)
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_entry) + "\n")

    return full_entry


def verify_audit_chain() -> tuple[bool, str]:
    """
    Walk the entire audit log and verify every HMAC.
    Returns (True, "OK") or (False, reason_string).
    Used by an admin endpoint to verify log integrity.
    """
    if not os.path.exists(AUDIT_LOG_FILE):
        return True, "Log file does not exist yet — chain is intact."

    prev_hash = "0" * 64
    line_num  = 0

    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_num += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False, f"Line {line_num}: invalid JSON"

            stored_hmac = entry.get("entry_hmac", "")
            stored_prev = entry.get("prev_entry_hmac", "")

            if stored_prev != prev_hash:
                return False, (
                    f"Line {line_num}: prev_entry_hmac mismatch — "
                    f"expected {prev_hash[:16]}... got {stored_prev[:16]}..."
                )

            # Recompute HMAC
            entry_data = {k: v for k, v in entry.items()
                          if k not in ("entry_hmac", "prev_entry_hmac")}
            expected_hmac = _compute_entry_hmac(entry_data, prev_hash)

            if not hmac.compare_digest(expected_hmac, stored_hmac):
                return False, (
                    f"Line {line_num}: HMAC tamper detected on "
                    f"submission {entry.get('submission_id', '?')}"
                )

            prev_hash = stored_hmac

    return True, f"Chain intact — {line_num} entries verified."


def generate_submission_id() -> str:
    """Generate a unique submission ID."""
    return f"BS-{uuid.uuid4().hex[:12].upper()}"