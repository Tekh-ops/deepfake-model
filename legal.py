"""
BharatShield Legal Document Generator
======================================
Generates court-ready legal documents from ML model outputs.

Usage:
    python bharatshield_legal_generator.py

Integration guide:
    Replace the DUMMY_* data classes at the bottom with real data
    from your ML pipeline and pass them into generate_all_documents().

Output:
    BharatShield_Legal_Package_<timestamp>.pdf  (all 6 docs in one file)
    Individual PDFs per document type in ./output/
"""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import Flowable


# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
NAVY       = colors.HexColor("#0C2340")
ACCENT     = colors.HexColor("#1A5276")
LIGHT_BLUE = colors.HexColor("#D6E8F7")
DANGER     = colors.HexColor("#A93226")
SUCCESS    = colors.HexColor("#1E8449")
WARNING    = colors.HexColor("#B7770D")
GRAY_DARK  = colors.HexColor("#2C3E50")
GRAY_MID   = colors.HexColor("#5D6D7E")
GRAY_LIGHT = colors.HexColor("#ECF0F1")
WHITE      = colors.white
BLACK      = colors.black


# ─────────────────────────────────────────────
# DATA MODELS  ← Replace with real ML pipeline output
# ─────────────────────────────────────────────

class MediaEvidence:
    """Represents a piece of suspect media and its ML analysis result."""
    def __init__(self):
        self.case_id             = f"BS-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
        self.media_filename      = "suspect_video_election_rally_2025.mp4"
        self.media_type          = "Video"          # Image / Video / Audio
        self.media_sha256        = hashlib.sha256(b"dummy_media_bytes_replace_with_real").hexdigest()
        self.capture_timestamp   = "2025-04-15T14:32:07+05:30"
        self.source_platform     = "WhatsApp (forwarded message)"
        self.source_url          = "https://web.whatsapp.com/share/xK9mP2 [archived]"
        self.originating_state   = "Karnataka"
        self.election_context    = "Karnataka State Legislative Assembly By-Election 2025"

        # ML Model Outputs
        self.cnn_score           = 0.924          # 0.0–1.0 probability of being synthetic
        self.temporal_score      = 0.887
        self.audio_score         = 0.951
        self.metadata_score      = 0.760
        self.fusion_score        = 0.916          # Final ensemble score
        self.verdict             = "LIKELY SYNTHETIC"  # LIKELY SYNTHETIC / UNCERTAIN / LIKELY AUTHENTIC
        self.verdict_color_tag   = "DANGER"        # DANGER / WARNING / SUCCESS
        self.model_version       = "BharatShield-DetectCore v2.1.3"
        self.inference_duration  = "4.7 seconds"
        self.salient_regions     = [
            "Face boundary blending artefacts at 00:03–00:07 (frames 72–168)",
            "Inconsistent ear-to-jaw texture gradient in frames 210–290",
            "Audio-lip sync deviation: +180ms offset detected at 00:11",
            "Compression signature anomaly: H.264 GOP structure inconsistent with stated recording device",
        ]

        # Chain of custody
        self.custody_log = [
            {"timestamp": "2025-04-15T14:32:07+05:30", "actor": "Field Officer Ramesh Kumar (Badge #KA-7721)",
             "action": "Media captured and SHA-256 hash computed via BharatShield Mobile App v3.0.1",
             "system_hash": self.media_sha256[:32] + "..."},
            {"timestamp": "2025-04-15T14:33:01+05:30", "actor": "BharatShield AI Engine (Auto)",
             "action": "Multi-modal inference completed. Result sealed with system signature.",
             "system_hash": hashlib.sha256(b"inference_result").hexdigest()[:32] + "..."},
            {"timestamp": "2025-04-15T15:10:44+05:30", "actor": "Inspector Priya Nair (Cyber Cell, Bengaluru)",
             "action": "Evidence reviewed. FIR support report generated.",
             "system_hash": hashlib.sha256(b"fir_report").hexdigest()[:32] + "..."},
            {"timestamp": "2025-04-15T16:05:00+05:30", "actor": "Nodal Officer Anand Sharma (MeitY)",
             "action": "Takedown notice issued to platform. Regulatory report filed.",
             "system_hash": hashlib.sha256(b"takedown_notice").hexdigest()[:32] + "..."},
        ]


class Complainant:
    """Person or body raising the complaint."""
    def __init__(self):
        self.name            = "Inspector Priya Nair"
        self.designation     = "Inspector, Cyber Crime Cell"
        self.organization    = "Bengaluru City Police"
        self.badge_id        = "KA-CCU-1194"
        self.contact_email   = "cybercell.blr@ksp.gov.in"
        self.contact_phone   = "+91-80-2294-3000"
        self.address         = "Cyber Crime Police Station, CID Headquarters, Carlton House, Palace Road, Bengaluru – 560 001"


class Subject:
    """Person or entity depicted in / responsible for the deepfake."""
    def __init__(self):
        self.name_depicted   = "Shri Arvind Patil (depicted without consent)"
        self.role_depicted   = "Member of Legislative Assembly, Constituency 47"
        self.alleged_creator = "Unknown — under investigation"
        self.platform_handle = "@arvind_patil_official (impersonation account)"


class NodaloOfficer:
    """MeitY nodal officer issuing the takedown."""
    def __init__(self):
        self.name         = "Shri Anand Sharma"
        self.designation  = "Director, Cyber Laws Division"
        self.ministry     = "Ministry of Electronics and Information Technology (MeitY)"
        self.email        = "nodalofficer-cyber@meity.gov.in"
        self.phone        = "+91-11-2430-1851"
        self.address      = "Electronics Niketan, 6 CGO Complex, New Delhi – 110 003"


# ─────────────────────────────────────────────
# STYLE HELPERS
# ─────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "doc_title":    S("doc_title",    fontName="Helvetica-Bold",   fontSize=16, textColor=NAVY,
                           alignment=TA_CENTER, spaceAfter=4),
        "doc_subtitle": S("doc_subtitle", fontName="Helvetica",        fontSize=10, textColor=GRAY_MID,
                           alignment=TA_CENTER, spaceAfter=12),
        "section_head": S("section_head", fontName="Helvetica-Bold",   fontSize=11, textColor=WHITE,
                           alignment=TA_LEFT,   spaceBefore=10, spaceAfter=4,
                           leftIndent=6, rightIndent=6, backColor=ACCENT,
                           borderPadding=(5,6,5,6)),
        "field_label":  S("field_label",  fontName="Helvetica-Bold",   fontSize=9,  textColor=GRAY_MID,
                           spaceBefore=4),
        "field_value":  S("field_value",  fontName="Helvetica",        fontSize=10, textColor=GRAY_DARK,
                           spaceAfter=2),
        "body":         S("body",         fontName="Helvetica",        fontSize=10, textColor=GRAY_DARK,
                           leading=15,    alignment=TA_JUSTIFY,        spaceAfter=6),
        "bullet":       S("bullet",       fontName="Helvetica",        fontSize=9.5, textColor=GRAY_DARK,
                           leftIndent=16, spaceAfter=3, leading=14),
        "verdict_text": S("verdict_text", fontName="Helvetica-Bold",   fontSize=14, textColor=DANGER,
                           alignment=TA_CENTER, spaceBefore=6, spaceAfter=6),
        "footer_text":  S("footer_text",  fontName="Helvetica-Oblique",fontSize=8,  textColor=GRAY_MID,
                           alignment=TA_CENTER),
        "mono":         S("mono",         fontName="Courier",          fontSize=8,  textColor=GRAY_DARK,
                           leading=12),
        "warning_box":  S("warning_box",  fontName="Helvetica-Bold",   fontSize=9,  textColor=WARNING,
                           alignment=TA_CENTER),
        "table_header": S("table_header", fontName="Helvetica-Bold",   fontSize=9,  textColor=WHITE),
        "table_cell":   S("table_cell",   fontName="Helvetica",        fontSize=9,  textColor=GRAY_DARK,
                           leading=13),
        "ref_num":      S("ref_num",      fontName="Courier-Bold",     fontSize=9,  textColor=ACCENT),
    }


class HeaderBanner(Flowable):
    """Draws the BharatShield document header with a coloured banner."""
    def __init__(self, doc_type_label, ref_number, classification="RESTRICTED – LAW ENFORCEMENT USE ONLY"):
        super().__init__()
        self.doc_type_label  = doc_type_label
        self.ref_number      = ref_number
        self.classification  = classification
        self.width           = A4[0] - 40*mm
        self.height          = 38*mm

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Navy banner
        c.setFillColor(NAVY)
        c.roundRect(0, h - 28*mm, w, 28*mm, 3*mm, fill=1, stroke=0)

        # BharatShield wordmark
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(8*mm, h - 13*mm, "BharatShield")
        c.setFont("Helvetica", 8)
        c.drawString(8*mm, h - 18*mm, "National Deepfake Detection & Misinformation Response Platform")

        # Document type on right
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(w - 6*mm, h - 11*mm, self.doc_type_label)
        c.setFont("Courier", 7.5)
        c.drawRightString(w - 6*mm, h - 17*mm, f"Ref: {self.ref_number}")

        # Thin accent line
        c.setStrokeColor(ACCENT)
        c.setLineWidth(1.5)
        c.line(0, h - 28.5*mm, w, h - 28.5*mm)

        # Classification bar
        c.setFillColor(DANGER)
        c.rect(0, 0, w, 7*mm, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(w / 2, 2.5*mm, self.classification)


def verdict_badge(styles, verdict, score):
    color_map = {"LIKELY SYNTHETIC": DANGER, "UNCERTAIN": WARNING, "LIKELY AUTHENTIC": SUCCESS}
    col = color_map.get(verdict, DANGER)
    data = [[
        Paragraph(f"VERDICT: {verdict}", styles["verdict_text"]),
        Paragraph(f"Fusion Score: {score:.1%}", styles["verdict_text"]),
    ]]
    t = Table(data, colWidths=["60%", "40%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#FDEDEC")),
        ("TEXTCOLOR",    (0,0), (-1,-1), col),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("BOX",          (0,0), (-1,-1), 1.5, col),
        ("ROWPADDING",   (0,0), (-1,-1), 8),
    ]))
    return t


def kv_table(rows, styles, col_widths=("35%", "65%")):
    """Renders a two-column key-value table."""
    data = [[Paragraph(k, styles["field_label"]),
             Paragraph(str(v), styles["field_value"])] for k, v in rows]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), GRAY_LIGHT),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("GRID",        (0,0), (-1,-1), 0.25, colors.HexColor("#BDC3C7")),
        ("ROWPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (0,-1), 8),
    ]))
    return t


def section(title, styles):
    return Paragraph(title.upper(), styles["section_head"])


def sig_block(name, designation, org, date_str, styles):
    data = [[
        Paragraph(f"<b>{name}</b><br/>{designation}<br/>{org}", styles["field_value"]),
        Paragraph(f"Signature: ___________________________<br/><br/>Date: {date_str}", styles["field_value"]),
    ]]
    t = Table(data, colWidths=["50%", "50%"])
    t.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 0.5, GRAY_MID),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("ROWPADDING", (0,0), (-1,-1), 8),
    ]))
    return t


# ─────────────────────────────────────────────
# DOCUMENT GENERATORS
# ─────────────────────────────────────────────

def doc1_evidence_package(evidence, complainant, styles):
    """BSA Section 63 Digital Evidence Package."""
    ref = f"EP-{evidence.case_id}"
    story = [
        HeaderBanner("DIGITAL EVIDENCE PACKAGE", ref),
        Spacer(1, 8*mm),
        Paragraph("Digital Evidence Package", styles["doc_title"]),
        Paragraph("Bharatiya Sakshya Adhiniyam 2023 · Section 63 · Hash-Certified Evidence", styles["doc_subtitle"]),
        HRFlowable(width="100%", thickness=0.5, color=GRAY_MID),
        Spacer(1, 4*mm),

        section("1. Case Identification", styles),
        kv_table([
            ("Case Reference", ref),
            ("BharatShield Case ID", evidence.case_id),
            ("Document Generated", datetime.now().strftime("%d %B %Y, %H:%M:%S IST")),
            ("Issuing Authority", f"{complainant.designation}, {complainant.organization}"),
        ], styles),
        Spacer(1, 4*mm),

        section("2. Media Under Examination", styles),
        kv_table([
            ("File Name",           evidence.media_filename),
            ("Media Type",          evidence.media_type),
            ("Capture Timestamp",   evidence.capture_timestamp),
            ("Source Platform",     evidence.source_platform),
            ("Source URL / ID",     evidence.source_url),
            ("Election Context",    evidence.election_context),
            ("Originating State",   evidence.originating_state),
        ], styles),
        Spacer(1, 4*mm),

        section("3. SHA-256 Hash Certificate (BSA S.63)", styles),
        Paragraph(
            "The following cryptographic hash was computed at the moment of media ingestion, "
            "prior to any processing or analysis. Any modification to the original file will "
            "produce a different hash value, thereby proving tampering. This satisfies the "
            "hash-based certification requirement under Section 63 of the Bharatiya Sakshya "
            "Adhiniyam, 2023.", styles["body"]),
        Spacer(1, 3*mm),
        Table([[Paragraph(f"SHA-256: {evidence.media_sha256}", styles["mono"])]],
              colWidths=["100%"],
              style=TableStyle([
                  ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#F2F3F4")),
                  ("BOX",         (0,0), (-1,-1), 0.5, ACCENT),
                  ("ROWPADDING",  (0,0), (-1,-1), 8),
              ])),
        Spacer(1, 4*mm),

        section("4. AI Detection Results", styles),
        kv_table([
            ("Model Version",        evidence.model_version),
            ("Inference Duration",   evidence.inference_duration),
            ("CNN Spatial Score",    f"{evidence.cnn_score:.1%}  (frame-level artefact detection)"),
            ("Temporal Score",       f"{evidence.temporal_score:.1%}  (inter-frame motion / A-V sync)"),
            ("Audio Forensics Score",f"{evidence.audio_score:.1%}  (synthetic voice / speaker ID)"),
            ("Metadata Score",       f"{evidence.metadata_score:.1%}  (encoding / provenance anomalies)"),
            ("Ensemble Fusion Score",f"{evidence.fusion_score:.1%}  (calibrated multi-modal ensemble)"),
        ], styles),
        Spacer(1, 4*mm),
        verdict_badge(styles, evidence.verdict, evidence.fusion_score),
        Spacer(1, 4*mm),

        section("5. Salient Detection Regions", styles),
        Paragraph("The following specific regions triggered the synthetic media classification:", styles["body"]),
    ]
    for i, region in enumerate(evidence.salient_regions, 1):
        story.append(Paragraph(f"  {i}.  {region}", styles["bullet"]))
    story += [
        Spacer(1, 4*mm),
        section("6. Legal Basis", styles),
        Paragraph(
            "This evidence package is prepared in accordance with:<br/>"
            "• <b>Bharatiya Sakshya Adhiniyam, 2023 – Section 63:</b> Digital records as primary evidence "
            "with hash-based certification ensuring chain-of-custody continuity.<br/>"
            "• <b>IT (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021:</b> "
            "Mandatory labelling and disclosure obligations for synthetic media.<br/>"
            "• <b>Indian Evidence Act (transitional):</b> Electronic records admissibility standards.",
            styles["body"]),
        Spacer(1, 6*mm),
        section("7. Certifying Officer", styles),
        Spacer(1, 3*mm),
        sig_block(complainant.name, complainant.designation,
                  complainant.organization, datetime.now().strftime("%d %B %Y"), styles),
        PageBreak(),
    ]
    return story


def doc2_takedown_notice(evidence, complainant, nodal, styles):
    """IT Rules 2021 Platform Takedown Notice."""
    ref = f"TDN-{evidence.case_id}"
    date_str = datetime.now().strftime("%d %B %Y")
    story = [
        HeaderBanner("PLATFORM TAKEDOWN NOTICE", ref),
        Spacer(1, 8*mm),
        Paragraph("Platform Takedown Notice", styles["doc_title"]),
        Paragraph(
            "IT (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021 · "
            "Rule 3(1)(b)(ii) & Rule 3(1)(b)(v)", styles["doc_subtitle"]),
        HRFlowable(width="100%", thickness=0.5, color=GRAY_MID),
        Spacer(1, 5*mm),

        kv_table([
            ("To",     "Grievance Officer / Authorised Representative"),
            ("Re",     "Mandatory Removal of Synthetic / Deepfake Media"),
            ("Date",   date_str),
            ("Ref No.", ref),
            ("Deadline", "Within 36 hours of receipt of this notice (24 hours for election-related content)"),
        ], styles),
        Spacer(1, 5*mm),

        section("1. Identification of Infringing Content", styles),
        kv_table([
            ("Platform",          evidence.source_platform),
            ("Content URL / ID",  evidence.source_url),
            ("Media Type",        evidence.media_type),
            ("Upload Handle",     "Unknown / Under Investigation"),
            ("Date Observed",     evidence.capture_timestamp),
        ], styles),
        Spacer(1, 4*mm),

        section("2. Grounds for Removal", styles),
        Paragraph(
            "The content identified above has been analysed by BharatShield, the National Deepfake "
            "Detection and Misinformation Response Platform, and has been classified as <b>likely "
            "synthetic (deepfake) media</b> with an authenticity fusion score of "
            f"<b>{evidence.fusion_score:.1%}</b>. The content:", styles["body"]),
        Paragraph("(a) Constitutes synthetic/AI-generated media in violation of <b>Rule 3(1)(b)(ii)</b> "
                  "(prohibition on false or misleading information).", styles["bullet"]),
        Paragraph("(b) Is being circulated in the context of an election and poses a direct threat to "
                  "electoral integrity under <b>Rule 3(1)(b)(v)</b>.", styles["bullet"]),
        Paragraph("(c) Depicts a named individual without their consent, constituting a violation of "
                  "privacy under <b>Section 66E of the Information Technology Act, 2000</b>.", styles["bullet"]),
        Spacer(1, 4*mm),

        section("3. Mandatory Actions Required", styles),
        Paragraph("<b>You are hereby directed to:</b>", styles["body"]),
        Paragraph("1. Remove or disable access to the content at the URL/ID specified above within "
                  "<b>36 hours</b> of receipt of this notice.", styles["bullet"]),
        Paragraph("2. Preserve all associated metadata, upload logs, IP addresses, and device identifiers "
                  "for a minimum of <b>180 days</b> for law enforcement access.", styles["bullet"]),
        Paragraph("3. Provide written acknowledgement of receipt and action taken to the nodal officer "
                  "within <b>24 hours</b>.", styles["bullet"]),
        Paragraph("4. Report the originating account to your Trust & Safety team and apply appropriate "
                  "platform-level sanctions.", styles["bullet"]),
        Spacer(1, 4*mm),

        section("4. Escalation", styles),
        Paragraph(
            "Failure to comply within the stipulated time will result in escalation to the Ministry of "
            "Electronics and Information Technology (MeitY) for action under <b>Section 69A of the "
            "Information Technology Act, 2000</b>, including potential blocking orders. "
            "Evidence Package Reference: EP-" + evidence.case_id + ".", styles["body"]),
        Spacer(1, 4*mm),

        section("5. Issuing Nodal Officer", styles),
        Spacer(1, 3*mm),
        sig_block(nodal.name, nodal.designation, nodal.ministry, date_str, styles),
        Spacer(1, 3*mm),
        kv_table([
            ("Contact Email", nodal.email),
            ("Contact Phone", nodal.phone),
            ("Office Address", nodal.address),
        ], styles),
        PageBreak(),
    ]
    return story


def doc3_fir_support_report(evidence, complainant, subject, styles):
    """FIR Support Report for law enforcement."""
    ref = f"FSR-{evidence.case_id}"
    story = [
        HeaderBanner("FIR SUPPORT REPORT", ref),
        Spacer(1, 8*mm),
        Paragraph("FIR Support Report", styles["doc_title"]),
        Paragraph(
            "Supporting document for First Information Report filing · "
            "IPC Sec. 499, 153A · IT Act Sec. 66C, 66D, 66E", styles["doc_subtitle"]),
        HRFlowable(width="100%", thickness=0.5, color=GRAY_MID),
        Spacer(1, 5*mm),

        section("1. Complainant Details", styles),
        kv_table([
            ("Name",         complainant.name),
            ("Designation",  complainant.designation),
            ("Organization", complainant.organization),
            ("Badge / ID",   complainant.badge_id),
            ("Contact",      f"{complainant.contact_email} | {complainant.contact_phone}"),
            ("Address",      complainant.address),
        ], styles),
        Spacer(1, 4*mm),

        section("2. Subject / Person Concerned", styles),
        kv_table([
            ("Person Depicted",   subject.name_depicted),
            ("Role / Office",     subject.role_depicted),
            ("Alleged Creator",   subject.alleged_creator),
            ("Platform Handle",   subject.platform_handle),
        ], styles),
        Spacer(1, 4*mm),

        section("3. Incident Summary", styles),
        Paragraph(
            f"On {evidence.capture_timestamp}, a video purportedly depicting "
            f"{subject.name_depicted} making inflammatory statements was identified in circulation "
            f"via {evidence.source_platform}. The content was flagged through BharatShield and "
            "subjected to multi-modal AI forensic analysis. The analysis conclusively determined "
            f"that the media is <b>{evidence.verdict}</b>, with a detection confidence of "
            f"<b>{evidence.fusion_score:.1%}</b>. The content is being circulated in the context "
            f"of {evidence.election_context}, posing a direct risk to public order and democratic integrity.",
            styles["body"]),
        Spacer(1, 4*mm),

        section("4. Applicable Legal Provisions", styles),
        Paragraph("The following offences are prima facie disclosed:", styles["body"]),
        kv_table([
            ("IPC Section 499",       "Defamation — false imputation of conduct to injure reputation"),
            ("IPC Section 153A",      "Promoting enmity between groups / communal incitement"),
            ("IPC Section 295A",      "Deliberate acts outraging religious feelings (if applicable)"),
            ("IT Act Section 66C",    "Identity theft — fraudulent use of another person's identity"),
            ("IT Act Section 66D",    "Cheating by personation using computer resource"),
            ("IT Act Section 66E",    "Violation of privacy — publishing private images without consent"),
            ("IT Act Section 67",     "Publishing obscene material in electronic form (if applicable)"),
            ("RPA Section 126",       "Election offence — publication of false statements about candidates"),
        ], styles),
        Spacer(1, 4*mm),

        section("5. Detection Evidence Summary", styles),
        kv_table([
            ("Evidence Package Ref.", f"EP-{evidence.case_id}"),
            ("SHA-256 (original)",    evidence.media_sha256),
            ("Fusion Score",          f"{evidence.fusion_score:.1%} (LIKELY SYNTHETIC)"),
            ("Model Version",         evidence.model_version),
        ], styles),
        Spacer(1, 4*mm),

        section("6. Recommended Immediate Actions", styles),
        Paragraph("1. Register FIR under the sections listed in Part 4 above.", styles["bullet"]),
        Paragraph("2. Issue preservation request to platform (Takedown Notice Ref: TDN-"
                  + evidence.case_id + " already dispatched).", styles["bullet"]),
        Paragraph("3. Initiate IP address and device identifier tracing via CERT-In.",  styles["bullet"]),
        Paragraph("4. Secure witness statements from individuals who received / forwarded the content.", styles["bullet"]),
        Paragraph("5. Coordinate with the Election Commission nodal officer for electoral violation proceedings.", styles["bullet"]),
        Spacer(1, 6*mm),

        section("7. Prepared By", styles),
        Spacer(1, 3*mm),
        sig_block(complainant.name, complainant.designation,
                  complainant.organization, datetime.now().strftime("%d %B %Y"), styles),
        PageBreak(),
    ]
    return story


def doc4_chain_of_custody(evidence, styles):
    """Chain of Custody Log."""
    ref = f"CCL-{evidence.case_id}"
    story = [
        HeaderBanner("CHAIN OF CUSTODY LOG", ref),
        Spacer(1, 8*mm),
        Paragraph("Chain of Custody Log", styles["doc_title"]),
        Paragraph(
            "Tamper-Resistant Audit Trail · BSA 2023 · Append-Only Signed Ledger",
            styles["doc_subtitle"]),
        HRFlowable(width="100%", thickness=0.5, color=GRAY_MID),
        Spacer(1, 5*mm),

        section("1. Evidence Identifier", styles),
        kv_table([
            ("Case ID",       evidence.case_id),
            ("Media File",    evidence.media_filename),
            ("Original Hash", evidence.media_sha256),
            ("Log Reference", ref),
        ], styles),
        Spacer(1, 4*mm),

        section("2. Custody Events", styles),
        Paragraph(
            "Each entry below is a cryptographically signed, append-only record. "
            "Any gap or modification to the sequence invalidates the chain.", styles["body"]),
        Spacer(1, 3*mm),
    ]

    # Build custody table
    table_data = [[
        Paragraph("#",              styles["table_header"]),
        Paragraph("Timestamp",      styles["table_header"]),
        Paragraph("Actor",          styles["table_header"]),
        Paragraph("Action",         styles["table_header"]),
        Paragraph("System Hash",    styles["table_header"]),
    ]]
    for i, entry in enumerate(evidence.custody_log, 1):
        table_data.append([
            Paragraph(str(i),              styles["table_cell"]),
            Paragraph(entry["timestamp"],  styles["table_cell"]),
            Paragraph(entry["actor"],      styles["table_cell"]),
            Paragraph(entry["action"],     styles["table_cell"]),
            Paragraph(entry["system_hash"], ParagraphStyle("mono_cell", fontName="Courier",
                                                           fontSize=7, textColor=ACCENT, leading=10)),
        ])

    t = Table(table_data, colWidths=["4%", "18%", "22%", "34%", "22%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),  (-1,0),  NAVY),
        ("TEXTCOLOR",    (0,0),  (-1,0),  WHITE),
        ("BACKGROUND",   (0,1),  (-1,-1), WHITE),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("GRID",         (0,0),  (-1,-1), 0.25, colors.HexColor("#BDC3C7")),
        ("VALIGN",       (0,0),  (-1,-1), "TOP"),
        ("ROWPADDING",   (0,0),  (-1,-1), 5),
        ("FONTSIZE",     (0,0),  (-1,-1), 8),
    ]))
    story += [
        t, Spacer(1, 5*mm),
        section("3. Integrity Statement", styles),
        Paragraph(
            "This log certifies that the digital evidence identified above has remained in an "
            "unaltered state from the point of initial capture to the present. The hash values "
            "recorded at each custody event may be independently verified against the media file "
            "using any standard SHA-256 utility. This document satisfies the chain-of-custody "
            "requirements for electronic evidence admissibility under <b>BSA 2023 Section 63</b> "
            "and the precedents established under <b>State (NCT of Delhi) v. Navjot Sandhu (2005)</b>.",
            styles["body"]),
        PageBreak(),
    ]
    return story


def doc5_regulatory_report(evidence, complainant, nodal, styles):
    """MeitY / ECI Regulatory Compliance Report."""
    ref = f"RCR-{evidence.case_id}"
    story = [
        HeaderBanner("REGULATORY COMPLIANCE REPORT", ref,
                     classification="CONFIDENTIAL – ELECTORAL INTEGRITY"),
        Spacer(1, 8*mm),
        Paragraph("Regulatory Compliance Report", styles["doc_title"]),
        Paragraph(
            "Submission to MeitY and Election Commission of India · "
            "IT Rules 2021 · Representation of the People Act, 1951",
            styles["doc_subtitle"]),
        HRFlowable(width="100%", thickness=0.5, color=GRAY_MID),
        Spacer(1, 5*mm),

        section("1. Reporting Authority", styles),
        kv_table([
            ("Submitted By",  f"{nodal.name}, {nodal.designation}"),
            ("Ministry",      nodal.ministry),
            ("Report Period", f"Incident Report – {datetime.now().strftime('%B %Y')}"),
            ("Report Ref",    ref),
        ], styles),
        Spacer(1, 4*mm),

        section("2. Incident Summary", styles),
        kv_table([
            ("Total Media Items Assessed",    "1 (this incident)"),
            ("Synthetic / Deepfake",          "1  (100%)"),
            ("Uncertain",                     "0"),
            ("Likely Authentic",              "0"),
            ("Takedown Notices Issued",       "1  (Ref: TDN-" + evidence.case_id + ")"),
            ("Law Enforcement FIR Filed",     "Pending  (Ref: FSR-" + evidence.case_id + ")"),
            ("Platform Response Received",    "Awaited (36-hr window active)"),
        ], styles),
        Spacer(1, 4*mm),

        section("3. Detection Metrics", styles),
        Paragraph(
            "The following modality-level scores informed the final fusion verdict. "
            "All scores are calibrated probability estimates (0.0 = definitely authentic, "
            "1.0 = definitely synthetic):", styles["body"]),
        kv_table([
            ("CNN Spatial",     f"{evidence.cnn_score:.1%}"),
            ("Temporal",        f"{evidence.temporal_score:.1%}"),
            ("Audio Forensics", f"{evidence.audio_score:.1%}"),
            ("Metadata",        f"{evidence.metadata_score:.1%}"),
            ("FUSION VERDICT",  f"{evidence.fusion_score:.1%} — {evidence.verdict}"),
        ], styles),
        Spacer(1, 4*mm),

        section("4. Platform Compliance Status", styles),
        kv_table([
            ("Platform",        evidence.source_platform),
            ("Notice Sent",     datetime.now().strftime("%d %B %Y %H:%M IST")),
            ("Deadline",        "36 hours from notice"),
            ("Status",          "PENDING"),
            ("Escalation Path", "MeitY → Section 69A blocking order if non-compliant"),
        ], styles),
        Spacer(1, 4*mm),

        section("5. Electoral Impact Assessment", styles),
        Paragraph(
            f"The content was identified during the <b>{evidence.election_context}</b>. "
            "Circulating synthetic media depicting a candidate or elected representative during "
            "an election period constitutes a direct threat to the integrity of the electoral "
            "process under <b>Section 126 of the Representation of the People Act, 1951</b>. "
            "The Election Commission of India is hereby notified for any additional action under "
            "its plenary powers.", styles["body"]),
        Spacer(1, 4*mm),

        section("6. Recommendations", styles),
        Paragraph("1. Mandate real-time BharatShield API integration with major platforms during election periods.", styles["bullet"]),
        Paragraph("2. Establish a dedicated election-period rapid-response channel with 4-hour takedown SLA.", styles["bullet"]),
        Paragraph("3. Issue public advisory on deepfake detection to voters via PIB.", styles["bullet"]),
        Paragraph("4. Review adequacy of penalties under current IT Rules for electoral deepfake offences.", styles["bullet"]),
        Spacer(1, 6*mm),

        section("7. Submitted By", styles),
        Spacer(1, 3*mm),
        sig_block(nodal.name, nodal.designation, nodal.ministry,
                  datetime.now().strftime("%d %B %Y"), styles),
        PageBreak(),
    ]
    return story


def doc6_expert_witness_statement(evidence, complainant, styles):
    """Expert Witness Statement for judicial proceedings."""
    ref = f"EWS-{evidence.case_id}"
    story = [
        HeaderBanner("EXPERT WITNESS STATEMENT", ref),
        Spacer(1, 8*mm),
        Paragraph("Expert Witness Statement", styles["doc_title"]),
        Paragraph(
            "Sworn technical statement for judicial proceedings · "
            "BSA 2023 · Indian Evidence Act (transitional)", styles["doc_subtitle"]),
        HRFlowable(width="100%", thickness=0.5, color=GRAY_MID),
        Spacer(1, 5*mm),

        section("1. Expert Identification", styles),
        kv_table([
            ("Name",           "Dr. Kavitha Rajan"),
            ("Designation",    "Principal Scientist, AI Forensics Division"),
            ("Organization",   "Centre for Development of Advanced Computing (C-DAC), Pune"),
            ("Qualifications", "Ph.D. Computer Vision (IISc, 2018) · M.Tech AI (IIT Bombay, 2014)"),
            ("Experience",     "11 years in digital media forensics and deepfake detection research"),
            ("Statement Ref",  ref),
        ], styles),
        Spacer(1, 4*mm),

        section("2. Scope of Statement", styles),
        Paragraph(
            "I have been requested to provide a technical expert statement regarding the "
            f"BharatShield analysis of the media file <b>{evidence.media_filename}</b> "
            f"(SHA-256: {evidence.media_sha256[:32]}...) in relation to Case ID "
            f"<b>{evidence.case_id}</b>. This statement is made on the basis of my professional "
            "expertise and my review of the evidence package, detection methodology, and model "
            "validation benchmarks.", styles["body"]),
        Spacer(1, 4*mm),

        section("3. Technical Methodology Explained", styles),
        Paragraph(
            "<b>3.1 Spatial CNN Analysis:</b> Convolutional neural networks (EfficientNet / Xception "
            "variants) examine individual video frames for pixel-level artefacts characteristic of "
            "generative AI synthesis — including boundary blending inconsistencies, unnatural texture "
            "gradients, and lighting direction mismatches.", styles["body"]),
        Paragraph(
            "<b>3.2 Temporal Transformer Analysis:</b> Transformer architectures process sequences "
            "of frames to detect motion inconsistencies and audio-visual synchronisation anomalies "
            "that persist across time — capturing synthesis signatures that frame-by-frame analysis "
            "may miss.", styles["body"]),
        Paragraph(
            "<b>3.3 Audio Forensics:</b> Dedicated models perform speaker verification and identify "
            "artefacts associated with neural text-to-speech and voice conversion systems.",
            styles["body"]),
        Paragraph(
            "<b>3.4 Metadata & Encoding Analysis:</b> The file's container metadata, compression "
            "parameters, and encoding signature are examined for inconsistencies with the stated "
            "recording device and provenance.", styles["body"]),
        Paragraph(
            "<b>3.5 Ensemble Fusion:</b> Outputs from all four modalities are combined using a "
            "calibrated ensemble (gradient-boosted logistic fusion). This multi-pathway approach "
            "reduces false positives arising from reliance on any single detection channel.",
            styles["body"]),
        Spacer(1, 4*mm),

        section("4. Model Validation & Known Limitations", styles),
        kv_table([
            ("Validation Dataset",   "FaceForensics++, DFDC, CelebDF-v2, WildDeepfake (combined 280,000+ samples)"),
            ("Reported Accuracy",    "94.1% F1-score on held-out test set (BharatShield v2.1.3 benchmarks)"),
            ("False Positive Rate",  "~3.2% on high-quality authentic media"),
            ("False Negative Rate",  "~5.1% on novel generation architectures not in training data"),
            ("Known Limitations",    "Performance may degrade on highly compressed media (<500kbps); "
                                     "adversarially crafted deepfakes designed to evade detection; "
                                     "media recorded under extreme low-light conditions."),
        ], styles),
        Spacer(1, 4*mm),

        section("5. Expert Opinion on This Case", styles),
        Paragraph(
            f"Having reviewed the evidence package and the detection output, it is my professional "
            f"opinion that the media file <b>{evidence.media_filename}</b> is, to a high degree of "
            f"technical certainty, a <b>synthetically generated (deepfake) video</b>. The fusion "
            f"score of <b>{evidence.fusion_score:.1%}</b> significantly exceeds the threshold for a "
            "'LIKELY SYNTHETIC' classification. The four specific salient regions identified by the "
            "system are consistent with known artefacts of current-generation face-swap and voice "
            "cloning technologies. I am prepared to appear before the Hon'ble Court to give oral "
            "evidence and be cross-examined on this statement.", styles["body"]),
        Spacer(1, 5*mm),

        section("6. Declaration", styles),
        Paragraph(
            "I, Dr. Kavitha Rajan, do solemnly affirm that the contents of this statement are true "
            "and correct to the best of my knowledge, information, and belief, and that nothing "
            "material has been concealed therefrom.", styles["body"]),
        Spacer(1, 6*mm),

        sig_block("Dr. Kavitha Rajan", "Principal Scientist, AI Forensics Division",
                  "C-DAC Pune", datetime.now().strftime("%d %B %Y"), styles),
    ]
    return story


# ─────────────────────────────────────────────
# PAGE TEMPLATE (header/footer on every page)
# ─────────────────────────────────────────────

def make_page_template(canvas, doc, case_id):
    canvas.saveState()
    w, h = A4
    # Footer
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GRAY_MID)
    canvas.drawString(20*mm, 12*mm,
                      f"BharatShield · Case {case_id} · RESTRICTED")
    canvas.drawRightString(w - 20*mm, 12*mm,
                           f"Page {doc.page}  |  Generated {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    canvas.setStrokeColor(GRAY_MID)
    canvas.setLineWidth(0.3)
    canvas.line(20*mm, 15*mm, w - 20*mm, 15*mm)
    canvas.restoreState()


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_all_documents(evidence=None, complainant=None, subject=None, nodal=None,
                           output_path="BharatShield_Legal_Package.pdf"):
    """
    Generate the complete BharatShield legal document PDF.

    Parameters
    ----------
    evidence    : MediaEvidence  — ML model outputs + media metadata
    complainant : Complainant    — reporting officer
    subject     : Subject        — person depicted / accused
    nodal       : NodaloOfficer  — MeitY nodal officer
    output_path : str            — output PDF path
    """
    # Use dummy data if not provided (for standalone testing)
    evidence    = evidence    or MediaEvidence()
    complainant = complainant or Complainant()
    subject     = subject     or Subject()
    nodal       = nodal       or NodaloOfficer()

    styles = build_styles()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm,  bottomMargin=22*mm,
        title=f"BharatShield Legal Package — {evidence.case_id}",
        author="BharatShield Platform",
        subject="Deepfake Detection Legal Documents",
    )

    # Cover page
    cover = [
        Spacer(1, 30*mm),
        Paragraph("BHARATSHIELD", ParagraphStyle("cover_brand", fontName="Helvetica-Bold",
                  fontSize=32, textColor=NAVY, alignment=TA_CENTER)),
        Paragraph("National Deepfake Detection & Misinformation Response Platform",
                  ParagraphStyle("cover_sub", fontName="Helvetica", fontSize=13,
                                 textColor=GRAY_MID, alignment=TA_CENTER, spaceAfter=4)),
        Spacer(1, 6*mm),
        HRFlowable(width="80%", thickness=2, color=ACCENT, hAlign="CENTER"),
        Spacer(1, 6*mm),
        Paragraph("LEGAL DOCUMENT PACKAGE",
                  ParagraphStyle("cover_type", fontName="Helvetica-Bold", fontSize=20,
                                 textColor=ACCENT, alignment=TA_CENTER, spaceAfter=8)),
        Spacer(1, 8*mm),
        Table([[
            Paragraph(f"Case ID", styles["field_label"]),
            Paragraph(evidence.case_id, styles["ref_num"]),
        ],[
            Paragraph("Generated", styles["field_label"]),
            Paragraph(datetime.now().strftime("%d %B %Y, %H:%M:%S IST"), styles["field_value"]),
        ],[
            Paragraph("Classification", styles["field_label"]),
            Paragraph("RESTRICTED — LAW ENFORCEMENT USE ONLY",
                      ParagraphStyle("danger_sm", fontName="Helvetica-Bold",
                                     fontSize=9, textColor=DANGER)),
        ]], colWidths=["35%", "65%"],
        style=TableStyle([
            ("BOX",        (0,0), (-1,-1), 0.5, GRAY_MID),
            ("INNERGRID",  (0,0), (-1,-1), 0.25, GRAY_LIGHT),
            ("ROWPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",(0,0), (0,-1), 10),
            ("BACKGROUND", (0,0), (0,-1), GRAY_LIGHT),
        ])),
        Spacer(1, 10*mm),
        Paragraph("DOCUMENTS INCLUDED",
                  ParagraphStyle("toc_head", fontName="Helvetica-Bold", fontSize=11,
                                 textColor=NAVY, alignment=TA_CENTER)),
        Spacer(1, 3*mm),
        Table([
            ["1", "Digital Evidence Package",         "BSA 2023 · Section 63"],
            ["2", "Platform Takedown Notice",          "IT Rules 2021 · Rule 3(1)(b)"],
            ["3", "FIR Support Report",                "IPC 499, 153A · IT Act 66C/D/E"],
            ["4", "Chain of Custody Log",              "BSA 2023 · Append-Only Ledger"],
            ["5", "Regulatory Compliance Report",      "MeitY / ECI Submission"],
            ["6", "Expert Witness Statement",          "Judicial Proceedings"],
        ], colWidths=["6%", "52%", "42%"],
        style=TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), GRAY_LIGHT),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [GRAY_LIGHT, WHITE]),
            ("GRID",        (0,0), (-1,-1), 0.25, colors.HexColor("#BDC3C7")),
            ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWPADDING",  (0,0), (-1,-1), 6),
            ("FONTNAME",    (1,0), (1,-1), "Helvetica-Bold"),
            ("TEXTCOLOR",   (1,0), (1,-1), NAVY),
            ("TEXTCOLOR",   (2,0), (2,-1), GRAY_MID),
        ])),
        PageBreak(),
    ]

    story = (
        cover
        + doc1_evidence_package(evidence, complainant, styles)
        + doc2_takedown_notice(evidence, complainant, nodal, styles)
        + doc3_fir_support_report(evidence, complainant, subject, styles)
        + doc4_chain_of_custody(evidence, styles)
        + doc5_regulatory_report(evidence, complainant, nodal, styles)
        + doc6_expert_witness_statement(evidence, complainant, styles)
    )

    doc.build(story, onFirstPage=lambda c, d: make_page_template(c, d, evidence.case_id),
                     onLaterPages=lambda c, d: make_page_template(c, d, evidence.case_id))

    print(f"[BharatShield] Legal package generated: {output_path}")
    print(f"[BharatShield] Case ID: {evidence.case_id}")
    print(f"[BharatShield] Documents: Evidence Package · Takedown Notice · FIR Support · "
          f"Chain of Custody · Regulatory Report · Expert Witness Statement")
    return output_path


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    output = generate_all_documents(output_path="./BharatShield_Legal_Package.pdf")
    print(f"[BharatShield] Done → {output}")