"""Narrow, evidence-backed answers for high-risk policy and card questions.

These guards use only retrieved passages and coarse segment signals. They do
not make approval decisions or attempt general natural-language entailment.
"""
from __future__ import annotations

import re

_CARD_COMPARISON = re.compile(
    r"\b(gold)\b.*\b(platinum)\b|\b(platinum)\b.*\b(gold)\b",
    re.IGNORECASE,
)
_CARD_CONTEXT = re.compile(r"\b(card|cards|credit|compare|comparison|differ|versus|vs\.?)\b", re.IGNORECASE)
_OFFER_QUESTION = re.compile(
    r"\b(waiv(?:e|er|ing)|retention|retain|retaining|offer|offers|eligible|eligibility)\b",
    re.IGNORECASE,
)

def _referenced(passages: list[dict], source: str, text: str) -> str | None:
    """Return a canonical source marker only if the matching excerpt has text."""
    for i, passage in enumerate(passages, 1):
        if passage.get("source") == source and text.lower() in passage.get("text", "").lower():
            return f"[S{i}]"
    return None


def conflicting_retention_answer(question: str, customer: dict, passages: list[dict]):
    """Avoid selecting one retention tier when the provided signals conflict."""
    if not _OFFER_QUESTION.search(question):
        return None
    if not any(p.get("source") == "retention_policy.txt" for p in passages):
        return None
    from .assistant import _retention_segment_signals

    segment = _retention_segment_signals(customer, passages)
    if "overlapping conditions" not in segment:
        return None
    waiver = _referenced(passages, "fee_waiver_policy.txt", "Eligibility")
    offers = _referenced(passages, "retention_policy.txt", "Retention Offers")
    sentences = [
        "I understand you want to explore the available options.",
        "The available tenure and credit-score indicators point to different "
        "retention segments, so I cannot determine one segment or recommend "
        "a specific offer from these records.",
    ]
    if waiver:
        sentences.append(
            "The fee waiver policy lists additional requirements, including "
            "payment-history and credit-score or verified-hardship checks "
            f"{waiver}; this dataset cannot establish that those conditions are met."
        )
    if offers:
        sentences.append(
            f"The retention policy lists different options by segment {offers}, "
            "but an offer must not be selected solely from conflicting indicators."
        )
    sentences.append(
        "Please ask an authorized retention representative to confirm the "
        "applicable segment, review the customer's concern and required "
        "records, and decide whether any option is appropriate."
    )
    markers = tuple(
        {"marker": f"[S{i}]", "source": p["source"], "line_start": p["line_start"],
         "line_end": p["line_end"], "chunk_id": p["chunk_id"]}
        for i, p in enumerate(passages, 1)
        if f"[S{i}]" in {waiver, offers}
    )
    return " ".join(sentences), markers


def _card_section(passages: list[dict], title: str):
    """Return only the matching credit-card product section, not loyalty tiers."""
    for i, passage in enumerate(passages, 1):
        if passage.get("source") != "product_summary.txt":
            continue
        text = passage.get("text", "")
        match = re.search(
            rf"(?m)^\s*\d+\.\s+{re.escape(title)} Credit Card\s*$",
            text,
        )
        if not match:
            continue
        rest = text[match.end():]
        following = re.search(r"(?m)^\s*\d+\.\s+[A-Z][^\n]+$", rest)
        section = rest[:following.start()] if following else rest
        fields = []
        for line in section.splitlines():
            clean = line.strip().removeprefix("-").strip()
            if re.match(r"(?i)^(annual fee|limit|reward rate|rewards|eligibility)\s*:", clean):
                fields.append(clean)
        if fields:
            return f"[S{i}]", fields, passage
    return None


def gold_platinum_comparison(question: str, passages: list[dict]):
    """Present source-extracted Gold/Platinum attributes with correct citations."""
    if not (_CARD_COMPARISON.search(question) and _CARD_CONTEXT.search(question)):
        return None
    gold = _card_section(passages, "Gold")
    platinum = _card_section(passages, "Platinum")
    if not (gold and platinum):
        return (
            "I cannot verify a complete Gold versus Platinum card comparison "
            "from the retrieved product excerpts. Please consult the current "
            "product summaries or an authorized bank representative.",
            (),
        )
    g_marker, g_fields, g_source = gold
    p_marker, p_fields, p_source = platinum
    text = (
        f"**Gold Credit Card** {g_marker}\n"
        + "\n".join(f"- {field}" for field in g_fields)
        + f"\n\n**Platinum Credit Card** {p_marker}\n"
        + "\n".join(f"- {field}" for field in p_fields)
        + "\n\nThese are product-card terms from the retrieved product summary. "
        "Loyalty-tier names and spending-based rewards are separate and should "
        "not be treated as automatic card benefits."
    )
    matches = (
        {"marker": g_marker, "source": g_source["source"],
         "line_start": g_source["line_start"], "line_end": g_source["line_end"],
         "chunk_id": g_source["chunk_id"]},
        {"marker": p_marker, "source": p_source["source"],
         "line_start": p_source["line_start"], "line_end": p_source["line_end"],
         "chunk_id": p_source["chunk_id"]},
    )
    return text, matches
