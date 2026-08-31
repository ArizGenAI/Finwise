"""
utils.py
--------
Small shared helpers: safe JSON parsing/validation for the LLM's structured
output, plus a couple of formatting utilities used by the Streamlit UI.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

REQUIRED_KEYS = [
    "financial_summary",
    "financial_health_score",
    "spending_analysis",
    "risk_level",
    "top_priorities",
    "budget_recommendations",
    "savings_strategy",
    "next_month_action_plan",
]

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if the model added them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first {...} JSON object substring from arbitrary text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def safe_parse_financial_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Safely parse the LLM's JSON output against the expected FinWise AI schema.

    Returns (parsed_dict, error_message):
      - On success: (dict, None)
      - On failure: (None, "human readable error")
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty response from the model."

    cleaned = _strip_markdown_fences(raw_text)

    # First attempt: parse as-is
    parsed = None
    for candidate in (cleaned, _extract_json_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if parsed is None:
        return None, "Could not parse the model's response as valid JSON."

    if not isinstance(parsed, dict):
        return None, "Parsed JSON is not an object."

    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    if missing:
        return None, f"JSON is missing required keys: {', '.join(missing)}"

    # Normalise / validate a few fields defensively
    try:
        parsed["financial_health_score"] = int(round(float(parsed["financial_health_score"])))
        parsed["financial_health_score"] = max(0, min(100, parsed["financial_health_score"]))
    except (TypeError, ValueError):
        parsed["financial_health_score"] = 0

    risk = str(parsed.get("risk_level", "")).strip().upper()
    parsed["risk_level"] = risk if risk in VALID_RISK_LEVELS else "MEDIUM"

    for list_key in (
        "spending_analysis",
        "top_priorities",
        "budget_recommendations",
        "savings_strategy",
        "next_month_action_plan",
    ):
        if not isinstance(parsed.get(list_key), list):
            parsed[list_key] = []

    return parsed, None


def format_currency(amount: float, currency: str = "USD") -> str:
    """Basic currency formatting for display in the dashboard."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "PKR": "Rs ", "INR": "₹", "AED": "AED "}
    symbol = symbols.get(currency, f"{currency} ")
    return f"{symbol}{amount:,.2f}"
