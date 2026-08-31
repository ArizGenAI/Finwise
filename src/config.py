"""
config.py
---------
Centralised configuration for FinWise AI.

Responsibilities:
- Load environment variables (OPENAI_API_KEY, model name, etc.) via python-dotenv.
- Hold static form options used by the Streamlit UI (expense categories,
  financial goals, currencies, cache options).

No LLM calls or Streamlit widgets live here — this module only exposes
plain Python constants and small helper functions so it can be imported
anywhere (app.py, chains.py, tests) without side effects other than
reading the .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the real .env next to app.py, and override any stale process env
# (e.g. a leftover placeholder from .env.example).
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE, override=True)

# ---------------------------------------------------------------------------
# API / model configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

_PLACEHOLDER_MARKERS = ("your-openai-api-key", "your-api-key", "sk-your-", "xxxx")


def has_api_key() -> bool:
    """Return True if a real OpenAI API key has been configured (not a placeholder)."""
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        return False
    lowered = OPENAI_API_KEY.lower()
    return not any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_NAME = "FinWise AI"
APP_TAGLINE = "AI-Powered Personal Financial Analysis & Smart Budget Assistant"

EDUCATIONAL_DISCLAIMER = (
    "⚠️ **Educational use only.** FinWise AI is a learning prototype. It does not "
    "provide guaranteed investment advice, execute financial transactions, or "
    "connect to real bank accounts. No outcome is guaranteed. Please consult a "
    "qualified financial professional before making real financial decisions."
)

# ---------------------------------------------------------------------------
# Form options
# ---------------------------------------------------------------------------
EXPENSE_CATEGORIES = [
    "housing_rent",
    "food",
    "transportation",
    "utilities",
    "education",
    "healthcare",
    "entertainment",
    "loan_debt",
    "subscriptions",
    "other",
]

EXPENSE_LABELS = {
    "housing_rent": "Housing / Rent",
    "food": "Food & Groceries",
    "transportation": "Transportation",
    "utilities": "Utilities",
    "education": "Education",
    "healthcare": "Healthcare",
    "entertainment": "Entertainment",
    "loan_debt": "Loan / Debt Payments",
    "subscriptions": "Subscriptions",
    "other": "Other",
}

FINANCIAL_GOALS = [
    "Save money",
    "Build an emergency fund",
    "Pay off debt",
    "Save for a vacation",
    "Start a business",
    "Improve budgeting habits",
]

CURRENCIES = ["USD", "EUR", "GBP", "PKR", "INR", "AED", "CAD", "AUD"]

CACHE_OPTIONS = ["In-Memory Cache", "SQLite Cache", "No Cache"]

# Financial health score bands (educational only)
SCORE_BANDS = [
    (80, 100, "Strong"),
    (60, 79, "Generally Healthy"),
    (40, 59, "Needs Improvement"),
    (0, 39, "High Attention"),
]


def score_band_label(score: float) -> str:
    """Map a 0-100 score to its educational band label."""
    for low, high, label in SCORE_BANDS:
        if low <= score <= high:
            return label
    return "Unknown"
