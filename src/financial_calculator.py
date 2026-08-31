"""
financial_calculator.py
------------------------
Pure, deterministic financial maths.

IMPORTANT: Nothing in this file talks to an LLM. Every function here is a
plain calculation — the same inputs always produce the same outputs. This
keeps a hard line between "facts computed by Python" and "insight generated
by the AI" (see chains.py / prompts.py), which the LLM is instructed to
respect and never contradict.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FinancialCalculation:
    """Container for all deterministic outputs of one budget calculation."""

    monthly_income: float
    expenses: Dict[str, float]
    savings: float
    total_expenses: float = field(init=False)
    remaining_income: float = field(init=False)
    savings_ratio: float = field(init=False)
    expense_ratio: float = field(init=False)
    debt_ratio: float = field(init=False)
    preliminary_score: float = field(init=False)

    def __post_init__(self):
        self.total_expenses = calculate_total_expenses(self.expenses)
        self.remaining_income = calculate_remaining_income(
            self.monthly_income, self.total_expenses
        )
        self.savings_ratio = calculate_savings_ratio(self.savings, self.monthly_income)
        self.expense_ratio = calculate_expense_ratio(
            self.total_expenses, self.monthly_income
        )
        self.debt_ratio = calculate_debt_ratio(
            self.expenses.get("loan_debt", 0.0), self.monthly_income
        )
        self.preliminary_score = calculate_preliminary_score(
            savings_ratio=self.savings_ratio,
            remaining_income=self.remaining_income,
            expense_ratio=self.expense_ratio,
            debt_ratio=self.debt_ratio,
        )

    def as_dict(self) -> dict:
        return {
            "monthly_income": self.monthly_income,
            "expenses": self.expenses,
            "savings": self.savings,
            "total_expenses": round(self.total_expenses, 2),
            "remaining_income": round(self.remaining_income, 2),
            "savings_ratio": round(self.savings_ratio, 2),
            "expense_ratio": round(self.expense_ratio, 2),
            "debt_ratio": round(self.debt_ratio, 2),
            "preliminary_score": round(self.preliminary_score, 1),
        }


def calculate_total_expenses(expenses: Dict[str, float]) -> float:
    """Sum every expense category."""
    return float(sum(max(0.0, v) for v in expenses.values()))


def calculate_remaining_income(monthly_income: float, total_expenses: float) -> float:
    """Income left over after all expenses."""
    return float(monthly_income - total_expenses)


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, guarding against a zero (or negative) income."""
    if denominator is None or denominator <= 0:
        return 0.0
    return numerator / denominator


def calculate_savings_ratio(savings: float, monthly_income: float) -> float:
    """Savings as a percentage of monthly income."""
    return _safe_div(savings, monthly_income) * 100


def calculate_expense_ratio(total_expenses: float, monthly_income: float) -> float:
    """Total expenses as a percentage of monthly income. Can exceed 100%."""
    return _safe_div(total_expenses, monthly_income) * 100


def calculate_debt_ratio(loan_debt_expense: float, monthly_income: float) -> float:
    """Debt/loan payments as a percentage of monthly income."""
    return _safe_div(loan_debt_expense, monthly_income) * 100


def calculate_preliminary_score(
    savings_ratio: float,
    remaining_income: float,
    expense_ratio: float,
    debt_ratio: float,
) -> float:
    """
    Weighted 0-100 heuristic used as a deterministic starting point before
    the LLM produces its own "financial_health_score". Weights:

      - Savings ratio        : 35%  (higher is better, capped at 30% ratio -> full marks)
      - Leftover / remaining  : 25%  (positive remaining income is rewarded)
      - Expense ratio         : 25%  (lower is better; >100% is heavily penalised)
      - Debt burden           : 15%  (lower is better)
    """
    # Savings component: 30% savings ratio or higher -> full 35 points
    savings_component = min(savings_ratio / 30.0, 1.0) * 35

    # Remaining income component: reward positive leftover, penalise negative
    if remaining_income > 0:
        remaining_component = 25
    elif remaining_income == 0:
        remaining_component = 12.5
    else:
        remaining_component = 0

    # Expense ratio component: 50% or below -> full marks; 100%+ -> zero
    if expense_ratio <= 50:
        expense_component = 25
    elif expense_ratio >= 100:
        expense_component = 0
    else:
        expense_component = 25 * (100 - expense_ratio) / 50

    # Debt component: 0% debt -> full marks; 40%+ debt -> zero
    if debt_ratio <= 0:
        debt_component = 15
    elif debt_ratio >= 40:
        debt_component = 0
    else:
        debt_component = 15 * (40 - debt_ratio) / 40

    score = savings_component + remaining_component + expense_component + debt_component
    return max(0.0, min(100.0, round(score, 1)))


def build_expense_breakdown_text(expenses: Dict[str, float], labels: Dict[str, str]) -> str:
    """Human-readable, comma-separated expense breakdown for prompt injection."""
    parts = []
    for key, value in expenses.items():
        if value and value > 0:
            label = labels.get(key, key)
            parts.append(f"{label}: {value:.2f}")
    return ", ".join(parts) if parts else "No expenses recorded"
