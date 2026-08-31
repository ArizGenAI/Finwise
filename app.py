"""
app.py
------
FinWise AI — Streamlit entry point.

Run with:  streamlit run app.py

Layout:
- Sidebar: branding, disclaimer, model settings, cache selection, reset button.
- Main page: tabbed input form -> financial overview metrics -> AI dashboard
  (health score, spending analysis, risk level, priorities, budget
  recommendations, savings strategy, next-month plan) -> streamed narrative.
"""

import streamlit as st

from src import config
from src.financial_calculator import FinancialCalculation, build_expense_breakdown_text
from src.cache_manager import configure_cache, current_cache_mode, clear_sqlite_cache
from src.chains import run_analysis_chain, run_message_demo, stream_recommendations
from src.utils import safe_parse_financial_json, format_currency

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="💰",
    layout="wide",
)

if "result" not in st.session_state:
    st.session_state.result = None
if "calc" not in st.session_state:
    st.session_state.calc = None
if "message_demo" not in st.session_state:
    st.session_state.message_demo = None


def reset_session():
    st.session_state.result = None
    st.session_state.calc = None
    st.session_state.message_demo = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(f"💰 {config.APP_NAME}")
    st.caption(config.APP_TAGLINE)
    st.info(config.EDUCATIONAL_DISCLAIMER)

    st.divider()
    st.subheader("⚙️ Model settings")
    model_name = st.text_input("OpenAI model", value=config.DEFAULT_MODEL_NAME)
    temperature = st.slider("Temperature", 0.0, 1.0, config.DEFAULT_TEMPERATURE, 0.05)

    if not config.has_api_key():
        st.warning(
            "No OPENAI_API_KEY detected. Copy `.env.example` to `.env` and add your key "
            "before running an analysis."
        )

    st.divider()
    st.subheader("🗄️ Caching")
    cache_choice = st.selectbox("Cache backend", config.CACHE_OPTIONS, index=0)
    if st.button("Apply cache setting", use_container_width=True):
        applied = configure_cache(cache_choice)
        st.success(f"Cache set to: {applied}")
    st.caption(f"Current cache mode: **{current_cache_mode()}**")
    if cache_choice == "SQLite Cache" and st.button("Clear SQLite cache file", use_container_width=True):
        removed = clear_sqlite_cache()
        st.success("SQLite cache file removed." if removed else "No cache file found.")

    st.divider()
    if st.button("🔄 Reset session", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.caption("Project: FinWise AI · LangChain + Streamlit FinTech Assignment")

# ---------------------------------------------------------------------------
# Main page — input form
# ---------------------------------------------------------------------------
st.title("💰 FinWise AI Dashboard")
st.caption(config.APP_TAGLINE)
st.warning(config.EDUCATIONAL_DISCLAIMER)

with st.form("budget_form"):
    st.subheader("📋 Your monthly financial information")

    col_income, col_savings, col_currency = st.columns(3)
    with col_income:
        monthly_income = st.number_input("Monthly income", min_value=0.0, value=5000.0, step=100.0)
    with col_savings:
        current_savings = st.number_input("Current monthly savings", min_value=0.0, value=500.0, step=50.0)
    with col_currency:
        currency = st.selectbox("Currency", config.CURRENCIES, index=0)

    financial_goal = st.selectbox("Financial goal", config.FINANCIAL_GOALS)

    st.markdown("**Monthly expenses**")
    with st.expander("Expand to enter all expense categories", expanded=True):
        expense_cols = st.columns(2)
        expenses = {}
        for i, key in enumerate(config.EXPENSE_CATEGORIES):
            with expense_cols[i % 2]:
                expenses[key] = st.number_input(
                    config.EXPENSE_LABELS[key], min_value=0.0, value=0.0, step=10.0, key=f"exp_{key}"
                )

    submitted = st.form_submit_button("🔍 Analyze my budget", use_container_width=True)

if submitted:
    if not config.has_api_key():
        st.error("Please configure OPENAI_API_KEY in your .env file before running an analysis.")
    else:
        calc = FinancialCalculation(
            monthly_income=monthly_income, expenses=expenses, savings=current_savings
        )
        st.session_state.calc = calc

        breakdown_text = build_expense_breakdown_text(expenses, config.EXPENSE_LABELS)
        chain_inputs = {
            "monthly_income": monthly_income,
            "total_expenses": calc.total_expenses,
            "remaining_income": calc.remaining_income,
            "savings": current_savings,
            "savings_ratio": round(calc.savings_ratio, 1),
            "expense_ratio": round(calc.expense_ratio, 1),
            "financial_goal": financial_goal,
            "expense_breakdown": breakdown_text,
        }
        st.session_state.chain_inputs = chain_inputs

        with st.spinner("FinWise AI is analyzing your finances..."):
            raw_response = run_analysis_chain(chain_inputs)
            parsed, error = safe_parse_financial_json(raw_response)

        if error:
            st.error(f"AI response could not be parsed: {error}")
            with st.expander("Raw model output (for debugging)"):
                st.code(raw_response)
        else:
            st.session_state.result = parsed

# ---------------------------------------------------------------------------
# Financial overview (deterministic Python numbers)
# ---------------------------------------------------------------------------
if st.session_state.calc:
    calc = st.session_state.calc
    st.subheader("📊 Financial Overview (calculated by Python)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monthly Income", format_currency(calc.monthly_income, currency))
    m2.metric("Total Expenses", format_currency(calc.total_expenses, currency))
    m3.metric(
        "Remaining Income",
        format_currency(calc.remaining_income, currency),
        delta=f"{calc.savings_ratio:.1f}% savings ratio",
    )
    m4.metric("Current Savings", format_currency(calc.savings, currency))

    with st.expander("See detailed ratios"):
        r1, r2, r3 = st.columns(3)
        r1.metric("Expense Ratio", f"{calc.expense_ratio:.1f}%")
        r2.metric("Debt Ratio", f"{calc.debt_ratio:.1f}%")
        r3.metric("Preliminary Score (rule-based)", f"{calc.preliminary_score:.0f}/100")

# ---------------------------------------------------------------------------
# AI Dashboard (structured JSON result)
# ---------------------------------------------------------------------------
if st.session_state.result:
    result = st.session_state.result
    st.subheader("🤖 AI Financial Analysis")

    score = result["financial_health_score"]
    risk = result["risk_level"]

    col_score, col_risk = st.columns([2, 1])
    with col_score:
        st.markdown(f"**Financial Health Score:** {score}/100 — {config.score_band_label(score)}")
        st.progress(score / 100)
    with col_risk:
        risk_color = {"LOW": "success", "MEDIUM": "warning", "HIGH": "error"}.get(risk, "info")
        getattr(st, risk_color)(f"Risk level: **{risk}**")

    st.markdown("**Summary**")
    st.write(result["financial_summary"])

    tabs = st.tabs(
        [
            "🔎 Spending Analysis",
            "🎯 Priorities",
            "💡 Budget Recommendations",
            "🏦 Savings Strategy",
            "📅 Next Month Plan",
        ]
    )

    with tabs[0]:
        for item in result["spending_analysis"]:
            with st.expander(f"📌 {item.get('category', 'Category')}"):
                st.write(f"**Observation:** {item.get('observation', '')}")
                st.write(f"**Recommendation:** {item.get('recommendation', '')}")

    with tabs[1]:
        for p in result["top_priorities"]:
            st.markdown(f"- {p}")

    with tabs[2]:
        for b in result["budget_recommendations"]:
            st.markdown(f"- {b}")

    with tabs[3]:
        for s in result["savings_strategy"]:
            st.markdown(f"- {s}")

    with tabs[4]:
        for a in result["next_month_action_plan"]:
            st.markdown(f"- {a}")

    st.divider()
    st.subheader("📝 Personalized Narrative Recommendation")
    if st.button("Stream my recommendation"):
        st.write_stream(stream_recommendations(st.session_state.chain_inputs))

    st.divider()
    with st.expander("🧪 SystemMessage / HumanMessage / AIMessage demo"):
        st.caption(
            "Demonstrates constructing a raw conversation with LangChain message "
            "objects, independent of the LLMChain used above."
        )
        if st.button("Run message demo"):
            with st.spinner("Calling the model..."):
                st.session_state.message_demo = run_message_demo(st.session_state.chain_inputs)
        if st.session_state.message_demo:
            demo = st.session_state.message_demo
            st.markdown(f"**SystemMessage:**\n\n> {demo['system'][:300]}...")
            st.markdown(f"**HumanMessage:**\n\n> {demo['human']}")
            st.markdown(f"**AIMessage:**\n\n> {demo['ai']}")

    st.caption(config.EDUCATIONAL_DISCLAIMER)
