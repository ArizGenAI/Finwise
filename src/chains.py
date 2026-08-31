"""
chains.py
---------
Wires up the LangChain LLM object and the chains used by the app:

- get_llm()                    -> builds a ChatOpenAI instance.
- get_analysis_chain()         -> reusable LLMChain (PromptTemplate -> JSON).
- run_message_demo()           -> shows SystemMessage / HumanMessage / AIMessage
                                   used directly in a conversation (learning
                                   objective requirement, independent of the
                                   chain above).
- stream_recommendations()     -> generator that streams the narrative
                                   recommendation chunk by chunk using
                                   ChatPromptTemplate + llm.stream().
"""

from typing import Dict, Generator

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src import config
from src.prompts import FINANCIAL_ANALYSIS_PROMPT, NARRATIVE_CHAT_TEMPLATE, SYSTEM_ROLE_TEXT


def get_llm(temperature: float = None, streaming: bool = False) -> ChatOpenAI:
    """
    Build a ChatOpenAI model instance.

    temperature: sampling temperature (defaults to config.DEFAULT_TEMPERATURE)
    streaming:   whether this instance will be used with .stream()
    """
    return ChatOpenAI(
        model=config.DEFAULT_MODEL_NAME,
        temperature=temperature if temperature is not None else config.DEFAULT_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
        streaming=streaming,
    )


def get_analysis_chain(llm: ChatOpenAI = None) -> LLMChain:
    """
    Build the reusable LLMChain that turns financial data (via the
    PromptTemplate in prompts.py) into a structured-JSON response.
    """
    llm = llm or get_llm()
    return LLMChain(llm=llm, prompt=FINANCIAL_ANALYSIS_PROMPT)


def run_analysis_chain(inputs: Dict) -> str:
    """
    Run the analysis LLMChain with the given financial inputs and return the
    raw text response (expected to be a JSON string — see utils.py for safe
    parsing).
    """
    chain = get_analysis_chain()
    result = chain.invoke(inputs)
    # LLMChain.invoke returns a dict with the prompt's output key ("text")
    return result["text"] if isinstance(result, dict) else str(result)


# ---------------------------------------------------------------------------
# Learning-objective demo: SystemMessage / HumanMessage / AIMessage used
# directly, outside of a chain, to show how a conversation is represented.
# ---------------------------------------------------------------------------
def run_message_demo(inputs: Dict) -> Dict[str, str]:
    """
    Demonstrates constructing a raw conversation with SystemMessage,
    HumanMessage, and (after the call) AIMessage. Returns a dict with each
    message's role/content so the UI can display them for teaching purposes.
    """
    llm = get_llm()

    system_msg = SystemMessage(content=SYSTEM_ROLE_TEXT)
    human_msg = HumanMessage(
        content=(
            f"In one short sentence, how does a savings ratio of "
            f"{inputs.get('savings_ratio', 0)}% and an expense ratio of "
            f"{inputs.get('expense_ratio', 0)}% look for someone whose goal is "
            f"'{inputs.get('financial_goal', 'improve budgeting')}'?"
        )
    )

    response = llm.invoke([system_msg, human_msg])
    ai_msg = AIMessage(content=response.content)

    return {
        "system": system_msg.content,
        "human": human_msg.content,
        "ai": ai_msg.content,
    }


# ---------------------------------------------------------------------------
# Streaming narrative recommendation
# ---------------------------------------------------------------------------
def stream_recommendations(inputs: Dict) -> Generator[str, None, None]:
    """
    Stream the narrative recommendation chunk by chunk.

    Usage in Streamlit:
        st.write_stream(stream_recommendations(inputs))
    """
    llm = get_llm(streaming=True)
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
