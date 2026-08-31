"""
cache_manager.py
-----------------
LangChain LLM response caching.

set_llm_cache(...) registers ONE global cache. Before each LLM call,
LangChain checks whether the exact same prompt (+ model + parameters) has
been seen before; if so, it returns the cached response instead of making a
new API call. This speeds up repeated prompts (e.g. re-running the same test
scenario) and reduces API cost.

Two backends are supported, matching the assignment's comparison table:

    InMemoryCache   -> lives in RAM, fastest, cleared on app restart,
                       good for a single Streamlit session.
    SQLiteCache     -> persisted to a .db file on disk, slightly slower,
                       survives restarts, good for reuse across sessions.
"""

import os
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

SQLITE_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".finwise_cache.db")

_CURRENT_CACHE_MODE = "None"


def configure_cache(mode: str) -> str:
    """
    Configure the global LangChain LLM cache.

    mode: one of "In-Memory Cache", "SQLite Cache", "No Cache"
    Returns the mode actually applied (for display in the UI).
    """
    global _CURRENT_CACHE_MODE

    if mode == "In-Memory Cache":
        set_llm_cache(InMemoryCache())
    elif mode == "SQLite Cache":
        set_llm_cache(SQLiteCache(database_path=SQLITE_CACHE_PATH))
    else:
        set_llm_cache(None)
        mode = "No Cache"

    _CURRENT_CACHE_MODE = mode
    return _CURRENT_CACHE_MODE


def current_cache_mode() -> str:
    """Return the currently configured cache mode label."""
    return _CURRENT_CACHE_MODE


def clear_sqlite_cache() -> bool:
    """Delete the on-disk SQLite cache file, if it exists. Returns True if removed."""
    if os.path.exists(SQLITE_CACHE_PATH):
        os.remove(SQLITE_CACHE_PATH)
        return True
    return False
