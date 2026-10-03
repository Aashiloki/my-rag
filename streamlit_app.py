"""
streamlit_app.py — Phase 6/7: Streamlit UI
Works locally and on Streamlit Cloud (free deploy).

HOW TO DEPLOY TO STREAMLIT CLOUD (free, 5 min):
1. Push your repo to GitHub
2. Go to share.streamlit.io
3. Connect repo → select streamlit_app.py
4. Add GROQ_API_KEY and API_BASE in Secrets
5. Done — you get a public URL

LEARNING: When deployed on Streamlit Cloud, this file runs on their
servers and calls your Railway/Render FastAPI URL. Set API_BASE to
your deployed backend URL, not localhost.
"""

import streamlit as st
import requests
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# LEARNING: os.environ.get() reads env vars — works both locally (.env)
# and on Streamlit Cloud (set in their Secrets UI).
# Fallback to localhost for local dev.
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="🔍", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  .source-badge {
    display: inline-block;
    background: #1a1a2e; color: #7c85e0;
    font-size: 11px; font-family: monospace;
    padding: 2px 8px; border-radius: 4px;
    margin: 2px 2px 0 0; border: 1px solid #2a2a4e;
  }
  .refusal-box {
    background: #1a1200; border: 1px solid #3d2e00;
    border-radius: 8px; padding: 12px 16px;
    color: #c9a227; font-size: 14px;
  }
  .stat-chip {
    display: inline-block;
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 6px; padding: 4px 10px;
    font-size: 12px; font-family: monospace;
    color: #8b949e; margin-right: 6px;
  }
</style>
""", unsafe_allow_html=True)

# Session state
if "messages"      not in st.session_state: st.session_state.messages      = []
if "total_tokens"  not in st.session_state: st.session_state.total_tokens  = 0
if "total_queries" not in st.session_state: st.session_state.total_queries = 0

def call_ingest():
    try:
        r = requests.post(f"{API_BASE}/ingest", timeout=120)
        return r.json().get("message", "Done.")
    except requests.exceptions.ConnectionError:
        return "❌ Cannot reach API. Is the backend running?"
    except Exception as e:
        return f"❌ {e}"

def call_ask(question):
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "top_k": 5},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Cannot reach API."}
    except Exception as e:
        return {"error": f"❌ {e}"}

def call_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json()
    except Exception:
        return {}

# Sidebar
with st.sidebar:
    st.title("🔍 RAG Chatbot")
    st.caption(f"API: {API_BASE}")
    st.divider()

    st.subheader("Documents")
    if st.button("⚙️ Ingest PDFs", use_container_width=True):
        with st.spinner("Building indexes..."):
            msg = call_ingest()
        (st.success if "❌" not in msg else st.error)(msg)
    st.caption("Run once after adding PDFs to /documents")
    st.divider()

    st.subheader("Session stats")
    stats = call_stats()
    if stats:
        st.metric("Queries",    stats.get("total_queries", 0))
        st.metric("Cache hits", stats.get("cache_hits", 0))
        total_tok = stats.get("total_prompt_tokens",0) + stats.get("total_completion_tokens",0)
        st.metric("Tokens",     total_tok)
        st.metric("Est. cost",  f"${stats.get('estimated_cost_usd',0):.5f}")
    else:
        st.caption("Start the API to see stats")
    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat area
st.header("Ask your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("is_refusal"):
                st.markdown(f'<div class="refusal-box">⚠️ {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
            if msg.get("sources"):
                badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in msg["sources"])
                st.markdown(badges, unsafe_allow_html=True)
            meta = msg.get("meta", {})
            if meta.get("prompt_tokens"):
                chips = (
                    f'<span class="stat-chip">prompt: {meta["prompt_tokens"]}</span>'
                    f'<span class="stat-chip">completion: {meta["completion_tokens"]}</span>'
                )
                if meta.get("cache_hit"):
                    chips += '<span class="stat-chip">⚡ cached</span>'
                st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = call_ask(prompt)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({"role": "assistant", "content": result["error"]})
        else:
            answer     = result.get("answer", "")
            sources    = result.get("sources", [])
            is_refusal = result.get("is_refusal", False)
            cache_hit  = result.get("cache_hit", False)
            p_tok      = result.get("prompt_tokens", 0)
            c_tok      = result.get("completion_tokens", 0)

            if is_refusal:
                st.markdown(f'<div class="refusal-box">⚠️ {answer}</div>', unsafe_allow_html=True)
            else:
                st.markdown(answer)

            if sources:
                badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
                st.markdown(badges, unsafe_allow_html=True)

            chips = f'<span class="stat-chip">prompt: {p_tok}</span><span class="stat-chip">completion: {c_tok}</span>'
            if cache_hit:
                chips += '<span class="stat-chip">⚡ cached</span>'
            st.markdown(chips, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": sources, "is_refusal": is_refusal,
                "meta": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "cache_hit": cache_hit}
            })
