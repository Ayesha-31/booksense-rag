# app.py
import time
import streamlit as st
from retriever import get_top_chunks
from generator import answer_with_context

# ---------- Page setup ----------
st.set_page_config(page_title="BookSense (making sense of books) — Review QA (RAG)", page_icon="📚", layout="centered")

# Subtle styling (safe, optional)
st.markdown(
    """
    <style>
      .answer-box {padding: 1rem 1.25rem; border: 1px solid #3a3a3a; border-radius: 12px;}
      .src-chip {font-size: 0.9rem; opacity: 0.85;}
      .muted {opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown("# 📚 BookSense")
st.markdown(
    "### Making sense of books through real reader voices.\n"
    "Ask emotional, thematic, or opinion-based questions grounded in real reviews. AI-Powered Book Review Q&A (RAG)"
)

# Quick “How it works”
with st.expander("🔎 How does BookSense work?"):
    st.markdown(
        """
        1. **Retrieves** the most relevant review snippets from a **Milvus** vector database.  
        2. **Builds context** and sends it with your question to **Groq’s Llama-3.1 model**.  
        3. **Generates** a concise answer **only from those snippets** (no hallucinations).  
        """
    )

# ---------- Sidebar: About ----------
with st.sidebar:
    st.header("📘 About")
    st.markdown(
        """
- **Tech**: Streamlit · Milvus · Groq (Llama-3.1-8B)
- **Data**: 130k+ review chunks
- **Method**: Retrieval-Augmented Generation (RAG)
- **Author**: Ayesha Tabassum Shaik
        """
    )
    st.divider()


# ---------- Example prompts ----------
st.markdown("#### 💡 Try a question")
col1, col2, col3 = st.columns(3)
examples = [
    "Convince me to read Game of Thrones",
    "Which books made readers cry the most?",
    "If I loved Harry Potter, what should I read next?",
]
if col1.button("Game of Thrones ❤️"):
    st.session_state["q"] = examples[0]
if col2.button("Most tear-jerking 😭"):
    st.session_state["q"] = examples[1]
if col3.button("Next Read!📝"):
    st.session_state["q"] = examples[2]

# ---------- Input controls ----------
q_default = "If I loved Harry Potter, what should I read next?"
q = st.text_input("Ask about any book or theme:", value=st.session_state.get("q", q_default))
k = st.slider("Top-K chunks (retrieval depth)", min_value=3, max_value=10, value=5, help="Higher may improve recall, but can add latency and cost.")

go = st.button("🔍 Search & Answer")

# ---------- Run query ----------
if go:
    t0 = time.time()
    try:
        with st.spinner("Retrieving relevant reviews…"):
            chunks = get_top_chunks(q, k=k)

        if not chunks:
            st.warning("No relevant sources found. Try rephrasing or increasing Top-K.")
        else:
            # Generate the grounded answer
            with st.spinner("Generating grounded answer…"):
                answer = answer_with_context(q, chunks)

            # ---------- Answer UI ----------
            st.subheader("Answer")
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            st.caption(f"Latency: {time.time() - t0:.2f}s • Top-K: {k}")

            # ---------- Sources ----------
            with st.expander("READ ACTUAL REVIEWS! (retrieved chunks)"):
                for i, c in enumerate(chunks, 1):
                    header = f"**[{i}] {c.get('book_title','Unknown Title')} — {c.get('authors','Unknown Author')}**"
                    score = c.get("score", None)
                    if score is not None:
                        header += f"  <span class='src-chip muted'>(score={score:.3f})</span>"
                    st.markdown(header, unsafe_allow_html=True)
                    st.write(c.get("chunk_text", ""))

    except Exception as e:
        st.error("Something went wrong while answering your question.")
        with st.expander("Show error details"):
            st.code(repr(e))
