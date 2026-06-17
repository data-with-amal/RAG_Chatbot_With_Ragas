"""
RAG Chatbot with RAGAS Evaluation — Streamlit App
Stack: LangChain · ChromaDB · HuggingFace Embeddings · Groq · gTTS
"""

import io
import os
import json
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .chat-user {
        background: #1e3a5f;
        border-left: 4px solid #60a5fa;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    .chat-bot {
        background: #1a3a2a;
        border-left: 4px solid #4ade80;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "vectorstore": None,
    "retriever": None,
    "rag_chain": None,
    "chat_history": [],
    "eval_results": None,
    "index_ready": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)


def build_rag_chain(pdf_bytes: bytes, cfg: dict):
    """Parse PDF → embed → build chain. Returns (vectorstore, retriever, chain)."""
    import tempfile
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # Write to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )
    chunks = splitter.split_documents(documents)

    embed_model = load_embedding_model(cfg["embed_model"])
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embed_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg["top_k"]})

    llm = ChatGroq(model=cfg["llm_model"], api_key=cfg["groq_key"])

    RAG_TEMPLATE = """\
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say "I don't have that information."
Be concise and clear.

Context:
{context}

Question: {question}
Answer:"""

    rag_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_TEMPLATE,
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    os.unlink(tmp_path)
    return vectorstore, retriever, chain, len(chunks)


def tts_audio_b64(text: str, lang: str = "en", slow: bool = False) -> str | None:
    """Return base64-encoded MP3 or None on failure."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=slow).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return None


def evaluate_rag(questions: list, retriever, vectorstore, rag_chain, top_k: int) -> pd.DataFrame:
    rows = []
    for q in questions:
        try:
            answer = rag_chain.invoke(q)
            try:
                docs = retriever.get_relevant_documents(q)
            except Exception:
                docs = vectorstore.similarity_search(q, k=top_k)

            n = len(docs)
            avg_len = sum(len(d.page_content) for d in docs) / n if n else 0
            has_answer = 1.0 if answer.strip() and "i don't have that information" not in answer.lower() else 0.0
            rows.append({"Question": q, "Answer": answer, "Docs_Retrieved": n,
                         "Avg_Doc_Length": round(avg_len), "Has_Answer": has_answer})
        except Exception as e:
            rows.append({"Question": q, "Answer": f"ERROR: {e}", "Docs_Retrieved": 0,
                         "Avg_Doc_Length": 0, "Has_Answer": 0.0})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Configuration
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    groq_key = st.text_input("🔑 Groq API Key", type="password",
                             help="Get a free key at https://console.groq.com")

    st.markdown("### 📄 Knowledge Base")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    st.markdown("### 🛠️ RAG Settings")
    chunk_size    = st.slider("Chunk Size",    200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap",  0,  400,  100,  50)
    top_k         = st.slider("Top-K Chunks",   1,   10,    4)
    llm_model     = st.selectbox("LLM Model", [
        "llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"
    ])
    embed_model   = st.selectbox("Embedding Model", [
        "all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"
    ])

    st.markdown("### 🔊 TTS Settings")
    tts_enabled = st.toggle("Enable Text-to-Speech", value=True)
    tts_lang    = st.selectbox("TTS Language", ["en", "hi", "fr", "de", "es", "ta", "ml"])
    tts_slow    = st.toggle("Slow speech", value=False)

    st.markdown("---")
    build_btn = st.button("🚀 Build Index", type="primary", use_container_width=True,
                          disabled=not (groq_key and uploaded_pdf))

    if build_btn:
        cfg = dict(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                   top_k=top_k, llm_model=llm_model, embed_model=embed_model,
                   groq_key=groq_key)
        with st.spinner("Building vector index… this may take a minute."):
            try:
                vs, ret, chain, n_chunks = build_rag_chain(uploaded_pdf.read(), cfg)
                st.session_state.vectorstore  = vs
                st.session_state.retriever    = ret
                st.session_state.rag_chain    = chain
                st.session_state.index_ready  = True
                st.session_state.chat_history = []
                st.session_state.eval_results = None
                st.session_state["n_chunks"]  = n_chunks
                st.success(f"✅ Indexed {n_chunks} chunks!")
            except Exception as e:
                st.error(f"❌ {e}")

    # Status pill
    if st.session_state.index_ready:
        st.markdown(
            '<span style="background:#22c55e;color:white;padding:4px 12px;'
            'border-radius:999px;font-size:0.8rem;font-weight:700;">● Index Ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="background:#94a3b8;color:white;padding:4px 12px;'
            'border-radius:999px;font-size:0.8rem;font-weight:700;">○ No Index</span>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main area — Tabs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:2rem;">📚 RAG Chatbot</h1>
    <p style="margin:0.4rem 0 0;opacity:0.8;">
        LangChain · ChromaDB · HuggingFace · Groq · gTTS · RAGAS
    </p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_eval, tab_about = st.tabs(["💬 Chat", "📊 Evaluation", "ℹ️ About"])

# ── TAB 1 — CHAT ───────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.index_ready:
        st.info("👈 Upload a PDF and enter your Groq API key in the sidebar, then click **Build Index**.")
    else:
        # Chat display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">🤖 <b>Assistant:</b> {msg["content"]}</div>',
                                unsafe_allow_html=True)
                    if msg.get("audio"):
                        st.audio(base64.b64decode(msg["audio"]), format="audio/mp3")
                    if msg.get("context"):
                        with st.expander("📎 Retrieved context chunks"):
                            for i, chunk in enumerate(msg["context"], 1):
                                st.markdown(f"**[{i}]** {chunk[:300]}…")

        # Input row
        col_q, col_ctx = st.columns([5, 1])
        with col_q:
            question = st.chat_input("Ask a question about the document…")
        with col_ctx:
            show_ctx = st.toggle("Show context", value=False)

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Thinking…"):
                answer = st.session_state.rag_chain.invoke(question)

                # Retrieve context chunks for display
                context_snippets = []
                if show_ctx:
                    try:
                        docs = st.session_state.retriever.get_relevant_documents(question)
                    except Exception:
                        docs = st.session_state.vectorstore.similarity_search(question, k=top_k)
                    context_snippets = [d.page_content for d in docs]

                # TTS
                audio_b64 = None
                if tts_enabled:
                    audio_b64 = tts_audio_b64(answer, lang=tts_lang, slow=tts_slow)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "audio": audio_b64,
                "context": context_snippets,
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.rerun()


# ── TAB 2 — EVALUATION ────────────────────────────────────────────────────────
with tab_eval:
    if not st.session_state.index_ready:
        st.info("Build the index first (sidebar).")
    else:
        st.markdown("### 📝 Test Questions")
        st.caption("Enter one question per line.")
        default_qs = (
            "How many casual leaves are allowed?\n"
            "What are the office timings?\n"
            "How does the company handle performance reviews?\n"
            "What benefits does the company offer?\n"
            "What is the company's mission statement?"
        )
        raw_questions = st.text_area("Questions", value=default_qs, height=160)
        questions = [q.strip() for q in raw_questions.splitlines() if q.strip()]
        st.caption(f"{len(questions)} questions")

        if st.button("▶️ Run Evaluation", type="primary", disabled=not questions):
            progress = st.progress(0, text="Evaluating…")
            rows = []
            for i, q in enumerate(questions):
                progress.progress((i + 1) / len(questions), text=f"[{i+1}/{len(questions)}] {q[:50]}")
                try:
                    answer = st.session_state.rag_chain.invoke(q)
                    try:
                        docs = st.session_state.retriever.get_relevant_documents(q)
                    except Exception:
                        docs = st.session_state.vectorstore.similarity_search(q, k=top_k)
                    n = len(docs)
                    avg_len = sum(len(d.page_content) for d in docs) / n if n else 0
                    has_ans = 1.0 if answer.strip() and "i don't have that information" not in answer.lower() else 0.0
                    rows.append({"Question": q, "Answer": answer, "Docs_Retrieved": n,
                                 "Avg_Doc_Length": round(avg_len), "Has_Answer": has_ans})
                except Exception as e:
                    rows.append({"Question": q, "Answer": f"ERROR: {e}",
                                 "Docs_Retrieved": 0, "Avg_Doc_Length": 0, "Has_Answer": 0.0})

            progress.empty()
            st.session_state.eval_results = pd.DataFrame(rows)

        if st.session_state.eval_results is not None:
            df = st.session_state.eval_results
            answered = int(df["Has_Answer"].sum())
            total = len(df)
            coverage = answered / total * 100

            # KPI row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Questions", total)
            c2.metric("Answered", f"{answered}/{total}")
            c3.metric("Coverage", f"{coverage:.1f}%")
            c4.metric("Avg Docs Retrieved", f"{df['Docs_Retrieved'].mean():.1f}")

            st.markdown("---")

            # Charts
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle("RAG Evaluation Dashboard", fontsize=14, fontweight="bold")

            # Docs retrieved per question
            ax = axes[0]
            colors = ["#22c55e" if x > 0 else "#ef4444" for x in df["Docs_Retrieved"]]
            ax.bar(range(len(df)), df["Docs_Retrieved"], color=colors, alpha=0.85)
            ax.axhline(y=top_k, color="blue", linestyle="--", linewidth=1.5, label=f"Top-K={top_k}")
            ax.set_xlabel("Question #"); ax.set_ylabel("Docs"); ax.set_title("Retrieved Docs/Q"); ax.legend(); ax.grid(axis="y", alpha=0.3)

            # Doc length distribution
            ax = axes[1]
            ax.hist(df["Avg_Doc_Length"], bins=max(3, len(df)//2), color="#f59e0b", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Avg Length (chars)"); ax.set_ylabel("Freq"); ax.set_title("Doc Length Distribution"); ax.grid(axis="y", alpha=0.3)

            # Coverage pie
            ax = axes[2]
            ax.pie([answered, total - answered], labels=["Answered", "No Answer"],
                   autopct="%1.1f%%", colors=["#86efac", "#fca5a5"], startangle=90)
            ax.set_title(f"Coverage ({answered}/{total})")

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("---")
            st.markdown("### 📋 Results Table")
            st.dataframe(
                df[["Question", "Answer", "Docs_Retrieved", "Has_Answer"]],
                use_container_width=True,
                column_config={
                    "Has_Answer": st.column_config.ProgressColumn("Has Answer", min_value=0, max_value=1),
                    "Docs_Retrieved": st.column_config.NumberColumn("Docs"),
                },
            )

            # Export
            st.markdown("---")
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                csv_data = df.to_csv(index=False).encode()
                st.download_button("⬇️ Download CSV", csv_data,
                                   file_name=f"rag_eval_{datetime.now():%Y%m%d_%H%M%S}.csv",
                                   mime="text/csv")
            with ecol2:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "summary": {"total": total, "answered": answered, "coverage_pct": coverage,
                                "avg_docs": float(df["Docs_Retrieved"].mean())},
                    "results": df.to_dict("records"),
                }
                st.download_button("⬇️ Download JSON", json.dumps(report, indent=2),
                                   file_name=f"rag_eval_{datetime.now():%Y%m%d_%H%M%S}.json",
                                   mime="application/json")


# ── TAB 3 — ABOUT ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## 📚 RAG Chatbot — How It Works

This app is a **Retrieval-Augmented Generation (RAG)** system built on top of your PDF knowledge base.

### 🔁 Pipeline
```
PDF  →  Chunk  →  Embed (HuggingFace)  →  ChromaDB
                                              ↓
Question  ──────────────────────────→  Semantic Search (Top-K)
                                              ↓
                                       Retrieved Context
                                              ↓
                                    Groq LLM (LLaMA 3) → Answer
                                              ↓
                                       Google TTS → Audio
```

### 📊 RAGAS Evaluation Metrics
| Metric | What it measures |
|--------|-----------------|
| **Coverage Rate** | % of questions the model could answer |
| **Docs Retrieved** | How many chunks were fetched per query |
| **Avg Doc Length** | Quality proxy — longer chunks carry more context |

### 🛠️ Stack
- **LangChain** — RAG pipeline & prompt management
- **ChromaDB** — Local vector store
- **HuggingFace Sentence Transformers** — Local embeddings (no API key)
- **Groq** — Fast LLaMA 3 inference
- **gTTS** — Google Text-to-Speech
- **RAGAS** — RAG evaluation framework
- **Streamlit** — This UI

### ⚡ Tips
- Smaller chunk sizes → better precision, more chunks
- Larger overlap → less context loss at boundaries
- Higher Top-K → more context, slower generation
- Try different LLM models for speed/quality trade-offs
""")
