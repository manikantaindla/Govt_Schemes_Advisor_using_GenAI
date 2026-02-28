import os
import json
import pandas as pd
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

load_dotenv()

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "data" / "index"
FAISS_INDEX = INDEX_DIR / "faiss.index"
META_PARQUET = INDEX_DIR / "meta.parquet"
LINKS_JSON = BASE / "data" / "sources" / "scheme_links.json"

st.set_page_config(page_title="Govt Scheme Advisor (Simple RAG)", layout="wide")
st.title("🇮🇳 Govt Scheme Advisor (AP + Telangana) — Simple RAG")
st.caption("Uses your downloaded govt PDFs → retrieves relevant lines → explains eligibility + benefits with citations.")


# -----------------------------
# Load Models / Index
# -----------------------------

@st.cache_resource
def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_index_and_meta():
    if not FAISS_INDEX.exists() or not META_PARQUET.exists():
        raise RuntimeError("Index not found. Run: py build_data.py")
    index = faiss.read_index(str(FAISS_INDEX))
    meta = pd.read_parquet(META_PARQUET)
    return index, meta


@st.cache_data
def load_scheme_links():
    if not LINKS_JSON.exists():
        return []
    return json.loads(LINKS_JSON.read_text(encoding="utf-8"))


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(query: str, top_k: int = 6):
    index, meta = load_index_and_meta()
    q = embedder().encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, top_k)

    out = []
    for i, idx in enumerate(ids[0]):
        row = meta.iloc[int(idx)].to_dict()
        out.append(
            {
                "score": float(scores[0][i]),
                "doc_id": str(row.get("doc_id", "")),
                "file_name": str(row.get("file_name", "")),
                "page_no": int(row.get("page_no", 0)),
                "text": row.get("text", ""),
            }
        )
    return out


# -----------------------------
# Link Matching (Deterministic)
# -----------------------------

def match_links_from_evidence(evidence: list[dict]):
    links_db = load_scheme_links()
    if not links_db:
        return []

    ev_doc_ids = {e["doc_id"].lower() for e in evidence}
    ev_files = {e["file_name"].lower() for e in evidence}

    matched = []

    for scheme in links_db:
        doc_ids = [d.lower() for d in scheme.get("doc_ids", [])]
        file_names = [f.lower() for f in scheme.get("file_names", [])]

        found = False

        # Match by doc_id
        if doc_ids and any(d in ev_doc_ids for d in doc_ids):
            found = True

        # Match by file_name
        elif file_names and any(f in ev_files for f in file_names):
            found = True

        # Fallback: scheme_name in filename
        else:
            scheme_name = scheme.get("scheme_name", "").lower()
            if scheme_name and any(scheme_name in f for f in ev_files):
                found = True

        if found:
            matched.append(scheme)

    return matched


# -----------------------------
# LLM Answer
# -----------------------------

def llm_answer(profile: dict, evidence: list[dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ GEMINI_API_KEY missing. Add it to .env"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    ev_text = "\n\n".join(
        [
            f"[{e['file_name']} | page {e['page_no']}]\n{e['text']}"
            for e in evidence
        ]
    )

    prompt = f"""
You are an India government scheme advisor for Andhra Pradesh and Telangana.

CRITICAL RULES:
- Use ONLY the evidence below for factual claims.
- Do NOT invent amounts, thresholds, eligibility rules.
- If evidence does not support the query, output ONLY: NOT FOUND
- DO NOT generate any links unless explicitly present in evidence.

User profile:
{profile}

Evidence:
{ev_text}

Write in language: {profile.get("language")}.

Output format (short and clean):

1) Scheme / Topic
2) Eligibility: Eligible / Maybe / Not sure + reason
3) Benefits (if supported)
4) Citations: [file | page]

If insufficient evidence, output ONLY: NOT FOUND
"""

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


# -----------------------------
# UI
# -----------------------------

c1, c2, c3, c4 = st.columns(4)
with c1:
    state = st.selectbox("State", ["Andhra Pradesh", "Telangana"])
with c2:
    age = st.number_input("Age", 0, 120, 25)
with c3:
    income = st.number_input("Annual income (₹)", 0, value=150000, step=1000)
with c4:
    category = st.selectbox("Category", ["General", "EWS", "OBC/BC", "SC", "ST"])

language = st.radio("Language", ["en", "te"], horizontal=True)

query_hint = st.text_input(
    "Optional keywords (pension, scholarship, housing, marriage, farmer)",
    value="pension eligibility benefits",
)

top_k = st.slider("Evidence chunks", 4, 12, 6)

MIN_SCORE = 0.22  # retrieval threshold

if st.button("Find schemes"):

    profile = {
        "state": state,
        "age": int(age),
        "annual_income": int(income),
        "category": category,
        "language": language,
    }

    query = f"{state} age {age} income {income} category {category} {query_hint}"
    evidence = retrieve(query=query, top_k=top_k)

    best_score = evidence[0]["score"] if evidence else 0.0

    if (not evidence) or (best_score < MIN_SCORE):
        st.error("NOT FOUND")
        st.caption(f"(No strong match in local PDFs. Best similarity score: {best_score:.3f})")
        st.stop()

    # LLM Answer
    st.subheader("✅ Answer")
    answer = llm_answer(profile, evidence)
    st.write(answer)

    # Deterministic Official Links
    matched_links = match_links_from_evidence(evidence)
    if matched_links:
        st.subheader("🔗 Official Links")
        for scheme in matched_links:
            st.markdown(f"**{scheme.get('scheme_name')}**")
            st.markdown(f"- Apply: {scheme.get('apply_link', 'Not found')}")
            for src in scheme.get("source_links", []):
                st.markdown(f"- Source: {src}")
    else:
        st.info("No official links mapped for this scheme.")

    # Evidence Viewer
    with st.expander("🔎 Evidence used"):
        for e in evidence:
            st.markdown(f"**[{e['file_name']} | page {e['page_no']}]** _(score {e['score']:.3f})_")
            st.write((e["text"][:900] + "…") if len(e["text"]) > 900 else e["text"])