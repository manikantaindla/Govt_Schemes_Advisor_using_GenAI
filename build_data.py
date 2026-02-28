import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
PDF_DIR = BASE / "data" / "sources" / "pdfs_raw"
EXTRACTED_DIR = BASE / "data" / "extracted"
INDEX_DIR = BASE / "data" / "index"

CHUNKS_JSONL = EXTRACTED_DIR / "chunks.jsonl"
META_PARQUET = INDEX_DIR / "meta.parquet"
FAISS_INDEX = INDEX_DIR / "faiss.index"


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200):
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        out.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return out


def parse_pdf(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        t = clean_text(t)
        if t:
            pages.append((i, t))
    return pages


def main():
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {PDF_DIR}")

    rows = []
    for pdf in pdfs:
        doc_id = pdf.stem
        pages = parse_pdf(pdf)
        for page_no, page_text in pages:
            for k, ck in enumerate(chunk_text(page_text), start=1):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "file_name": pdf.name,
                        "page_no": int(page_no),
                        "chunk_no": int(k),
                        "text": ck,
                    }
                )

    # Write chunks.jsonl
    with CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Build embeddings + FAISS
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [r["text"] for r in rows]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(FAISS_INDEX))
    pd.DataFrame(rows).to_parquet(META_PARQUET, index=False)

    print(f"✅ PDFs processed: {len(pdfs)}")
    print(f"✅ Chunks created: {len(rows)}")
    print(f"✅ Saved: {CHUNKS_JSONL}")
    print(f"✅ Saved: {FAISS_INDEX}")
    print(f"✅ Saved: {META_PARQUET}")


if __name__ == "__main__":
    main()