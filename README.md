🇮🇳 Govt Scheme Advisor (AP + Telangana) – RAG Based AI System
📌 Overview

Govt Scheme Advisor is a Retrieval-Augmented Generation (RAG) based AI application that helps users understand government welfare schemes in Andhra Pradesh and Telangana.

The system retrieves relevant information from official government PDF documents and generates clear, structured explanations about:

Eligibility criteria

Benefits

Required documents

Application process

It reduces the difficulty of navigating complex government scheme documents.



🚀 Problem Statement

Government scheme PDFs are:

Long and difficult to understand

Written in formal/legal language

Hard to search manually

Users often struggle to:

Find the correct scheme

Understand eligibility

Identify benefits clearly

This project solves that using AI-powered semantic search + explanation generation.



🧠 Architecture (RAG Pipeline)
Government PDFs
      ↓
Text Extraction (PyPDF)
      ↓
Chunking
      ↓
Embeddings (Sentence Transformers)
      ↓
FAISS Vector Index
      ↓
User Query
      ↓
Semantic Retrieval
      ↓
Google Gemini (LLM)
      ↓
Structured Answer with Context




🛠 Tech Stack

Python

Streamlit (Frontend UI)

FAISS (Vector Similarity Search)

Sentence-Transformers (all-MiniLM-L6-v2) (Embeddings)

Google Generative AI (Gemini) (Answer Generation)

Pandas / NumPy

PyPDF



🔍 Key Features

✅ Semantic search over government PDFs
✅ Accurate context retrieval using FAISS
✅ LLM-generated explanations with citations
✅ Simple and clean Streamlit interface
✅ Supports Andhra Pradesh & Telangana schemes

<img width="2437" height="1209" alt="Screenshot 2026-02-28 095156" src="https://github.com/user-attachments/assets/1ccbac66-0286-4962-a492-541861c0ee47" />

🏠 Home Page

<img width="2381" height="962" alt="Screenshot 2026-02-28 094529" src="https://github.com/user-attachments/assets/aeb95f59-c3fc-4b82-8d9e-67994faa3777" />


🔎 Retrieval Results

(Add screenshot here)

🤖 AI Generated Explanation

(Add screenshot here)

📂 Project Structure
govt-scheme-advisor/
│
├── app.py
├── ingestion/
│   ├── build_index.py
│
├── data/
│   ├── raw_pdfs/
│   ├── index/
│       ├── faiss.index
│       ├── meta.parquet
│
├── requirements.txt
└── README.md
⚙️ How It Works

Government PDFs are collected and stored locally.

Text is extracted and split into chunks.

Each chunk is converted into vector embeddings.

FAISS stores vectors for fast similarity search.

User enters a query.

Relevant chunks are retrieved.

Gemini generates a structured explanation using retrieved context.

🎯 Example Use Case

User Query:

Who is eligible for Post-Matric Scholarship in Telangana?

System Output:

Eligibility criteria

Income limits

Eligible categories

Benefits provided

Official references

📊 Learning Outcomes

Through this project, I learned:

Practical implementation of RAG systems

Vector databases and similarity search

Prompt engineering for structured outputs

Handling real-world government documents

End-to-end AI application development

🔮 Future Improvements

Add multilingual support (Telugu + English)

Deploy publicly

Add chatbot-style UI

Improve citation formatting

Add scheme filtering by category

👤 Author

Manikanta
AI & Data Enthusiast.
