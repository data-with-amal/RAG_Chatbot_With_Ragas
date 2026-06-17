<<<<<<< HEAD
# 📚 RAG Chatbot — Streamlit App

A Retrieval-Augmented Generation chatbot with Text-to-Speech and RAGAS evaluation,
converted from the original Jupyter notebook.

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

## 🔑 API Key
Get a free **Groq API key** at https://console.groq.com  
Enter it in the sidebar when the app starts.

## 🛠️ Usage
1. Paste your Groq API key in the sidebar
2. Upload a PDF (e.g. an HR manual)
3. Adjust RAG settings if needed (chunk size, Top-K, model)
4. Click **Build Index** → wait ~30–60 s for embeddings
5. Go to the **Chat** tab and ask questions
6. Go to the **Evaluation** tab to run RAGAS-style metrics

## 📁 Project Structure
```
rag_streamlit/
├── app.py           # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## ⚡ Deployment Options

### Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io → New app
3. Point to `app.py` and deploy

### Local Docker
```bash
docker run -p 8501:8501 \
  -v $(pwd):/app \
  python:3.11-slim \
  bash -c "pip install -r /app/requirements.txt && streamlit run /app/app.py"
```
=======
# RAG_Chatbot
RAG chatbot that answers questions from a PDF knowledge base using LangChain, ChromaDB, HuggingFace embeddings, and LLaMA 3.1 via Groq. Features Google Text-to-Speech for voice responses and a RAGAS-style evaluation pipeline to measure retrieval and answer quality.
>>>>>>> 62dbe430f64fd7f2319e9248a89e4e5cb067776c
