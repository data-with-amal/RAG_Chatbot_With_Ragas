# 📚 Knowledge Base RAG Chatbot with Text-to-Speech

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from a PDF knowledge base (e.g., an HR manual), speaks its answers aloud, and is evaluated using both custom metrics and RAGAS.

## Overview

This project implements an end-to-end RAG pipeline: a PDF is loaded, chunked, and embedded into a local vector store. User questions are answered by retrieving the most relevant chunks and passing them to an LLM, which generates a grounded response. The response can optionally be converted to speech. The pipeline's retrieval and generation quality is evaluated with a custom scoring function and the RAGAS framework, with results visualized on a dashboard and exported to CSV/JSON.

## Features

- **PDF ingestion** — loads and parses a PDF knowledge base page by page
- **Chunking** — splits documents into overlapping chunks for fine-grained retrieval
- **Local embeddings** — uses a HuggingFace sentence-transformer model, no API key required
- **Vector search** — ChromaDB for fast local semantic search
- **LLM-powered answers** — Groq-hosted LLaMA 3.1 for fast, free inference
- **Grounded prompting** — answers are restricted to retrieved context to reduce hallucination
- **Text-to-speech** — answers can be read aloud using Google TTS (gTTS), no API key required
- **Evaluation suite** — custom evaluation function plus RAGAS metrics (context precision/recall, faithfulness, answer relevancy)
- **Evaluation dashboard** — matplotlib visualizations of retrieval and answer quality
- **Exportable reports** — evaluation results saved as timestamped CSV and JSON files

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Text-to-Speech | gTTS (Google Text-to-Speech) |
| Evaluation | RAGAS, pandas |
| Visualization | matplotlib |

## How It Works

1. **Load & Chunk** — `PyPDFLoader` reads the PDF; `RecursiveCharacterTextSplitter` splits it into overlapping chunks (default: 1000 chars, 100 overlap).
2. **Embed & Index** — each chunk is embedded with a local HuggingFace model and stored in a Chroma vector store.
3. **Retrieve & Generate** — for each question, the top-k most similar chunks are retrieved and inserted into a prompt template, which the Groq LLM uses to generate a grounded answer.
4. **Speak** — the answer text is optionally converted to audio via gTTS and played inline.
5. **Evaluate** — a set of test questions is run through the pipeline, scored for retrieval/answer quality, visualized on a dashboard, and exported as CSV/JSON reports.

## Project Structure

```
.
├── rag_chatbot_with_ragas.ipynb   # Main notebook: pipeline, chatbot, evaluation
└── README.md
```

## Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
pip install langchain langchain-community langchain-core langchain-chroma langchain-groq \
    pypdf sentence-transformers chromadb gtts ragas pandas matplotlib
```

### Configuration

Set your Groq API key as an environment variable (recommended) or enter it when prompted at runtime:

```bash
export GROQ_API_KEY="your-key-here"
```

Update the `PDF_PATH` variable in the notebook's configuration cell to point to your own PDF knowledge base.

Key configuration options:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 4 | Chunks retrieved per query |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq LLM model |
| `TTS_LANGUAGE` | `en` | gTTS language code |

## Usage

Run the notebook cells in order. Once the pipeline is set up, ask questions using the `ask()` function:

```python
# Ask a question with spoken response
ask("Where is this company located?")

# Show retrieved context alongside the answer
ask("What is the HR manual about?", show_context=True)

# Text-only mode
ask("What are the working hours?", speak=False)

# Batch mode
questions = [
    "Who is the CEO?",
    "What are the leave policies?",
    "What is the code of conduct?",
]
for q in questions:
    ask(q, speak=True)
```

## Evaluation

The notebook includes two layers of evaluation:

1. **Custom evaluation (`evaluate_rag_simple`)** — runs a list of test questions through the pipeline and reports the number of documents retrieved, average document length, and whether a usable answer was returned (i.e., not "I don't have that information").
2. **RAGAS metrics** — using the same Groq LLM, computes:
   - **Context Precision** — fraction of retrieved chunks that are relevant
   - **Context Recall** — fraction of relevant chunks that were retrieved
   - **Faithfulness** — whether the answer is grounded in the retrieved context
   - **Answer Relevancy** — whether the answer addresses the question

Results are visualized in a four-panel dashboard (documents retrieved, document length distribution, answer coverage, and per-question breakdown) and exported as timestamped `rag_evaluation_results_*.csv` and JSON report files.

## Notes

- Embeddings run fully locally, so no API key is required for that step.
- gTTS makes a lightweight request to a public endpoint and requires no API key.
- Replace the sample `test_questions` and `expected_answers` with ones relevant to your own PDF before running the evaluation.

