<p align="center">
  <img src="https://img.shields.io/badge/RAG-Roadmap-blue?style=for-the-badge&logo=github&logoColor=white" alt="RAG Roadmap">
  <img src="https://img.shields.io/badge/From%20Fundamentals%20to%20Production-Ready%20Agentic%20RAG-green?style=for-the-badge" alt="Full Stack RAG">
  <img src="https://img.shields.io/badge/10%20Modules-50+%20Topics-orange?style=for-the-badge" alt="Topics">
  <img src="https://img.shields.io/badge/Hands--On-Labs-red?style=for-the-badge" alt="Hands-On">
</p>

<h1 align="center">🔍 Retrieval-Augmented Generation (RAG)<br>Comprehensive Course Curriculum</h1>

<h3 align="center">From Fundamentals to Production-Ready Agentic RAG Systems</h3>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" height="40" alt="Python"/>
  &nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/langchain/langchain-original.svg" width="40" height="40" alt="LangChain"/>
  &nbsp;&nbsp;
  <img src="https://img.icons8.com/fluency/48/database.png" width="40" height="40" alt="Vector DB"/>
  &nbsp;&nbsp;
  <img src="https://img.icons8.com/fluency/48/artificial-intelligence.png" width="40" height="40" alt="AI"/>
  &nbsp;&nbsp;
  <img src="https://img.icons8.com/fluency/48/flow-chart.png" width="40" height="40" alt="LangGraph"/>
</p>

---

## 📋 Table of Contents

- [🎯 Course Overview](#-course-overview)
- [🗺️ Module Map](#-module-map)
- [✅ Prerequisites](#-prerequisites)
- [📘 Module 1 — RAG Fundamentals and Architecture](#-module-1--rag-fundamentals-and-architecture)
- [📗 Module 2 — Document Processing and Chunking](#-module-2--document-processing-and-chunking)
- [📙 Module 3 — Embeddings and Vector Representations](#-module-3--embeddings-and-vector-representations)
- [📕 Module 4 — Vector Stores](#-module-4--vector-stores)
- [🔍 Module 5 — Basic Retrieval Techniques](#-module-5--basic-retrieval-techniques)
- [🚀 Module 6 — Advanced Retrieval Techniques](#-module-6--advanced-retrieval-techniques)
- [🧠 Module 7 — Advanced RAG Patterns](#-module-7--advanced-rag-patterns)
- [🤖 Module 8 — Agentic RAG with LangGraph](#-module-8--agentic-rag-with-langgraph)
- [📊 Module 9 — Evaluating RAG using RAGAS](#-module-9--evaluating-rag-using-ragas)
- [🏆 Module 10 — Capstone Project with Deployment](#-module-10--capstone-project-with-deployment)
- [🧰 Bonus — Tools & Frameworks Quick Reference](#-bonus--tools--frameworks-quick-reference)
- [🤝 Contributing](#-contributing)
- [⭐ Star History](#-star-history)

---

## 🎯 Course Overview

This curriculum provides a **structured, end-to-end path** through Retrieval-Augmented Generation — beginning with core concepts and document processing, advancing through embeddings, vector stores, and retrieval techniques, and culminating in **agentic RAG systems** built with LangGraph and a **deployable capstone project**.

> 💡 **Philosophy**: Every module includes **hands-on labs**, **code examples**, and **comparison exercises**. Don't just read — build.

### What You'll Achieve

| Milestone | Outcome |
|-----------|---------|
| 🏗️ Architecture | Understand RAG data flow, when to use RAG vs fine-tuning |
| 📄 Document Processing | Ingest, chunk, and metadata-tag documents from any source |
| 🧮 Embeddings | Select, compare, and implement the right embedding model |
| 🗄️ Vector Stores | Build CRUD applications with Chroma, FAISS, Qdrant, Pinecone, Milvus |
| 🔎 Retrieval | Master similarity search, MMR, hybrid search, and re-ranking |
| 🧠 Advanced Patterns | Implement RAG Fusion, HyDE, CRAG, Self-RAG, and Graph RAG |
| 🤖 Agentic RAG | Build autonomous RAG agents with LangGraph |
| 📏 Evaluation | Measure faithfulness, relevance, precision, recall with RAGAS |
| 🚀 Production | Deploy a monitored, optimized, secure RAG pipeline |

---

## 🗺️ Module Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG LEARNING PATH                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: FOUNDATIONS                                          │
│  ┌──────────────┐  ┌───────────────────┐  ┌────────────────┐  │
│  │  Module 1    │──▶  │    Module 2      │──▶│   Module 3     │  │
│  │  RAG         │     │  Document Proc.  │   │  Embeddings    │  │
│  │  Fundamentals│     │  & Chunking      │   │  & Vectors     │  │
│  └──────────────┘  └───────────────────┘  └────────────────┘  │
│          │                                        │              │
│          ▼                                        ▼              │
│  PHASE 2: STORAGE & RETRIEVAL                                  │
│  ┌──────────────┐  ┌───────────────────┐  ┌────────────────┐  │
│  │  Module 4    │──▶  │    Module 5      │──▶│   Module 6     │  │
│  │  Vector      │     │  Basic           │   │  Advanced      │  │
│  │  Stores      │     │  Retrieval       │   │  Retrieval     │  │
│  └──────────────┘  └───────────────────┘  └────────────────┘  │
│                                                   │              │
│                                                   ▼              │
│  PHASE 3: ADVANCED PATTERNS & AGENTS                           │
│  ┌──────────────┐  ┌───────────────────┐                       │
│  │  Module 7    │──▶  │    Module 8      │                      │
│  │  Advanced    │     │  Agentic RAG     │                      │
│  │  RAG Patterns│     │  with LangGraph  │                      │
│  └──────────────┘  └───────────────────┘                       │
│                            │                                    │
│                            ▼                                    │
│  PHASE 4: EVALUATION & PRODUCTION                              │
│  ┌──────────────┐  ┌───────────────────┐                       │
│  │  Module 9    │──▶  │    Module 10     │                      │
│  │  RAG         │     │  Capstone        │                      │
│  │  Evaluation  │     │  & Deployment    │                      │
│  └──────────────┘  └───────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Prerequisites

Before starting this roadmap, ensure you have a solid grasp of:

| Prerequisite | Why It's Needed | Recommended Resource |
|-------------|-----------------|---------------------|
| 🐍 **Python Fundamentals** | All implementations are in Python | [Python Official Docs](https://docs.python.org/3/tutorial/) |
| 🧠 **GenAI Fundamentals** | Understanding LLMs, tokens, prompts | [GenAI Roadmap](https://github.com/AdilShamim8/GenAI-Roadmap-with-Notes-and-Projects) |
| 🔗 **LangChain** | Core framework for RAG pipelines | [LangChain Docs](https://python.langchain.com/) |
| 🕸️ **LangGraph** | Building agentic RAG workflows | [LangGraph Docs](https://langchain-ai.github.io/langgraph/) |

> ⚠️ **Note**: If you're new to LangChain or LangGraph, complete their quickstart tutorials before Module 8.

---

## 📘 Module 1 — RAG Fundamentals and Architecture

> 🎯 **Goal**: Understand what RAG is, why it exists, and how it compares to alternative approaches.

### 1.1 Introduction to RAG

#### 1.1.1 What is RAG and the Problems It Solves

Large Language Models are powerful but inherently limited. RAG addresses four critical shortcomings:

| Problem | Description | How RAG Helps |
|---------|-------------|---------------|
| 🎭 **Hallucination** | LLMs generate plausible but fabricated answers | Grounds generation in retrieved facts |
| 📅 **Knowledge Cutoff** | Training data has a fixed cutoff date | Provides access to up-to-date information |
| 🏥 **Domain Gaps** | General training lacks specialized knowledge | Injects domain-specific documents at query time |
| 🔍 **No Source Attribution** | Impossible to verify where an answer came from | Every answer traces back to retrieved sources |

> 💡 **Key Insight**: RAG doesn't change the model's weights — it changes the **input context** at inference time, making it cost-effective and immediately updateable.

```
Without RAG:   User Query ──────────────────▶ LLM ──▶ Response (may hallucinate)
                                              ↑
                                         Stale weights only

With RAG:      User Query ──▶ Retriever ──▶ LLM ──▶ Response (grounded)
                                ↑
                          Knowledge Base
                       (up-to-date docs)
```

#### 1.1.2 Core Components

```
┌──────────────────────────────────────────────────────┐
│                  RAG ARCHITECTURE                     │
│                                                      │
│   ┌─────────────┐    ┌────────────┐   ┌───────────┐ │
│   │  Knowledge  │───▶│ Retriever  │──▶│ Generator │ │
│   │    Base     │    │            │   │   (LLM)   │ │
│   └─────────────┘    └────────────┘   └───────────┘ │
│        ▲                   ▲               │         │
│        │                   │               ▼         │
│   Documents &         Query +          Grounded      │
│   Embeddings         Context          Response       │
└──────────────────────────────────────────────────────┘
```

- **Knowledge Base**: Your document corpus, chunked and embedded into a vector store
- **Retriever**: Finds the most relevant chunks for a given query
- **Generator**: An LLM that synthesizes the retrieved context into a coherent answer

#### 1.1.3 RAG Data Flow

```
Step-by-step end-to-end flow:

 1. User submits query
        │
        ▼
 2. Query is embedded using the same embedding model
        │
        ▼
 3. Vector store performs similarity search
        │
        ▼
 4. Top-k relevant document chunks are retrieved
        │
        ▼
 5. Chunks are injected into the LLM prompt as context
        │
        ▼
 6. LLM generates a response grounded in the retrieved context
        │
        ▼
 7. Response is returned to the user (with source citations)
```

### 1.2 RAG vs Fine-Tuning vs Prompt Engineering

| Dimension | 🟢 Prompt Engineering | 🔵 RAG | 🟣 Fine-Tuning |
|-----------|----------------------|--------|----------------|
| **Cost** | 💰 Lowest | 💰💰 Medium | 💰💰💰 Highest |
| **Latency** | ⚡ Fastest | ⚡⚡ Moderate | ⚡⚡⚡ Inference same, but training hours |
| **Accuracy (domain)** | ❌ Low for deep domain | ✅ High (if docs exist) | ✅✅ Very High |
| **Updatability** | ✅ Instant | ✅ Instant (update docs) | ❌ Requires retraining |
| **Maintenance** | ✅ Easy | 🟡 Moderate | ❌ Complex |
| **Source Attribution** | ❌ No | ✅ Yes | ❌ No |
| **Hallucination Control** | 🟡 Limited | ✅ Good | 🟡 Better than prompting |
| **Data Requirements** | None | Document corpus | Hundreds-Thousands of examples |

**Decision Framework:**

```
                    ┌─────────────────────┐
                    │  What's your goal?   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     Need real-time      Need domain         Need to change
     data / sources?     knowledge?          model behavior
              │               │               │
              ▼               ▼               ▼
          ┌───────┐     ┌───────────┐    ┌──────────┐
          │  RAG  │     │Have docs? │    │Fine-Tune │
          └───────┘     └─────┬─────┘    └──────────┘
                             │
                    ┌────────┼────────┐
                    ▼                 ▼
                  Yes                No
                    │                 │
                    ▼                 ▼
                 ┌───────┐     ┌───────────┐
                 │  RAG  │     │ Fine-Tune │
                 └───────┘     └───────────┘

    Quick task framing / formatting? → Prompt Engineering
```

> 📝 **Hands-On**: For each approach, write a 3-sentence summary of a use case where it would be the optimal choice.

---

## 📗 Module 2 — Document Processing and Chunking

> 🎯 **Goal**: Master document ingestion, splitting strategies, and metadata management — the foundation of every RAG pipeline.

### 2.1 Document Loaders in LangChain

LangChain provides 100+ document loaders. Here are the most essential:

| Loader | Source | Key Features |
|--------|--------|-------------|
| `PyPDFLoader` | PDF | Fast, basic text extraction |
| `PDFMiner` | PDF | Better layout handling |
| `PDFPlumber` | PDF | Table extraction support |
| `Unstructured` | PDF | Best for complex layouts, images + text |
| `WebBaseLoader` | Web | BeautifulSoup-based HTML parsing |
| `RecursiveUrlLoader` | Web | Crawls linked pages recursively |
| `CSVLoader` | CSV | Row-by-row document creation |
| `JSONLoader` | JSON | jq-schema based extraction |
| `TextLoader` | .txt | Simple text file loading |

```python
# ─── PDF Loading ───────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("technical_report.pdf")
pages = loader.load()  # Each page becomes a Document
print(f"Loaded {len(pages)} pages")
print(pages[0].metadata)  # {'source': '...', 'page': 0}

# ─── Web Loading ───────────────────────────────────────────
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.example.com/guide")
docs = loader.load()

# ─── Structured Data ───────────────────────────────────────
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("products.csv", metadata_columns=["category", "price"])
rows = loader.load()  # Each row = 1 Document

# ─── JSON Loading ──────────────────────────────────────────
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader("data.json", jq_schema=".messages[].content")
json_docs = loader.load()
```

> 🔬 **Hands-On**: Load documents from **5 different sources** (PDF, web, CSV, JSON, .txt) and inspect the `page_content` and `metadata` of each.

### 2.2 Text Splitting Strategies

Choosing the right chunking strategy is one of the **highest-leverage decisions** in RAG.

#### 2.2.1 RecursiveCharacterTextSplitter ⭐ (Recommended Default)

```
How it works:
┌─────────────────────────────────────────────────────┐
│  1. Try splitting on "\n\n" (paragraphs)           │
│  2. If chunks too large → split on "\n" (lines)    │
│  3. If still too large → split on ". " (sentences)  │
│  4. If still too large → split on " " (words)      │
│  5. If still too large → split on "" (characters)   │
│                                                     │
│  This hierarchical approach preserves structure!    │
└─────────────────────────────────────────────────────┘
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Max characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchical
    length_function=len,
    is_separator_regex=False,
)

chunks = splitter.split_documents(docs)
```

#### 2.2.2 CharacterTextSplitter & TokenTextSplitter

```python
# Character-based (simpler, less structure-aware)
from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=100)

# Token-based (precise token counting)
from langchain_text_splitters import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)  # In tokens, not chars
```

#### 2.2.3 Semantic Chunking with SemanticChunker 🧠

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # Options below
    breakpoint_threshold_amount=75,          # Threshold value
)

# Breakpoint threshold types:
# ┌────────────────────┬──────────────────────────────────────┐
# │ Type               │ How it decides where to split        │
# ├────────────────────┼──────────────────────────────────────┤
# │ "percentile"       │ Top Nth percentile of distance drops │
# │ "standard_deviation"│ Mean + N*std of distances           │
# │ "gradient"         │ Largest gradient changes in distance │
# │ "interquartile"    │ Beyond IQR * multiplier              │
# └────────────────────┴──────────────────────────────────────┘

semantic_chunks = splitter.split_documents(docs)
```

#### 2.2.4 MarkdownTextSplitter

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = splitter.split_text(markdown_text)
# Each chunk inherits header metadata: {"Header 1": "...", "Header 2": "..."}
```

#### 2.2.5 Code Splitters

```python
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# Python
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)

# JavaScript
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=2000, chunk_overlap=200
)
```

#### 2.2.6 LLM-Based Text Splitting

```python
# Uses an LLM to determine semantically meaningful split points
# More expensive but produces the highest quality chunks
from langchain.chains import LLMChain
# Strategy: Ask LLM "Where should this text be split for best retrieval?"
```

> 🔬 **Hands-On**: Apply **all chunking strategies** to the same document. Compare chunk count, average chunk size, and semantic coherence.

### 2.3 Chunking Best Practices

#### Optimal Chunk Sizes by Use Case

| Use Case | Recommended Size | Rationale |
|----------|-----------------|-----------|
| 🔢 **Factoid queries** (who, what, when) | 128–256 tokens | Short, precise answers need tight chunks |
| 📖 **General RAG** | 256–512 tokens | Balance of context and precision |
| 📊 **Complex analysis** | 512–1024 tokens | Longer reasoning needs more context |

#### Overlap Strategies

```
┌─────────────────────────────────────────────────────┐
│  Overlap: 10–20% of chunk_size (recommended)        │
│                                                     │
│  Chunk 1: [████████████████░░░░]                    │
│  Chunk 2:          [░░░░████████████████░░░░]       │
│  Chunk 3:                   [░░░░████████████████]  │
│                           overlap     overlap       │
│                                                     │
│  Why? Prevents losing facts that fall on boundaries │
└─────────────────────────────────────────────────────┘
```

#### How Chunk Size Affects Retrieval Quality

```
Too Small (<100 tokens):
  ✅ Precise matching
  ❌ Lost context, fragmented answers
  ❌ More chunks = slower search

Too Large (>1500 tokens):
  ✅ Rich context
  ❌ Diluted relevance, noise in context window
  ❌ Wastes LLM token budget

Just Right (256–512 tokens for general RAG):
  ✅ Balanced precision and context
  ✅ Efficient retrieval and generation
```

#### Handling Tables and Structured Data

- Use `Unstructured` loader for complex table extraction
- Consider markdown representation of tables
- Preserve header context in chunk metadata
- For very large tables: chunk by rows with headers repeated

> 🔬 **Lab**: Optimize chunking for a **PDF technical document** — experiment with 3 different `chunk_size`/`chunk_overlap` combinations and measure retrieval quality manually.

### 2.4 Metadata Management

```python
from langchain_core.documents import Document

# ─── Adding Custom Metadata ────────────────────────────
doc = Document(
    page_content="Revenue grew 23% YoY...",
    metadata={
        "source": "Q3_2024_Earnings.pdf",
        "page": 12,
        "department": "Finance",
        "date": "2024-10-15",
        "doc_version": "2.1",
        "confidence": 0.95,
    }
)

# ─── Metadata Filtering in Retrieval ───────────────────
# Filter by any metadata field during search
results = vectorstore.similarity_search(
    "revenue growth",
    k=5,
    filter={"department": "Finance"}  # Only search Finance docs
)

# ─── Source Tracking & Document Lineage ────────────────
# Every chunk should trace back to: source file → page → section → paragraph

# ─── Document Versioning Strategies ────────────────────
# Option 1: Timestamp-based metadata
metadata["ingested_at"] = "2024-10-15T10:30:00Z"
metadata["doc_version"] = "2.1"

# Option 2: Hash-based deduplication
import hashlib
content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
metadata["content_hash"] = content_hash
```

> 🔬 **Hands-On**: Build a **metadata-enriched document pipeline** that automatically adds source, page number, section headers, ingestion timestamp, and content hash to every chunk.

---

## 📙 Module 3 — Embeddings and Vector Representations

> 🎯 **Goal**: Understand how text becomes vectors, choose the right embedding model, and implement embeddings in LangChain.

### 3.1 How Embeddings Work

```
┌──────────────────────────────────────────────────────────────────┐
│                    EMBEDDING SPACE                               │
│                                                                  │
│        🐕 Dog                                                    │
│       ╱                                                          │
│     🐕 Puppy ──── 🐕 Canine                                      │
│       ╲                                                          │
│        🐱 Cat ──── 🐱 Kitten                                    │
│                                                                  │
│   🚗 Car ────── 🚙 SUV ────── 🏎️ Sports Car                     │
│                                                                  │
│   Distance = Semantic Similarity                                 │
│   d("dog", "puppy") ≈ 0.12   (very close)                      │
│   d("dog", "car")  ≈ 0.89   (very far)                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Distance Metrics

| Metric | Formula Concept | When to Use | Range |
|--------|----------------|-------------|-------|
| **Cosine Similarity** | Angle between vectors | Default choice, scale-invariant | [-1, 1] → often [0, 1] |
| **Euclidean Distance (L2)** | Straight-line distance | When magnitude matters | [0, ∞) |
| **Dot Product** | Projection of one onto other | Pre-normalized vectors | (-∞, ∞) |

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def dot_product(v1, v2):
    return np.dot(v1, v2)
```

#### Embedding Dimensions Trade-offs

```
Dimensions:  384 ──────▶ 768 ──────▶ 1536 ──────▶ 3072

Quality:      🟡          🟢          🟢🟢          🟢🟢🟢
Speed:        🟢🟢🟢      🟢🟢         🟢             🟡
Storage:      🟢🟢🟢      🟢🟢         🟢             🟡
Cost:         🟢🟢🟢      🟢🟢         🟢             🟡

Rule of thumb: Start with 1536 (OpenAI small), upgrade if needed.
```

### 3.2 Embedding Models Overview

#### OpenAI Embeddings

| Model | Dimensions | Cost | Best For |
|-------|-----------|------|----------|
| `text-embedding-3-small` | 1536 (default) | $0.02 / 1M tokens | **Best value** — most RAG use cases |
| `text-embedding-3-large` | 3072 (default) | $0.13 / 1M tokens | **Highest quality** — precision-critical apps |

```python
from langchain_openai import OpenAIEmbeddings

# Small (recommended starting point)
embeddings_small = OpenAIEmbeddings(model="text-embedding-3-small")

# Large (when quality justifies 6.5x cost)
embeddings_large = OpenAIEmbeddings(model="text-embedding-3-large")
```

#### Open-Source Alternatives

```python
# ─── Embedding Gemma through Ollama Python Lib ─────────
import ollama

response = ollama.embeddings(
    model="gemma:2b",
    prompt="What is retrieval-augmented generation?"
)
embedding_vector = response["embedding"]

# ─── Embedding Gemma through LangChain-Ollama ──────────
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="gemma:2b")
vector = embeddings.embed_query("What is RAG?")
```

### 3.3 Choosing the Right Embedding Model

#### MTEB Leaderboard

> 🏆 [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — the definitive benchmark for embedding models.

**What to look for:**

| Criterion | Why It Matters |
|-----------|---------------|
| **Retrieval score** | Directly measures search quality |
| **Model size** | Affects inference speed and memory |
| **Max sequence length** | Must accommodate your chunk sizes |
| **Language support** | Multilingual vs English-only |
| **License** | Commercial use restrictions |

#### Cost vs Quality Decision Framework

```
┌─────────────────────────────────────────────────┐
│         EMBEDDING MODEL DECISION TREE           │
│                                                 │
│   Budget constrained?                           │
│      ├── Yes → Open-source (Gemma, BGE, E5)    │
│      └── No → OpenAI text-embedding-3-small    │
│                                                 │
│   Need absolute best quality?                   │
│      ├── Yes → OpenAI text-embedding-3-large   │
│      └── No → small is sufficient              │
│                                                 │
│   Data privacy / air-gapped?                    │
│      ├── Yes → Ollama + local model            │
│      └── No → Cloud APIs are fine              │
│                                                 │
│   Multilingual?                                 │
│      ├── Yes → multilingual-e5-large           │
│      └── No → English-specific models           │
└─────────────────────────────────────────────────┘
```

> 🔬 **Hands-On**: Embed the same 10 queries with **3 different models**. Compare cosine similarity rankings.

### 3.4 LangChain Embeddings Implementation

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─── embed_documents: For your corpus ──────────────────
doc_vectors = embeddings.embed_documents([
    "RAG stands for Retrieval-Augmented Generation",
    "Vector stores enable similarity search",
    "Chunking affects retrieval quality",
])
print(f"3 documents embedded, each {len(doc_vectors[0])}-dimensional")

# ─── embed_query: For user queries ─────────────────────
query_vector = embeddings.embed_query("What is RAG?")
print(f"Query embedded into {len(query_vector)}-dimensional vector")

# ⚠️ Always use embed_query for queries and embed_documents for docs
# Some models handle them differently internally
```

---

## 📕 Module 4 — Vector Stores

> 🎯 **Goal**: Understand the vector store landscape, select the right store, and perform full CRUD operations.

### 4.1 Vector Store Comparison and Selection

#### Free Vector Store Landscape

| Store | Type | Best For | Free Tier | Scalability |
|-------|------|----------|-----------|-------------|
| **Chroma** 🟢 | Embedded | Learning, prototyping | Unlimited (self-hosted) | Single-node |
| **FAISS** 🔵 | Library | High-performance search | Unlimited | Single-node, GPU |
| **Qdrant** 🟣 | Server | Production | 1GB cloud forever | Distributed cluster |
| **Pinecone** 🟠 | Managed | Zero-ops deployment | 100K vectors | Fully managed |
| **Milvus** 🔴 | Server | Billion-scale | Unlimited (self-hosted) | Distributed cluster |

#### Selection Decision Tree

```
What's your use case?
│
├── Learning / Prototyping
│   └── Chroma or FAISS (zero setup)
│
├── Production, small-medium scale
│   ├── Want managed? → Pinecone free tier
│   └── Self-host OK? → Qdrant
│
└── Production, massive scale (billions)
    └── Milvus

Need GPU acceleration? → FAISS (GPU-enabled)
```

### 4.2 Vector Store Operations (CRUD)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ═══════════════════════════════════════════════════════
# CREATE — Adding Documents
# ═══════════════════════════════════════════════════════
from langchain_core.documents import Document

docs = [
    Document(page_content="RAG reduces hallucination", metadata={"topic": "basics"}),
    Document(page_content="Chunking affects retrieval", metadata={"topic": "chunking"}),
    Document(page_content="Vector stores enable search", metadata={"topic": "storage"}),
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_collection",
    persist_directory="./chroma_db",
)

# ─── Or add texts directly ────────────────────────────
ids = vectorstore.add_texts(
    texts=["New document about embeddings", "Another about retrieval"],
    metadatas=[{"topic": "embeddings"}, {"topic": "retrieval"}],
)

# ═══════════════════════════════════════════════════════
# READ — Similarity Search
# ═══════════════════════════════════════════════════════

# Basic similarity search
results = vectorstore.similarity_search("What reduces hallucination?", k=2)

# Search with relevance scores
results_with_scores = vectorstore.similarity_search_with_score(
    "What reduces hallucination?", k=2
)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content}")

# ═══════════════════════════════════════════════════════
# UPDATE — Modifying Documents
# ═══════════════════════════════════════════════════════

# Update: delete old + add new (most vector stores)
vectorstore.delete(ids=[old_id])
vectorstore.add_texts(
    texts=["Updated content about RAG"],
    metadatas=[{"topic": "basics", "updated": True}],
    ids=[old_id],  # Reuse the same ID
)

# ═══════════════════════════════════════════════════════
# DELETE — Removing Documents
# ═══════════════════════════════════════════════════════

# Delete by ID
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])

# Delete by filter (where supported)
vectorstore.delete(filter={"topic": "outdated"})
```

> 🔬 **Lab**: Build a **full CRUD vector store application** — create a collection, add documents, search, update a document, delete by filter, and verify operations.

---

## 🔍 Module 5 — Basic Retrieval Techniques

> 🎯 **Goal**: Master the three fundamental retrieval strategies — similarity search, threshold filtering, and MMR.

### 5.1 Similarity Search Fundamentals

```python
# ─── Basic Similarity Search ────────────────────────────
results = vectorstore.similarity_search(
    query="How does RAG reduce hallucination?",
    k=4,  # Number of results
)

# ─── Search with Relevance Scores ───────────────────────
results_with_scores = vectorstore.similarity_search_with_score(
    query="How does RAG reduce hallucination?",
    k=4,
)

# Lower score = more similar (for L2 distance)
# Higher score = more similar (for cosine similarity)
# Check your vector store's distance metric!
```

> 🔬 **Hands-On**: Implement basic similarity search and analyze how `k` affects result diversity.

### 5.2 Similarity Score Threshold

```python
# ─── Why thresholds matter ──────────────────────────────
# Without threshold: you ALWAYS get k results, even if none are relevant
# With threshold: you only get results above a quality bar

from langchain_community.vectorstores import Chroma

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Only return results above this score
        "k": 10,                 # Max results to consider
    }
)

# ⚠️ Not all vector stores support score_threshold natively
# Verify compatibility with your chosen store
```

```
Without Threshold:                    With Threshold (0.7):
┌──────────────────────────┐         ┌──────────────────────────┐
│ Result 1: score=0.92 ✅  │         │ Result 1: score=0.92 ✅  │
│ Result 2: score=0.85 ✅  │         │ Result 2: score=0.85 ✅  │
│ Result 3: score=0.61 ❌  │         │  ── (filtered out) ──    │
│ Result 4: score=0.45 ❌  │         │  ── (filtered out) ──    │
└──────────────────────────┘         └──────────────────────────┘
  All k results returned               Only relevant results
```

> 🔬 **Hands-On**: Implement threshold-based retrieval and experiment with different threshold values.

### 5.3 MMR (Maximal Marginal Relevance)

```
Standard Similarity:              MMR (Maximal Marginal Relevance):
┌───────────────────────┐        ┌───────────────────────┐
│  🟢🟢🟢               │        │  🟢    🔵              │
│  🟢🟢🟢  ← cluster   │        │     🟡       🟣       │
│     of similar docs   │        │  diverse results      │
└───────────────────────┘        └───────────────────────┘
  High relevance                     Balanced relevance
  Low diversity                      + diversity
```

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,              # Final results to return
        "fetch_k": 20,       # Candidates to fetch first (must be > k)
        "lambda_mult": 0.5,  # 0.0 = max diversity, 1.0 = max relevance
    }
)

results = retriever.invoke("Explain RAG architecture")
```

| `lambda_mult` | Behavior | Use Case |
|---------------|----------|----------|
| 0.0 | Maximum diversity | Exploratory research, broad coverage |
| 0.5 | Balanced | General RAG queries |
| 1.0 | Maximum relevance | Precise factoid answers (same as similarity) |

> 🔬 **Hands-On**: Compare MMR vs standard similarity search on a multi-topic document collection.

### 5.4 Hybrid Search

```
┌─────────────────────────────────────────────────────────┐
│                  HYBRID SEARCH                          │
│                                                         │
│  Dense Vectors (Semantic)  +  Sparse Vectors (Keyword)  │
│         │                          │                    │
│         ▼                          ▼                    │
│  "How does RAG      ←→     "RAG" "reduce"              │
│   reduce errors?"           "errors"                    │
│                                                         │
│  Captures meaning        Captures exact terms           │
│  ────────────────        ────────────────               │
│  Combined → Best of both worlds                         │
└─────────────────────────────────────────────────────────┘
```

```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

# ─── Sparse Retriever (Keyword - BM25) ─────────────────
bm25_retriever = BM25Retriever.from_documents(docs, k=5)

# ─── Dense Retriever (Semantic) ────────────────────────
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ─── Combine with EnsembleRetriever ────────────────────
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # alpha: 0.4 keyword, 0.6 semantic
)

results = ensemble_retriever.invoke("What is RAG fusion?")
```

**When Hybrid Search Outperforms Pure Semantic:**

| Scenario | Why Hybrid Wins |
|----------|----------------|
| Product names / codes | Exact keyword match is critical |
| Medical / legal terms | Precise terminology matters |
| Abbreviations | Sparse catches exact matches |
| Mixed queries | "What is the ROI of RAG?" — needs both concepts |

> 🔬 **Lab**: Compare **semantic vs keyword vs hybrid** search on a technical documentation corpus. Measure precision for 10 test queries.

---

## 🚀 Module 6 — Advanced Retrieval Techniques

> 🎯 **Goal**: Move beyond basic retrieval to sophisticated strategies that handle real-world RAG challenges.

### 6.1 Contextual Compression

```
Problem: Retrieved chunks may contain mostly irrelevant content
Solution: Compress chunks to only the relevant parts before sending to LLM

Before Compression:
┌──────────────────────────────────────────┐
│ The company was founded in 2010. It has  │
│ offices in New York, London, and Tokyo.  │
│ Revenue grew 45% in Q3. The CEO is Jane. │  ← 4 facts, user asked about revenue
│ Employee count reached 5000 this year.   │
└──────────────────────────────────────────┘

After Compression:
┌──────────────────────┐
│ Revenue grew 45% Q3. │  ← Only what's relevant
└──────────────────────┘
```

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ─── Option 1: LLM-Based Extraction ───────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

# ─── Option 2: Embeddings Filter (Faster) ─────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76,
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

# ─── Option 3: Pipeline (Chain Compressors) ────────────
from langchain_community.document_transformers import EmbeddingsRedundantFilter

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, embeddings_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

> 🔬 **Hands-On**: Implement a contextual compression pipeline. Compare token usage with and without compression.

### 6.2 Parent Document Retriever

```
The Chunk-Size Dilemma:
┌────────────────────────────────────────────────────┐
│  Small chunks → Better matching, but lost context  │
│  Large chunks → Rich context, but diluted matching │
│                                                    │
│  Solution: Retrieve small chunks, return large ones│
└────────────────────────────────────────────────────┘

Architecture:
┌──────────────────────────────────────────┐
│  Original Document (Parent)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Child 1  │ │ Child 2  │ │ Child 3  │ │
│  │ (small)  │ │ (small)  │ │ (small)  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │             │             │       │
│  Vector Store    Vector Store  Vector Store│
│  (embeddings)    (embeddings)  (embeddings)│
└──────────────────────────────────────────┘

  Query → Match Child 2 → Return Parent Document (full context)
```

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Child splitter (small chunks for matching)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Parent splitter (large chunks for context)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Docstore to hold parent documents
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,       # Stores child embeddings
    docstore=docstore,              # Stores parent documents
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Index documents
retriever.add_documents(docs)

# Retrieve → returns PARENT documents
results = retriever.invoke("What is RAG?")
# Even though a child chunk matched, you get the full parent context
```

> 🔬 **Hands-On**: Build a parent-child document system. Compare answer quality vs. flat chunking.

### 6.3 Self-Query Retriever

```
Natural Language → Structured Query

User: "Show me finance documents from 2024 about revenue"
                │
                ▼
        ┌───────────────┐
        │  Self-Query   │
        │  Retriever    │
        └───────┬───────┘
                │
    ┌───────────┼───────────┐
    ▼                       ▼
Semantic Part          Metadata Filter
"revenue"              dept="Finance" AND year=2024
    │                       │
    └───────────┬───────────┘
                ▼
        Filtered Similarity Search
```

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="department",
        description="Department that produced the document",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="Year the document was published",
        type="integer",
    ),
    AttributeInfo(
        name="category",
        description="Document category: report, memo, presentation",
        type="string",
    ),
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Company internal documents and reports",
    metadata_field_info=metadata_field_info,
    enable_limit=True,  # Allow "show me 3 documents..."
)

# The LLM automatically converts:
# "finance documents from 2024 about revenue"
# → semantic: "revenue", filter: department="Finance", year=2024
results = retriever.invoke("finance documents from 2024 about revenue")
```

> 🔬 **Hands-On**: Build a **product catalog search** with self-query — filter by price, category, brand from natural language.

### 6.4 Multi-Query Retriever

```
User Query: "How does RAG work?"
        │
        ├──▶ LLM generates variations:
        │     1. "What is the architecture of RAG?"
        │     2. "Explain RAG pipeline and components"
        │     3. "How do retrieval and generation work together in RAG?"
        │
        ├──▶ Each variation → separate retrieval
        │     Query 1 → [doc_a, doc_b, doc_c]
        │     Query 2 → [doc_d, doc_b, doc_e]
        │     Query 3 → [doc_a, doc_f, doc_g]
        │
        └──▶ Union + deduplicate → richer context
              [doc_a, doc_b, doc_c, doc_d, doc_e, doc_f, doc_g]
```

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
)

# Enable logging to see generated queries
import logging
logging.setLevel(logging.DEBUG)

results = retriever.invoke("How does RAG work?")
# The LLM generates 3 query variations automatically
# All results are combined and deduplicated
```

> 🔬 **Hands-On**: Implement multi-query retrieval. Log the generated queries and analyze recall improvement.

### 6.5 Re-ranking Strategies

```
Two-Stage Retrieval:

  Stage 1: Fast Initial Retrieval (top-50)
  ┌────────────────────────────────────────────┐
  │ Retrieve many candidates with cheap method  │
  │ (vector similarity, BM25, etc.)             │
  └────────────────────┬───────────────────────┘
                       │
                       ▼
  Stage 2: Re-rank with Cross-Encoder (top-5)
  ┌────────────────────────────────────────────┐
  │ Score each (query, doc) pair jointly        │
  │ Cross-encoder considers query+doc together  │
  │ Much more accurate but slower               │
  └────────────────────────────────────────────┘
```

```python
# ─── Cross-Encoder Re-ranker ────────────────────────────
from langchain_community.document_transformers import (
    CrossEncoderReranker,
)
from langchain.retrievers import ContextualCompressionRetriever

# ─── Cohere Rerank Integration ─────────────────────────
from langchain_cohere import CohereRerank

compressor = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5,
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

results = compression_retriever.invoke("Explain RAG architecture")
# Returns top-5 re-ranked results from initial top-20
```

> 🔬 **Hands-On**: Add re-ranking to an existing pipeline. Compare retrieval precision before and after.

---

## 🧠 Module 7 — Advanced RAG Patterns

> 🎯 **Goal**: Implement cutting-edge RAG patterns that handle edge cases, improve relevance, and extend RAG to new modalities.

### 7.1 RAG Fusion

```
┌──────────────────────────────────────────────────────────┐
│                    RAG FUSION PIPELINE                    │
│                                                          │
│  Query: "Benefits of RAG"                                │
│      │                                                   │
│      ├──▶ Q1: "Advantages of RAG"         ──▶ [R1, R2]  │
│      ├──▶ Q2: "Why use retrieval augmentation" ──▶ [R3]  │
│      ├──▶ Q3: "RAG benefits over fine-tuning" ──▶ [R4,R1]│
│      └──▶ Q4: "Pros of RAG architecture"  ──▶ [R2, R5]  │
│                      │                                    │
│                      ▼                                    │
│            Reciprocal Rank Fusion (RRF)                   │
│            ┌─────────────────────────┐                   │
│            │ R1: 1/1 + 1/4 = 1.25   │                   │
│            │ R2: 1/2 + 1/1 = 1.50   │ ← highest rank   │
│            │ R3: 1/1 = 1.00         │                   │
│            │ R4: 1/1 = 1.00         │                   │
│            │ R5: 1/2 = 0.50         │                   │
│            └─────────────────────────┘                   │
│                      │                                    │
│                      ▼                                    │
│              Fused Ranking → LLM                          │
└──────────────────────────────────────────────────────────┘

RRF Formula: score(d) = Σ 1/(k + rank_i(d))  where k typically = 60
```

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Step 1: Generate multiple queries
# Step 2: Retrieve for each query
# Step 3: Apply Reciprocal Rank Fusion
# Step 4: Return fused results

# RRF Implementation
def reciprocal_rank_fusion(list_of_doc_lists, k=60):
    fused_scores = {}
    for doc_list in list_of_doc_lists:
        for rank, doc in enumerate(doc_list):
            doc_key = doc.page_content
            if doc_key not in fused_scores:
                fused_scores[doc_key] = {"score": 0, "doc": doc}
            fused_scores[doc_key]["score"] += 1 / (k + rank + 1)

    sorted_docs = sorted(
        fused_scores.values(), key=lambda x: x["score"], reverse=True
    )
    return [item["doc"] for item in sorted_docs]
```

**Trade-offs:**

| Aspect | Advantage | Disadvantage |
|--------|-----------|--------------|
| Comprehensiveness | Multiple perspectives on the query | — |
| Robustness | Less sensitive to query phrasing | — |
| Latency | — | Multiple LLM calls + retrievals |
| Cost | — | More API calls per query |

> 🔬 **Hands-On**: Build a complete RAG Fusion pipeline with RRF scoring.

### 7.2 HyDE (Hypothetical Document Embeddings)

```
Problem: Short queries and long documents live in different
         parts of embedding space → poor retrieval

Solution: Generate a hypothetical answer, embed THAT,
          then search with the hypothetical document

┌─────────────────────────────────────────────┐
│  Query: "What is CRAG?"                     │
│       │                                     │
│       ▼                                     │
│  LLM generates hypothetical answer:         │
│  "CRAG stands for Corrective RAG. It is a   │
│   framework that grades retrieved documents  │
│   for relevance and falls back to web search │
│   when retrieval fails..."                   │
│       │                                     │
│       ▼                                     │
│  Embed hypothetical answer                  │
│       │                                     │
│       ▼                                     │
│  Search vector store with HyDE embedding    │
│  → Much closer to actual document vectors!  │
└─────────────────────────────────────────────┘
```

```python
from langchain.retrievers import HypotheticalDocumentEmbedder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

hyde_retriever = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="instruction",  # Key in the prompt template
)

results = hyde_retriever.invoke("What is CRAG?")
# Under the hood: LLM generates hypothetical answer → embed it → search
```

> 🔬 **Hands-On**: Implement HyDE and compare retrieval quality against standard embedding search.

### 7.3 Corrective RAG (CRAG)

```
┌──────────────────────────────────────────────────────────┐
│                  CRAG (Corrective RAG)                    │
│                                                          │
│  Query                                                   │
│    │                                                     │
│    ▼                                                     │
│  ┌──────────────┐                                        │
│  │   Retrieve   │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │  Relevance   │──── All relevant? ────▶ Generate       │
│  │   Grading    │                                        │
│  └──────┬───────┘                                        │
│         │ Not all relevant                               │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │  Query       │──── Rewritten query ──▶ Re-retrieve   │
│  │  Rewriting   │                                        │
│  └──────┬───────┘                                        │
│         │ Still not enough                               │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │  Web Search  │──── Fallback to external knowledge     │
│  │   Fallback   │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│     Generate (with corrected context)                    │
└──────────────────────────────────────────────────────────┘
```

```python
# CRAG with LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

class RAGState(TypedDict):
    query: str
    documents: List[Document]
    relevance_scores: List[str]
    rewritten_query: str
    generation: str
    web_results: List[Document]

def retrieve(state: RAGState):
    docs = retriever.invoke(state["query"])
    return {"documents": docs}

def grade_documents(state: RAGState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    scored = []
    for doc in state["documents"]:
        prompt = f"""Is this document relevant to the query?
        Query: {state['query']}
        Document: {doc.page_content[:500]}
        Answer 'relevant' or 'irrelevant'."""
        result = llm.invoke(prompt).content
        scored.append(result)
    return {"relevance_scores": scored}

def rewrite_query(state: RAGState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Rewrite this query for better retrieval:
    Original: {state['query']}"""
    rewritten = llm.invoke(prompt).content
    return {"rewritten_query": rewritten}

def web_search_fallback(state: RAGState):
    # Use Tavily, SerpAPI, or similar
    from langchain_community.tools import TavilySearchResults
    search = TavilySearchResults(max_results=3)
    results = search.invoke(state["rewritten_query"] or state["query"])
    return {"web_results": results}

def generate(state: RAGState):
    # Combine relevant docs + web results, generate answer
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"Context: {context}\n\nQuestion: {state['query']}\n\nAnswer:"
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    answer = llm.invoke(prompt).content
    return {"generation": answer}

# Build the graph
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("web_search", web_search_fallback)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_action, {
    "generate": "generate",
    "rewrite": "rewrite",
})
workflow.add_edge("rewrite", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

> 🔬 **Hands-On**: Build CRAG with LangGraph. Test with a query where initial retrieval fails.

### 7.4 Self-RAG

```
┌──────────────────────────────────────────────────────────┐
│                    SELF-RAG                               │
│                                                          │
│  Self-RAG introduces REFLECTION TOKENS:                  │
│                                                          │
│  1. Retrieve?:  Should I retrieve for this query?        │
│  2. IsRelevant: Is the retrieved doc relevant?           │
│  3. IsSupported: Is the generation supported by docs?    │
│  4. IsUseful: Is the response useful to the user?        │
│                                                          │
│  Flow:                                                   │
│  Query → Classify complexity →                           │
│    Simple? → Direct LLM response (no retrieval)          │
│    Complex? → Retrieve → Grade → Generate → Verify       │
│      ↓                                                   │
│    Not supported? → Re-retrieve or correct               │
│      ↓                                                   │
│    Self-correction loop until quality threshold met      │
└──────────────────────────────────────────────────────────┘
```

**Key Capabilities:**
- 🔍 **Query complexity classification**: Skip retrieval for simple queries
- ✅ **Self-reflection**: Verify generation is grounded in retrieved context
- 🔄 **Self-correction**: Re-retrieve or regenerate when quality is low
- 📊 **Reflection tokens**: Explicit quality indicators at each step

### 7.5 Graph RAG

```
┌──────────────────────────────────────────────────────────┐
│                    GRAPH RAG                              │
│                                                          │
│  When Vector RAG struggles with:                         │
│  • Multi-hop reasoning (A→B→C)                          │
│  • Relationship queries                                  │
│  • Global community understanding                        │
│                                                          │
│  Graph RAG adds:                                         │
│  ┌─────────────┐       ┌─────────────┐                  │
│  │  Knowledge   │       │   Vector    │                  │
│  │   Graph      │  +    │   Store     │                  │
│  │  (Neo4j)    │       │  (Chroma)   │                  │
│  └──────┬──────┘       └──────┬──────┘                  │
│         │                      │                         │
│         └──────────┬───────────┘                         │
│                    ▼                                     │
│           Combined Retrieval                             │
│           (traverse relationships + semantic search)      │
└──────────────────────────────────────────────────────────┘
```

```python
# ─── Adding Data to Neo4j Graph Database ───────────────
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password",
)

# ─── Working with Multi-Hop Queries ────────────────────
# Query: "Who is the CEO of the company that acquired the firm
#         that developed the technology mentioned in this paper?"
#
# Vector RAG: Struggles — needs 3 hops across documents
# Graph RAG: Follows edges (paper→technology→firm→company→CEO)

# ─── When Graph RAG Outperforms Vector RAG ─────────────
# ✅ Multi-hop reasoning questions
# ✅ Relationship-heavy domains (organizational charts, supply chains)
# ✅ Community-level summaries
# ✅ Queries requiring traversal logic
```

### 7.6 Multi-Modal RAG 🔜

> 🚧 **Coming Soon** — Extending RAG to images, tables, and mixed-content documents.

---

## 🤖 Module 8 — Agentic RAG with LangGraph

> 🎯 **Goal**: Build autonomous RAG agents that reason, use tools, and execute multi-step retrieval strategies.

### 8.1 Introduction to Agentic RAG

```
┌──────────────────────────────────────────────────────────┐
│          TRADITIONAL RAG vs AGENTIC RAG                  │
│                                                          │
│  Traditional RAG:                                        │
│  Query → Retrieve → Generate (fixed pipeline)            │
│                                                          │
│  Agentic RAG:                                            │
│  Query → Agent decides:                                  │
│    ├── Need retrieval? → Search tool                     │
│    ├── Need different source? → Switch knowledge base    │
│    ├── Need clarification? → Ask user                    │
│    ├── Need web search? → Fallback tool                  │
│    ├── Need calculation? → Code execution tool           │
│    └── Have enough info? → Generate answer               │
│                                                          │
│  Key Difference: The AGENT decides the retrieval        │
│  strategy, not a fixed pipeline.                         │
└──────────────────────────────────────────────────────────┘
```

**Key Capabilities:**

| Capability | Description |
|-----------|-------------|
| 🧠 **Reasoning** | Agent decides whether, how, and when to retrieve |
| 🔧 **Tool Use** | RAG is one of many tools the agent can invoke |
| 🔄 **Multi-step Execution** | Retrieve → evaluate → retrieve more → generate |
| 🎯 **Adaptive** | Changes strategy based on intermediate results |

### 8.2 RAG as a Tool for Agents

```python
from langchain.tools import create_retriever_tool

# ─── Creating Retriever Tools ──────────────────────────
# Tool 1: Company knowledge base
company_tool = create_retriever_tool(
    retriever=company_vectorstore.as_retriever(search_kwargs={"k": 5}),
    name="company_knowledge_base",
    description="Search internal company documents, policies, and reports. "
                "Use this when asked about company-specific information.",
)

# Tool 2: Technical documentation
tech_tool = create_retriever_tool(
    retriever=tech_vectorstore.as_retriever(search_kwargs={"k": 5}),
    name="technical_docs",
    description="Search technical documentation, API references, and code samples. "
                "Use this for technical questions about products and APIs.",
)

# ─── When Agents Should Retrieve vs Respond Directly ────
# The agent decides based on:
# 1. Tool descriptions — clear descriptions help the agent choose
# 2. Query type — factual → retrieve, creative → generate directly
# 3. Confidence — if unsure, retrieve; if confident, answer directly

# ─── Agent with Multiple Knowledge Base Tools ──────────
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent

tools = [company_tool, tech_tool]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# The agent will choose the right tool based on the query
```

> 🔬 **Hands-On**: Build an agent with a retrieval tool. Test queries that should and shouldn't trigger retrieval.

### 8.3 LangGraph Fundamentals for RAG

```
┌──────────────────────────────────────────────────────────┐
│              LANGGRAPH CORE CONCEPTS                     │
│                                                          │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│  │  STATE  │────▶│  NODE   │────▶│  EDGE   │           │
│  └─────────┘     └─────────┘     └─────────┘           │
│                                                          │
│  State:  TypedDict that flows through the graph         │
│  Node:   A function that processes/transforms state     │
│  Edge:   Connection between nodes (can be conditional)  │
│                                                          │
│  RAG State Schema:                                       │
│  ┌────────────────────────────────────┐                 │
│  │ class RAGState(TypedDict):        │                 │
│  │     query: str                    │                 │
│  │     documents: List[Document]     │                 │
│  │     generation: str               │                 │
│  │     relevance: str                │                 │
│  │     iterations: int               │                 │
│  └────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────┘
```

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
import operator

# ─── Step 1: Define State Schema ───────────────────────
class AgentRAGState(TypedDict):
    query: str
    documents: List[Document]
    generation: str
    relevance_grade: str
    hallucination_grade: str
    iterations: int

# ─── Step 2: Define Node Functions ─────────────────────
def retrieve(state: AgentRAGState):
    docs = retriever.invoke(state["query"])
    return {"documents": docs}

def generate(state: AgentRAGState):
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"Context: {context}\n\nQuestion: {state['query']}\n\nAnswer:"
    answer = llm.invoke(prompt).content
    return {"generation": answer}

def grade_relevance(state: AgentRAGState):
    prompt = f"""Is this document relevant to the query?
    Query: {state['query']}
    Docs: {[d.page_content[:200] for d in state['documents']]}
    Answer 'relevant' or 'irrelevant'."""
    grade = llm.invoke(prompt).content
    return {"relevance_grade": grade}

# ─── Step 3: Define Conditional Routing ─────────────────
def should_generate(state: AgentRAGState):
    if state["relevance_grade"] == "relevant":
        return "generate"
    elif state.get("iterations", 0) >= 3:
        return "generate"  # Give up after 3 tries
    else:
        return "rewrite"

# ─── Step 4: Build the Graph ────────────────────────────
workflow = StateGraph(AgentRAGState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_relevance)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite_query)

# Add edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", should_generate, {
    "generate": "generate",
    "rewrite": "rewrite",
})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
```

> 🔬 **Hands-On**: Build an agentic RAG graph from scratch with retrieval, grading, and conditional routing.

### 8.4 Agentic RAG Design Patterns

#### Pattern 1: ReAct with RAG

```
┌──────────────────────────────────────────────────────────┐
│  ReAct (Reason + Act) Pattern with RAG                  │
│                                                          │
│  Thought: I need to find information about RAG metrics  │
│  Action: search_knowledge_base("RAG evaluation metrics") │
│  Observation: [retrieved documents about RAGAS...]      │
│  Thought: I found info about RAGAS but need more on     │
│           faithfulness specifically                      │
│  Action: search_knowledge_base("faithfulness metric")   │
│  Observation: [more specific docs...]                   │
│  Thought: Now I have enough information to answer       │
│  Answer: Based on the retrieved documents...            │
└──────────────────────────────────────────────────────────┘
```

#### Pattern 2: Plan-and-Execute with Retrieval

```
┌──────────────────────────────────────────────────────────┐
│  Plan-and-Execute Pattern                                │
│                                                          │
│  Query: "Compare RAG vs fine-tuning for medical QA"     │
│                                                          │
│  Step 1: PLAN                                            │
│    Sub-task 1: Retrieve info about RAG for medical QA    │
│    Sub-task 2: Retrieve info about fine-tuning for QA    │
│    Sub-task 3: Compare the two approaches               │
│                                                          │
│  Step 2: EXECUTE each sub-task                           │
│    Execute task 1 → results_1                            │
│    Execute task 2 → results_2                            │
│                                                          │
│  Step 3: SYNTHESIZE                                      │
│    Combine results_1 + results_2 → comparative answer    │
└──────────────────────────────────────────────────────────┘
```

#### Pattern 3: Reflection / Self-Correction

```
┌──────────────────────────────────────────────────────────┐
│  Reflection Pattern                                      │
│                                                          │
│  Generate → Critique → Improve                           │
│                                                          │
│  1. Generate answer from retrieved context               │
│  2. Critique: Is this answer grounded? Complete?         │
│     ├── Yes → Return answer                              │
│     └── No → Identify gaps → Re-retrieve → Regenerate   │
│                                                          │
│  Loop until quality threshold met or max iterations      │
└──────────────────────────────────────────────────────────┘
```

> 🔬 **Hands-On**: Implement a **ReAct RAG agent** using LangGraph with tools for retrieval and web search fallback.

---

## 📊 Module 9 — Evaluating RAG using RAGAS

> 🎯 **Goal**: Measure what matters — systematically evaluate every component of your RAG pipeline.

### 9.1 RAG Evaluation

```
┌──────────────────────────────────────────────────────────┐
│               RAG EVALUATION METRICS                      │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │              GENERATION QUALITY                  │     │
│  │  • Faithfulness: Is the answer grounded in      │     │
│  │    retrieved context?                            │     │
│  │  • Relevance: Is the answer relevant to the     │     │
│  │    query?                                        │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │              RETRIEVAL QUALITY                   │     │
│  │  • Context Precision: Are retrieved chunks      │     │
│  │    relevant and noise-free?                      │     │
│  │  • Context Recall: Did we retrieve all the      │     │
│  │    necessary information?                        │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │              ROBUSTNESS                          │     │
│  │  • Noise Sensitivity: How does irrelevant       │     │
│  │    context affect generation?                    │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

| Metric | What It Measures | Score Range | When It Matters |
|--------|-----------------|-------------|----------------|
| **Faithfulness** | Answer grounded in context? | 0–1 | Always — prevents hallucination |
| **Relevance** | Answer addresses the query? | 0–1 | Always — ensures usefulness |
| **Context Precision** | Retrieved docs are relevant? | 0–1 | When retrieval noise is an issue |
| **Context Recall** | All needed info retrieved? | 0–1 | When completeness matters |
| **Noise Sensitivity** | Robustness to irrelevant docs | 0–1 | When retrieval is imprecise |

```python
# ─── RAGAS Framework Implementation ────────────────────
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# ─── Creating Evaluation Dataset ───────────────────────
# Format: list of dictionaries with these keys

eval_data = {
    "question": [
        "What is RAG?",
        "How does chunking affect retrieval?",
        "When should I use MMR?",
    ],
    "answer": [
        "RAG is Retrieval-Augmented Generation...",
        "Smaller chunks improve matching precision...",
        "MMR is useful when you need diverse results...",
    ],
    "contexts": [
        ["RAG combines retrieval with generation..."],  # Retrieved chunks
        ["Chunk size affects precision and recall..."],
        ["MMR balances relevance and diversity..."],
    ],
    "ground_truth": [  # Reference answers
        "RAG is a technique that combines document retrieval with LLM generation",
        "Smaller chunks improve matching but may lose context; larger chunks provide context but dilute relevance",
        "Use MMR when query results need diversity, like summarization or exploratory search",
    ],
}

dataset = Dataset.from_dict(eval_data)

# ─── Evaluate ──────────────────────────────────────────
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print(results)
# Output example:
# {'faithfulness': 0.85, 'answer_relevancy': 0.92,
#  'context_precision': 0.78, 'context_recall': 0.71}
```

```python
# ─── Evaluating Individual Metrics ─────────────────────
from ragas.metrics import faithfulness

# Single sample evaluation
score = faithfulness.score(
    question="What is RAG?",
    answer="RAG combines retrieval with generation",
    contexts=["RAG is a framework that retrieves documents..."],
)
```

> 🔬 **Lab**: Evaluate **each metric individually** on your RAG pipeline, then evaluate the full API. Identify the weakest component and iterate.

---

## 🏆 Module 10 — Capstone Project with Deployment

> 🎯 **Goal**: Build, evaluate, and deploy a production-ready RAG system.

### 10.1 Project Assignment: Build a Production-Ready RAG System

```
┌──────────────────────────────────────────────────────────┐
│              CAPSTONE ARCHITECTURE                       │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │         MULTI-DOCUMENT INGESTION PIPELINE        │     │
│  │  PDF ──▶ Web ──▶ CSV ──▶ JSON ──▶ Chunking     │     │
│  │          ──▶ Embedding ──▶ Vector Store          │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │         HYBRID RETRIEVAL WITH RE-RANKING         │     │
│  │  BM25 (sparse) + Dense vectors → Ensemble        │     │
│  │  → Cross-encoder re-ranker → Top-K              │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │         AGENTIC WORKFLOW WITH LANGGRAPH          │     │
│  │  Retrieve → Grade → Rewrite/Search → Generate    │     │
│  │  → Verify → Return                               │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │         EVALUATION SUITE WITH RAGAS              │     │
│  │  Faithfulness · Relevance · Precision · Recall   │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │         MONITORING WITH LANGSMITH                │     │
│  │  Trace every step · Track metrics · Debug        │     │
│  └──────────────────────┬──────────────────────────┘     │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │              DEPLOYMENT                          │     │
│  │  API (FastAPI) · Container (Docker) · Cloud     │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### 10.2 Production RAG Good Practices

#### Pipeline Optimization

| Area | Strategy | Impact |
|------|----------|--------|
| **Embedding** | Batch embed documents, cache query embeddings | 🟢🟢🟢 |
| **Retrieval** | Tune `k`, use hybrid search, pre-filter by metadata | 🟢🟢 |
| **Re-ranking** | Only re-rank top-20, not entire corpus | 🟢🟢 |
| **Generation** | Use cheaper model (gpt-4o-mini) for simple queries | 🟢🟢🟢 |
| **Chunking** | Pre-compute and persist chunks | 🟢🟢 |

#### Caching Strategies

```python
# ─── Semantic Caching ──────────────────────────────────
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# ─── Redis Caching for Production ──────────────────────
from langchain_community.cache import RedisCache
from redis import Redis

set_llm_cache(RedisCache(redis_=Redis(host="localhost", port=6379)))

# ─── Embedding Caching ─────────────────────────────────
from langchain.embeddings import CacheBackedEmbeddings

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_store=RedisStore(),
)
```

#### Cost Optimization

```
┌────────────────────────────────────────────────────────┐
│              COST OPTIMIZATION CHECKLIST               │
│                                                        │
│  ☐ Use text-embedding-3-small unless quality gap is   │
│    measurable in your evaluation                       │
│  ☐ Cache frequent query embeddings                     │
│  ☐ Use gpt-4o-mini for routing / grading / simple     │
│    queries; reserve gpt-4o for complex generation     │
│  ☐ Compress context before sending to LLM             │
│  ☐ Implement tiered retrieval (cheap first, expensive │
│    only when needed)                                   │
│  ☐ Set max_tokens to prevent runaway generation       │
│  ☐ Monitor token usage per query                      │
└────────────────────────────────────────────────────────┘
```

#### Monitoring & Debugging

```python
# ─── LangSmith Integration ─────────────────────────────
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "rag-production"

# Every LLM call, retrieval, and chain step is now traced
# View at: https://smith.langchain.com

# Key metrics to monitor:
# • Latency per pipeline stage
# • Token usage per query
# • Retrieval quality (relevance scores)
# • Error rates and types
# • User feedback / thumbs up-down
```

#### Common Pitfalls and Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Bad chunking** | Irrelevant or incomplete answers | Optimize chunk_size, use semantic chunking, parent document retriever |
| **Stale data** | Outdated answers | Implement incremental indexing, version control |
| **No fallback** | Empty answers when retrieval fails | Add CRAG-style web search fallback |
| **Context overflow** | LLM ignores relevant chunks | Use compression, re-ranking, smaller `k` |
| **Missing metadata** | Can't filter effectively | Enrich documents with metadata at ingestion |
| **No evaluation** | Can't measure improvement | Implement RAGAS evaluation suite |

#### Security and Compliance

```
┌────────────────────────────────────────────────────────┐
│            SECURITY & COMPLIANCE CHECKLIST             │
│                                                        │
│  ☐ Access Control: RBAC on knowledge bases             │
│  ☐ PII Detection: Redact before embedding              │
│  ☐ Data Encryption: At rest and in transit             │
│  ☐ Audit Logging: Track all queries and responses      │
│  ☐ Rate Limiting: Prevent abuse                       │
│  ☐ Input Validation: Sanitize user queries             │
│  ☐ GDPR Compliance: Right to deletion in vector store  │
│  ☐ SOC 2: Document data handling procedures            │
│  ☐ Prompt Injection Defense: Validate retrieved content │
│  ☐ Data Residency: Ensure vectors stored in correct    │
│    region                                              │
└────────────────────────────────────────────────────────┘
```

---

## 🧰 Bonus — Tools & Frameworks Quick Reference

### Essential Stack

| Category | Tool | Purpose |
|----------|------|---------|
| **Framework** | LangChain | RAG pipeline orchestration |
| **Agents** | LangGraph | Agentic RAG workflows |
| **Embeddings** | OpenAI / Ollama | Text vectorization |
| **Vector Store** | Chroma / Qdrant | Similarity search |
| **Re-ranking** | Cohere Rerank | Result quality boost |
| **Evaluation** | RAGAS | Pipeline metrics |
| **Monitoring** | LangSmith | Tracing & debugging |
| **Graph DB** | Neo4j | Graph RAG |
| **Deployment** | FastAPI + Docker | Production serving |

### Key LangChain Imports Cheat Sheet

```python
# ─── Document Loading ──────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader

# ─── Text Splitting ────────────────────────────────────
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker

# ─── Embeddings ────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# ─── Vector Stores ─────────────────────────────────────
from langchain_community.vectorstores import Chroma, FAISS, Qdrant

# ─── Retrievers ────────────────────────────────────────
from langchain.retrievers import (
    MultiQueryRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
    EnsembleRetriever,
    SelfQueryRetriever,
    BM25Retriever,
)
from langchain_community.retrievers import TavilySearchAPIRetriever

# ─── Tools ─────────────────────────────────────────────
from langchain.tools import create_retriever_tool

# ─── Re-ranking ────────────────────────────────────────
from langchain_cohere import CohereRerank
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
)

# ─── Evaluation ────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
```

---

## 🗓️ Suggested Learning Schedule

| Week | Module | Focus Area | Deliverable |
|------|--------|-----------|-------------|
| 1 | Module 1 | RAG Fundamentals | Architecture diagram + comparison matrix |
| 2 | Module 2 | Document Processing | Multi-source document pipeline |
| 3 | Module 3 | Embeddings | Model comparison report |
| 3 | Module 4 | Vector Stores | Full CRUD application |
| 4 | Module 5 | Basic Retrieval | Similarity + MMR + Hybrid demo |
| 5 | Module 6 | Advanced Retrieval | Compression + Parent-Doc + Self-Query |
| 6 | Module 7 | Advanced Patterns | CRAG + HyDE + RAG Fusion implementations |
| 7 | Module 8 | Agentic RAG | LangGraph RAG agent with tools |
| 8 | Module 9 | Evaluation | RAGAS evaluation suite |
| 9–10 | Module 10 | Capstone | Production RAG system deployed |

---

## 📚 Recommended Resources

| Resource | Type | Link |
|----------|------|------|
| LangChain Docs | Documentation | [python.langchain.com](https://python.langchain.com/) |
| LangGraph Docs | Documentation | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| RAGAS Docs | Documentation | [docs.ragas.io](https://docs.ragas.io/) |
| MTEB Leaderboard | Benchmark | [huggingface.co/spaces/mteb](https://huggingface.co/spaces/mteb/leaderboard) |
| LangSmith | Monitoring | [smith.langchain.com](https://smith.langchain.com/) |
| Qdrant Docs | Documentation | [qdrant.tech/documentation](https://qdrant.tech/documentation/) |
| Neo4j GraphRAG | Documentation | [neo4j.com/labs/genai-ecosystem](https://neo4j.com/labs/genai-ecosystem/) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

---

## ⭐ Star History

If this roadmap helped you, please consider giving it a ⭐!

---

<p align="center">
  <strong>Happy Learning! 🚀</strong><br>
  <em>From RAG Fundamentals to Production-Ready Agentic Systems</em>
</p>
