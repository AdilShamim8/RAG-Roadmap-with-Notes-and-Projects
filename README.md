# RAG Roadmap — From Fundamentals to Production-Ready Agentic Systems

<div align="center">

<img src="https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-4A90D9?style=for-the-badge&logo=databricks&logoColor=white" />
<img src="https://img.shields.io/badge/LangChain-Powered-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" />
<img src="https://img.shields.io/badge/LangGraph-Agentic%20RAG-FF6B35?style=for-the-badge" />

<br/><br/>

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/RAG-Roadmap?style=social)](https://github.com/YOUR_USERNAME/RAG-Roadmap)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/RAG-Roadmap?style=social)](https://github.com/YOUR_USERNAME/RAG-Roadmap)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Last Updated](https://img.shields.io/badge/last%20updated-June%202026-brightgreen)](https://github.com/YOUR_USERNAME/RAG-Roadmap)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<br/>

**A comprehensive, end-to-end learning path for Retrieval-Augmented Generation —**
**from core concepts and document processing to agentic RAG systems, evaluation, and production deployment.**

</div>

---

## 🆕 What's New in 2026

> **This roadmap has been updated for June 2026** to reflect the latest advancements in the RAG ecosystem, vector databases, embedding models, and agentic frameworks.

### Key Updates:

- ✅ **LangGraph v1.1.10** – Type-safe streaming, type-safe invoke, Pydantic and dataclass coercion; now the backbone for agentic RAG workflows
- ✅ **Agentic RAG patterns** – ReAct, Plan-and-Execute, and Reflection patterns all covered with LangGraph implementation
- ✅ **Graph RAG with Neo4j** – Knowledge graph integration, multi-hop queries, and hybrid vector + graph retrieval
- ✅ **RAGAS Evaluation Framework** – Full suite: faithfulness, relevance, context precision, context recall, and noise sensitivity
- ✅ **OpenAI text-embedding-3 series** – `text-embedding-3-small` and `text-embedding-3-large` as default embedding models
- ✅ **Qdrant Cloud free tier** – 1 GB cloud-forever; now a recommended production vector store option
- ✅ **Corrective RAG (CRAG)** – Web search fallback, document grading, and query rewriting via LangGraph
- ✅ **Self-RAG** – Reflection tokens, self-correction, and query complexity classification
- ✅ **Multimodal RAG** – Extending retrieval to images, tables, and mixed-content documents *(coming soon)*
- ✅ **Capstone Project + Deployment** – Multi-document pipeline, hybrid retrieval, agentic workflow, RAGAS evaluation, LangSmith monitoring *(coming soon)*

---

##  Table of Contents

- [Overview](#-overview)
- [2026 RAG Landscape](#-2026-rag-landscape)
- [Prerequisites](#-prerequisites)
- [Module Map](#-module-map)
- [Module 1 — RAG Fundamentals and Architecture](#-module-1--rag-fundamentals-and-architecture)
- [Module 2 — Document Processing and Chunking](#-module-2--document-processing-and-chunking)
- [Module 3 — Embeddings and Vector Representations](#-module-3--embeddings-and-vector-representations)
- [Module 4 — Vector Stores](#-module-4--vector-stores)
- [Module 5 — Basic Retrieval Techniques](#-module-5--basic-retrieval-techniques)
- [Module 6 — Advanced Retrieval Techniques](#-module-6--advanced-retrieval-techniques)
- [Module 7 — Advanced RAG Patterns](#-module-7--advanced-rag-patterns)
- [Module 8 — Agentic RAG with LangGraph](#-module-8--agentic-rag-with-langgraph)
- [Module 9 — Evaluating RAG with RAGAS](#-module-9--evaluating-rag-with-ragas)
- [Module 10 — Capstone Project with Deployment](#-module-10--capstone-project-with-deployment)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Code Examples](#-code-examples)
- [Top Resources](#-top-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

##  Overview

This repository provides a **structured, step-by-step curriculum** for mastering Retrieval-Augmented Generation (RAG) — one of the most impactful patterns in production AI engineering today.

RAG addresses core limitations of large language models: hallucination, knowledge cutoffs, and the inability to cite sources. This roadmap takes you from understanding those problems to building production-grade systems that solve them.

### What you will build:

| Milestone | Description |
|-----------|-------------|
| 📄 Document Pipelines | Load, parse, chunk, and enrich documents from PDFs, URLs, CSVs, and more |
| 🧠 Embedding Workflows | Compare and implement embedding models for semantic search |
| 🗄️ Vector Store CRUD | Build full create/read/update/delete vector store applications |
| 🔎 Retrieval Systems | Similarity search, MMR, hybrid search, contextual compression, self-query |
| 🤖 Advanced RAG Patterns | RAG Fusion, HyDE, CRAG, Self-RAG, Graph RAG |
| 🕸️ Agentic RAG | LangGraph-based agents with retrieval tools, ReAct loops, and self-correction |
| 📊 RAG Evaluation | RAGAS-based evaluation pipelines with all key metrics |
| 🚀 Production Deployment | Monitoring with LangSmith, caching, cost optimization, security |

---

## 🌐 2026 RAG Landscape

### Why RAG Is Foundational in 2026

RAG has moved from an experimental technique to the **default architecture** for enterprise AI systems that require accurate, traceable, and domain-specific responses. Key forces driving this:

1. **Hallucination remains unsolved at scale** — even frontier models hallucinate; RAG provides a grounding mechanism
2. **Knowledge cutoffs still exist** — retrieval gives models access to live or proprietary information
3. **Regulatory requirements** — industries like finance and healthcare demand source attribution and auditability
4. **Context window limits** — while context windows have grown, retrieval is still more cost-efficient for large corpora
5. **Agentic AI requires reliable memory** — RAG is the backbone of agent long-term memory and tool-augmented reasoning

### RAG Architecture Evolution

| Generation | Approach | Key Limitation |
|------------|----------|----------------|
| **Naive RAG** | Chunk → Embed → Retrieve → Generate | Poor precision, no diversity control |
| **Advanced RAG** | Query expansion, re-ranking, compression | Higher latency, more complexity |
| **Modular RAG** | Swappable retrieval components | Integration overhead |
| **Agentic RAG** (2025–2026) | Agents that decide when/how to retrieve | Reasoning overhead, requires orchestration |
| **Graph RAG** (2025–2026) | Knowledge graphs + vector retrieval | Graph construction cost |

### Key Tools in the Ecosystem (June 2026)

| Category | Tools |
|----------|-------|
| **Orchestration** | LangChain v1.2.x, LlamaIndex, Haystack |
| **Agentic Frameworks** | LangGraph v1.1.10, AutoGen, CrewAI |
| **Vector Stores** | Chroma, FAISS, Qdrant, Pinecone, Milvus, Weaviate |
| **Embedding Models** | OpenAI text-embedding-3, Ollama/Gemma, Cohere, HuggingFace |
| **Re-ranking** | Cohere Rerank, FlashRank, cross-encoders |
| **Evaluation** | RAGAS, TruLens, DeepEval |
| **Observability** | LangSmith, Arize, Weights & Biases |
| **Graph Databases** | Neo4j, Amazon Neptune |

---

## 🛠️ Prerequisites

Before starting this course, you should be comfortable with:

- **Python fundamentals** — functions, classes, file I/O, virtual environments
- **GenAI fundamentals** — what LLMs are, how prompting works, basic model API usage
- **LangChain basics** — chains, prompts, LLM wrappers (covered in the [GenAI Roadmap](https://github.com/AdilShamim8/GenAI-Roadmap-with-Notes-and-Projects))
- **LangGraph basics** — state, nodes, edges (covered in the [Agentic AI Roadmap](https://github.com/AdilShamim8/Agentic-AI-Roadmap-with-Notes-Using-LangGraph))

---

## 🗺️ Module Map

```
RAG Roadmap — 10 Modules
│
├── Module 1  — RAG Fundamentals and Architecture          [Week 1]
├── Module 2  — Document Processing and Chunking           [Week 2]
├── Module 3  — Embeddings and Vector Representations      [Week 3]
├── Module 4  — Vector Stores                              [Week 4]
├── Module 5  — Basic Retrieval Techniques                 [Week 5]
├── Module 6  — Advanced Retrieval Techniques              [Week 6]
├── Module 7  — Advanced RAG Patterns                      [Week 7–8]
├── Module 8  — Agentic RAG with LangGraph                 [Week 9]
├── Module 9  — Evaluating RAG with RAGAS                  [Week 10]
└── Module 10 — Capstone Project with Deployment           [Week 11–12]
```

---

## 📦 Module 1 — RAG Fundamentals and Architecture

> **Duration:** ~1 week | **Goal:** Understand what RAG is, why it exists, and how data flows through the system

### 1.1 Introduction to RAG

#### 1.1.1 What is RAG and the Problems It Solves

- **Hallucination reduction** — grounding generation in retrieved evidence
- **Knowledge cutoff limitations** — retrieving live or proprietary data the LLM was never trained on
- **Domain-specific knowledge injection** — plugging in enterprise or niche documents without fine-tuning
- **Source attribution and transparency** — every generated claim can be traced to a source chunk

#### 1.1.2 Core Components

```
Knowledge Base  →  Retriever  →  Generator
     ↑                               ↓
 [Documents]               [Final Response + Sources]
```

#### 1.1.3 RAG Data Flow

```
User Query
    │
    ▼
[Query Embedding]
    │
    ▼
[Vector Store Similarity Search]
    │
    ▼
[Top-K Relevant Chunks]
    │
    ▼
[Prompt Assembly: Query + Context]
    │
    ▼
[LLM Generation]
    │
    ▼
[Final Answer with Citations]
```

---

### 1.2 RAG vs Fine-Tuning vs Prompt Engineering

| Dimension | Prompt Engineering | RAG | Fine-Tuning |
|-----------|-------------------|-----|-------------|
| **Cost** | Very Low | Low–Medium | High |
| **Latency** | Lowest | Medium | Low |
| **Knowledge Freshness** | Static | Dynamic ✅ | Static |
| **Domain Accuracy** | Limited | High ✅ | Very High |
| **Source Attribution** | ❌ | ✅ | ❌ |
| **Maintenance** | Easy | Medium | High |
| **Data Required** | None | Documents | Labeled examples |

#### Decision Framework

- ✅ Use **Prompt Engineering** → Simple tasks, no special domain knowledge needed
- ✅ Use **RAG** → Dynamic, large, or proprietary knowledge bases; source attribution required
- ✅ Use **Fine-Tuning** → Specific output style/format; consistent domain-specific tone; fixed, curated dataset

---

## 📄 Module 2 — Document Processing and Chunking

> **Duration:** ~1 week | **Goal:** Load and prepare any document type for embedding and retrieval

### 2.1 Document Loaders in LangChain

| Loader | Source Type | Notes |
|--------|-------------|-------|
| `PyPDFLoader` | PDF | Fast, page-level splits |
| `PDFMinerLoader` | PDF | Better text extraction for complex layouts |
| `PDFPlumberLoader` | PDF | Best for tables and structured PDFs |
| `UnstructuredLoader` | PDF, DOCX, PPTX, HTML | Multi-format powerhouse |
| `WebBaseLoader` | Web pages | HTML parsing |
| `RecursiveUrlLoader` | Entire sites | Crawls links recursively |
| `CSVLoader` | CSV | Row-per-document loading |
| `JSONLoader` | JSON | jq-based field selection |
| `TextLoader` | .txt | Simplest loader |

**🧪 Hands-on:** Load documents from 5 different sources (PDF, URL, CSV, JSON, plain text)

---

### 2.2 Text Splitting Strategies

#### RecursiveCharacterTextSplitter *(Recommended Default)*

- Hierarchical separator logic: `["\n\n", "\n", " ", ""]`
- Parameters: `chunk_size`, `chunk_overlap`
- Respects document structure before falling back to raw character splitting

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
```

#### Other Splitters

| Splitter | Best For |
|----------|----------|
| `CharacterTextSplitter` | Simple single-separator splits |
| `TokenTextSplitter` | Token-budget-aware splitting |
| `SemanticChunker` | Meaning-based breakpoints (percentile, std_dev, gradient) |
| `MarkdownTextSplitter` | Markdown headers as natural boundaries |
| `PythonCodeTextSplitter` | Python code (functions, classes) |
| `JavaScriptTextSplitter` | JS/TS code |
| LLM-based splitting | Semantic boundaries via LLM prompting |

**🧪 Hands-on:** Compare all chunking strategies on the same PDF document

---

### 2.3 Chunking Best Practices

#### Optimal Chunk Sizes by Use Case

| Use Case | Chunk Size | Overlap |
|----------|-----------|---------|
| Factoid Q&A | 128–256 tokens | 10–15% |
| General RAG | 256–512 tokens | 10–20% |
| Complex Analysis | 512–1024 tokens | 15–20% |
| Summarization | 1024–2048 tokens | 5–10% |

- **Overlap recommendation:** 10–20% of chunk size (e.g., 50 tokens overlap for a 512-token chunk)
- **Tables and structured data:** Extract separately; preserve structure or use specialized table parsers
- **Context preservation:** Use `ParentDocumentRetriever` to retrieve larger context around small matched chunks

**🧪 Lab:** Optimize chunking strategy for a PDF technical document — measure retrieval quality at each configuration

---

### 2.4 Metadata Management

- Adding metadata fields to documents (`source`, `page`, `author`, `date`, `category`)
- Using metadata for **filtering** at retrieval time (e.g., `filter={"category": "legal"}`)
- Source tracking and document lineage
- Document versioning strategies

**🧪 Hands-on:** Build a metadata-enriched document pipeline with filter-based retrieval

---

## 🧠 Module 3 — Embeddings and Vector Representations

> **Duration:** ~1 week | **Goal:** Understand embeddings and select the right model for your use case

### 3.1 How Embeddings Work

- Text → dense numerical vector (e.g., 1536 dimensions)
- Semantically similar texts map to nearby points in vector space
- **Distance metrics:**
  - **Cosine Similarity** — angle between vectors; most common for RAG
  - **Euclidean (L2)** — absolute distance; sensitive to magnitude
  - **Dot Product** — fast; best when vectors are normalized
- Higher dimensions → more expressive but more memory/compute

---

### 3.2 Embedding Models Overview

#### OpenAI Embeddings

| Model | Cost | Dimensions | Best For |
|-------|------|-----------|----------|
| `text-embedding-3-small` | $0.02 / 1M tokens | 1536 | Best value — recommended default |
| `text-embedding-3-large` | $0.13 / 1M tokens | 3072 | Highest quality, precision-critical tasks |
| `text-embedding-ada-002` | $0.10 / 1M tokens | 1536 | Legacy; prefer `text-embedding-3-small` |

#### Open-Source / Local Embedding Options

| Method | Model | Notes |
|--------|-------|-------|
| `langchain-ollama` | Gemma, Llama, Nomic | Full local inference |
| `HuggingFaceEmbeddings` | `all-MiniLM-L6-v2`, `bge-large-en` | Free, self-hosted |
| `CohereEmbeddings` | `embed-english-v3.0` | Strong MTEB scores |

---

### 3.3 Choosing the Right Embedding Model

- **MTEB Leaderboard** — the industry benchmark for embedding model evaluation
- Dimension trade-offs:

```
384 dims  → Fast, lightweight (MiniLM)
768 dims  → Balanced (BGE-base)
1536 dims → High quality (OpenAI text-embedding-3-small)
3072 dims → Highest quality (OpenAI text-embedding-3-large)
```

- **Decision framework:** Start with `text-embedding-3-small`; evaluate on your domain; upgrade to `text-embedding-3-large` only if quality gap is significant

**🧪 Hands-on:** Compare embedding models on sample queries using cosine similarity scores

---

### 3.4 LangChain Embeddings Implementation

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a list of documents (batch)
doc_vectors = embeddings.embed_documents(["text one", "text two"])

# Embed a single user query
query_vector = embeddings.embed_query("What is RAG?")
```

- `embed_documents()` → used during **indexing** (batch processing)
- `embed_query()` → used during **retrieval** (single query, may apply query prefix)

---

## 🗄️ Module 4 — Vector Stores

> **Duration:** ~1 week | **Goal:** Store, search, and manage embeddings at scale

### 4.1 Vector Store Comparison and Selection

#### Free Vector Store Landscape

| Store | Type | Best For | Free Tier |
|-------|------|----------|-----------|
| **Chroma** | Embedded | Learning, prototyping | Unlimited (self-hosted) |
| **FAISS** | Library | High-performance local search | Unlimited |
| **Qdrant** | Server | Production | 1 GB cloud forever |
| **Pinecone** | Managed | Zero-ops cloud deployment | 100K vectors |
| **Milvus** | Server | Billion-scale | Unlimited (self-hosted) |
| **Weaviate** | Server | Hybrid search native | 14-day free cloud trial |

#### Selection Guide

```
Learning / Prototyping  →  Chroma or FAISS
Local high-performance  →  FAISS
Production (free tier)  →  Qdrant Cloud
Production (zero-ops)   →  Pinecone
Enterprise / Billion-scale → Milvus
Hybrid search native    →  Weaviate
```

---

### 4.2 Vector Store Operations (CRUD)

| Operation | Method | Description |
|-----------|--------|-------------|
| **Create** | `add_documents()`, `add_texts()` | Index new documents/texts |
| **Read** | `similarity_search()`, `similarity_search_with_score()` | Query the store |
| **Update** | `add_documents()` with same ID | Overwrite existing entries |
| **Delete** | `delete(ids=[...])` | Remove by ID or metadata filter |

**🧪 Lab:** Build a full CRUD vector store application with Chroma and then with Qdrant

---

## 🔎 Module 5 — Basic Retrieval Techniques

> **Duration:** ~1 week | **Goal:** Master the core retrieval methods every RAG system needs

### 5.1 Similarity Search Fundamentals

- How vector similarity search works under the hood (ANN — Approximate Nearest Neighbors)
- `similarity_search(query, k=4)` — returns top-k documents
- `similarity_search_with_score(query, k=4)` — returns documents + distance scores
- Configuring `k` and `fetch_k` parameters

**🧪 Hands-on:** Basic similarity search implementation with score inspection

---

### 5.2 Similarity Score Threshold

- Why raw top-k retrieval can return irrelevant results
- Setting a minimum relevance threshold to filter out noise
- LangChain `search_type="similarity_score_threshold"` with `score_threshold=0.75`

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75, "k": 5}
)
```

**🧪 Hands-on:** Implement threshold-based retrieval and compare result quality vs raw top-k

---

### 5.3 MMR — Maximal Marginal Relevance

- Balances **relevance** (close to query) with **diversity** (spread across topics)
- **Lambda parameter:** `0` = pure diversity, `1` = pure relevance; default `0.5`
- **`fetch_k`** = how many candidates to fetch; **`k`** = how many to return after MMR scoring

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)
```

- **Best for:** Summarization tasks, complex multi-faceted queries, avoiding repetitive context

**🧪 Hands-on:** Compare MMR vs standard similarity search on a diverse document corpus

---

### 5.4 Hybrid Search

- **Dense vectors** (semantic similarity) + **Sparse vectors** (keyword/BM25 matching)
- Pure semantic search fails on: product codes, proper nouns, abbreviations, exact terms
- Pure keyword search fails on: paraphrased queries, conceptual questions

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(docs, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # alpha: 0 = all BM25, 1 = all vector
)
```

**🧪 Lab:** Compare semantic vs keyword vs hybrid retrieval on the same document set — measure precision and recall

---

## ⚙️ Module 6 — Advanced Retrieval Techniques

> **Duration:** ~1 week | **Goal:** Solve real-world retrieval challenges with sophisticated techniques

### 6.1 Contextual Compression

**Problem:** Retrieved chunks often contain irrelevant padding around the relevant sentence.

**Solution:** Compress retrieved documents to extract only the portion relevant to the query.

| Compressor | Method | Speed |
|-----------|--------|-------|
| `LLMChainExtractor` | LLM extracts relevant sentences | Slow, highest quality |
| `EmbeddingsFilter` | Embedding similarity filter | Fast, good quality |
| `DocumentCompressorPipeline` | Chain multiple compressors | Configurable |

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

**🧪 Hands-on:** Implement a contextual compression pipeline and measure token reduction

---

### 6.2 Parent Document Retriever

**Problem:** Small chunks improve retrieval precision, but lose surrounding context for generation.

**Solution:** Index small child chunks for matching; return their large parent chunks for generation.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

**🧪 Hands-on:** Build a parent-child document system; compare context quality vs standard retrieval

---

### 6.3 Self-Query Retriever

- Parses natural language into both a **semantic query** and **metadata filter** automatically
- LLM extracts structured filter conditions from the user's question
- Define `AttributeInfo` to describe the metadata schema

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="category", description="Product category", type="string"),
    AttributeInfo(name="price", description="Price in USD", type="float"),
]

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, "Product catalog", metadata_field_info
)
# "Show me electronics under $500" → filter={"category": "electronics", "price": {"$lt": 500}}
```

**🧪 Hands-on:** Product catalog search with self-query and metadata filters

---

### 6.4 Multi-Query Retriever

- Uses an LLM to generate **multiple reformulations** of the original query
- Runs all queries in parallel and merges results (deduplicating by document ID)
- Improves recall by covering different phrasings of the same information need

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
# "What are RAG limitations?" → also generates:
# "What are the drawbacks of retrieval-augmented generation?"
# "Challenges when using RAG systems in production?"
```

**🧪 Hands-on:** Implement multi-query retrieval; measure recall improvement vs single-query

---

### 6.5 Re-ranking Strategies

**Why re-rank?** Vector similarity is fast but imprecise. A two-stage approach retrieves broadly, then re-ranks accurately.

| Stage | Method | Characteristics |
|-------|--------|----------------|
| Stage 1: Retrieve | Vector similarity | Fast, approximate |
| Stage 2: Re-rank | Cross-encoder / Cohere | Slow, precise |

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)
```

- **Cross-encoders** — compare query and document jointly; highest accuracy
- **Cohere Rerank** — managed API; easy to integrate; `rerank-english-v3.0` is current best

**🧪 Hands-on:** Add Cohere re-ranking to an existing pipeline; measure MRR improvement

---

## 🚀 Module 7 — Advanced RAG Patterns

> **Duration:** ~2 weeks | **Goal:** Implement cutting-edge RAG architectures for complex use cases

### 7.1 RAG Fusion

**Concept:** Generate multiple query variants → retrieve for each → combine results using Reciprocal Rank Fusion (RRF)

```
User Query
    │
    ▼
[LLM: Generate N query variants]
    │
    ├── Query 1 → [Retrieve top-k]
    ├── Query 2 → [Retrieve top-k]
    └── Query N → [Retrieve top-k]
                        │
                        ▼
              [Reciprocal Rank Fusion]
                        │
                        ▼
                [Unified Ranked List]
                        │
                        ▼
                  [LLM Generation]
```

- **Reciprocal Rank Fusion:** `RRF(d) = Σ 1 / (k + rank(d))` for each query list (k=60 default)
- **Trade-off:** Better comprehensiveness vs higher latency (N × retrieval calls)

**🧪 Hands-on:** Build a complete RAG Fusion pipeline with configurable N queries

---

### 7.2 HyDE — Hypothetical Document Embeddings

**Problem:** Queries and documents live in different embedding spaces (short question vs long answer).

**Solution:** Generate a hypothetical answer first → embed the answer → use that embedding for retrieval.

```
Query → [LLM: Generate hypothetical answer] → [Embed answer] → [Search]
# Result: finds real documents near the hypothetical answer's embedding
```

```python
from langchain.chains import HypotheticalDocumentEmbedder

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search"
)
retriever = vectorstore.as_retriever(embeddings=hyde_embeddings)
```

**🧪 Hands-on:** Implement HyDE with LangChain; compare retrieval quality vs standard embedding

---

### 7.3 Corrective RAG (CRAG)

**Concept:** Grade retrieved documents for relevance; fall back to web search or query rewriting when retrieval quality is poor.

```
Query → Retrieve → [Relevance Grader]
                        │
               ┌────────┴────────┐
          Relevant           Not Relevant
               │                 │
           Generate          [Rewrite Query]
                                 │
                          [Web Search Fallback]
                                 │
                             Generate
```

- Built with **LangGraph** as a stateful workflow
- Relevance grading: LLM scores each retrieved document (relevant / not relevant / ambiguous)
- Web search fallback using Tavily, SerpAPI, or DuckDuckGo

**🧪 Hands-on:** Build CRAG with LangGraph — grade documents, rewrite queries, and integrate web search

---

### 7.4 Self-RAG

**Concept:** Model generates retrieval decisions and self-critiques its own output using special reflection tokens.

- **Retrieve token** — decides whether retrieval is needed for this generation step
- **ISREL token** — grades whether retrieved passages are relevant
- **ISSUP token** — grades whether the generation is supported by the retrieved text
- **ISUSE token** — grades whether the final output is useful to the user
- **Query complexity classification** — simple factoid vs complex multi-hop

**Key insight:** Not all queries need retrieval — Self-RAG avoids unnecessary retrieval for questions the model can answer from parametric knowledge.

---

### 7.5 Graph RAG

**Concept:** Build a **knowledge graph** over your documents → query via graph traversal + vector search.

```
Documents
    │
    ▼
[Entity & Relationship Extraction]
    │
    ▼
[Neo4j Knowledge Graph]
    │        │
Graph Query  Vector Query
    │        │
    └────────┘
         │
    [Combined Results]
         │
    [LLM Generation]
```

- **Neo4j GraphRAG** — official integration with LangChain
- **Adding data:** Parse PDF → extract entities/relationships → load into Neo4j
- **Multi-hop queries:** "Who works at the company that acquired OpenAI?" (requires 2+ graph hops)
- **When to prefer Graph RAG over Vector RAG:**
  - Highly interconnected data (organizations, people, events)
  - Queries require multi-hop reasoning
  - Relationships between entities are as important as the entities themselves

**🧪 Hands-on:** Build Graph RAG with Neo4j; compare multi-hop query accuracy vs vector RAG

---

### 7.6 Multimodal RAG *(Coming Soon)*

- Extending RAG to images, tables, and mixed-content documents
- Image captioning + text embedding for unified retrieval
- Table extraction and structured retrieval
- Cross-modal retrieval strategies

---

## 🤖 Module 8 — Agentic RAG with LangGraph

> **Duration:** ~1 week | **Goal:** Build autonomous RAG agents that reason about when and how to retrieve

### 8.1 Introduction to Agentic RAG

| Aspect | Traditional RAG | Agentic RAG |
|--------|----------------|-------------|
| **Retrieval** | Always retrieves, single step | Decides whether/when to retrieve |
| **Queries** | Single query | Iterative, adaptive queries |
| **Correction** | No self-correction | Can detect and fix poor retrieval |
| **Tools** | One retriever | Multiple tools (retriever, web, calculator, etc.) |
| **Workflow** | Fixed pipeline | Dynamic, condition-based graph |

**Market significance:** Over 55% of enterprise AI projects in 2026 include agentic capabilities, with RAG as the primary knowledge source.

---

### 8.2 RAG as a Tool for Agents

```python
from langchain.tools.retriever import create_retriever_tool

rag_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(),
    name="search_company_docs",
    description="Search internal company documentation. Use this for questions about company policies, products, or procedures."
)

# Agent decides when to call this tool
agent = create_react_agent(llm, tools=[rag_tool, web_search_tool])
```

- Agent decides: retrieve vs answer directly from parametric knowledge
- Multiple knowledge base tools (e.g., "search legal docs", "search product catalog")

**🧪 Hands-on:** Build an agent with a retrieval tool; observe tool-use decision patterns

---

### 8.3 LangGraph Fundamentals for RAG

#### State Schema for Agentic RAG

```python
from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search_needed: bool
    iterations: int
```

#### RAG Graph Structure

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("rewrite_query", rewrite_query_node)

# Add conditional edges
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"generate": "generate", "rewrite": "rewrite_query", "web_search": "web_search"}
)

app = workflow.compile()
```

**🧪 Hands-on:** Build a complete agentic RAG graph from scratch with LangGraph

---

### 8.4 Agentic RAG Design Patterns

#### ReAct Pattern with RAG

```
Thought: I need to find information about X.
Action: search_docs("X")
Observation: [Retrieved chunks about X]
Thought: The documents mention Y, let me search for more.
Action: search_docs("Y related to X")
Observation: [More context]
Thought: I now have enough context to answer.
Final Answer: [Generated response with citations]
```

#### Available Patterns

| Pattern | Description | Best For |
|---------|-------------|---------|
| **ReAct** | Reason → Act → Observe loop | Open-ended Q&A, multi-hop |
| **Plan-and-Execute** | Plan all steps → execute sequentially | Complex structured tasks |
| **Reflection** | Generate → critique → refine | High-accuracy requirements |
| **CRAG** | Retrieve → grade → conditionally fix | Noisy or unreliable corpora |

**🧪 Hands-on:** Implement a ReAct RAG agent with LangGraph

---

## 📊 Module 9 — Evaluating RAG with RAGAS

> **Duration:** ~1 week | **Goal:** Rigorously measure and improve your RAG system with the RAGAS framework

### 9.1 RAG Evaluation Framework

Evaluating RAG requires measuring both **retrieval quality** and **generation quality** independently.

#### Core RAGAS Metrics

| Metric | Measures | Range |
|--------|----------|-------|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucinations) | 0–1 |
| **Answer Relevance** | Is the answer relevant to the question? | 0–1 |
| **Context Precision** | Are retrieved chunks precise and not noisy? | 0–1 |
| **Context Recall** | Did retrieval find all relevant information? | 0–1 |
| **Noise Sensitivity** | Does the system degrade when irrelevant docs are added? | 0–1 |

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    noise_sensitivity
)
from datasets import Dataset

data = {
    "question": ["What is RAG?"],
    "answer": ["RAG stands for Retrieval-Augmented Generation..."],
    "contexts": [["RAG is a framework that combines..."]],
    "ground_truth": ["RAG stands for Retrieval-Augmented Generation..."]
}

dataset = Dataset.from_dict(data)
results = evaluate(dataset, metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
])
print(results)
```

#### Creating Evaluation Datasets

- **Synthetic generation:** Use an LLM to generate Q&A pairs from your documents
- **Human annotation:** Domain experts annotate ground truth answers
- **Production traces:** Collect real user queries + feedback for evaluation

**🧪 Lab:** Evaluate each RAGAS metric individually; then run end-to-end evaluation on a full RAG pipeline

---

## 🏆 Module 10 — Capstone Project with Deployment *(Coming Soon)*

> **Duration:** ~2 weeks | **Goal:** Build and deploy a production-ready RAG system end-to-end

### 10.1 Project Assignment: Build a Production-Ready RAG System

Build a complete RAG system incorporating everything from this course:

- **Multi-document ingestion pipeline**
  - PDF, web pages, CSVs
  - Metadata enrichment and tagging
  - Incremental indexing (add new docs without reindexing)

- **Hybrid retrieval with re-ranking**
  - BM25 + vector search (EnsembleRetriever)
  - Cohere re-ranking stage
  - Contextual compression

- **Agentic workflow with LangGraph**
  - Document relevance grading
  - Web search fallback (CRAG)
  - Multi-query expansion

- **Evaluation suite with RAGAS**
  - All five metrics
  - Evaluation dataset generation
  - Regression testing

- **Monitoring with LangSmith**
  - Trace every retrieval and generation step
  - Cost tracking per query
  - Latency monitoring

- **Deployment** *(TBD)*
  - FastAPI + LangServe
  - Docker containerization
  - Cloud deployment (AWS / GCP / Azure)

---

### 10.2 Production RAG Best Practices

#### Pipeline Optimization

- **Async retrieval** — parallelize multiple retrieval calls
- **Batch embedding** — embed documents in batches, not one by one
- **Index optimization** — choose the right HNSW parameters for your dataset size

#### Caching Strategies

```python
from langchain.cache import InMemoryCache, SQLiteCache
import langchain

# Cache LLM calls (saves cost on repeated queries)
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# Semantic caching — cache similar (not just identical) queries
from langchain_community.cache import GPTCache
```

#### Cost Optimization

| Strategy | Savings |
|----------|---------|
| Use `text-embedding-3-small` instead of `text-embedding-3-large` | ~85% cheaper |
| Cache repeated LLM calls | 30–70% LLM cost reduction |
| Compress context before sending to LLM | Fewer output tokens |
| Use `gpt-4o-mini` for grading/routing tasks | 10–20x cheaper than `gpt-4o` |
| Batch indexing jobs | API call efficiency |

#### Monitoring & Debugging

- **LangSmith** — full trace of every chain step, token counts, latency
- **Log retrieval scores** — track average similarity scores to detect drift
- **Alert on low faithfulness** — set RAGAS faithfulness threshold alerts

#### Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Chunks too large → irrelevant context | Reduce chunk_size; try 256–512 tokens |
| Chunks too small → missing context | Use ParentDocumentRetriever |
| High hallucination | Add faithfulness grading; use CRAG |
| Slow retrieval | Reduce k; use FAISS instead of Chroma for large corpora |
| Missing relevant docs | Multi-query retriever; increase k and re-rank |
| High cost | Cache aggressively; use smaller embedding model |
| No source attribution | Always include `doc.metadata["source"]` in prompt |

#### Security and Compliance

- **PII in documents** — scrub before indexing or use access controls
- **Prompt injection via documents** — validate retrieved content before injecting into prompts
- **Access control** — metadata-based filtering to restrict document access by user role
- **Audit trail** — LangSmith traces provide full retrieval + generation audit logs

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher (recommended: 3.11+)
- pip package manager
- OpenAI API key (or other LLM provider)
- Optional: Cohere API key (re-ranking), Tavily API key (web search)

### Installation

1. **Clone this repository:**

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Roadmap.git
cd RAG-Roadmap
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**

```bash
cat > .env << EOF
# LLM Providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Vector Stores (optional)
PINECONE_API_KEY=your_pinecone_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url

# Re-ranking
COHERE_API_KEY=your_cohere_api_key

# Web Search Fallback
TAVILY_API_KEY=your_tavily_api_key

# Observability
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-roadmap
EOF
```

5. **Verify installation:**

```bash
python -c "import langchain; import langchain_openai; print('Setup complete!')"
```

---

## 📁 Project Structure

```
RAG-Roadmap/
│
├── module_01_fundamentals/          # RAG Fundamentals and Architecture
│   ├── 01_what_is_rag.ipynb        # RAG concepts and data flow
│   ├── 02_rag_vs_finetuning.ipynb  # Comparison framework
│   └── notes.md                    # Module notes
│
├── module_02_document_processing/   # Document Processing and Chunking
│   ├── 01_document_loaders.ipynb   # 5-source hands-on loading
│   ├── 02_text_splitters.ipynb     # All splitting strategies
│   ├── 03_chunking_lab.ipynb       # Optimization lab
│   ├── 04_metadata_pipeline.ipynb  # Metadata enrichment
│   └── notes.md
│
├── module_03_embeddings/            # Embeddings and Vector Representations
│   ├── 01_how_embeddings_work.ipynb
│   ├── 02_embedding_models.ipynb   # OpenAI, Ollama, HuggingFace
│   ├── 03_model_comparison.ipynb   # MTEB evaluation
│   ├── 04_langchain_embeddings.ipynb
│   └── notes.md
│
├── module_04_vector_stores/         # Vector Stores
│   ├── 01_chroma_basics.ipynb
│   ├── 02_faiss_basics.ipynb
│   ├── 03_qdrant_basics.ipynb
│   ├── 04_pinecone_basics.ipynb
│   ├── 05_crud_application.ipynb   # Full CRUD lab
│   └── notes.md
│
├── module_05_basic_retrieval/       # Basic Retrieval Techniques
│   ├── 01_similarity_search.ipynb
│   ├── 02_score_threshold.ipynb
│   ├── 03_mmr_retrieval.ipynb
│   ├── 04_hybrid_search.ipynb
│   └── notes.md
│
├── module_06_advanced_retrieval/    # Advanced Retrieval Techniques
│   ├── 01_contextual_compression.ipynb
│   ├── 02_parent_document_retriever.ipynb
│   ├── 03_self_query_retriever.ipynb
│   ├── 04_multi_query_retriever.ipynb
│   ├── 05_reranking.ipynb
│   └── notes.md
│
├── module_07_advanced_patterns/     # Advanced RAG Patterns
│   ├── 01_rag_fusion.ipynb
│   ├── 02_hyde.ipynb
│   ├── 03_corrective_rag.ipynb     # CRAG with LangGraph
│   ├── 04_self_rag.ipynb
│   ├── 05_graph_rag.ipynb          # Neo4j GraphRAG
│   └── notes.md
│
├── module_08_agentic_rag/           # Agentic RAG with LangGraph
│   ├── 01_intro_agentic_rag.ipynb
│   ├── 02_rag_as_tool.ipynb
│   ├── 03_langgraph_fundamentals.ipynb
│   ├── 04_react_rag_agent.ipynb
│   └── notes.md
│
├── module_09_evaluation/            # Evaluating RAG with RAGAS
│   ├── 01_ragas_metrics.ipynb      # Each metric individually
│   ├── 02_evaluation_dataset.ipynb # Dataset generation
│   ├── 03_pipeline_evaluation.ipynb # End-to-end eval
│   └── notes.md
│
├── module_10_capstone/              # Capstone Project (Coming Soon)
│   ├── README.md
│   └── project_spec.md
│
├── resources/                       # Curated learning resources
│   ├── papers.md                   # Key research papers
│   ├── courses.md                  # Recommended courses
│   └── tools.md                   # Tools and platforms
│
├── requirements.txt                 # All Python dependencies
├── .env.example                    # Environment variables template
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## Code Examples

### End-to-End Basic RAG Pipeline

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load documents
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed and store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. Build RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
Cite your sources using [Source: filename, page X] format.

Context:
{context}

Question: {question}

Answer:""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query
response = rag_chain.invoke("What are the main findings?")
print(response)
```

---

### Hybrid Search with EnsembleRetriever

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Dense retriever (semantic)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Sparse retriever (keyword / BM25)
bm25_retriever = BM25Retriever.from_documents(chunks, k=5)

# Hybrid: 60% semantic, 40% keyword
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
)

results = hybrid_retriever.invoke("What is the revenue in Q3?")
```

---

### Agentic RAG with LangGraph (CRAG Pattern)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: bool

def retrieve(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def grade_documents(state: RAGState) -> RAGState:
    # Grade each document for relevance
    relevant_docs = [doc for doc in state["documents"]
                     if relevance_grader.invoke({
                         "question": state["question"],
                         "document": doc.page_content
                     })["score"] == "yes"]
    web_search = len(relevant_docs) < 2
    return {"documents": relevant_docs, "web_search": web_search}

def decide_to_generate(state: RAGState) -> str:
    return "web_search" if state["web_search"] else "generate"

def web_search(state: RAGState) -> RAGState:
    results = tavily_client.search(state["question"])
    web_docs = [Document(page_content=r["content"]) for r in results["results"]]
    return {"documents": state["documents"] + web_docs}

def generate(state: RAGState) -> RAGState:
    answer = rag_chain.invoke({
        "question": state["question"],
        "context": "\n\n".join(d.page_content for d in state["documents"])
    })
    return {"generation": answer}

# Build the graph
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate,
                               {"web_search": "web_search", "generate": "generate"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
result = app.invoke({"question": "What is corrective RAG?"})
print(result["generation"])
```

---

### RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Build evaluation dataset
eval_data = {
    "question": ["What is the capital of France?", "Explain RAG in simple terms."],
    "answer": ["Paris is the capital of France.", "RAG retrieves relevant documents..."],
    "contexts": [
        ["France is a country in Europe. Its capital is Paris."],
        ["RAG stands for Retrieval-Augmented Generation. It combines..."]
    ],
    "ground_truth": [
        "The capital of France is Paris.",
        "RAG is a technique that retrieves relevant information..."
    ]
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(results.to_pandas())
```

---

## 📖 Top Resources

### Essential Research Papers

| Paper | Year | Why Read It |
|-------|------|-------------|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | The original RAG paper by Facebook AI |
| [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) | 2020 | Foundational retrieval-integrated pretraining |
| [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) | 2023 | Self-reflective RAG with special tokens |
| [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | 2024 | CRAG — relevance grading + web fallback |
| [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) | 2024 | GraphRAG by Microsoft Research |
| [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496) | 2022 | Hypothetical Document Embeddings |
| [RAGAS: Automated Evaluation of RAG Pipelines](https://arxiv.org/abs/2309.15217) | 2023 | RAGAS evaluation framework |
| [Engineering the RAG Stack (2026)](https://arxiv.org/pdf/2601.05264) | 2026 | Comprehensive 2026 RAG architecture review |

---

### Official Documentation

- 📘 [LangChain RAG Docs](https://python.langchain.com/docs/tutorials/rag/)
- 📘 [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- 📘 [LangSmith Platform](https://smith.langchain.com/)
- 📘 [RAGAS Documentation](https://docs.ragas.io/)
- 📘 [Chroma Documentation](https://docs.trychroma.com/)
- 📘 [Qdrant Documentation](https://qdrant.tech/documentation/)
- 📘 [Pinecone Documentation](https://docs.pinecone.io/)
- 📘 [Neo4j GraphRAG Python Docs](https://neo4j.com/docs/neo4j-graphrag-python/current/)

---

### Recommended Courses

| Course | Platform | Level | Free? |
|--------|----------|-------|-------|
| [LangChain: Chat with Your Data](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/) | DeepLearning.AI | Beginner | ✅ |
| [Building and Evaluating Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/) | DeepLearning.AI | Intermediate | ✅ |
| [Advanced Retrieval for AI with Chroma](https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/) | DeepLearning.AI | Intermediate | ✅ |
| [Knowledge Graphs for RAG](https://learn.deeplearning.ai/courses/knowledge-graphs-rag/) | DeepLearning.AI | Intermediate | ✅ |
| [LangChain Academy — RAG Track](https://academy.langchain.com/) | LangChain | All Levels | Partial |
| [Vector Databases: from Embeddings to Applications](https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications/) | DeepLearning.AI | Beginner | ✅ |

---

### YouTube Channels & Playlists

- 🎥 [LangChain Official](https://www.youtube.com/@LangChain)
- 🎥 [DeepLearning.AI](https://www.youtube.com/@Deeplearningai)
- 🎥 [Sam Witteveen — RAG deep dives](https://www.youtube.com/@samwitteveenai)
- 🎥 [AI Jason — LangGraph tutorials](https://www.youtube.com/@AIJasonZ)
- 🎥 [Prompt Engineering](https://www.youtube.com/@engineerprompt)

---

### Tools and Platforms

| Tool | Purpose | Cost |
|------|---------|------|
| [LangSmith](https://smith.langchain.com/) | Tracing, observability, evaluation | Free tier |
| [Chroma](https://www.trychroma.com/) | Local vector store | Free (self-hosted) |
| [Qdrant Cloud](https://cloud.qdrant.io/) | Production vector store | 1 GB free forever |
| [Pinecone](https://www.pinecone.io/) | Managed vector store | 100K vectors free |
| [Cohere Rerank](https://cohere.com/rerank) | Re-ranking API | Free trial |
| [Tavily](https://tavily.com/) | Web search for RAG agents | Free tier |
| [RAGAS](https://ragas.io/) | RAG evaluation | Open source |
| [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | Embedding model benchmarks | Free |
| [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) | Graph database for Graph RAG | Free tier |

---

## 🤝 Contributing

Contributions are welcome and encouraged! If you'd like to add notes, fix errors, add new hands-on notebooks, or expand any module:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/module-X-improvement`
3. Make your changes and add clear commit messages: `git commit -m 'Add: Self-RAG notebook for Module 7'`
4. Push to the branch: `git push origin feature/module-X-improvement`
5. Open a Pull Request with a description of what you changed

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- Website: [Adil Shamim](https://adilshamim.me/)
- GitHub: [Adil Shamim](https://github.com/AdilShamim8)
- Create an issue in this repository for questions or suggestions

---

<p align="center">
  <a href="https://github.com/AdilShamim8">
    <img src="https://img.shields.io/badge/GitHub-AdilShamim8-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://www.linkedin.com/in/adilshamim8">
    <img src="https://img.shields.io/badge/LinkedIn-AdilShamim8-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://www.kaggle.com/adilshamim8">
    <img src="https://img.shields.io/badge/Kaggle-AdilShamim8-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://x.com/adil_shamim8">
    <img src="https://img.shields.io/badge/Twitter%2FX-@adil__shamim8-000000?style=for-the-badge&logo=x&logoColor=white" alt="Twitter/X Profile"/>
  </a>
  <span style="opacity:.6">&nbsp;</span>

  <a href="https://adilshamim8.medium.com/">
    <img src="https://img.shields.io/badge/Medium-AdilShamim8-12100E?style=for-the-badge&logo=medium&logoColor=white" alt="Medium Profile"/>
  </a>
</p>
<div align="center">
  
⭐ **If you find this repository helpful, please consider giving it a star!** ⭐

</div>
