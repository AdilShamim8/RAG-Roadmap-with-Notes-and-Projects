<div align="center">

# RAG Roadmap with Notes and Projects

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green?logo=langchain)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange)](https://www.langchain.com/langgraph)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-purple)](https://docs.ragas.io/)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/RAG-Roadmap?style=social)](https://github.com/YOUR_USERNAME/RAG-Roadmap)

**A structured, end-to-end curriculum for Retrieval-Augmented Generation —**  
**from core concepts and document processing to production-ready Agentic RAG systems.**

[📚 Start Learning](#-module-1--rag-fundamentals-and-architecture) • [🗺️ Module Map](#️-module-map) • [⚙️ Setup](#️-setup--installation) • [🤝 Contributing](#-contributing)

</div>

---

## 📖 About This Repository

This repository provides a structured learning path for developers who want to master **Retrieval-Augmented Generation (RAG)** end-to-end. It contains curated notes, hands-on labs, code examples, and implementation guides that cover the complete RAG stack — from document ingestion and embeddings to advanced retrieval patterns, agentic workflows with LangGraph, and production deployment.

Whether you are new to RAG or looking to move from prototype to production, this roadmap gives you a clear, step-by-step path.

> ✅ **Covers:** Document Processing · Embeddings · Vector Stores · Advanced Retrieval · RAG Fusion · HyDE · CRAG · Self-RAG · Graph RAG · Agentic RAG · RAGAS Evaluation · Capstone Deployment

---

## ✅ Prerequisites

Before starting this curriculum, you should be comfortable with:

| Prerequisite | Why It Matters |
|---|---|
| 🐍 Python Fundamentals | All code examples are in Python |
| 🤖 GenAI Fundamentals | Core LLM concepts underpin RAG |
| 🦜 LangChain | Used for loaders, retrievers, chains |
| 🕸️ LangGraph | Used for agentic RAG workflows |

---

## 🗺️ Module Map

```
RAG Roadmap
│
├── Module 1  ──  RAG Fundamentals and Architecture
├── Module 2  ──  Document Processing and Chunking
├── Module 3  ──  Embeddings and Vector Representations
├── Module 4  ──  Vector Stores
├── Module 5  ──  Basic Retrieval Techniques
├── Module 6  ──  Advanced Retrieval Techniques
├── Module 7  ──  Advanced RAG Patterns
├── Module 8  ──  Agentic RAG with LangGraph
├── Module 9  ──  Evaluating RAG with RAGAS
└── Module 10 ──  Capstone Project with Deployment
```

---

## 📚 Module 1 — RAG Fundamentals and Architecture

### 1.1 Introduction to RAG

#### 1.1.1 What is RAG and the Problems It Solves
- 🛡️ **Hallucination reduction** — grounding outputs in retrieved facts
- 📅 **Knowledge cutoff limitations** — injecting up-to-date external knowledge
- 🏢 **Domain-specific knowledge injection** — querying private/proprietary data
- 🔗 **Source attribution and transparency** — traceable, cite-able answers

#### 1.1.2 Core Components
```
Knowledge Base  ──►  Retriever  ──►  Generator  ──►  Response
```

#### 1.1.3 RAG Data Flow
- End-to-end flow from user query → document retrieval → context injection → generation

---

### 1.2 RAG vs Fine-Tuning vs Prompt Engineering

| Approach | Cost | Latency | Accuracy | Maintenance |
|---|---|---|---|---|
| Prompt Engineering | 💚 Low | 💚 Fast | 🟡 Limited | 💚 Easy |
| RAG | 🟡 Medium | 🟡 Medium | 💚 High | 🟡 Moderate |
| Fine-Tuning | 🔴 High | 💚 Fast | 💚 High | 🔴 Complex |

> 📌 **Decision Framework:** Use prompt engineering for simple tasks, RAG when you need up-to-date/domain-specific knowledge, and fine-tuning when you need style/behavior adaptation at scale.

---

## 📄 Module 2 — Document Processing and Chunking

### 2.1 Document Loaders in LangChain

| Loader Type | Examples |
|---|---|
| 📕 PDF | `PyPDFLoader`, `PDFMiner`, `PDFPlumber`, `Unstructured` |
| 🌐 Web | `WebBaseLoader`, `RecursiveUrlLoader` |
| 🗄️ Structured | `CSVLoader`, `JSONLoader` |
| 📝 Text | `TextLoader` |

> 🧪 **Hands-on:** Load documents from 5 different sources

---

### 2.2 Text Splitting Strategies

| Splitter | Best For |
|---|---|
| `RecursiveCharacterTextSplitter` | Recommended default — hierarchical separator logic |
| `CharacterTextSplitter` | Simple delimiter-based splitting |
| `TokenTextSplitter` | Token-aware splitting for LLM context windows |
| `SemanticChunker` | Semantic breakpoints (percentile / std_deviation / gradient) |
| `MarkdownTextSplitter` | Structured markdown documents |
| Code Splitters | Python, JavaScript, and more |
| LLM-Based Splitting | Intelligent boundary detection using an LLM |

> 🧪 **Hands-on:** Compare chunking strategies on the same document

---

### 2.3 Chunking Best Practices

| Use Case | Recommended Chunk Size | Overlap |
|---|---|---|
| Factoid / Q&A | 128–256 tokens | 10–20% |
| General RAG | 256–512 tokens | 10–20% |
| Complex Analysis | 512–1024 tokens | 15–20% |

- How chunk size directly affects retrieval quality
- Handling tables and structured data inside chunks
- Preserving cross-chunk context

> 🧪 **Lab:** Optimize chunking for a PDF technical document

---

### 2.4 Metadata Management

- Adding rich metadata to `Document` objects
- Metadata filtering during retrieval
- Source tracking and document lineage
- Document versioning strategies

> 🧪 **Hands-on:** Build a metadata-enriched document ingestion pipeline

---

## 🔢 Module 3 — Embeddings and Vector Representations

### 3.1 How Embeddings Work
- Vector representations of text in high-dimensional space
- Semantic similarity in embedding space
- Distance metrics: **cosine similarity**, **Euclidean**, **dot product**
- Embedding dimensions and their trade-offs

---

### 3.2 Embedding Models Overview

| Provider | Model | Cost | Notes |
|---|---|---|---|
| OpenAI | `text-embedding-3-small` | $0.02 / 1M tokens | ✅ Best value |
| OpenAI | `text-embedding-3-large` | $0.13 / 1M tokens | 🏆 Highest quality |
| Ollama (local) | `Gemma` (via `ollama` lib) | Free | Privacy-first |
| Ollama (local) | `Gemma` (via `langchain-ollama`) | Free | LangChain native |

---

### 3.3 Choosing the Right Embedding Model
- MTEB leaderboard — what metrics to look for
- Dimension trade-offs: `384 → 768 → 1536 → 3072`
- Cost vs quality decision framework

> 🧪 **Hands-on:** Compare embedding models on sample queries

---

### 3.4 LangChain Embeddings Implementation
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a batch of documents
doc_vectors = embeddings.embed_documents(["doc 1 text", "doc 2 text"])

# Embed a single query
query_vector = embeddings.embed_query("What is RAG?")
```

---

## 🗃️ Module 4 — Vector Stores

### 4.1 Vector Store Comparison and Selection

| Store | Type | Best For | Free Tier |
|---|---|---|---|
| **Chroma** | Embedded | Learning, prototyping | ✅ Unlimited (self-hosted) |
| **FAISS** | Library | High-performance local search | ✅ Unlimited |
| **Qdrant** | Server | Production workloads | ✅ 1 GB cloud forever |
| **Pinecone** | Managed | Zero-ops deployment | ✅ 100K vectors |
| **Milvus** | Server | Billion-scale search | ✅ Unlimited (self-hosted) |

---

### 4.2 Vector Store Operations (CRUD)

| Operation | Description |
|---|---|
| ➕ **Create** | `add_documents()`, `add_texts()` |
| 🔍 **Read** | `similarity_search()`, `similarity_search_with_score()` |
| ✏️ **Update** | Modify existing documents by ID |
| 🗑️ **Delete** | Remove documents by ID or metadata filter |

> 🧪 **Lab:** Build a full CRUD vector store application

---

## 🔍 Module 5 — Basic Retrieval Techniques

### 5.1 Similarity Search Fundamentals
- How vector similarity search works under the hood
- `similarity_search()` method and `k` parameter configuration
- Returning relevance scores alongside results

> 🧪 **Hands-on:** Basic similarity search implementation

---

### 5.2 Similarity Score Threshold
- Why thresholds matter for quality control (filtering noise)
- Setting appropriate thresholds per use case
- LangChain `similarity_score_threshold` search type

> 🧪 **Hands-on:** Implement threshold-based retrieval

---

### 5.3 MMR — Maximal Marginal Relevance

| Parameter | Effect |
|---|---|
| `lambda = 0` | Maximum diversity |
| `lambda = 1` | Maximum relevance |
| `fetch_k` | Candidates to retrieve before MMR re-ranking |
| `k` | Final documents returned |

> 🧪 **Hands-on:** Compare MMR vs standard similarity search

---

### 5.4 Hybrid Search
- Dense vectors (semantic) + Sparse vectors (BM25 keyword)
- When hybrid search outperforms pure semantic search
- `BM25Retriever` in LangChain
- Weighting strategies with the `alpha` parameter
- `EnsembleRetriever` for combining retrievers

> 🧪 **Lab:** Compare semantic vs keyword vs hybrid retrieval on the same document set

---

## 🚀 Module 6 — Advanced Retrieval Techniques

### 6.1 Contextual Compression
- **Why compress?** — Retrieved chunks often contain irrelevant filler text
- `LLMChainExtractor` — LLM extracts only the relevant passage
- `EmbeddingsFilter` — Fast embedding-based relevance filtering
- `DocumentCompressorPipeline` — Chain multiple compressors together

> 🧪 **Hands-on:** Implement a contextual compression pipeline

---

### 6.2 Parent Document Retriever
- ⚖️ The chunk-size dilemma: small chunks for precise matching, large chunks for full context
- **Child chunks** used for retrieval → **parent chunks** returned as context
- `InMemoryStore` and `docstore` configuration

> 🧪 **Hands-on:** Build a parent-child document retrieval system

---

### 6.3 Self-Query Retriever
- Natural language → structured metadata filter queries
- `AttributeInfo` schema definition
- Automatic filter extraction from user queries

> 🧪 **Hands-on:** Product catalog search with self-query retriever

---

### 6.4 Multi-Query Retriever
- Query expansion using an LLM to generate query variations
- Improving recall by retrieving across multiple query perspectives

> 🧪 **Hands-on:** Implement multi-query retrieval

---

### 6.5 Re-Ranking Strategies
- Two-stage pipeline: **retrieve → re-rank**
- Cross-encoder re-rankers for higher precision
- Cohere Rerank API integration
- RAG Fusion re-ranking (Reciprocal Rank Fusion)

> 🧪 **Hands-on:** Add re-ranking to an existing RAG pipeline

---

## ⚡ Module 7 — Advanced RAG Patterns

### 7.1 RAG Fusion
```
Query ──► LLM generates N query variations
        ──► Retrieve for each variation
        ──► Reciprocal Rank Fusion (RRF)
        ──► Final ranked document set ──► Generator
```
- Trade-offs: comprehensiveness vs latency

> 🧪 **Hands-on:** Build a RAG Fusion pipeline from scratch

---

### 7.2 HyDE — Hypothetical Document Embeddings
```
User Query ──► LLM generates hypothetical answer
           ──► Embed hypothetical answer
           ──► Search vector store with hypothesis embedding
           ──► Retrieve real documents ──► Generator
```
- Bridges the semantic gap between short queries and long documents

> 🧪 **Hands-on:** Implement HyDE with LangChain

---

### 7.3 Corrective RAG (CRAG)
- Document relevance grading after initial retrieval
- Web search fallback mechanism when retrieval quality is low
- Query rewriting when retrieval fails

> 🧪 **Hands-on:** Build CRAG with LangGraph

---

### 7.4 Self-RAG
- Reflection tokens and self-correction loops
- Query complexity classification before retrieval
- Dynamic retrieval — retrieve only when needed

---

### 7.5 Graph RAG
- Knowledge graphs combined with vector RAG
- Neo4j GraphRAG approach
- Adding data to a graph database from PDF documents
- Multi-hop query handling
- When Graph RAG outperforms vector RAG

---

### 7.6 Multi-Modal RAG *(Coming Soon)*
- Extending RAG to images, tables, and mixed-content documents
- Vision-language models for document understanding

---

## 🤖 Module 8 — Agentic RAG with LangGraph

### 8.1 Introduction to Agentic RAG

| | Traditional RAG | Agentic RAG |
|---|---|---|
| Query handling | Single-pass | Multi-step reasoning |
| Retrieval | Fixed pipeline | Dynamic tool use |
| Error handling | None | Self-correction loops |
| Flexibility | Low | High |

---

### 8.2 RAG as a Tool for Agents
- Creating retriever tools with `create_retriever_tool`
- When agents should retrieve vs respond directly from parametric memory
- Multiple knowledge base tools for multi-domain agents

> 🧪 **Hands-on:** Build an agent with a retrieval tool

---

### 8.3 LangGraph Fundamentals for RAG

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class RAGState(TypedDict):
    query: str
    documents: list
    generation: str

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge("retrieve", "generate")
```

- State, nodes, edges, and conditional routing
- State schemas for RAG using `TypedDict`

> 🧪 **Hands-on:** Build an agentic RAG graph from scratch

---

### 8.4 Agentic RAG Design Patterns

| Pattern | Description |
|---|---|
| ⚛️ **ReAct** | Reason + Act loop with retrieval tool |
| 📋 **Plan-and-Execute** | Multi-step planning before retrieval |
| 🔄 **Reflection / Self-Correction** | Critique and revise generated answers |

> 🧪 **Hands-on:** Implement a ReAct RAG agent

---

## 📊 Module 9 — Evaluating RAG with RAGAS

### 9.1 Core Evaluation Metrics

| Metric | What It Measures |
|---|---|
| 🎯 **Faithfulness** | Is the answer grounded in the retrieved context? |
| 🔍 **Answer Relevance** | Does the answer address the user's question? |
| 📐 **Context Precision** | Are retrieved chunks relevant? |
| 📦 **Context Recall** | Are all necessary chunks retrieved? |
| 🔊 **Noise Sensitivity** | How much does irrelevant context hurt the answer? |

---

### 9.2 RAGAS Implementation
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

> 🧪 **Lab:** Evaluate each metric individually, then run a full evaluation suite on your RAG pipeline

---

## 🏗️ Module 10 — Capstone Project with Deployment *(Coming Soon)*

### 10.1 Project: Build a Production-Ready RAG System

- [ ] Multi-document ingestion pipeline
- [ ] Hybrid retrieval with re-ranking
- [ ] Agentic workflow with LangGraph
- [ ] Evaluation suite with RAGAS
- [ ] Monitoring with LangSmith
- [ ] Deployment (TBD)

---

### 10.2 Production RAG Best Practices

| Area | Topics |
|---|---|
| ⚡ **Pipeline Optimization** | Async retrieval, batching, parallelism |
| 💾 **Caching Strategies** | Semantic caching, embedding cache |
| 💰 **Cost Optimization** | Model selection, token budgets, reuse |
| 🔭 **Monitoring & Debugging** | LangSmith tracing, logging, alerting |
| ⚠️ **Common Pitfalls** | Chunk too small/large, missing metadata, stale index |
| 🔐 **Security & Compliance** | PII filtering, access control, data governance |

---

## 📁 Repository Structure

```
RAG-Roadmap/
│
├── module_01_fundamentals/
│   ├── intro_to_rag.ipynb
│   └── rag_vs_finetuning.ipynb
│
├── module_02_document_processing/
│   ├── document_loaders.ipynb
│   ├── text_splitting.ipynb
│   ├── chunking_best_practices.ipynb
│   └── metadata_management.ipynb
│
├── module_03_embeddings/
│   ├── how_embeddings_work.ipynb
│   ├── embedding_models_overview.ipynb
│   └── langchain_embeddings.ipynb
│
├── module_04_vector_stores/
│   ├── vector_store_comparison.ipynb
│   └── crud_operations.ipynb
│
├── module_05_basic_retrieval/
│   ├── similarity_search.ipynb
│   ├── score_threshold.ipynb
│   ├── mmr_retrieval.ipynb
│   └── hybrid_search.ipynb
│
├── module_06_advanced_retrieval/
│   ├── contextual_compression.ipynb
│   ├── parent_document_retriever.ipynb
│   ├── self_query_retriever.ipynb
│   ├── multi_query_retriever.ipynb
│   └── reranking.ipynb
│
├── module_07_advanced_patterns/
│   ├── rag_fusion.ipynb
│   ├── hyde.ipynb
│   ├── corrective_rag.ipynb
│   ├── self_rag.ipynb
│   └── graph_rag.ipynb
│
├── module_08_agentic_rag/
│   ├── intro_agentic_rag.ipynb
│   ├── rag_as_tool.ipynb
│   ├── langgraph_fundamentals.ipynb
│   └── react_rag_agent.ipynb
│
├── module_09_evaluation/
│   ├── ragas_metrics.ipynb
│   └── full_evaluation_pipeline.ipynb
│
├── module_10_capstone/         # Coming Soon
│   └── README.md
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/RAG-Roadmap.git
cd RAG-Roadmap
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```env
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
```

### 5. Launch Jupyter
```bash
jupyter notebook
```

---

## 📦 Key Dependencies

```txt
langchain
langchain-openai
langchain-community
langchain-ollama
langgraph
chromadb
faiss-cpu
qdrant-client
ragas
sentence-transformers
pypdf
unstructured
tiktoken
python-dotenv
jupyter
```

---

## 📈 Learning Path Recommendation

```
Week 1  ──  Modules 1–2   (Fundamentals + Document Processing)
Week 2  ──  Modules 3–4   (Embeddings + Vector Stores)
Week 3  ──  Modules 5–6   (Basic + Advanced Retrieval)
Week 4  ──  Module 7      (Advanced RAG Patterns)
Week 5  ──  Module 8      (Agentic RAG with LangGraph)
Week 6  ──  Modules 9–10  (Evaluation + Capstone)
```

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how to get involved:

1. 🍴 **Fork** this repository
2. 🌿 **Create** a feature branch: `git checkout -b feature/your-feature`
3. 💾 **Commit** your changes: `git commit -m "Add: your feature description"`
4. 📤 **Push** to your branch: `git push origin feature/your-feature`
5. 🔁 **Open** a Pull Request

Please ensure your notebooks:
- Run cleanly from top to bottom
- Include markdown explanations alongside code
- Follow the existing module naming conventions

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

If this roadmap helps you, consider giving it a ⭐ — it helps others find it too!

---

<div align="center">

**Built for learners. Structured for engineers. Ready for production.**

[![Follow on GitHub](https://img.shields.io/github/followers/YOUR_USERNAME?style=social)](https://github.com/YOUR_USERNAME)

</div>
