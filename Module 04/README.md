# Module 4: Vector Stores 

## Where Your Embeddings Live and Get Searched

---

> **Before I explain — let me ask:**
>
> *"What do you think a vector store is?"*
>
> A beginner might say: *"It's a database that stores vectors."*
>
> That's correct but misses the magic: **a vector store is a database optimized for finding similar vectors at lightning speed** — even when you have billions of them.
>
> Searching 1 billion vectors with brute force = hours.
> Searching 1 billion vectors with a vector store = milliseconds.
>
> **That speed difference is the entire reason vector stores exist.**

---

# The Big Picture: Why Vector Stores Matter

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  THE FUNDAMENTAL PROBLEM:                                        │
│                                                                  │
│  You have:    1 million document chunks                          │
│  Each chunk:  1536-dimensional vector                            │
│  User query:  1 vector                                           │
│                                                                  │
│  Task: Find top 5 most similar vectors out of 1 million          │
│                                                                  │
│  Naive approach: Compare query to every single vector             │
│  → 1 million cosine similarity calculations                       │
│  → Each calc = 1536 multiplications                              │
│  → Total: 1.5 BILLION operations                                 │
│  → Time: Several seconds (too slow!)                              │
│                                                                  │
│  Vector store approach: Use special index (HNSW, IVF)            │
│  → Time: ~10 milliseconds                                         │
│  → 100-1000x faster!                                              │
│                                                                  │
│  THIS SPEEDUP = THE ENTIRE POINT OF VECTOR STORES                │
└──────────────────────────────────────────────────────────────────┘
```

---

# PART 1: What is a Vector Store?

---

## 1. First Principles: Database vs Vector Store

### Traditional Database

```
SQL Database query:
SELECT * FROM users WHERE age = 25;

→ Looks for EXACT matches
→ Uses indexes like B-trees
→ Optimized for: equality, ranges, sorting
```

### Vector Store

```
Vector store query:
"Find documents similar to this query vector"

→ Looks for APPROXIMATE matches (similar, not equal)
→ Uses indexes like HNSW, IVF (different math!)
→ Optimized for: nearest neighbor search in high dimensions
```

**Key insight:** Vector stores answer a fundamentally different question:
- SQL: *"Find the rows where X equals Y"*
- Vector Store: *"Find the rows most similar to Y"*

This requires completely different data structures and algorithms.

---

## 2. How Vector Stores Work (Simplified)

### The Brute Force Problem

```
You have 1 million 1536-dimensional vectors.
Query comes in as 1 vector.

NAIVE APPROACH:
for each of 1,000,000 stored vectors:
    calculate cosine_similarity(query, stored_vector)
sort by similarity
return top 5

Cost: O(N × D) where N=1M, D=1536
     = 1.5 billion operations
     = ~5-10 seconds  ❌ TOO SLOW

You need to respond in milliseconds, not seconds.
```

### The Solution: Approximate Nearest Neighbor (ANN) Indexes

Vector stores use clever data structures to **approximate** the nearest neighbors without comparing to every vector:

```
SMART APPROACH (HNSW algorithm):
1. Build a graph where similar vectors are connected
2. Start at a random vector
3. Move to neighbor that's closer to query
4. Repeat until you can't get closer
5. Return that vector (and its neighbors)

Cost: O(log N) instead of O(N)
     For 1M vectors: ~20 operations instead of 1M!
     Result: 10-50ms instead of 5-10 seconds
```

**The tradeoff:**
```
Brute force:     100% accurate but SLOW
ANN (HNSW):     ~95-99% accurate but FAST

This 1-5% accuracy loss = acceptable for retrieval
Because semantic search is already approximate anyway!
```

> Don't worry about understanding HNSW deeply yet — just know:
> **Vector stores use special algorithms to find similar vectors FAST.**

---

# PART 2: The Free Vector Store Landscape

---

## 3. Reading the Comparison Table

Let me explain each store from your image with deep understanding:

```
┌──────────┬──────────┬──────────────────────┬─────────────────────────┐
│ Store    │ Type     │ Best For             │ Free Tier               │
├──────────┼──────────┼──────────────────────┼─────────────────────────┤
│ Chroma   │ Embedded │ Learning, prototyping│ Unlimited (self-hosted) │
│ FAISS    │ Library  │ High-perf. search    │ Unlimited               │
│ Qdrant   │ Server   │ Production           │ 1GB cloud forever       │
│ Pinecone │ Managed  │ Zero-ops deployment  │ 100K vectors            │
│ Milvus   │ Server   │ Billion-scale        │ Unlimited (self-hosted) │
└──────────┴──────────┴──────────────────────┴─────────────────────────┘
```

### Understanding the "Type" Column

```
EMBEDDED:
  Runs INSIDE your Python process
  No separate server needed
  Like SQLite for vectors
  Pros: Zero setup, easy to start
  Cons: Doesn't scale across machines
  Best for: Prototypes, small apps, learning

LIBRARY:
  A Python library you import
  Stores vectors in RAM (or disk files)
  No database server
  Pros: Maximum speed, minimal overhead
  Cons: You manage persistence yourself
  Best for: Research, custom pipelines

SERVER:
  Runs as a separate process/container
  You connect over network (HTTP/gRPC)
  Like PostgreSQL but for vectors
  Pros: Multi-app access, production features
  Cons: More setup, infrastructure to manage
  Best for: Production apps, team projects

MANAGED:
  Cloud service (someone else runs it)
  You just use an API
  Like AWS RDS but for vectors
  Pros: Zero ops, scales automatically
  Cons: Costs money beyond free tier, vendor lock-in
  Best for: Production with limited DevOps capacity
```

---

## 4. Deep Dive: Each Vector Store

---

### 🟢 Chroma — The Learning Default

```
WHAT IT IS:
  Open-source embedded vector database
  Designed specifically for LLM applications
  Easy-first API, no infrastructure needed

WHY START HERE:
  ✅ Zero setup — pip install and go
  ✅ Persists to disk automatically
  ✅ Excellent LangChain integration
  ✅ Built-in embedding function support
  ✅ Free forever (open source)

WHEN TO USE:
  ✅ Learning RAG fundamentals
  ✅ Prototypes and POCs
  ✅ Small to medium applications (< 1M vectors)
  ✅ Single-machine deployment
  ✅ Local development

WHEN NOT TO USE:
  ❌ Need to share across multiple servers
  ❌ Very high concurrent query load
  ❌ Massive scale (> 10M vectors)
```

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store (persists to disk)
vectorstore = Chroma(
    collection_name="my_documents",       # Like a "table" name
    embedding_function=embeddings,
    persist_directory="./chroma_db",       # Saves to disk
)

# That's it — you're ready to use it!
# No servers, no configuration files, no setup
```

---

### 🔵 FAISS — The Speed Champion

```
WHAT IT IS:
  Facebook AI Similarity Search library
  Pure C++ optimization for maximum speed
  Multiple indexing algorithms built-in

WHY USE IT:
  ✅ Fastest similarity search available
  ✅ GPU support for massive speedups
  ✅ Many index types (Flat, IVF, HNSW, PQ)
  ✅ Battle-tested at Meta/Facebook scale
  ✅ Completely free, no service needed

WHEN TO USE:
  ✅ Need maximum search speed
  ✅ Research and experimentation
  ✅ Custom RAG pipelines
  ✅ GPU-accelerated search
  ✅ Static knowledge base (rare updates)

WHEN NOT TO USE:
  ❌ Need server/multi-app access (it's a library)
  ❌ Complex metadata filtering (limited support)
  ❌ Frequent updates and deletions (rebuild required)
  ❌ Want simple API (lower-level than others)
```

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create FAISS index from documents
texts = ["Document 1 content", "Document 2 content", "Document 3 content"]
vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
loaded_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Required for security
)

# FAISS is a LIBRARY:
# Vectors live in RAM during usage
# Disk files are for persistence between sessions
# Restart your app = reload from disk
```

---

### 🟣 Qdrant — The Production Sweet Spot

```
WHAT IT IS:
  Production-ready vector database
  Written in Rust (fast and safe)
  Server-based with REST and gRPC APIs

WHY USE IT:
  ✅ Production-grade reliability
  ✅ Rich metadata filtering
  ✅ Self-host or cloud (1GB free forever!)
  ✅ Excellent performance
  ✅ Distributed mode for scale
  ✅ Built-in payload (metadata) indexing

WHEN TO USE:
  ✅ Production applications
  ✅ Need complex metadata filtering
  ✅ Team projects (multi-app access)
  ✅ Want to start free, scale to paid
  ✅ Self-hosting comfortable

WHEN NOT TO USE:
  ❌ Just learning (Chroma is simpler)
  ❌ Single-machine prototype (Chroma is enough)
  ❌ Massive billion-scale (Milvus is better)
```

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

# Option 1: Self-hosted via Docker
# docker run -p 6333:6333 qdrant/qdrant

client = QdrantClient(url="http://localhost:6333")

# Option 2: Qdrant Cloud (1GB free forever)
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding=embeddings,
)

# Production features built-in:
# - Snapshots for backups
# - Replication for HA
# - Quantization for storage optimization
# - Hybrid search (vector + keyword)
```

---

### 🟡 Pinecone — The Zero-Ops Cloud

```
WHAT IT IS:
  Fully managed cloud vector database
  No infrastructure to manage
  Original "vector database as a service"

WHY USE IT:
  ✅ Zero infrastructure work
  ✅ Auto-scaling
  ✅ Industry-leading performance
  ✅ 100K vectors free tier
  ✅ Excellent monitoring/observability

WHEN TO USE:
  ✅ Small team, no DevOps capacity
  ✅ Want to focus only on application logic
  ✅ Need enterprise SLAs
  ✅ Budget for managed services

WHEN NOT TO USE:
  ❌ Free tier too small for production
  ❌ Want vendor independence
  ❌ Cost-sensitive (gets expensive at scale)
  ❌ Data sovereignty/privacy requirements
  ❌ Air-gapped deployments
```

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone client
pc = Pinecone(api_key="your-pinecone-api-key")

# Create or connect to index
index_name = "my-rag-index"
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

# Pricing reality check:
# Free:    100K vectors (good for prototype only)
# Starter: $70/month for ~5M vectors
# Standard:$150+/month with more features
# → Plan budget carefully
```

---

### 🔴 Milvus — The Billion-Scale Beast

```
WHAT IT IS:
  Open-source distributed vector database
  Built for massive scale (billions of vectors)
  Used by major tech companies in production

WHY USE IT:
  ✅ Handles billions of vectors
  ✅ Horizontal scaling
  ✅ GPU acceleration
  ✅ Cloud-native architecture
  ✅ Multiple index types
  ✅ Free self-hosted, managed Zilliz option

WHEN TO USE:
  ✅ Massive scale (100M+ vectors)
  ✅ Need horizontal scaling
  ✅ Have DevOps capacity
  ✅ Multi-tenant SaaS applications
  ✅ Complex deployments

WHEN NOT TO USE:
  ❌ Small projects (overkill)
  ❌ No Kubernetes expertise
  ❌ Single-machine deployments
  ❌ Quick prototypes
```

```python
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to Milvus
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={
        "host": "localhost",
        "port": "19530",
    },
    collection_name="my_documents",
)

# Or use Milvus Lite (embedded version for development)
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./milvus_demo.db"},
    collection_name="my_documents",
)
```

---

## 5. Decision Framework: Which Vector Store?

```
DECISION TREE FOR VECTOR STORE SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

START
   │
   ▼
Are you just learning RAG?
   │
   ├── YES ──→ Use CHROMA
   │           Simplest, no infrastructure, perfect for learning
   │
   ▼
Is this a prototype or POC?
   │
   ├── YES ──→ Use CHROMA or FAISS
   │           Both are simple, free, run locally
   │
   ▼
Production application?
   │
   ├── How many vectors?
   │   │
   │   ├── < 100K vectors
   │   │   │
   │   │   ├── No DevOps team? → PINECONE (free tier)
   │   │   └── Have DevOps?    → QDRANT (self-hosted)
   │   │
   │   ├── 100K - 100M vectors
   │   │   │
   │   │   ├── No DevOps?      → PINECONE (paid)
   │   │   └── Have DevOps?    → QDRANT
   │   │
   │   └── > 100M vectors (massive scale)
   │       │
   │       └── Use MILVUS or managed Zilliz Cloud
   │
   ▼
Special requirements?
   │
   ├── Maximum search speed     → FAISS (with GPU)
   ├── Complex metadata filters → QDRANT
   ├── Need server-based        → QDRANT or MILVUS
   ├── Want fully managed       → PINECONE
   └── Data must stay on-prem   → CHROMA, FAISS, or self-hosted QDRANT
```

---

# PART 3: Vector Store Operations (CRUD)

---

## 6. Understanding CRUD for Vectors

```
CRUD = Create, Read, Update, Delete
The four fundamental operations on any database

For vector stores:

CREATE:  Add new documents → embed → store vectors
READ:    Search by similarity (with optional filters)
UPDATE:  Modify existing documents (re-embed and replace)
DELETE:  Remove documents (by ID or filter)

Each operation has unique challenges in vector stores
that don't exist in traditional databases.
```

---

## 7. CREATE — Adding Documents

### The Creation Process

```
Document → Embedding Model → Vector → Store in DB

Behind the scenes:
1. Take your text
2. Call embedding model (API call or local inference)
3. Receive vector (e.g., 1536 numbers)
4. Store in vector database with:
   - Unique ID
   - The original text (page_content)
   - Metadata (any extra info)
   - The vector itself
```

### Method 1: Add Texts (Simple Strings)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="hr_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add simple texts
texts = [
    "Employees get 15 vacation days per year.",
    "Sick leave is 10 days annually.",
    "Maternity leave is 12 weeks paid.",
    "Remote work is allowed 3 days per week.",
]

# add_texts: Automatically embeds and stores
ids = vectorstore.add_texts(texts=texts)
print(f"Added {len(ids)} documents")
print(f"Generated IDs: {ids}")
# Output: ['doc-abc123', 'doc-def456', 'doc-ghi789', 'doc-jkl012']

# With custom IDs (recommended for production)
custom_ids = ["policy-1", "policy-2", "policy-3", "policy-4"]
ids = vectorstore.add_texts(
    texts=texts,
    ids=custom_ids,  # YOU control the IDs
)
# Why custom IDs matter:
# - Can update by ID later
# - Can delete by ID
# - Can track which document is which
```

### Method 2: Add Texts with Metadata

```python
texts = [
    "Employees get 15 vacation days per year.",
    "Sick leave is 10 days annually.",
    "Remote work is allowed 3 days per week.",
]

metadatas = [
    {
        "source":     "hr_handbook.pdf",
        "page":       12,
        "category":   "leave_policy",
        "version":    "2024",
        "department": "hr",
    },
    {
        "source":     "hr_handbook.pdf",
        "page":       14,
        "category":   "leave_policy",
        "version":    "2024",
        "department": "hr",
    },
    {
        "source":     "remote_work_policy.pdf",
        "page":       3,
        "category":   "work_policy",
        "version":    "2024",
        "department": "hr",
    },
]

ids = vectorstore.add_texts(
    texts=texts,
    metadatas=metadatas,
    ids=["leave-1", "leave-2", "remote-1"],
)
print(f"Added {len(ids)} docs with metadata")
```

### Method 3: Add Documents (LangChain Document Objects)

```python
from langchain_core.documents import Document

# Document objects bundle text + metadata together
documents = [
    Document(
        page_content="Employees get 15 vacation days per year.",
        metadata={
            "source":   "hr_handbook.pdf",
            "page":     12,
            "category": "leave_policy",
        }
    ),
    Document(
        page_content="Sick leave is 10 days annually.",
        metadata={
            "source":   "hr_handbook.pdf",
            "page":     14,
            "category": "leave_policy",
        }
    ),
]

# add_documents: Takes Document objects directly
ids = vectorstore.add_documents(
    documents=documents,
    ids=["leave-1", "leave-2"],
)
```

### Method 4: Build Vector Store from Scratch

```python
# Create new vector store and populate in one step
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="hr_docs",
    persist_directory="./chroma_db",
)

# Same pattern works for from_texts:
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    ids=custom_ids,
    collection_name="hr_docs",
    persist_directory="./chroma_db",
)
```

### Batch Processing for Efficiency

```python
# When adding many documents, batch them!
# Why? Each embedding API call has network overhead

def add_documents_in_batches(
    vectorstore,
    documents: list,
    batch_size: int = 100,
):
    """Add documents in batches for efficiency."""
    all_ids = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: "
              f"docs {i} to {i + len(batch)}")

        try:
            batch_ids = vectorstore.add_documents(batch)
            all_ids.extend(batch_ids)
        except Exception as e:
            print(f"Batch failed: {e}")
            # In production: retry logic, dead letter queue, etc.

    print(f"\n✅ Total documents added: {len(all_ids)}")
    return all_ids

# Why batch_size=100?
# - Small enough to fit in API limits
# - Large enough to minimize overhead
# - OpenAI accepts up to 2048 inputs per request
# - But 100 is safer for retry logic
```

---

## 8. READ — Similarity Search

### Basic Similarity Search

```python
# Find documents similar to a query
query = "How many vacation days do I get?"

results = vectorstore.similarity_search(
    query=query,
    k=3,  # Return top 3 most similar
)

# What happens under the hood:
# 1. Embed the query: vector = embed(query)
# 2. Search index for k nearest vectors
# 3. Return the original documents

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")
    print()
```

### Similarity Search with Scores

```python
# Get similarity scores too — critical for understanding quality
results_with_scores = vectorstore.similarity_search_with_score(
    query=query,
    k=3,
)

for doc, score in results_with_scores:
    print(f"Score:   {score:.4f}")
    print(f"Content: {doc.page_content}")
    print()

# IMPORTANT: Score interpretation depends on the vector store!
#
# Chroma:    Returns DISTANCE (lower = more similar)
#            Range: [0, 2] for cosine distance
#            score=0.1 → very similar
#            score=1.5 → not similar
#
# FAISS:     Returns DISTANCE (lower = more similar)
#            L2 distance by default
#
# Pinecone:  Returns SIMILARITY (higher = more similar)
#            Range: [-1, 1] for cosine
#            score=0.95 → very similar
#            score=0.10 → not similar
#
# Qdrant:    Returns SIMILARITY (higher = more similar)
#
# ALWAYS check the documentation for your specific vector store!
```

### Search with Metadata Filtering

```python
# This is the SUPERPOWER of metadata!
# Search only within specific subsets of your data

# Example 1: Filter by category
results = vectorstore.similarity_search(
    query="How many days off can I take?",
    k=3,
    filter={"category": "leave_policy"},  # Only leave-related docs
)

# Example 2: Filter by multiple fields
results = vectorstore.similarity_search(
    query="Remote work guidelines",
    k=3,
    filter={
        "department": "hr",
        "version":    "2024",
    },
)

# Example 3: Complex filters (Chroma syntax)
results = vectorstore.similarity_search(
    query="Recent policies",
    k=5,
    filter={
        "$and": [
            {"department": {"$eq": "hr"}},
            {"version":    {"$gte": "2024"}},
        ]
    }
)

# Why filtering matters:
# Without filter:  Search 1M vectors → may retrieve irrelevant
# With filter:     Search 10K vectors → much more precise
# AND much faster!
```

### Similarity Search by Vector

```python
# Sometimes you have a pre-computed vector
# Useful for: re-ranking, multi-step retrieval, custom logic

# Embed manually
query_vector = embeddings.embed_query("vacation policy")

# Search by vector directly (no re-embedding!)
results = vectorstore.similarity_search_by_vector(
    embedding=query_vector,
    k=3,
)

# Use case: Cache query embeddings to save costs
# Same query asked 100 times → embed once, search 100 times
```

### MMR (Maximum Marginal Relevance) Search

```python
# Problem with regular similarity search:
# Top 5 results might all be VERY SIMILAR to each other
# → Repetitive information in your context
# → LLM doesn't get diverse perspectives

# Solution: MMR — balance relevance with DIVERSITY
results = vectorstore.max_marginal_relevance_search(
    query=query,
    k=5,                # Return 5 documents
    fetch_k=20,         # First fetch 20 candidates
    lambda_mult=0.5,    # Balance: 0=max diversity, 1=max relevance
)

# How MMR works:
# 1. Fetch top fetch_k (e.g., 20) by similarity
# 2. Pick the most relevant one
# 3. For next pick: balance relevance vs how different
#    from already-picked documents
# 4. Repeat until k documents selected
#
# Result: 5 documents that are BOTH relevant AND diverse
#
# Use when:
# - Long documents with redundant info
# - Want diverse perspectives
# - Avoiding the "lost in similar context" problem
```

### Retriever Interface (LangChain Standard)

```python
# Convert vector store to a Retriever object
# This is the standard interface for RAG pipelines

retriever = vectorstore.as_retriever(
    search_type="similarity",          # or "mmr", "similarity_score_threshold"
    search_kwargs={
        "k": 5,
        "filter": {"department": "hr"},
    }
)

# Now use the standard invoke() interface
documents = retriever.invoke("What is the vacation policy?")

# This is what RAG chains expect:
# Retriever → produces Documents → fed to LLM
```

---

## 9. UPDATE — Modifying Documents

### The Update Challenge

```
WHY UPDATES ARE TRICKY IN VECTOR STORES:

The vector represents the OLD text content.
If you change the text, you must:
1. Re-embed the new text (new vector!)
2. Replace the old vector
3. Update the metadata

Some vector stores don't support true "update":
→ You delete + re-add instead

This is why custom IDs are CRITICAL:
→ Without them, you can't reliably update
```

### Update Pattern 1: Delete + Re-Add

```python
# This works in any vector store

# Step 1: Delete old version
vectorstore.delete(ids=["policy-1"])

# Step 2: Add updated version with same ID
updated_doc = Document(
    page_content="Employees get 20 vacation days per year (updated 2025).",
    metadata={
        "source":   "hr_handbook.pdf",
        "page":     12,
        "category": "leave_policy",
        "version":  "2025",       # Updated version
        "updated":  "2025-01-15",
    }
)

vectorstore.add_documents(
    documents=[updated_doc],
    ids=["policy-1"],  # Same ID = it's the same logical document
)

print("✅ Document updated")
```

### Update Pattern 2: Upsert (Chroma supports this)

```python
# Chroma has true upsert capability
# "Upsert" = Update if exists, Insert if not

vectorstore.update_documents(
    ids=["policy-1"],
    documents=[updated_doc],
)

# Behind the scenes:
# - Re-embeds the new text
# - Replaces the old vector
# - Keeps the same ID
# - Updates metadata
```

### Production-Grade Update Function

```python
import hashlib
from datetime import datetime

def smart_update_document(
    vectorstore,
    doc_id: str,
    new_content: str,
    new_metadata: dict,
) -> dict:
    """
    Intelligently update a document only if changed.
    Saves embedding API costs!
    """

    # Step 1: Get existing document
    try:
        existing = vectorstore.get(ids=[doc_id])
        if existing["ids"]:
            old_content = existing["documents"][0]
            old_metadata = existing["metadatas"][0]
        else:
            old_content = None
            old_metadata = {}
    except:
        old_content = None
        old_metadata = {}

    # Step 2: Check if content actually changed
    new_hash = hashlib.md5(new_content.encode()).hexdigest()
    old_hash = old_metadata.get("content_hash", "")

    if old_hash == new_hash:
        print(f"⏭️  Skip {doc_id}: content unchanged")
        return {"status": "skipped", "id": doc_id}

    # Step 3: Add timestamp and hash to metadata
    new_metadata.update({
        "content_hash":  new_hash,
        "updated_at":    datetime.now().isoformat(),
        "previous_hash": old_hash if old_hash else None,
    })

    # Step 4: Perform update (delete + add)
    if old_content is not None:
        vectorstore.delete(ids=[doc_id])

    new_doc = Document(
        page_content=new_content,
        metadata=new_metadata,
    )
    vectorstore.add_documents(documents=[new_doc], ids=[doc_id])

    action = "updated" if old_content else "created"
    print(f"✅ {action.capitalize()}: {doc_id}")
    return {"status": action, "id": doc_id}

# Usage
result = smart_update_document(
    vectorstore=vectorstore,
    doc_id="policy-1",
    new_content="Employees get 20 vacation days per year.",
    new_metadata={"source": "hr_handbook.pdf", "page": 12},
)
```

---

## 10. DELETE — Removing Documents

### Delete by ID (Most Common)

```python
# Delete a single document
vectorstore.delete(ids=["policy-1"])

# Delete multiple documents
vectorstore.delete(ids=["policy-1", "policy-2", "policy-3"])

print("Documents deleted")
```

### Delete by Filter

```python
# Delete all documents matching certain criteria
# Critical for: removing outdated docs, GDPR compliance, etc.

# Chroma example: Delete all docs from old version
vectorstore.delete(where={"version": "2023"})

# Delete all documents from a specific source
vectorstore.delete(where={"source": "old_handbook.pdf"})

# Delete based on access level
vectorstore.delete(where={"access_level": "deprecated"})

# Complex filter
vectorstore.delete(
    where={
        "$and": [
            {"department": {"$eq": "hr"}},
            {"version":    {"$lt":  "2024"}},
        ]
    }
)
```

### Delete Entire Collection

```python
# Nuclear option: delete the whole collection
vectorstore.delete_collection()

# When to use:
# - Resetting during development
# - Major schema changes
# - Switching embedding models (must re-embed everything)

# In Chroma:
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection(name="hr_docs")
```

### Soft Delete Pattern (Recommended for Production)

```python
# Instead of actually deleting, mark as deleted
# Why? Allows recovery, audit trail, gradual rollout

def soft_delete(vectorstore, doc_id: str):
    """Mark document as deleted without actually removing."""

    # Get existing document
    existing = vectorstore.get(ids=[doc_id])
    if not existing["ids"]:
        return False

    # Update metadata to mark as deleted
    old_content = existing["documents"][0]
    old_metadata = existing["metadatas"][0]

    old_metadata.update({
        "is_deleted":  True,
        "deleted_at":  datetime.now().isoformat(),
    })

    # Re-add with updated metadata
    vectorstore.delete(ids=[doc_id])
    vectorstore.add_documents(
        documents=[Document(
            page_content=old_content,
            metadata=old_metadata,
        )],
        ids=[doc_id],
    )

    return True

# Then filter out deleted docs during retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"is_deleted": {"$ne": True}}
    }
)
```

---

# PART 4: Full CRUD Lab — Production-Ready Application

---

## 11. Lab: Complete CRUD Vector Store Application

```python
"""
Production-Ready Vector Store Application
─────────────────────────────────────────
Full CRUD operations with:
- Custom IDs and metadata
- Update tracking
- Soft deletes
- Error handling
- Logging
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import logging
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Production-grade vector store manager with full CRUD support.

    Features:
    - Automatic ID generation
    - Content hash tracking (skip re-embedding unchanged docs)
    - Soft delete with recovery
    - Comprehensive metadata
    - Filter-based operations
    - Statistics and monitoring
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./vectordb",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.collection_name = collection_name

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        logger.info(f"Initialized vector store: {collection_name}")

    # ═══════════════════════════════════════════════════════════════
    # CREATE OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    def _generate_id(self, prefix: str = "doc") -> str:
        """Generate unique document ID."""
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def _content_hash(self, content: str) -> str:
        """Generate content hash for change detection."""
        return hashlib.md5(content.encode()).hexdigest()

    def _enrich_metadata(
        self,
        content: str,
        user_metadata: Dict[str, Any],
        doc_id: str,
    ) -> Dict[str, Any]:
        """Add system metadata to user metadata."""
        return {
            **user_metadata,
            "doc_id":       doc_id,
            "content_hash": self._content_hash(content),
            "created_at":   datetime.now().isoformat(),
            "updated_at":   datetime.now().isoformat(),
            "char_count":   len(content),
            "is_deleted":   False,
        }

    def create(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Create a single document."""
        metadata = metadata or {}
        doc_id = doc_id or self._generate_id()

        enriched_metadata = self._enrich_metadata(content, metadata, doc_id)

        document = Document(
            page_content=content,
            metadata=enriched_metadata,
        )

        self.vectorstore.add_documents(
            documents=[document],
            ids=[doc_id],
        )

        logger.info(f"✅ Created document: {doc_id}")
        return doc_id

    def create_batch(
        self,
        contents: List[str],
        metadatas: Optional[List[Dict]] = None,
        batch_size: int = 100,
    ) -> List[str]:
        """Create multiple documents in batches."""
        metadatas = metadatas or [{}] * len(contents)
        all_ids = []

        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = [self._generate_id() for _ in batch_contents]

            documents = [
                Document(
                    page_content=content,
                    metadata=self._enrich_metadata(content, meta, doc_id),
                )
                for content, meta, doc_id in zip(
                    batch_contents, batch_metadatas, batch_ids
                )
            ]

            try:
                self.vectorstore.add_documents(documents, ids=batch_ids)
                all_ids.extend(batch_ids)
                logger.info(
                    f"Batch {i//batch_size + 1}: added {len(batch_ids)} docs"
                )
            except Exception as e:
                logger.error(f"Batch failed at index {i}: {e}")

        logger.info(f"✅ Total created: {len(all_ids)} documents")
        return all_ids

    # ═══════════════════════════════════════════════════════════════
    # READ OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    def read_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        result = self.vectorstore.get(ids=[doc_id])

        if not result["ids"]:
            return None

        return Document(
            page_content=result["documents"][0],
            metadata=result["metadatas"][0],
        )

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None,
        include_deleted: bool = False,
    ) -> List[tuple]:
        """Semantic search with optional filtering."""

        # Always exclude deleted unless explicitly requested
        if not include_deleted:
            if filter is None:
                filter = {}
            filter["is_deleted"] = False

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter if filter else None,
        )

        logger.info(f"🔍 Query: '{query[:50]}...' returned {len(results)} results")
        return results

    def search_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """Diversity-aware search using MMR."""

        if filter is None:
            filter = {}
        filter["is_deleted"] = False

        results = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter if filter else None,
        )

        logger.info(f"🎯 MMR query returned {len(results)} diverse results")
        return results

    # ═══════════════════════════════════════════════════════════════
    # UPDATE OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    def update(
        self,
        doc_id: str,
        new_content: Optional[str] = None,
        new_metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Update a document. Skips re-embedding if content unchanged."""

        existing = self.read_by_id(doc_id)
        if not existing:
            logger.warning(f"Document {doc_id} not found for update")
            return {"status": "not_found", "id": doc_id}

        # Determine what to update
        final_content = new_content if new_content else existing.page_content
        final_metadata = {**existing.metadata}

        if new_metadata:
            final_metadata.update(new_metadata)

        # Check if content actually changed
        new_hash = self._content_hash(final_content)
        if new_hash == existing.metadata.get("content_hash"):
            # Only metadata changed — still need to update
            if new_metadata:
                final_metadata["updated_at"] = datetime.now().isoformat()
                self.vectorstore.delete(ids=[doc_id])
                self.vectorstore.add_documents(
                    documents=[Document(
                        page_content=final_content,
                        metadata=final_metadata,
                    )],
                    ids=[doc_id],
                )
                logger.info(f"📝 Metadata-only update: {doc_id}")
                return {"status": "metadata_updated", "id": doc_id}
            else:
                logger.info(f"⏭️  No changes for: {doc_id}")
                return {"status": "no_change", "id": doc_id}

        # Content changed — full update
        final_metadata.update({
            "content_hash": new_hash,
            "updated_at":   datetime.now().isoformat(),
        })

        self.vectorstore.delete(ids=[doc_id])
        self.vectorstore.add_documents(
            documents=[Document(
                page_content=final_content,
                metadata=final_metadata,
            )],
            ids=[doc_id],
        )

        logger.info(f"🔄 Updated: {doc_id}")
        return {"status": "updated", "id": doc_id}

    # ═══════════════════════════════════════════════════════════════
    # DELETE OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    def soft_delete(self, doc_id: str) -> bool:
        """Mark document as deleted (recoverable)."""
        result = self.update(
            doc_id=doc_id,
            new_metadata={
                "is_deleted":  True,
                "deleted_at":  datetime.now().isoformat(),
            }
        )
        success = result["status"] in ("updated", "metadata_updated")
        if success:
            logger.info(f"🗑️  Soft deleted: {doc_id}")
        return success

    def hard_delete(self, doc_id: str) -> bool:
        """Permanently remove document."""
        try:
            self.vectorstore.delete(ids=[doc_id])
            logger.info(f"💥 Hard deleted: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed for {doc_id}: {e}")
            return False

    def delete_by_filter(self, filter: Dict) -> int:
        """Delete all documents matching a filter."""
        # Get matching IDs first
        results = self.vectorstore.get(where=filter)
        ids_to_delete = results["ids"]

        if ids_to_delete:
            self.vectorstore.delete(ids=ids_to_delete)
            logger.info(f"🗑️  Deleted {len(ids_to_delete)} docs by filter")

        return len(ids_to_delete)

    def restore(self, doc_id: str) -> bool:
        """Restore a soft-deleted document."""
        result = self.update(
            doc_id=doc_id,
            new_metadata={
                "is_deleted": False,
                "restored_at": datetime.now().isoformat(),
            }
        )
        return result["status"] in ("updated", "metadata_updated")

    # ═══════════════════════════════════════════════════════════════
    # STATISTICS & MONITORING
    # ═══════════════════════════════════════════════════════════════

    def stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        all_docs = self.vectorstore.get()
        total = len(all_docs["ids"])

        # Count active vs deleted
        active = sum(
            1 for meta in all_docs["metadatas"]
            if not meta.get("is_deleted", False)
        )
        deleted = total - active

        return {
            "collection_name": self.collection_name,
            "total_documents": total,
            "active":          active,
            "soft_deleted":    deleted,
        }


# ═══════════════════════════════════════════════════════════════════
# USAGE DEMO
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Initialize the manager
    manager = VectorStoreManager(
        collection_name="hr_policies",
        persist_directory="./demo_vectordb",
    )

    # ─── CREATE ────────────────────────────────────────────────
    print("\n══════════ CREATE ══════════")

    id1 = manager.create(
        content="Employees receive 15 vacation days per year.",
        metadata={"category": "vacation", "department": "hr"},
        doc_id="vacation-policy-001",
    )

    # Batch create
    contents = [
        "Sick leave is 10 days annually.",
        "Parental leave is 12 weeks paid.",
        "Bereavement leave is 5 days.",
    ]
    metadatas = [
        {"category": "sick_leave", "department": "hr"},
        {"category": "parental_leave", "department": "hr"},
        {"category": "bereavement", "department": "hr"},
    ]
    batch_ids = manager.create_batch(contents, metadatas)

    # ─── READ ──────────────────────────────────────────────────
    print("\n══════════ READ ══════════")

    # By ID
    doc = manager.read_by_id("vacation-policy-001")
    print(f"Found: {doc.page_content}")

    # By similarity
    results = manager.search(
        query="How many days off for vacation?",
        k=3,
    )
    for doc, score in results:
        print(f"[{score:.4f}] {doc.page_content}")

    # By similarity with filter
    print("\n--- Filtered search ---")
    filtered_results = manager.search(
        query="time off",
        k=5,
        filter={"category": "sick_leave"},
    )
    for doc, score in filtered_results:
        print(f"[{score:.4f}] {doc.page_content}")

    # ─── UPDATE ────────────────────────────────────────────────
    print("\n══════════ UPDATE ══════════")

    result = manager.update(
        doc_id="vacation-policy-001",
        new_content="Employees receive 20 vacation days per year (updated 2025).",
        new_metadata={"version": "2025"},
    )
    print(f"Update result: {result}")

    # Verify update
    updated = manager.read_by_id("vacation-policy-001")
    print(f"After update: {updated.page_content}")
    print(f"Metadata: {updated.metadata}")

    # ─── DELETE ────────────────────────────────────────────────
    print("\n══════════ DELETE ══════════")

    # Soft delete
    manager.soft_delete("vacation-policy-001")

    # Verify excluded from default search
    results_after_delete = manager.search("vacation policy", k=5)
    print(f"After soft delete: {len(results_after_delete)} results")

    # Restore
    manager.restore("vacation-policy-001")
    results_after_restore = manager.search("vacation policy", k=5)
    print(f"After restore: {len(results_after_restore)} results")

    # ─── STATS ─────────────────────────────────────────────────
    print("\n══════════ STATS ══════════")
    stats = manager.stats()
    print(f"Collection statistics: {stats}")
```

---

# PART 5: Failure Modes and Interview Prep

---

## 12. Failure Modes 🚨

```
FAILURE 1: Score Direction Confusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Code assumes "higher score = better"
           but Chroma returns DISTANCE (lower = better)
Result:    Filtering keeps WORST matches instead of best
Fix:       Read docs for YOUR vector store carefully
           Test with known queries to verify direction

FAILURE 2: Embedding Model Mismatch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Stored vectors with Model A, queries with Model B
Result:    Completely random retrieval (math just doesn't work)
Fix:       Store embedding model name in collection metadata
           Validate model match on every query

FAILURE 3: No Custom IDs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Let vector store auto-generate UUIDs
Result:    Can't reliably update or delete specific documents
Fix:       Always assign meaningful, stable IDs
           Use a naming scheme: "source_chunk_index"

FAILURE 4: Duplicate Vectors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Re-ingest same documents → duplicates in DB
Result:    Same content retrieved multiple times in top-k
Fix:       Implement content hashing
           Check before adding: "Does this hash exist?"

FAILURE 5: Index Not Built/Updated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Added vectors but didn't refresh index (FAISS)
Result:    New documents not searchable
Fix:       Understand your store's indexing behavior
           Some require explicit index building

FAILURE 6: Memory Explosion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   FAISS loads entire index into RAM
           10M vectors × 1536 dims × 4 bytes = 60GB RAM!
Result:    Out of memory crashes
Fix:       Use quantization (PQ index)
           Or use disk-based stores (Qdrant, Chroma)

FAILURE 7: Missing Persistence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Chroma without persist_directory
Result:    All data lost when process ends
Fix:       Always specify persist_directory
           Test app restart preserves data

FAILURE 8: Filter Syntax Variations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Filter syntax differs across vector stores
           Chroma uses $and, $eq, $gte
           Qdrant uses different syntax
           Pinecone uses different syntax
Result:    Code breaks when switching stores
Fix:       Abstract filter logic in your application layer
           Don't tightly couple to one store's syntax
```

---

## 13. Interview Questions and Answers

### Q1: "Why do we need a vector store instead of a regular database?"

**Strong Answer:**
> "Vector stores solve a fundamentally different problem from traditional databases. SQL databases excel at exact-match queries using B-tree indexes — find rows where age equals 25. Vector stores excel at approximate similarity queries in high-dimensional space — find documents most similar to this query vector. The math is different: traditional indexes can't efficiently search 1536-dimensional space. Vector stores use specialized algorithms like HNSW and IVF that achieve O(log N) search through approximate nearest neighbor techniques. Without these, searching 1 million embeddings would take seconds with brute force. With HNSW, it takes milliseconds with 95-99% accuracy — an acceptable tradeoff for semantic search, which is itself approximate."

---

### Q2: "How would you choose between Chroma, Qdrant, and Pinecone?"

**Strong Answer:**
> "It depends on three factors: scale, ops capacity, and budget. I start with Chroma for prototyping because it's embedded — no servers, zero setup, runs in-process. For production with under 1M vectors and a small team without DevOps capacity, I choose Pinecone for its managed simplicity, despite the cost beyond the free tier. For production where I have DevOps capacity and want self-hosting, I choose Qdrant — it's Rust-based, very fast, has excellent metadata filtering, and offers 1GB cloud free forever for hybrid deployments. For massive scale beyond 100M vectors, I'd evaluate Milvus. Critically, I lock this decision in early because switching vector stores requires re-embedding everything — a costly migration."

---

### Q3: "What's the difference between similarity_search and max_marginal_relevance_search?"

**Strong Answer:**
> "similarity_search returns the top-k most similar documents by cosine similarity. The problem: those top-k results might all be near-duplicates, providing redundant context to the LLM. MMR — Maximum Marginal Relevance — solves this by balancing relevance with diversity. It first fetches a larger candidate set, then iteratively picks documents that are relevant to the query but different from already-selected documents. The lambda_mult parameter controls the tradeoff: 1.0 = pure relevance (same as similarity search), 0.0 = pure diversity. I use MMR when documents have redundant content, when I want diverse perspectives in the LLM context, or when retrieving from sources with many similar chunks. The cost: slightly slower since you fetch fetch_k candidates instead of just k."

---

### Q4: "How do you handle document updates in a vector store?"

**Strong Answer:**
> "Updates are tricky because the vector represents the OLD text. Changing text requires re-embedding. My approach has three layers. First, use custom IDs from day one — never let the vector store auto-generate them, because you can't reliably update what you can't reference. Second, implement content hashing — calculate MD5 of content, compare to stored hash, skip re-embedding if unchanged. This saves significant API costs at scale. Third, for the actual update: most vector stores require delete + re-add since true update isn't always supported. I wrap this in a single function that's idempotent. For production, I also implement soft deletes with an 'is_deleted' metadata flag and filter it out during retrieval — this preserves audit trails and enables recovery."

---

## 14. Common Mistakes ⚠️

```
MISTAKE 1: "All vector stores have the same API"
TRUTH:     APIs differ significantly. Chroma's filter syntax
           differs from Qdrant's differs from Pinecone's.
           Abstract this in your application layer.

MISTAKE 2: "I can switch vector stores easily"
TRUTH:     Different stores have different index types, different
           score interpretations, different filter syntax.
           Switching = re-embedding + code changes.

MISTAKE 3: "Higher k always gives better answers"
TRUTH:     Higher k = more context = more noise to the LLM.
           Sweet spot is usually k=3-5 for most use cases.
           Test with actual queries to find optimal k.

MISTAKE 4: "Vector search is exact"
TRUTH:     Modern ANN algorithms (HNSW) are APPROXIMATE.
           They're 95-99% accurate, not 100%. Acceptable
           for semantic search but understand the tradeoff.

MISTAKE 5: "Metadata filtering is free"
TRUTH:     Post-filtering (filter after vector search) is fast.
           Pre-filtering (filter then search) requires special
           index support and can be slower in some stores.

MISTAKE 6: "I don't need to specify IDs"
TRUTH:     Without custom IDs, you can't reliably update or
           delete specific documents. ALWAYS use custom IDs
           in production.

MISTAKE 7: "Embedded vector stores scale to production"
TRUTH:     Chroma in embedded mode is great for development
           but doesn't scale to multi-server deployments.
           Plan migration to server-based stores early.
```

---

## 15. Engineer Mindset 🔧

```
WHAT A REAL RAG ENGINEER THINKS ABOUT:

1. "Lock in vector store early"
   → Switching stores = re-embedding + code changes
   → Choose based on production needs, not just prototype
   → Document the choice and reasoning

2. "Always use custom IDs"
   → Naming scheme: "{source}_{chunk_index}_{version}"
   → Enables reliable updates and deletes
   → Critical for incremental ingestion pipelines

3. "Plan for updates from day one"
   → Implement content hashing for change detection
   → Use soft deletes for audit trails
   → Build idempotent ingestion (safe to re-run)

4. "Metadata is your friend"
   → Add rich metadata at ingestion time
   → Enables filtering, access control, versioning
   → Design schema upfront, validate on ingest

5. "Monitor what you store"
   → Track collection size growth
   → Monitor query latency over time
   → Alert on retrieval quality degradation

6. "Test with real queries"
   → Don't just trust benchmark numbers
   → Build evaluation set with your actual use cases
   → Measure precision@k for your domain

7. "Plan for re-embedding"
   → Embedding models will improve, you'll want to upgrade
   → Keep original text accessible (not just vectors)
   → Build re-embedding pipeline as part of ops

8. "Budget for costs"
   → Embedding API calls during ingestion
   → Storage costs (1M × 1536 dims × 4 bytes = 6GB)
   → Query latency at scale
   → Managed service tiers as you grow
```

---

## 16. Exercises 📝

### Beginner

```
1. Set up Chroma locally:
   - Add 10 documents with metadata
   - Search for similar documents
   - Verify persistence (restart Python, reload)

2. Compare similarity_search vs similarity_search_with_score:
   - Run both on same query
   - Print scores
   - Understand: is your store using distance or similarity?

3. Practice metadata filtering:
   - Create documents with category metadata
   - Search with and without filter
   - Verify filter actually reduces results

4. Implement create, read, delete:
   - Build a function to add a document
   - Build a function to retrieve it by ID
   - Build a function to delete it
   - Test the full cycle
```

### Intermediate

```
5. Build an idempotent ingestion pipeline:
   - Calculate content hash before adding
   - Skip documents that already exist (by hash)
   - Run the pipeline twice, verify no duplicates

6. Implement soft delete pattern:
   - Add 'is_deleted' metadata flag
   - Build delete function that just sets the flag
   - Build retriever that filters out deleted docs
   - Build restore function

7. Compare MMR vs similarity search:
   - Add 20 documents about similar topics
   - Query with similarity_search (k=5)
   - Query with max_marginal_relevance_search (k=5)
   - Compare: which gives more diverse results?

8. Evaluate vector store performance:
   - Ingest 1000 documents
   - Measure: ingestion time per document
   - Measure: query latency for different k values
   - Plot results
```

### Advanced

```
9. Build a vector store abstraction layer:
   - Define interface: create, read, update, delete, search
   - Implement for Chroma
   - Implement for FAISS
   - Verify same client code works with both

10. Implement document versioning:
    - Each document has version metadata
    - Store multiple versions of same logical doc
    - Search can filter to latest version
    - Build "rollback to previous version" function

11. Build a hybrid retrieval system:
    - Use metadata filter for hard constraints (department, date)
    - Use vector similarity for soft ranking
    - Implement re-ranking based on metadata signals
    - Compare to pure vector search

12. Production migration exercise:
    - Start with Chroma (embedded)
    - Migrate to Qdrant (server-based)
    - Verify same documents return same search results
    - Document the migration steps and challenges
```

---

## 17. Resources 📚

```
OFFICIAL DOCUMENTATION:
🔗 Chroma:    https://docs.trychroma.com/
🔗 FAISS:     https://github.com/facebookresearch/faiss/wiki
🔗 Qdrant:    https://qdrant.tech/documentation/
🔗 Pinecone:  https://docs.pinecone.io/
🔗 Milvus:    https://milvus.io/docs

LANGCHAIN INTEGRATIONS:
🔗 Vector Stores Overview
   https://python.langchain.com/docs/integrations/vectorstores/

🔗 Vector Store Concepts
   https://python.langchain.com/docs/concepts/vectorstores/

ALGORITHMS & PAPERS:
📄 "Efficient and Robust Approximate Nearest Neighbor Search
    Using Hierarchical Navigable Small World Graphs" (HNSW)
   https://arxiv.org/abs/1603.09320

📄 "Billion-scale similarity search with GPUs" (FAISS)
   https://arxiv.org/abs/1702.08734

BENCHMARKS:
🔗 ANN Benchmarks (compare algorithms)
   https://ann-benchmarks.com/

🔗 Vector Database Benchmark
   https://github.com/qdrant/vector-db-benchmark

TUTORIALS:
🔗 LangChain Vector Store Tutorial
   https://python.langchain.com/docs/tutorials/retrievers/

🔗 Building RAG with Chroma
   https://docs.trychroma.com/getting-started
```

---

## Module 4 Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 4 MENTAL MODEL                          │
│                                                                 │
│  WHAT:    Specialized DB optimized for finding similar          │
│           vectors in high-dimensional space — fast.            │
│                                                                 │
│  WHY:     Brute force search of 1M vectors = seconds            │
│           Vector store with HNSW = milliseconds                 │
│           Speed enables real-time semantic search.             │
│                                                                 │
│  CHOOSE:  Learning?      → Chroma (embedded, simple)            │
│           Production?    → Qdrant (self-hosted) or Pinecone     │
│           Massive scale? → Milvus                                │
│           Need speed?    → FAISS                                 │
│                                                                 │
│  CRUD:    CREATE  - Use custom IDs, add rich metadata          │
│           READ    - similarity_search vs MMR for diversity      │
│           UPDATE  - Delete + re-add (true update rare)          │
│           DELETE  - Prefer soft delete in production            │
│                                                                 │
│  GOLDEN RULES:                                                  │
│  1. Lock in vector store at design time                        │
│  2. Always use custom IDs (never auto-generated)               │
│  3. Add metadata for filtering and access control              │
│  4. Use content hashing for idempotent updates                 │
│  5. Implement soft deletes for production safety               │
│  6. Know your store's score direction (distance vs similarity) │
└─────────────────────────────────────────────────────────────────┘
```