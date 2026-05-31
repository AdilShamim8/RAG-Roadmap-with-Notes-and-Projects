# Module 6: Advanced Retrieval Techniques 

## Taking RAG from Good to Production-Grade

---

> **Before I explain — let me ask:**
>
> *"What's the limitation of basic retrieval that we covered in Module 5?"*
>
> A beginner might say: *"Sometimes it returns irrelevant results."*
>
> That's true — but the deeper insight is this:
>
> **Basic retrieval treats every query the same way. But queries are not equal:**
> - Some have complex filter conditions
> - Some need multiple perspectives
> - Some return chunks that are too small or too big
> - Some retrieve relevant docs but with noise
>
> **Advanced retrieval techniques are about adapting to query complexity.**
> They're the difference between a toy RAG and a production system that actually works.

---

# The Big Picture: Why Advanced Retrieval Matters

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  THE EVOLUTION OF RAG QUALITY:                                   │
│                                                                  │
│  Level 1 (Basic):  similarity_search                             │
│    Quality: ~60% useful retrievals                               │
│    Problem: Noise, missing context, no filtering                 │
│                                                                  │
│  Level 2 (Hybrid): + BM25, MMR, thresholds                       │
│    Quality: ~75% useful retrievals                               │
│    Problem: Still suffers from chunk-size dilemma                │
│                                                                  │
│  Level 3 (Advanced): + compression, multi-query, re-ranking      │
│    Quality: ~90% useful retrievals                               │
│    Result: Production-grade RAG                                  │
│                                                                  │
│  Each technique solves a SPECIFIC failure mode of basic RAG.     │
│  Stack them strategically based on your needs.                  │
└──────────────────────────────────────────────────────────────────┘
```

---

# PART 1: Contextual Compression

---

## 1. Why Compress Retrieved Context?

### The Problem: Noise in Retrieved Chunks

```
SCENARIO: User asks "What is our refund policy?"

You retrieve top-3 chunks (each 500 tokens):

CHUNK 1 (500 tokens):
"Our company was founded in 1995... [200 tokens of history]
 ...we believe in customer satisfaction... [100 tokens of mission]
 Our REFUND POLICY allows returns within 30 days. [200 tokens
 of other policies]"

CHUNK 2 (500 tokens):
"Customer service hours are 9-5 EST... [most about hours]
 ...returns are handled by our team..."

CHUNK 3 (500 tokens):
"Product warranties last 1 year... [warranty info]
 Refunds for damaged items can be issued..."

TOTAL: 1,500 tokens sent to LLM
RELEVANT: ~200 tokens
NOISE:    ~1,300 tokens (87% noise!)

PROBLEMS:
❌ LLM may get confused by irrelevant context
❌ Higher API cost (paying for noise)
❌ Longer response time
❌ Risk of "lost in the middle" problem
```

### The Solution: Post-Retrieval Compression

```
COMPRESSION = Filter/extract only the RELEVANT parts
              of retrieved chunks BEFORE sending to LLM

After compression:
TOTAL: 200 tokens (only the relevant parts)
NOISE: 0 tokens
COST:  85% lower
QUALITY: Higher (LLM only sees relevant info)
```

---

## 2. Architecture: How Compression Fits In

```
┌────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE FLOW                       │
└────────────────────────────────────────────────────────────┘

         User Query
             │
             ▼
  ┌──────────────────────┐
  │  Base Retriever      │  ← Fetches top-k chunks
  │  (similarity_search) │     with all their noise
  └──────────────────────┘
             │
             ▼
       [chunk1, chunk2, chunk3]  ← 1,500 tokens
             │
             ▼
  ┌──────────────────────┐
  │ ContextualCompression │  ← NEW LAYER
  │ Retriever             │
  │                       │
  │ For each chunk:       │
  │  - Extract relevant   │
  │    parts only         │
  │  - Or drop entirely   │
  │    if irrelevant      │
  └──────────────────────┘
             │
             ▼
       [compressed_chunks]  ← 200 tokens
             │
             ▼
         To LLM →  Generation
```

---

## 3. LLMChainExtractor — LLM-Based Compression

### How It Works

```
For each retrieved chunk, an LLM is asked:
"Given this query and this chunk, extract ONLY the parts
 that are relevant to the query. If nothing is relevant,
 return NO_OUTPUT."

Pros: Highest quality extraction (LLM understands meaning)
Cons: Slow and expensive (one LLM call per chunk)
```

### Implementation

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

# Setup base vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(
        page_content="Our company was founded in 1995 by John Smith. "
                    "We are headquartered in New York. "
                    "Our refund policy allows returns within 30 days "
                    "with original receipt. We also offer extended warranties.",
        metadata={"id": "doc1"}
    ),
    Document(
        page_content="Customer service is available 9-5 EST Monday-Friday. "
                    "For refund requests, contact support@company.com. "
                    "Our agents are trained extensively.",
        metadata={"id": "doc2"}
    ),
    Document(
        page_content="Product warranties last 1 year from purchase date. "
                    "Damaged items can be refunded or exchanged within 14 days.",
        metadata={"id": "doc3"}
    ),
]

vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    collection_name="compression_demo",
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ─────────────────────────────────────────────────────
# Setup LLM-based extractor
# ─────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Compressor: uses LLM to extract relevant parts
compressor = LLMChainExtractor.from_llm(llm)

# Wrap base retriever with compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# ─────────────────────────────────────────────────────
# Compare with and without compression
# ─────────────────────────────────────────────────────
query = "What is the refund policy?"

print("═" * 70)
print("WITHOUT COMPRESSION (full chunks)")
print("═" * 70)
original_docs = base_retriever.invoke(query)
total_chars_before = 0
for i, doc in enumerate(original_docs):
    print(f"\nChunk {i+1} ({len(doc.page_content)} chars):")
    print(f"  {doc.page_content}")
    total_chars_before += len(doc.page_content)
print(f"\nTotal: {total_chars_before} characters")

print("\n" + "═" * 70)
print("WITH COMPRESSION (only relevant parts)")
print("═" * 70)
compressed_docs = compression_retriever.invoke(query)
total_chars_after = 0
for i, doc in enumerate(compressed_docs):
    print(f"\nChunk {i+1} ({len(doc.page_content)} chars):")
    print(f"  {doc.page_content}")
    total_chars_after += len(doc.page_content)
print(f"\nTotal: {total_chars_after} characters")
print(f"\nReduction: {(1 - total_chars_after/total_chars_before)*100:.1f}%")
```

**Expected output:**
```
WITHOUT: ~600 characters (lots of noise about company history, warranties)
WITH:    ~150 characters (only refund-related sentences)
Reduction: ~75%
```

---

## 4. EmbeddingsFilter — Fast Embedding-Based Filtering

### How It Works

```
Instead of using an LLM to extract relevant parts,
use EMBEDDING SIMILARITY to filter at the chunk level.

For each retrieved chunk:
  similarity = cosine(query_embedding, chunk_embedding)
  if similarity < threshold:
    DROP the chunk entirely
  else:
    KEEP the chunk (whole)

This is MUCH faster than LLMChainExtractor:
- No LLM calls
- Just embedding comparisons (fast)
- But less precise (whole chunks, not parts)
```

### Implementation

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Setup embeddings filter
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.75,  # Drop chunks below this similarity
)

# Wrap base retriever
filter_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever,
)

# Test
query = "What is the refund policy?"
results = filter_retriever.invoke(query)

print(f"Retrieved {len(results)} relevant chunks (filtered)")
for doc in results:
    print(f"  {doc.page_content[:100]}...")
```

### Comparison: LLM Extractor vs Embeddings Filter

```
┌─────────────────┬───────────────────┬───────────────────┐
│ Aspect          │ LLMChainExtractor │ EmbeddingsFilter  │
├─────────────────┼───────────────────┼───────────────────┤
│ Granularity     │ Sentence-level    │ Chunk-level       │
│ Speed           │ Slow (LLM calls)  │ Fast (embeddings) │
│ Cost            │ High              │ Low               │
│ Quality         │ Best              │ Good              │
│ Use case        │ Critical accuracy │ High volume       │
└─────────────────┴───────────────────┴───────────────────┘

RULE OF THUMB:
- Production with budget: LLMChainExtractor
- Production high-volume:  EmbeddingsFilter
- Best practice:           Combine both in pipeline!
```

---

## 5. DocumentCompressorPipeline — Chaining Compressors

### The Power of Combination

```
WHY CHAIN COMPRESSORS?

Each compressor has strengths and weaknesses:
- EmbeddingsRedundantFilter: Removes duplicate chunks (fast)
- EmbeddingsFilter:          Drops irrelevant chunks (fast)
- LLMChainExtractor:         Extracts relevant sentences (slow but precise)

OPTIMAL PIPELINE (fast → slow):
1. First, remove duplicates (cheap)
2. Then, filter irrelevant chunks (cheap)
3. Finally, extract sentences from remaining (expensive but small input)

Result: Maximum quality with minimum cost!
```

### Implementation

```python
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    EmbeddingsRedundantFilter,
    LLMChainExtractor,
)
from langchain_text_splitters import CharacterTextSplitter

# ─────────────────────────────────────────────────────
# Build the compression pipeline
# ─────────────────────────────────────────────────────

# Step 1: Optional - split long docs into smaller pieces for filtering
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator=". "
)

# Step 2: Remove redundant/duplicate chunks
redundant_filter = EmbeddingsRedundantFilter(
    embeddings=embeddings,
    similarity_threshold=0.95,  # If 95% similar, consider duplicate
)

# Step 3: Filter out irrelevant chunks (fast)
relevant_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.7,
)

# Step 4: Extract specific relevant sentences (slow but precise)
llm_extractor = LLMChainExtractor.from_llm(llm)

# Compose pipeline
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        splitter,              # Step 1: Split
        redundant_filter,      # Step 2: Remove duplicates
        relevant_filter,       # Step 3: Filter irrelevant
        llm_extractor,         # Step 4: Extract relevant
    ]
)

# Use in retriever
pipeline_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever,
)

# Test
query = "What is the refund policy?"
results = pipeline_retriever.invoke(query)

print(f"Pipeline produced {len(results)} highly-relevant chunks")
for i, doc in enumerate(results):
    print(f"\n{i+1}. {doc.page_content}")
```

### Hands-On Lab: Full Contextual Compression Pipeline

```python
"""
COMPLETE CONTEXTUAL COMPRESSION DEMO
Shows the value at each stage of the pipeline.
"""

import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    EmbeddingsRedundantFilter,
    LLMChainExtractor,
)
from langchain_core.documents import Document

# Setup with noisy documents (realistic scenario)
noisy_docs = [
    Document(page_content=(
        "Welcome to our company! We were founded in 1995 by entrepreneur "
        "John Smith. Our headquarters are in New York City. "
        "REFUND POLICY: Customers may return any unopened product within "
        "30 days of purchase for a full refund. "
        "We pride ourselves on quality customer service."
    )),
    Document(page_content=(
        "Our products are manufactured in Italy using the finest materials. "
        "We use sustainable practices in all our operations. "
        "Customer satisfaction is our top priority. "
        "Returns are accepted within 30 days with original packaging."
    )),
    Document(page_content=(
        "Subscribe to our newsletter for monthly deals. "
        "Follow us on social media. We have offices in 12 countries. "
        "Our team consists of 5,000 employees worldwide."
    )),
    Document(page_content=(
        "Standard shipping takes 5-7 business days. Express shipping "
        "is available for an additional fee. International orders "
        "may take longer. We ship to over 80 countries."
    )),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vectorstore = Chroma.from_documents(
    noisy_docs,
    embedding=embeddings,
    collection_name="compression_lab",
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def measure_retrieval(name, retriever, query):
    """Measure tokens and time for a retriever."""
    start = time.time()
    docs = retriever.invoke(query)
    elapsed = time.time() - start

    total_chars = sum(len(d.page_content) for d in docs)

    return {
        "name":        name,
        "num_docs":    len(docs),
        "total_chars": total_chars,
        "time_s":      elapsed,
        "docs":        docs,
    }

query = "What is the refund policy?"

# Test 1: No compression
no_comp = measure_retrieval("No Compression", base_retriever, query)

# Test 2: Only embeddings filter
embed_only_retriever = ContextualCompressionRetriever(
    base_compressor=EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.6
    ),
    base_retriever=base_retriever,
)
embed_only = measure_retrieval("EmbeddingsFilter Only", embed_only_retriever, query)

# Test 3: Only LLM extractor
llm_only_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=base_retriever,
)
llm_only = measure_retrieval("LLMExtractor Only", llm_only_retriever, query)

# Test 4: Full pipeline
pipeline = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsRedundantFilter(
            embeddings=embeddings,
            similarity_threshold=0.95
        ),
        EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=0.6
        ),
        LLMChainExtractor.from_llm(llm),
    ]
)
full_pipeline_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever,
)
pipeline_result = measure_retrieval("Full Pipeline", full_pipeline_retriever, query)

# Print comparison
print("\n" + "═" * 80)
print(f"COMPRESSION COMPARISON: '{query}'")
print("═" * 80)
print(f"\n{'Strategy':<25} {'Docs':<8} {'Chars':<10} {'Time':<10}")
print("-" * 60)

for result in [no_comp, embed_only, llm_only, pipeline_result]:
    print(f"{result['name']:<25} "
          f"{result['num_docs']:<8} "
          f"{result['total_chars']:<10} "
          f"{result['time_s']:.2f}s")

print(f"\nReduction with full pipeline: "
      f"{(1 - pipeline_result['total_chars']/no_comp['total_chars'])*100:.1f}%")

# Show what each strategy returned
print("\n" + "═" * 80)
print("CONTENT COMPARISON")
print("═" * 80)
for result in [no_comp, pipeline_result]:
    print(f"\n--- {result['name']} ---")
    for i, doc in enumerate(result["docs"]):
        print(f"{i+1}. {doc.page_content[:150]}...")
```

---

# PART 2: Parent Document Retriever

---

## 6. The Chunk-Size Dilemma

### The Fundamental Tradeoff

```
SMALL CHUNKS (100-200 tokens):
✅ HIGH RETRIEVAL PRECISION
   Sharp, focused embeddings
   Match specific facts well
❌ MISSING CONTEXT
   Chunk alone may be incomprehensible
   LLM doesn't have enough context

LARGE CHUNKS (1000+ tokens):
✅ RICH CONTEXT
   Complete ideas preserved
   LLM has everything it needs
❌ POOR RETRIEVAL PRECISION
   Diluted embeddings
   Topic drift within chunks

THE DILEMMA:
You need SMALL for matching but LARGE for context.
Can you have both? YES!
```

### The Solution: Parent Document Retriever

```
GENIUS IDEA:
Use SMALL chunks for retrieval/matching.
Use LARGE chunks (parents) for LLM context.

How it works:
1. Split documents into LARGE parent chunks
2. Split each parent into SMALL child chunks
3. Store child chunks in vector store
4. Store parent chunks in separate document store
5. Each child knows its parent (via metadata)

Retrieval:
1. User query → search SMALL children (precise matching)
2. Find which PARENT each child belongs to
3. Return the PARENTS to the LLM (rich context)

Best of both worlds!
```

### Visual Example

```
DOCUMENT: "Refund Policy Manual" (3000 tokens)

PARENT CHUNKS (1000 tokens each):
┌──────────────────────────────────────────┐
│ Parent 1: General Refund Policy          │  ← LLM sees this
│ "Our refund policy allows returns within  │
│  30 days... [full context]"               │
└──────────────────────────────────────────┘
        │
        ├── Child A (200 tokens): "Returns within 30 days"  ← Embedded
        ├── Child B (200 tokens): "Refund processing time"  ← Embedded
        ├── Child C (200 tokens): "Required documentation"   ← Embedded
        ├── Child D (200 tokens): "Exceptions to policy"     ← Embedded
        └── Child E (200 tokens): "International returns"    ← Embedded

QUERY: "How long does refund take?"
       ↓
Matches Child B (precise!)
       ↓
Returns Parent 1 (full context to LLM!)
       ↓
LLM has complete policy context, not just one sentence
```

---

## 7. Hands-On: Build a Parent-Child System

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector store for CHILD chunks (small, for retrieval)
child_vectorstore = Chroma(
    collection_name="children",
    embedding_function=embeddings,
    persist_directory="./parent_child_db",
)

# Document store for PARENT chunks (large, for context)
# InMemoryStore is fine for development; use Redis for production
parent_docstore = InMemoryStore()

# ─────────────────────────────────────────────────────
# Configure splitters
# ─────────────────────────────────────────────────────

# Parent splitter: LARGE chunks (for LLM context)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)

# Child splitter: SMALL chunks (for precise retrieval)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

# ─────────────────────────────────────────────────────
# Create Parent Document Retriever
# ─────────────────────────────────────────────────────
retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,    # Where children live
    docstore=parent_docstore,          # Where parents live
    child_splitter=child_splitter,     # How to split into children
    parent_splitter=parent_splitter,   # How to split into parents
    search_kwargs={"k": 4},            # Retrieve top 4 children
)

# ─────────────────────────────────────────────────────
# Add documents (long policy document)
# ─────────────────────────────────────────────────────
long_policy_doc = Document(
    page_content="""
    REFUND POLICY MANUAL — Version 2024

    1. OVERVIEW
    Our company believes in customer satisfaction. This policy outlines
    the conditions under which we provide refunds to customers.

    2. ELIGIBILITY
    Customers may return any unopened product within 30 days of purchase
    for a full refund. The product must be in original condition with
    all packaging intact. Receipt or proof of purchase is required.

    3. PROCESSING TIME
    Refunds are typically processed within 7-10 business days after
    we receive the returned item. The refund will be issued to the
    original payment method. Bank processing may add additional time
    of 3-5 business days.

    4. REQUIRED DOCUMENTATION
    To process a refund, you must provide:
    - Original receipt or order number
    - Photo ID matching the purchase
    - Product in original packaging
    - Completed return form (available online)

    5. EXCEPTIONS
    The following items cannot be returned:
    - Personalized or custom-made items
    - Perishable goods
    - Digital downloads after activation
    - Items marked "Final Sale"

    6. INTERNATIONAL RETURNS
    International customers must arrange return shipping at their
    own expense. Customs fees and shipping costs are non-refundable.
    Processing time for international returns is 14-21 business days.

    7. CONTACT INFORMATION
    For refund inquiries, contact our customer service team at
    refunds@company.com or call 1-800-REFUND-1 during business hours.
    """,
    metadata={"source": "policy_manual.pdf", "category": "refunds"}
)

# This automatically:
# 1. Splits into parents (1000 tokens each)
# 2. Splits each parent into children (200 tokens each)
# 3. Embeds children → stores in vectorstore
# 4. Stores parents → in docstore with IDs
# 5. Links children to parents via metadata
retriever.add_documents([long_policy_doc])

# ─────────────────────────────────────────────────────
# Test retrieval
# ─────────────────────────────────────────────────────
query = "How long does it take to get my refund?"

# Standard retrieval would return small children
# Parent retriever returns the FULL parent chunks!
results = retriever.invoke(query)

print(f"Query: {query}\n")
print(f"Returned {len(results)} PARENT chunks (with full context):\n")
for i, doc in enumerate(results):
    print(f"─── Parent {i+1} ({len(doc.page_content)} chars) ───")
    print(doc.page_content)
    print()

# Compare with retrieving CHILDREN directly
print("\n" + "═" * 70)
print("FOR COMPARISON: What if we retrieved CHILDREN directly?")
print("═" * 70)
child_results = child_vectorstore.similarity_search(query, k=4)
print(f"\nChildren retrieved (small, possibly missing context):")
for i, doc in enumerate(child_results):
    print(f"\n─── Child {i+1} ({len(doc.page_content)} chars) ───")
    print(doc.page_content)
```

### Key Benefits Summary

```
PARENT DOCUMENT RETRIEVER BENEFITS:

1. Best of both worlds:
   - Precise retrieval (small children matched)
   - Rich context (large parents returned)

2. No more "chunk size" agonizing:
   - Set children small enough for good embeddings
   - Set parents large enough for complete context

3. Better LLM answers:
   - Context preserves logical flow
   - Avoids "fragments without context"

4. Production-ready pattern:
   - Used by many advanced RAG systems
   - Simple to implement with LangChain
```

---

# PART 3: Self-Query Retriever

---

## 8. The Problem: Mixed Natural Language + Structured Filters

### Real-World Scenario

```
USER QUERY: "Show me electronics under $500 with rating above 4 stars"

WITH BASIC RAG:
similarity_search("Show me electronics under $500 with rating above 4 stars")
→ Searches semantically
→ Returns products that "look like" the query
→ Doesn't actually filter by price < 500
→ Doesn't actually filter by rating > 4
→ POOR RESULTS

WHAT WE WANT:
1. Extract structured filter: category=electronics, price<500, rating>4
2. Apply filter to candidates
3. Then semantically search remaining candidates

This is exactly what Self-Query Retriever does!
```

### How Self-Query Works

```
USER QUERY: "Show me electronics under $500 with rating above 4 stars"
                            │
                            ▼
              ┌──────────────────────────┐
              │   LLM ANALYZES QUERY     │
              │                          │
              │ Splits into:             │
              │ - Semantic query: "show  │
              │   me electronics"        │
              │ - Filters:               │
              │   * category = electronics│
              │   * price < 500          │
              │   * rating > 4           │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  VECTOR STORE QUERY      │
              │  WITH AUTO-EXTRACTED     │
              │  FILTERS                 │
              └──────────────────────────┘
                            │
                            ▼
                Filtered + semantic results
```

---

## 9. Hands-On: Product Catalog with Self-Query

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────
# Step 1: Setup product catalog with rich metadata
# ─────────────────────────────────────────────────────
products = [
    Document(
        page_content="Wireless Bluetooth Headphones with active noise cancellation",
        metadata={
            "name":     "SonicPro Headphones",
            "category": "electronics",
            "price":    299,
            "rating":   4.5,
            "brand":    "SonicPro",
            "in_stock": True,
        }
    ),
    Document(
        page_content="Smart watch with heart rate monitor and GPS",
        metadata={
            "name":     "FitTracker Pro",
            "category": "electronics",
            "price":    450,
            "rating":   4.3,
            "brand":    "FitTracker",
            "in_stock": True,
        }
    ),
    Document(
        page_content="4K Ultra HD Smart Television with HDR support",
        metadata={
            "name":     "VisionMax 55",
            "category": "electronics",
            "price":    899,
            "rating":   4.7,
            "brand":    "VisionMax",
            "in_stock": True,
        }
    ),
    Document(
        page_content="Premium leather wallet with RFID protection",
        metadata={
            "name":     "Classic Wallet",
            "category": "accessories",
            "price":    89,
            "rating":   4.2,
            "brand":    "LeatherCraft",
            "in_stock": True,
        }
    ),
    Document(
        page_content="Wireless charging pad with fast charging support",
        metadata={
            "name":     "ChargeFast Pad",
            "category": "electronics",
            "price":    49,
            "rating":   3.8,
            "brand":    "ChargeFast",
            "in_stock": True,
        }
    ),
    Document(
        page_content="Premium running shoes with cushioned soles",
        metadata={
            "name":     "RunPro X1",
            "category": "footwear",
            "price":    180,
            "rating":   4.6,
            "brand":    "RunPro",
            "in_stock": False,
        }
    ),
    Document(
        page_content="Compact mirrorless camera with 24MP sensor",
        metadata={
            "name":     "PhotoMaster Z1",
            "category": "electronics",
            "price":    1299,
            "rating":   4.8,
            "brand":    "PhotoMaster",
            "in_stock": True,
        }
    ),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    products,
    embedding=embeddings,
    collection_name="products",
)

# ─────────────────────────────────────────────────────
# Step 2: Define metadata schema for the LLM
# ─────────────────────────────────────────────────────
# AttributeInfo tells the LLM what fields exist and what they mean

metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The category of the product. "
                   "One of: electronics, accessories, footwear",
        type="string",
    ),
    AttributeInfo(
        name="price",
        description="The price of the product in USD",
        type="float",
    ),
    AttributeInfo(
        name="rating",
        description="The customer rating from 1.0 to 5.0",
        type="float",
    ),
    AttributeInfo(
        name="brand",
        description="The brand or manufacturer of the product",
        type="string",
    ),
    AttributeInfo(
        name="in_stock",
        description="Whether the product is currently in stock",
        type="boolean",
    ),
]

# Description of what the document content represents
document_content_description = "Product description and features"

# ─────────────────────────────────────────────────────
# Step 3: Create the SelfQueryRetriever
# ─────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True,  # Shows the auto-generated query!
)

# ─────────────────────────────────────────────────────
# Step 4: Test with various queries
# ─────────────────────────────────────────────────────

test_queries = [
    # Pure semantic
    "show me headphones",

    # With numerical filter
    "electronics under $500",

    # With range filter
    "products rated above 4.5",

    # Multiple filters
    "in-stock electronics under $300 with rating above 4",

    # Complex query
    "wireless devices from SonicPro brand",
]

for query in test_queries:
    print("\n" + "═" * 70)
    print(f"🔍 QUERY: {query}")
    print("═" * 70)

    results = self_query_retriever.invoke(query)

    print(f"\nFound {len(results)} matching products:")
    for doc in results:
        print(f"\n  📦 {doc.metadata['name']}")
        print(f"     Category: {doc.metadata['category']}")
        print(f"     Price:    ${doc.metadata['price']}")
        print(f"     Rating:   {doc.metadata['rating']}/5.0")
        print(f"     Stock:    {'Yes' if doc.metadata['in_stock'] else 'No'}")
```

### What's Happening Under the Hood

```
For query "electronics under $500 with rating above 4":

LLM generates this structured query:
{
  "query": "electronics",                # Semantic part
  "filter": {                            # Structured filters
    "and": [
      {"eq": {"category": "electronics"}},
      {"lt": {"price": 500}},
      {"gt": {"rating": 4}}
    ]
  }
}

This is then executed as:
vectorstore.similarity_search(
    query="electronics",
    filter={
        "$and": [
            {"category": {"$eq": "electronics"}},
            {"price":    {"$lt": 500}},
            {"rating":   {"$gt": 4}}
        ]
    }
)

The LLM does the heavy lifting of converting natural language
to structured filters automatically!
```

---

# PART 4: Multi-Query Retriever

---

## 10. The Problem: Single Query, Limited Perspective

### Why One Query Isn't Enough

```
USER QUERY: "How do I improve my model's performance?"

This single query has MANY possible angles:
- "How to optimize ML model accuracy?"
- "Techniques for reducing model overfitting?"
- "Best practices for hyperparameter tuning?"
- "How to improve neural network training?"
- "Methods for feature engineering?"

If we only embed the ORIGINAL query, we might miss
documents that match better with these alternative phrasings.

RESULT: Lower recall, missing relevant documents
```

### How Multi-Query Solves This

```
USER QUERY: "How do I improve my model's performance?"
              │
              ▼
    ┌──────────────────────────────────┐
    │  LLM GENERATES VARIATIONS:       │
    │                                  │
    │  1. "How to optimize ML model    │
    │      accuracy?"                  │
    │  2. "What techniques reduce      │
    │      overfitting?"               │
    │  3. "Best practices for          │
    │      hyperparameter tuning?"     │
    │  4. "Methods to improve neural   │
    │      network training"           │
    └──────────────────────────────────┘
              │
              ▼
    Each variation searches independently
              │
              ▼
    ┌──────────────────────────────────┐
    │  UNION of all results            │
    │  (deduplicated)                  │
    │                                  │
    │  Captures docs that match ANY    │
    │  of the variations               │
    └──────────────────────────────────┘
              │
              ▼
    Higher recall, more comprehensive results
```

---

## 11. Hands-On: Implement Multi-Query Retrieval

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
import logging

# Enable logging to SEE the generated queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# ─────────────────────────────────────────────────────
# Setup knowledge base
# ─────────────────────────────────────────────────────
ml_docs = [
    Document(page_content=(
        "To optimize machine learning model accuracy, focus on data quality "
        "and proper feature engineering. Clean data is more valuable than "
        "complex models."
    )),
    Document(page_content=(
        "Hyperparameter tuning is critical for model performance. Use "
        "grid search, random search, or Bayesian optimization to find "
        "optimal hyperparameters."
    )),
    Document(page_content=(
        "Regularization techniques like L1, L2, and dropout help prevent "
        "overfitting in neural networks, especially when training data "
        "is limited."
    )),
    Document(page_content=(
        "Cross-validation provides reliable performance estimates and "
        "helps identify if your model is overfitting to training data."
    )),
    Document(page_content=(
        "Data augmentation creates additional training samples by applying "
        "transformations, significantly improving model generalization."
    )),
    Document(page_content=(
        "Transfer learning leverages pre-trained models to achieve better "
        "performance with less training data and computation time."
    )),
    Document(page_content=(
        "Ensemble methods like bagging and boosting combine multiple models "
        "to achieve better predictive performance than any single model."
    )),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    ml_docs,
    embedding=embeddings,
    collection_name="multi_query_demo",
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ─────────────────────────────────────────────────────
# Create Multi-Query Retriever
# ─────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    # By default, generates 3 alternative queries
)

# ─────────────────────────────────────────────────────
# Compare basic vs multi-query
# ─────────────────────────────────────────────────────
query = "How do I improve my model's performance?"

print("═" * 70)
print(f"QUERY: {query}")
print("═" * 70)

# Basic retrieval (single query)
print("\n--- BASIC RETRIEVAL (single query) ---")
basic_results = base_retriever.invoke(query)
print(f"Returned {len(basic_results)} docs:")
for i, doc in enumerate(basic_results):
    print(f"  {i+1}. {doc.page_content[:80]}...")

# Multi-query retrieval
print("\n--- MULTI-QUERY RETRIEVAL ---")
# This will print the generated queries thanks to logging
multi_results = multi_query_retriever.invoke(query)
print(f"\nReturned {len(multi_results)} docs (deduplicated):")
for i, doc in enumerate(multi_results):
    print(f"  {i+1}. {doc.page_content[:80]}...")

# Calculate the recall improvement
basic_ids = set(id(d) for d in basic_results)
multi_ids = set(id(d) for d in multi_results)
new_docs = multi_ids - basic_ids

print(f"\n📈 Multi-query added {len(new_docs)} unique docs")
print(f"   Total recall improvement: {len(new_docs)/len(basic_results)*100:.0f}%")
```

### Custom Multi-Query with Custom Prompt

```python
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# Customize the query generation prompt
custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Generate 5 different
versions of the user's question to retrieve relevant documents from a vector
database. By generating multiple perspectives, you help overcome limitations
of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}

Alternative questions:"""
)

custom_multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=custom_prompt,
)

# Now generates 5 queries instead of default 3
results = custom_multi_query_retriever.invoke(query)
```

---

# PART 5: Re-ranking Strategies

---

## 12. Why Re-ranking?

### The Two-Stage Retrieval Pattern

```
THE INSIGHT:
- Embedding-based retrieval is FAST but APPROXIMATE
- We can use a SLOWER but MORE ACCURATE model to re-rank top results

STAGE 1: Retrieval (fast, approximate)
  Vector search → top 50 candidates
  ~50ms for millions of docs

STAGE 2: Re-ranking (slow, accurate)
  Run expensive model on 50 candidates → top 5
  ~500ms but very accurate

Total: 550ms but much higher quality than just stage 1
```

### Why Two Stages?

```
WHY NOT JUST USE THE SLOW MODEL FROM THE START?

Slow model on millions of docs = HOURS
Fast retrieval on millions of docs = MILLISECONDS
Slow model on 50 docs = MILLISECONDS

Two-stage gives you the speed of fast retrieval
AND the accuracy of slow re-ranking.
This is the production pattern for high-quality RAG.
```

---

## 13. Cross-Encoder Re-rankers

### Bi-Encoder vs Cross-Encoder

```
BI-ENCODER (regular embeddings):
  Embed query separately → query_vector
  Embed doc separately → doc_vector
  Compare: cosine(query_vector, doc_vector)

  Fast: encode once, reuse
  Less accurate: vectors are independent

CROSS-ENCODER (re-ranker):
  Take BOTH query and doc together
  Feed into transformer model
  Output: relevance score

  Slow: must run model for every (query, doc) pair
  Very accurate: model sees both together

  Better at understanding subtle relevance
```

### Implementation with Hugging Face Cross-Encoder

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ─────────────────────────────────────────────────────
# Setup cross-encoder for re-ranking
# ─────────────────────────────────────────────────────

# This downloads a pre-trained cross-encoder model
# BAAI/bge-reranker-base is a popular, free choice
cross_encoder_model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base"
)

# Create the re-ranker compressor
reranker = CrossEncoderReranker(
    model=cross_encoder_model,
    top_n=3,  # Re-rank and keep top 3 from candidates
)

# Wrap the base retriever
# Note: base retriever should return MORE candidates (e.g., 20)
# than what re-ranker keeps (e.g., 3)
base_retriever_for_rerank = vectorstore.as_retriever(
    search_kwargs={"k": 20}  # Fetch many candidates
)

rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever_for_rerank,
)

# ─────────────────────────────────────────────────────
# Compare with and without re-ranking
# ─────────────────────────────────────────────────────
query = "How can I prevent my model from overfitting?"

print("--- WITHOUT RE-RANKING (top 5 by embedding) ---")
basic_results = vectorstore.similarity_search(query, k=5)
for i, doc in enumerate(basic_results):
    print(f"  {i+1}. {doc.page_content[:100]}...")

print("\n--- WITH CROSS-ENCODER RE-RANKING (top 3 of 20 candidates) ---")
reranked_results = rerank_retriever.invoke(query)
for i, doc in enumerate(reranked_results):
    print(f"  {i+1}. {doc.page_content[:100]}...")
```

---

## 14. Cohere Rerank — Managed Re-ranking

### What is Cohere Rerank?

```
Cohere offers a managed re-ranking API service.

Benefits:
✅ No model to host or maintain
✅ Best-in-class accuracy
✅ Fast (cloud-hosted)
✅ Free tier available

Drawbacks:
❌ Costs money beyond free tier
❌ Requires API key
❌ Data leaves your infrastructure
```

### Implementation

```python
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

# Setup Cohere reranker
# Requires: pip install langchain-cohere
# Get API key from https://cohere.com/

cohere_reranker = CohereRerank(
    model="rerank-english-v3.0",  # Or "rerank-multilingual-v3.0"
    top_n=3,                       # Keep top 3 after re-ranking
    cohere_api_key="your-cohere-api-key",
)

# Wrap base retriever
cohere_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

# Use it
results = cohere_retriever.invoke("How to prevent overfitting?")
for doc in results:
    print(f"Score: {doc.metadata.get('relevance_score', 'N/A')}")
    print(f"Content: {doc.page_content[:100]}...")
    print()
```

---

## 15. Hands-On Lab: Add Re-ranking to Pipeline

```python
"""
COMPLETE RE-RANKING COMPARISON
Compare base retrieval vs cross-encoder vs Cohere.
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import time

# Setup knowledge base
docs = [
    Document(page_content="Regularization techniques like L1, L2, and dropout "
                         "help prevent overfitting in neural networks."),
    Document(page_content="Cross-validation gives reliable performance "
                         "estimates and helps identify overfitting."),
    Document(page_content="Data augmentation creates training samples through "
                         "transformations to improve generalization."),
    Document(page_content="Early stopping prevents overfitting by stopping "
                         "training when validation loss increases."),
    Document(page_content="Batch normalization stabilizes training and acts "
                         "as a form of regularization."),
    Document(page_content="Ensemble methods reduce overfitting by combining "
                         "multiple models' predictions."),
    Document(page_content="Feature selection reduces overfitting by removing "
                         "irrelevant or redundant features."),
    Document(page_content="Increasing training data is the most effective way "
                         "to reduce overfitting."),
    Document(page_content="Model complexity should match dataset size to "
                         "prevent overfitting."),
    Document(page_content="Hyperparameter tuning optimizes model parameters "
                         "for better performance."),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    collection_name="rerank_demo",
)

# Setup three retrievers
basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Cross-encoder re-ranker
cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base"
)
cross_encoder_retriever = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(
        model=cross_encoder, top_n=3
    ),
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

# Comparison function
def evaluate(name, retriever, query):
    start = time.time()
    results = retriever.invoke(query)
    elapsed = time.time() - start

    print(f"\n--- {name} (took {elapsed*1000:.0f}ms) ---")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content[:100]}...")

query = "What technique helps prevent overfitting by using validation data?"

print("═" * 70)
print(f"QUERY: {query}")
print("═" * 70)

evaluate("Basic Retrieval", basic_retriever, query)
evaluate("Cross-Encoder Re-ranking", cross_encoder_retriever, query)

# Expected: Cross-encoder ranks the cross-validation doc higher
# because it more accurately understands the query intent
```

---

# PART 6: Failure Modes and Interview Prep

---

## 16. Failure Modes 🚨

```
FAILURE 1: Over-Compression
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Threshold too strict → drops relevant chunks
Result:    Empty context, "I don't know" responses
Fix:       Tune compression thresholds carefully
           Always test with realistic queries

FAILURE 2: Parent Chunks Too Large
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Parent chunks of 5000+ tokens
Result:    Hits context window, expensive
Fix:       Parent: 800-1500 tokens is typical
           Child: 150-300 tokens for precision

FAILURE 3: Self-Query Wrong Schema
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   AttributeInfo descriptions are vague
Result:    LLM generates wrong filters
Fix:       Be SPECIFIC in descriptions
           Include example values where helpful

FAILURE 4: Multi-Query Generating Bad Variations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   LLM generates off-topic variations
Result:    Retrieves irrelevant docs, wastes compute
Fix:       Customize the prompt for your domain
           Validate generated queries before use

FAILURE 5: Re-ranking Too Few Candidates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Re-rank top 5, keep top 3
Result:    Re-ranker has nothing to work with
Fix:       Re-rank should have 4-10x more candidates
           If keeping 3, re-rank from 20+

FAILURE 6: LLMChainExtractor Too Slow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   One LLM call per chunk
Result:    Slow response times, high costs
Fix:       Use EmbeddingsFilter first to reduce chunks
           Then apply LLMChainExtractor to fewer chunks
```

---

## 17. Interview Questions and Answers

### Q1: "What is contextual compression and when would you use it?"

**Strong Answer:**
> "Contextual compression filters or extracts only the relevant parts of retrieved chunks before sending them to the LLM. The problem it solves: retrieved chunks often contain a lot of noise — a 500-token chunk might have only 50 tokens relevant to the query. Without compression, the LLM gets distracted by noise, costs are higher, and we risk 'lost in the middle' issues. LangChain offers two main compressors: LLMChainExtractor uses an LLM to extract relevant sentences — most accurate but slow and expensive. EmbeddingsFilter uses embedding similarity to filter whole chunks — fast and cheap but less precise. In production, I chain them in a pipeline: first remove duplicates with EmbeddingsRedundantFilter, then filter irrelevant chunks with EmbeddingsFilter, then extract relevant sentences with LLMChainExtractor. This combines speed and quality."

---

### Q2: "Explain the chunk size dilemma and how Parent Document Retriever solves it."

**Strong Answer:**
> "The chunk size dilemma is a fundamental tension: small chunks give precise retrieval because embeddings are sharp and focused on one idea, but they lack context — a 200-token chunk may be incomprehensible alone. Large chunks provide rich context but produce diluted embeddings that match less precisely. Parent Document Retriever elegantly solves this by using two chunk sizes. Small child chunks (200 tokens) are embedded and stored in the vector store for precise retrieval matching. Large parent chunks (1000 tokens) are stored separately in a document store. When a query matches a child chunk, we return its parent chunk to the LLM. This gives us the best of both worlds: precise retrieval matching AND rich context for generation."

---

### Q3: "How does Self-Query Retriever work and when is it useful?"

**Strong Answer:**
> "Self-Query Retriever uses an LLM to translate natural language queries into structured filters automatically. For a query like 'electronics under $500 with rating above 4', it extracts: semantic query = 'electronics', filters = {category=electronics, price<500, rating>4}. Then it performs filtered vector search. This is essential when your data has rich metadata that users want to filter by naturally. You define AttributeInfo objects describing each metadata field — name, type, and description. The LLM uses these to generate appropriate filter syntax. It's particularly powerful for product catalogs, document management with metadata tags, and any structured-data RAG where users express filter intent in natural language."

---

### Q4: "What's the difference between bi-encoder and cross-encoder re-ranking?"

**Strong Answer:**
> "Bi-encoders — what regular embeddings use — process query and document independently, producing separate vectors that we compare via cosine similarity. They're fast because you can pre-compute document embeddings, but they're less accurate because the vectors are independent. Cross-encoders take query and document together as input, allowing the model to attend to interactions between them. This is much more accurate but slow because you must run the model for every (query, document) pair. The production pattern: use bi-encoders for fast first-stage retrieval to get top 20-50 candidates, then use cross-encoders to re-rank and keep the top 3-5. This gives you the speed of bi-encoders for the large set and the accuracy of cross-encoders for the final ranking. Popular cross-encoders: BAAI/bge-reranker-base (free, open source) or Cohere Rerank API (managed service)."

---

## 18. Common Mistakes ⚠️

```
MISTAKE 1: "Use all advanced techniques together"
TRUTH:     Each adds latency and cost. Choose strategically
           based on actual failure modes you're seeing.

MISTAKE 2: "Multi-query always improves recall"
TRUTH:     Bad variations dilute results. Quality of
           generated queries depends on LLM and prompt.

MISTAKE 3: "Parent retriever fixes all chunk size issues"
TRUTH:     Still need to pick good parent/child sizes.
           Bad sizes mean bad retrieval, even with this pattern.

MISTAKE 4: "Cross-encoders are always better than bi-encoders"
TRUTH:     Slower and more expensive. Use only for top-k re-ranking.
           Never use as first-stage retriever.

MISTAKE 5: "Self-query understands all filters"
TRUTH:     LLM may misinterpret complex filters.
           Always test with edge cases.

MISTAKE 6: "LLMChainExtractor is always best"
TRUTH:     Slow and expensive. EmbeddingsFilter often
           gives 80% of quality at 5% of the cost.
```

---

## 19. Engineer Mindset 🔧

```
WHAT A REAL RAG ENGINEER THINKS ABOUT:

1. "Identify the failure mode first"
   → Don't add techniques randomly
   → Diagnose: chunks too noisy? → compression
   → Diagnose: missing context? → parent retriever
   → Diagnose: complex queries? → self-query
   → Diagnose: top results wrong? → re-ranking

2. "Cost-quality tradeoff for each technique"
   → Compression: +latency, +cost, +quality
   → Multi-query: +latency, +cost, +recall
   → Re-ranking: +latency, +cost, +precision
   → Choose based on your priorities

3. "Build measurement before optimization"
   → Without metrics, you can't tell what helps
   → Track: precision@k, recall@k, latency, cost
   → A/B test new techniques

4. "Stack techniques in optimal order"
   Production-grade pipeline:
   1. Hybrid search (semantic + BM25)
   2. Multi-query for high-recall first stage
   3. Re-rank with cross-encoder to get precision
   4. Contextual compression to reduce noise
   5. Parent document retrieval for context

5. "Self-query needs good metadata"
   → Investment in metadata quality pays off
   → Update schema as new fields become useful
   → Validate LLM-generated filters in production

6. "Pre-compute what you can"
   → Embedding documents: once, at ingestion
   → BM25 indexes: build at ingestion
   → Only re-rank at query time (must be live)

7. "Monitor and iterate"
   → User feedback (thumbs up/down) reveals failures
   → Log retrieval quality metrics
   → Periodic re-evaluation with test sets
```

---

## 20. Exercises 📝

### Beginner

```
1. Implement basic contextual compression:
   - Setup vector store with 10 docs
   - Add EmbeddingsFilter with threshold 0.7
   - Compare results with/without compression

2. Build a parent-child retriever:
   - Create 3 long documents (1000+ tokens each)
   - Setup ParentDocumentRetriever
   - Query and verify you get parents, not children

3. Try self-query with 5 products:
   - Define AttributeInfo for category, price, rating
   - Test queries: "show electronics", "items under $100"
   - Print the generated filters

4. Try multi-query retrieval:
   - Enable logging to see generated queries
   - Compare basic vs multi-query results
   - Count unique docs retrieved by each
```

### Intermediate

```
5. Build a full compression pipeline:
   - EmbeddingsRedundantFilter (remove duplicates)
   - EmbeddingsFilter (drop irrelevant)
   - LLMChainExtractor (extract sentences)
   - Measure token reduction at each stage

6. Optimize parent-child chunk sizes:
   - Test child=100, parent=500
   - Test child=200, parent=1000
   - Test child=300, parent=1500
   - Which gives best retrieval quality?

7. Customize self-query for your domain:
   - Define rich AttributeInfo with examples
   - Test with edge cases:
     * Multiple filters
     * Range filters
     * Boolean filters
   - Document which queries work/fail

8. Add re-ranking to existing pipeline:
   - Start with hybrid retrieval
   - Add cross-encoder re-ranking
   - Measure precision improvement
```

### Advanced

```
9. Build a production-grade retrieval pipeline:
   - Hybrid search (BM25 + semantic)
   - Multi-query expansion
   - Parent document retrieval
   - Cross-encoder re-ranking
   - LLMChainExtractor compression
   - Comprehensive logging and metrics

10. A/B test advanced techniques:
    - Create test set with ground truth
    - Test combinations of techniques
    - Plot quality improvement per technique
    - Identify minimum techniques for 90% quality

11. Build a query router:
    - Classify queries (factual, analytical, lookup)
    - Route to different retrieval strategies
    - Compare unified vs routed approach

12. Implement custom re-ranking:
    - Build feature extractor (query-doc features)
    - Train custom re-ranker (XGBoost or similar)
    - Compare to cross-encoder baseline
```

---

## 21. Resources 📚

```
LANGCHAIN DOCS:
🔗 Contextual Compression
   https://python.langchain.com/docs/how_to/contextual_compression/

🔗 Parent Document Retriever
   https://python.langchain.com/docs/how_to/parent_document_retriever/

🔗 Self-Query
   https://python.langchain.com/docs/how_to/self_query/

🔗 Multi-Query
   https://python.langchain.com/docs/how_to/MultiQueryRetriever/

🔗 Cross-Encoder Reranker
   https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/

PAPERS:
📄 "Lost in the Middle: How Language Models Use Long Contexts"
   https://arxiv.org/abs/2307.03172

📄 "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
   (Cross-encoder origins)
   https://arxiv.org/abs/1908.10084

📄 "BGE-M3: One Model for Multiple Languages and Modalities"
   https://arxiv.org/abs/2402.03216

MODELS:
🔗 BAAI/bge-reranker-base (Hugging Face)
   https://huggingface.co/BAAI/bge-reranker-base

🔗 Cohere Rerank
   https://cohere.com/rerank

TUTORIALS:
🔗 Advanced RAG Series (LangChain Blog)
   https://blog.langchain.com/tag/rag/

🔗 LlamaIndex Advanced Retrieval
   https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval/
```

---

## Module 6 Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 6 MENTAL MODEL                          │
│                                                                 │
│  Each technique solves a specific failure mode:                │
│                                                                 │
│  CONTEXTUAL COMPRESSION                                         │
│    Problem: Retrieved chunks have too much noise               │
│    Solution: Extract/filter only relevant parts                │
│    Tools: LLMChainExtractor (slow, accurate)                   │
│           EmbeddingsFilter (fast, less precise)                │
│           Pipeline: Combine for best results                   │
│                                                                 │
│  PARENT DOCUMENT RETRIEVER                                      │
│    Problem: Small chunks = precise but no context              │
│             Large chunks = context but imprecise                │
│    Solution: Small children retrieve, large parents return      │
│    Use when: Documents have logical structure                  │
│                                                                 │
│  SELF-QUERY RETRIEVER                                           │
│    Problem: Mixed natural language + structured filters         │
│    Solution: LLM extracts filters automatically                │
│    Use when: Rich metadata + user filter intent                │
│                                                                 │
│  MULTI-QUERY RETRIEVER                                          │
│    Problem: Single query misses relevant variations            │
│    Solution: LLM generates query variations                    │
│    Use when: Recall matters more than latency                  │
│                                                                 │
│  RE-RANKING                                                     │
│    Problem: Embedding similarity is approximate                │
│    Solution: Use slow accurate model on top candidates          │
│    Tools: Cross-encoder (free) or Cohere Rerank (managed)      │
│                                                                 │
│  GOLDEN RULES:                                                  │
│  1. Diagnose failure mode BEFORE choosing technique            │
│  2. Each adds latency/cost — use strategically                 │
│  3. Stack in order: hybrid → multi-query → re-rank → compress  │
│  4. Always measure quality improvement                          │
│  5. Production = combination of techniques, not one alone      │
└─────────────────────────────────────────────────────────────────┘
```