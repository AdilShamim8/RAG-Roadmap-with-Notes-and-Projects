# Module 5: Basic Retrieval Techniques 

## The Art of Finding the Right Information

---

> **Before I explain — let me ask:**
>
> *"What do you think 'retrieval' means in RAG?"*
>
> A beginner might say: *"Searching the vector database for similar documents."*
>
> That's the mechanic — but the real challenge is **finding the RIGHT documents, not just SIMILAR ones**.
>
> Here's the critical insight: **The LLM can only be as good as what you retrieve.**
> Bad retrieval = bad answer, always. No prompt engineering can fix it.
>
> This module is about mastering the different retrieval strategies — because **one size does not fit all queries.**

---

# The Big Picture: Why Retrieval Strategies Matter

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  THE RETRIEVAL QUALITY EQUATION:                                 │
│                                                                  │
│  RAG Quality = f(Retrieval Quality, LLM Quality)                 │
│                                                                  │
│  Even GPT-4 with perfect prompts CANNOT answer correctly         │
│  if you give it WRONG context.                                   │
│                                                                  │
│  Retrieval failures = silent failures                             │
│  → LLM still confidently generates an answer                     │
│  → User can't tell the context was wrong                         │
│  → Hallucination disguised as fact                                │
│                                                                  │
│  Different queries need different strategies:                    │
│  - Specific fact?     → Pure similarity                          │
│  - Complex analysis?  → MMR for diversity                        │
│  - Exact terms?       → Keyword search                            │
│  - Mixed?             → Hybrid search                             │
│                                                                  │
│  MASTER OF RETRIEVAL = MASTER OF RAG                              │
└──────────────────────────────────────────────────────────────────┘
```

---

# PART 1: Similarity Search Fundamentals

---

## 1. How Vector Similarity Search Works (End-to-End)

### The Complete Mental Model

```
USER QUERY → "What is the refund policy?"
                       │
                       ▼
         ┌─────────────────────────────┐
         │ 1. EMBED THE QUERY          │
         │ Use SAME embedding model    │
         │ that embedded documents     │
         │                             │
         │ "What is the refund policy?"│
         │           ↓                 │
         │ [0.23, -0.18, 0.41, ...]    │
         │ (1536-dim vector)           │
         └─────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ 2. SEARCH VECTOR INDEX      │
         │ Compare query vector to     │
         │ all stored document vectors │
         │ using cosine similarity     │
         │                             │
         │ HNSW algorithm finds        │
         │ approximate nearest         │
         │ neighbors in O(log N) time  │
         └─────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ 3. RANK BY SIMILARITY       │
         │                             │
         │ Doc A: 0.94 ← most similar  │
         │ Doc B: 0.91                 │
         │ Doc C: 0.87                 │
         │ Doc D: 0.82                 │
         │ Doc E: 0.76                 │
         │ ...                         │
         └─────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ 4. RETURN TOP-K             │
         │                             │
         │ if k=3, return top 3 docs:  │
         │ [Doc A, Doc B, Doc C]       │
         │                             │
         │ These become CONTEXT for    │
         │ the LLM to answer with      │
         └─────────────────────────────┘
```

### Why Order Matters

```
Top-k results are presented to the LLM in order.

LLMs often pay more attention to:
- Beginning of context (primacy effect)
- End of context (recency effect)
- Less attention to MIDDLE (lost in the middle problem!)

So if k=10 and the BEST chunk is at position 5:
The LLM might IGNORE it!

Best practice: Use small k (3-5) so all retrieved
docs get attention. Quality > Quantity.
```

---

## 2. The similarity_search() Method

### Basic Usage Pattern

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# THE simplest possible retrieval
results = vectorstore.similarity_search(
    query="What is the vacation policy?",
    k=4,  # Return top 4 most similar documents
)

# Returns: List[Document]
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Content:  {doc.page_content[:200]}")
    print(f"Source:   {doc.metadata.get('source')}")
    print()

# What you DON'T see:
# - The similarity scores
# - How close/far apart these are
# - Whether any of them are actually relevant!
```

### The Hidden Problem with Basic similarity_search

```
PROBLEM: similarity_search ALWAYS returns k results.

Even if NONE of them are relevant!

Example:
Query: "What is quantum entanglement?"
Knowledge Base: Only contains HR documents

similarity_search returns 4 documents:
  Result 1: "Vacation policy..."  (similarity: 0.31)
  Result 2: "Sick leave..."       (similarity: 0.29)
  Result 3: "Dress code..."       (similarity: 0.27)
  Result 4: "Office hours..."     (similarity: 0.25)

ALL of these are IRRELEVANT!
But the LLM gets them as context and may hallucinate.

This is why we need similarity_search_with_score
to actually evaluate relevance.
```

---

## 3. Search with Relevance Scores

### Getting Scores Back

```python
# Use _with_score variant to see relevance values
results_with_scores = vectorstore.similarity_search_with_score(
    query="What is the vacation policy?",
    k=4,
)

# Returns: List[Tuple[Document, float]]
for doc, score in results_with_scores:
    print(f"Score:    {score:.4f}")
    print(f"Content:  {doc.page_content[:100]}")
    print()
```

### Score Interpretation — The Critical Trap

```
DIFFERENT VECTOR STORES RETURN DIFFERENT SCORES!

CHROMA returns DISTANCE (lower = better):
  0.0  → identical
  0.3  → very similar
  1.0  → not similar
  2.0  → opposite

PINECONE returns SIMILARITY (higher = better):
  1.0  → identical
  0.9  → very similar
  0.5  → somewhat similar
  0.0  → not similar

QDRANT returns SIMILARITY (higher = better):
  Same as Pinecone

FAISS depends on the metric:
  L2:        distance (lower = better)
  Inner Prod: similarity (higher = better)
  Cosine:    similarity (higher = better)

ALWAYS check your vector store's documentation!
Or run a test with a known similar pair to verify direction.
```

### Verifying Score Direction (Always Do This)

```python
# Quick sanity check — what direction is your score?
def check_score_direction(vectorstore):
    """Test if higher or lower score means more similar."""

    # Add a unique test document
    vectorstore.add_texts(
        texts=["This is a unique test document about purple elephants"],
        ids=["test-direction-check"]
    )

    # Search for EXACT match (should be most similar)
    results = vectorstore.similarity_search_with_score(
        query="This is a unique test document about purple elephants",
        k=1
    )

    score = results[0][1]
    print(f"Score for exact match: {score:.4f}")

    if score < 0.5:
        print("→ LOWER scores mean MORE similar (DISTANCE)")
        print("→ Use score < threshold to filter")
    else:
        print("→ HIGHER scores mean MORE similar (SIMILARITY)")
        print("→ Use score > threshold to filter")

    # Clean up
    vectorstore.delete(ids=["test-direction-check"])

check_score_direction(vectorstore)
```

---

## 4. Configuring k and Search Parameters

### Choosing the Right k

```
THE k TRADEOFF:

Too Small (k=1):
  ❌ Single chunk might miss complete answer
  ❌ No diversity in context
  ❌ If wrong chunk retrieved → completely wrong answer
  ✅ Fastest, cheapest

Too Large (k=20):
  ❌ Most retrieved chunks are irrelevant noise
  ❌ Lost in the middle — LLM ignores middle chunks
  ❌ Higher token cost
  ❌ Slower response
  ✅ Higher recall (more likely to include relevant doc)

Sweet Spot (k=3 to k=7):
  ✅ Multiple chunks for context
  ✅ All chunks get LLM attention
  ✅ Reasonable cost
  ✅ Allows for re-ranking if needed
```

### Decision Framework for k

```
Query Type                      → Recommended k
─────────────────────────────────────────────────
Factoid (single fact)           → k=3
General Q&A                     → k=4-5
Multi-aspect questions          → k=5-7
Analytical/summary              → k=7-10
"Show me everything about X"    → k=10-15 (with MMR)
```

### Hands-On: Basic Similarity Search Implementation

```python
"""
Complete similarity search implementation
with all the bells and whistles.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Tuple

class SimilaritySearchEngine:
    """Production-quality similarity search wrapper."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma_db",
    ):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def setup_demo_data(self):
        """Add demo HR documents for testing."""
        docs = [
            Document(
                page_content="Employees are entitled to 15 vacation days per year.",
                metadata={"category": "vacation", "id": "v1"}
            ),
            Document(
                page_content="Vacation days can be carried over up to 5 days.",
                metadata={"category": "vacation", "id": "v2"}
            ),
            Document(
                page_content="Sick leave allows 10 days per year with doctor's note.",
                metadata={"category": "sick_leave", "id": "s1"}
            ),
            Document(
                page_content="Maternity leave is 12 weeks paid for primary caregiver.",
                metadata={"category": "parental", "id": "p1"}
            ),
            Document(
                page_content="Paternity leave is 4 weeks paid for secondary caregiver.",
                metadata={"category": "parental", "id": "p2"}
            ),
            Document(
                page_content="Remote work is allowed 3 days per week with approval.",
                metadata={"category": "remote_work", "id": "r1"}
            ),
            Document(
                page_content="Office hours are 9 AM to 5 PM Monday through Friday.",
                metadata={"category": "office_hours", "id": "o1"}
            ),
            Document(
                page_content="Dress code is business casual with Fridays being casual.",
                metadata={"category": "dress_code", "id": "d1"}
            ),
        ]

        ids = [doc.metadata["id"] for doc in docs]
        self.vectorstore.add_documents(documents=docs, ids=ids)
        print(f"✅ Added {len(docs)} demo documents")

    def basic_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """Simple similarity search."""
        return self.vectorstore.similarity_search(query=query, k=k)

    def search_with_analysis(
        self,
        query: str,
        k: int = 4,
    ) -> dict:
        """Comprehensive search with score analysis."""

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
        )

        # Analyze score distribution
        scores = [score for _, score in results]
        analysis = {
            "query":             query,
            "num_results":       len(results),
            "best_score":        min(scores),    # Chroma: lower = better
            "worst_score":       max(scores),
            "score_range":       max(scores) - min(scores),
            "results":           [
                {
                    "rank":     i + 1,
                    "score":    score,
                    "category": doc.metadata.get("category"),
                    "content":  doc.page_content,
                }
                for i, (doc, score) in enumerate(results)
            ]
        }

        return analysis

    def display_results(self, analysis: dict):
        """Pretty print search results."""
        print(f"\n{'═'*70}")
        print(f"🔍 Query: {analysis['query']}")
        print(f"{'═'*70}")
        print(f"Score range: {analysis['best_score']:.4f} (best) "
              f"to {analysis['worst_score']:.4f} (worst)")
        print(f"Range spread: {analysis['score_range']:.4f}")
        print(f"\n📋 Results (k={analysis['num_results']}):")

        for r in analysis["results"]:
            # Create visual score bar (Chroma: lower = better)
            # Normalize 0-2 to 0-20 chars
            normalized = max(0, min(20, int((2 - r["score"]) * 10)))
            bar = "█" * normalized + "░" * (20 - normalized)

            print(f"\n  Rank {r['rank']}: [{bar}] score={r['score']:.4f}")
            print(f"    Category: {r['category']}")
            print(f"    Content:  {r['content']}")


# ── Demo ───────────────────────────────────────────────────
engine = SimilaritySearchEngine(collection_name="hr_demo")
engine.setup_demo_data()

# Test 1: Direct query
analysis = engine.search_with_analysis(
    query="How many vacation days do I get?",
    k=3,
)
engine.display_results(analysis)

# Test 2: Indirect query (semantic match)
analysis = engine.search_with_analysis(
    query="time off for new parents",
    k=3,
)
engine.display_results(analysis)

# Test 3: Off-topic query (should show high scores = low similarity)
analysis = engine.search_with_analysis(
    query="quantum physics and black holes",
    k=3,
)
engine.display_results(analysis)
# Notice: Even off-topic returns 3 results!
# Scores will be high (low similarity) — this is the threshold problem
```

---

# PART 2: Similarity Score Threshold

---

## 5. Why Thresholds Matter

### The "Always Returns Something" Problem

```
SCENARIO: Customer support chatbot for a coffee company

User asks: "How do I configure my Kubernetes cluster?"

Without threshold:
  similarity_search returns top-3 results:
  - "Our coffee beans are sourced ethically" (sim=0.31)
  - "Brewing instructions for espresso" (sim=0.28)
  - "Coffee subscription pricing" (sim=0.25)

  ALL IRRELEVANT but still returned!
  LLM gets these as context.
  LLM tries to answer Kubernetes question with coffee docs.
  Result: hallucination or nonsense response.

With threshold (sim > 0.5):
  No results meet threshold.
  System responds: "I don't have information about that topic.
                   Please ask about our coffee products."

  This is the CORRECT behavior!
```

### The Trust-Building Effect of Thresholds

```
GOOD RAG SYSTEMS KNOW WHEN TO SAY "I DON'T KNOW"

Without thresholds:
  100% of queries get an answer
  But many answers are hallucinated
  Users lose trust over time

With thresholds:
  85% of queries get a confident answer
  15% get "I don't have that information"
  Users TRUST the answers they DO get

A confident "I don't know" > a confident hallucination
```

---

## 6. Setting Appropriate Thresholds

### How to Determine the Right Threshold

```
STEP-BY-STEP THRESHOLD CALIBRATION:

Step 1: Collect ~50 representative queries
        - Real user queries from logs
        - Mix of in-domain and out-of-domain

Step 2: Run similarity search for each
        Record the top-1 similarity score

Step 3: Manually label each result
        Relevant (1) or Not Relevant (0)

Step 4: Find the threshold that maximizes:
        - Precision (% of returned that are relevant)
        - Recall (% of relevant that are returned)

Step 5: Tune based on use case:
        - Customer support: High precision (95%+)
          (false positives = wrong answers = lost trust)
        - Research assistant: High recall (80%+)
          (false negatives = missing valuable docs)
```

### Threshold Guidelines (Rough Starting Points)

```
For OpenAI text-embedding-3-small (cosine similarity):

> 0.85   → Excellent match, almost certainly relevant
0.70-0.85 → Strong match, very likely relevant
0.55-0.70 → Moderate match, possibly relevant
0.40-0.55 → Weak match, often irrelevant
< 0.40   → No semantic relationship

RECOMMENDED STARTING THRESHOLDS:
  Strict (high precision):    > 0.75
  Balanced:                   > 0.65
  Lenient (high recall):      > 0.55

For Chroma (which returns DISTANCE, lower = better):
  Use these inversely:
  Strict:    distance < 0.5
  Balanced:  distance < 0.7
  Lenient:   distance < 0.9

ALWAYS calibrate for your specific:
- Embedding model
- Document domain
- Query types
```

---

## 7. LangChain similarity_score_threshold Search Type

### Method 1: Using as_retriever with threshold

```python
# LangChain's standard way: retriever with threshold filter
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,                    # Fetch up to 10 candidates
        "score_threshold": 0.7,     # Only keep those with score >= 0.7
                                    # (LangChain normalizes scores)
    }
)

results = retriever.invoke("What is the vacation policy?")
print(f"Returned {len(results)} results above threshold")

# IMPORTANT: LangChain normalizes scores to [0, 1] where higher = better
# This is consistent across vector stores!
# But the actual threshold value still depends on your embeddings
```

### Method 2: Manual Threshold Implementation

```python
def search_with_threshold(
    vectorstore,
    query: str,
    k: int = 10,
    distance_threshold: float = 0.7,  # For Chroma (lower = better)
) -> List[Document]:
    """
    Manual threshold implementation — gives you full control.
    Use when you need score visibility.
    """

    # Fetch more than needed (we'll filter)
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
    )

    # Filter by threshold
    filtered = [
        doc for doc, score in results
        if score < distance_threshold  # Chroma: lower = better
    ]

    print(f"Fetched {len(results)} candidates")
    print(f"Filtered to {len(filtered)} above threshold")

    return filtered


# Better: return scores too for transparency
def search_with_threshold_and_scores(
    vectorstore,
    query: str,
    k: int = 10,
    distance_threshold: float = 0.7,
) -> List[Tuple[Document, float]]:
    """Return filtered results WITH scores."""

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
    )

    filtered = [
        (doc, score) for doc, score in results
        if score < distance_threshold
    ]

    return filtered
```

### Production-Grade Threshold Retriever

```python
class ThresholdRetriever:
    """
    Production-quality retrieval with threshold and fallback.
    """

    def __init__(
        self,
        vectorstore,
        primary_threshold: float = 0.6,    # Strict threshold
        fallback_threshold: float = 0.8,   # Lenient fallback
        min_results: int = 1,
        max_results: int = 5,
    ):
        self.vectorstore = vectorstore
        self.primary_threshold = primary_threshold
        self.fallback_threshold = fallback_threshold
        self.min_results = min_results
        self.max_results = max_results

    def retrieve(self, query: str) -> dict:
        """
        Smart retrieval with threshold and fallback strategy.
        """

        # Step 1: Fetch generously
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.max_results * 3,  # Fetch 3x to allow filtering
        )

        # Step 2: Try strict threshold first
        strict_results = [
            (doc, score) for doc, score in results
            if score < self.primary_threshold
        ][:self.max_results]

        if len(strict_results) >= self.min_results:
            return {
                "status":    "strict_match",
                "threshold": self.primary_threshold,
                "results":   strict_results,
                "message":   f"Found {len(strict_results)} confident matches"
            }

        # Step 3: Fallback to lenient threshold
        lenient_results = [
            (doc, score) for doc, score in results
            if score < self.fallback_threshold
        ][:self.max_results]

        if len(lenient_results) >= self.min_results:
            return {
                "status":    "lenient_match",
                "threshold": self.fallback_threshold,
                "results":   lenient_results,
                "message":   f"Found {len(lenient_results)} possible matches "
                            f"(lower confidence)"
            }

        # Step 4: No matches at all
        return {
            "status":    "no_match",
            "threshold": self.fallback_threshold,
            "results":   [],
            "message":   "No relevant information found. "
                         "Try rephrasing your query."
        }


# Usage
retriever = ThresholdRetriever(
    vectorstore=vectorstore,
    primary_threshold=0.6,    # Chroma distance
    fallback_threshold=0.8,
    min_results=1,
    max_results=5,
)

# Test with different queries
for query in [
    "vacation policy",              # In-domain
    "is the office dog-friendly?",  # Borderline
    "configure my router",          # Completely off-topic
]:
    result = retriever.retrieve(query)
    print(f"\n🔍 Query: {query}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    for doc, score in result["results"]:
        print(f"  [{score:.3f}] {doc.page_content[:80]}")
```

---

# PART 3: MMR (Maximal Marginal Relevance)

---

## 8. The Problem MMR Solves: Redundancy

### The Diversity Problem in Similarity Search

```
SCENARIO: User asks "Tell me about Python programming"

Your knowledge base has 100 chunks about Python.
Many chunks discuss similar topics.

similarity_search returns top-5:

Result 1: "Python is a high-level programming language..."   (0.95)
Result 2: "Python is a popular high-level language..."        (0.94)
Result 3: "Python is widely used as a high-level language..." (0.93)
Result 4: "Python is a versatile high-level language..."      (0.92)
Result 5: "Python is a powerful high-level language..."       (0.91)

ALL 5 SAY ESSENTIALLY THE SAME THING!

The LLM gets 5 paraphrases of the same fact.
No information about:
  - Python's history
  - Python libraries
  - Python use cases
  - Python syntax

Result: Shallow, repetitive answer.
```

### How MMR Fixes This

```
MMR (Maximal Marginal Relevance) BALANCES TWO GOALS:
1. RELEVANCE: How similar is the document to the query?
2. DIVERSITY: How different is it from already-selected docs?

MMR Algorithm:
━━━━━━━━━━━━━━

Step 1: Fetch larger candidate pool (fetch_k = 20)

Step 2: Pick THE most relevant document (highest similarity to query)
        → This is your first result

Step 3: For each remaining candidate, calculate:
        MMR_score = λ × similarity(query, doc)
                    - (1-λ) × max_similarity(doc, already_selected)

        - First term: How relevant to query (we want HIGH)
        - Second term: How similar to already-picked (we want LOW)
        - λ controls the tradeoff

Step 4: Pick the document with highest MMR_score

Step 5: Repeat until k documents selected

Result: k documents that are both RELEVANT and DIVERSE!
```

### The Lambda Parameter — Visual Intuition

```
λ = 1.0  → PURE RELEVANCE (same as similarity_search)
            All results may be near-duplicates

λ = 0.7  → MOSTLY RELEVANCE, some diversity
            Default for many use cases

λ = 0.5  → BALANCED RELEVANCE AND DIVERSITY
            Good for general use

λ = 0.3  → MOSTLY DIVERSITY, some relevance
            Risk: less relevant results

λ = 0.0  → PURE DIVERSITY (ignores query!)
            Just picks documents most different from each other

VISUAL EXAMPLE:
Query: "Python programming"
Candidates: 10 docs about Python

λ=1.0:  [Python intro 1, Python intro 2, Python intro 3, ...]
        All similar intros, no breadth

λ=0.5:  [Python intro, Python history, Python libraries,
         Python web frameworks, Python data science]
        Different aspects of Python

λ=0.0:  [Python intro, Cobra snake species, Programming pizza,
         Math operations, Office furniture]
        Maximally different but mostly irrelevant!
```

---

## 9. fetch_k vs k Parameters

```
TWO IMPORTANT PARAMETERS IN MMR:

k:        How many documents to RETURN
fetch_k:  How many documents to FETCH for consideration

Why two parameters?
━━━━━━━━━━━━━━━━━━━

MMR needs a POOL of candidates to choose from for diversity.
If you only fetch 5 candidates and need 5 diverse results,
you have NO room to optimize diversity!

GOOD: fetch_k = 20, k = 5
       Fetch 20 candidates, intelligently pick 5 diverse ones

BAD:   fetch_k = 5, k = 5
       Just return the top-5, no diversity benefit

GUIDELINES:
fetch_k should be 3-5x larger than k

For k=3:   fetch_k = 10-15
For k=5:   fetch_k = 15-25
For k=10:  fetch_k = 30-50

Tradeoff:
  Higher fetch_k → Better diversity but slower
  Lower fetch_k  → Faster but less diversity benefit
```

---

## 10. MMR Use Cases

```
WHEN TO USE MMR:

✅ DOCUMENT SUMMARIZATION
   "Summarize what you know about X"
   Need diverse perspectives, not duplicates

✅ COMPLEX MULTI-FACETED QUERIES
   "Tell me everything about our product"
   Need coverage of multiple aspects

✅ KNOWLEDGE BASE WITH REDUNDANT CONTENT
   Multiple sources discuss same topics
   MMR avoids "lost in similar chunks"

✅ EXPLORATORY RESEARCH
   "What do we know about climate change?"
   User wants breadth of information

✅ EDUCATIONAL CONTENT GENERATION
   Need to cover topic from multiple angles


WHEN TO STICK WITH PURE SIMILARITY:

❌ SPECIFIC FACTUAL QUERIES
   "What is the date of the meeting?"
   You want the SPECIFIC answer, not diverse opinions

❌ SMALL KNOWLEDGE BASE
   <100 chunks: diversity may sacrifice relevance

❌ UNIQUE DOCUMENT PER TOPIC
   If your docs are already diverse, MMR adds overhead
```

---

## 11. Hands-On: Compare MMR vs Standard Similarity

```python
"""
Compare similarity_search vs MMR on the same data
to see the diversity benefit in action.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create knowledge base with intentional redundancy
# (similar to real-world situations)
redundant_docs = [
    # Multiple chunks about vacation (similar content)
    Document(page_content="Employees receive 15 vacation days annually.",
             metadata={"topic": "vacation", "id": "v1"}),
    Document(page_content="Annual vacation allowance is 15 days per employee.",
             metadata={"topic": "vacation", "id": "v2"}),
    Document(page_content="Each employee gets 15 paid vacation days per year.",
             metadata={"topic": "vacation", "id": "v3"}),
    Document(page_content="The company provides 15 days of vacation yearly.",
             metadata={"topic": "vacation", "id": "v4"}),

    # Diverse but related topics
    Document(page_content="Sick leave is 10 days per year with doctor's note.",
             metadata={"topic": "sick_leave", "id": "s1"}),
    Document(page_content="Maternity leave is 12 weeks paid.",
             metadata={"topic": "parental", "id": "p1"}),
    Document(page_content="Remote work allowed 3 days per week.",
             metadata={"topic": "remote", "id": "r1"}),
    Document(page_content="Vacation requests must be submitted 2 weeks ahead.",
             metadata={"topic": "vacation_process", "id": "vp1"}),
    Document(page_content="Unused vacation can be paid out at year end.",
             metadata={"topic": "vacation_payout", "id": "vp2"}),
]

vectorstore = Chroma.from_documents(
    documents=redundant_docs,
    embedding=embeddings,
    collection_name="mmr_demo",
    persist_directory="./mmr_demo_db",
)

# ─────────────────────────────────────────────────────
# Standard Similarity Search
# ─────────────────────────────────────────────────────
print("═" * 70)
print("STANDARD SIMILARITY SEARCH")
print("═" * 70)

query = "Tell me about vacation policies"

similarity_results = vectorstore.similarity_search(query=query, k=4)

print(f"\n🔍 Query: {query}\n")
print("Top 4 results (notice the redundancy):")
for i, doc in enumerate(similarity_results):
    print(f"\n  Rank {i+1}: [{doc.metadata['topic']}]")
    print(f"    {doc.page_content}")


# ─────────────────────────────────────────────────────
# MMR Search (Diversity-Aware)
# ─────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("MMR SEARCH (DIVERSITY-AWARE)")
print("═" * 70)

mmr_results = vectorstore.max_marginal_relevance_search(
    query=query,
    k=4,                # Same number of results
    fetch_k=15,         # But fetch more candidates
    lambda_mult=0.5,    # Balance relevance and diversity
)

print(f"\n🔍 Query: {query}\n")
print("Top 4 results (notice the diversity):")
for i, doc in enumerate(mmr_results):
    print(f"\n  Rank {i+1}: [{doc.metadata['topic']}]")
    print(f"    {doc.page_content}")


# ─────────────────────────────────────────────────────
# Compare Lambda Values
# ─────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("EFFECT OF LAMBDA PARAMETER")
print("═" * 70)

for lambda_val in [0.0, 0.3, 0.5, 0.7, 1.0]:
    print(f"\n─── λ = {lambda_val} ───")

    results = vectorstore.max_marginal_relevance_search(
        query=query,
        k=4,
        fetch_k=15,
        lambda_mult=lambda_val,
    )

    # Show diversity by listing unique topics
    topics = [doc.metadata['topic'] for doc in results]
    unique_topics = len(set(topics))

    print(f"  Topics returned: {topics}")
    print(f"  Unique topics:   {unique_topics}/4")

    if lambda_val == 1.0:
        print("  → Pure relevance (same as similarity_search)")
    elif lambda_val == 0.0:
        print("  → Pure diversity (ignores query relevance)")
    else:
        print(f"  → Balanced (relevance weight = {lambda_val})")
```

**Expected output:**
```
λ = 1.0: [vacation, vacation, vacation, vacation] - 1 unique topic
λ = 0.5: [vacation, vacation_process, vacation_payout, sick_leave] - 4 unique
λ = 0.0: [vacation, sick_leave, remote, parental] - 4 unique (may lose vacation focus)
```

---

# PART 4: Hybrid Search

---

## 12. Why Pure Semantic Search Isn't Enough

### The Semantic Search Weakness

```
SEMANTIC SEARCH IS GREAT FOR:
✅ "How do I get my money back?"  → finds "refund policy"
✅ Conceptual queries
✅ Paraphrased queries
✅ Synonym handling

SEMANTIC SEARCH STRUGGLES WITH:
❌ Exact product codes:    "SKU-12345"
❌ Specific people names:  "Dr. Sarah Chen"
❌ Technical IDs:          "Error code 0x80004005"
❌ Rare jargon:            "thrombocytopenia"
❌ Acronyms:              "RAG vs FAANG vs ICBM"
❌ Recent terminology:    Words not seen during training
```

### Real-World Example

```
Knowledge base has product manuals.
User asks: "How do I install software version 3.2.1?"

SEMANTIC SEARCH:
Embeddings don't strongly encode the specific version number.
Returns docs about software installation in general.
Misses the version-specific document!

KEYWORD SEARCH (BM25):
Sees "3.2.1" as a rare, specific term.
Boosts documents containing exact match "3.2.1".
Finds the version-specific document!

HYBRID SEARCH (Combine both):
Gets the best of both worlds:
- Semantic finds "installation guide" docs
- Keyword finds "version 3.2.1" specific docs
- Combined ranking returns BEST of both
```

---

## 13. Sparse vs Dense Vectors

### Two Fundamentally Different Representations

```
DENSE VECTORS (Embeddings):
━━━━━━━━━━━━━━━━━━━━━━━━━━

Text: "How do I get a refund?"
Dense vector: [0.23, -0.18, 0.41, 0.62, ..., 0.31]
              (1536 numbers, all non-zero)

Properties:
- Every dimension has a value
- Encodes SEMANTIC meaning
- Captured by neural networks
- Good for: meaning, paraphrasing


SPARSE VECTORS (BM25, TF-IDF):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Text: "How do I get a refund?"
Sparse vector: [0, 0, 0, ..., 0.45, 0, 0, ..., 0.32, 0, ...]
               (50,000+ dimensions, mostly zeros)

Each dimension = one word in vocabulary
Value = importance of that word

Properties:
- Mostly zeros (sparse!)
- Dimensions correspond to specific WORDS
- Calculated by mathematical formulas (not neural)
- Good for: exact terms, rare words


DENSE vs SPARSE:
┌──────────────┬─────────────────┬─────────────────┐
│ Aspect       │ Dense           │ Sparse          │
├──────────────┼─────────────────┼─────────────────┤
│ Dimensions   │ 384-3072        │ 50,000+         │
│ Most values  │ Non-zero        │ Zero            │
│ Captures     │ Meaning         │ Word presence   │
│ Method       │ Neural network  │ Math formula    │
│ Vocab        │ No vocab        │ Fixed vocab     │
│ Rare words   │ Smoothed        │ Highlighted     │
│ Synonyms     │ Match well      │ Don't match     │
│ Exact terms  │ Don't match well│ Match perfectly │
└──────────────┴─────────────────┴─────────────────┘
```

---

## 14. BM25 — The Keyword Search Algorithm

### What is BM25?

```
BM25 = Best Matching 25 (the 25th iteration of the algorithm)

It's a SOPHISTICATED keyword matching algorithm that:
1. Gives weight to terms based on rarity (rare = more important)
2. Considers term frequency in document (more occurrences = more relevant)
3. Normalizes for document length (long docs don't unfairly score higher)

This is what powers most modern search engines!
Including ElasticSearch, Solr, and many others.
```

### BM25 Formula (Simplified)

```
For each query term, BM25 calculates a score:

BM25(query, doc) = Σ IDF(term) × TF_normalized(term, doc)
                   for each term in query

Where:
  IDF(term) = log(N / df(term))
              N = total documents
              df = how many docs contain the term
              → Rare terms get HIGHER IDF

  TF_normalized = considers term frequency AND document length

INTUITION:
- If "refund" appears in only 5 of 10,000 docs → very high IDF
- If "the" appears in all 10,000 docs → near-zero IDF (useless)
- Doc with "refund" mentioned 5 times → higher TF score
- But not proportionally (saturates) → prevents keyword stuffing
```

### BM25Retriever in LangChain

```python
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# BM25 doesn't need embeddings — it's a pure text algorithm
documents = [
    Document(page_content="Refund policy allows returns within 30 days."),
    Document(page_content="Product SKU-12345 has been discontinued."),
    Document(page_content="Customer service hours are 9-5 EST."),
    Document(page_content="To install software version 3.2.1, follow these steps."),
    Document(page_content="Returns require original receipt and packaging."),
]

# Create BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3  # Return top 3

# Query with exact terms
results = bm25_retriever.invoke("software version 3.2.1")
for doc in results:
    print(f"BM25: {doc.page_content}")

# BM25 will rank the version-specific doc HIGHEST
# because "3.2.1" is a very rare term (high IDF)
```

---

## 15. EnsembleRetriever — Combining Dense and Sparse

### How Ensemble Works

```
ENSEMBLE RETRIEVER PROCESS:

User Query → "How to install software version 3.2.1?"
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────┐                ┌──────────────┐
│ Dense        │                │ Sparse       │
│ (Embeddings) │                │ (BM25)       │
│              │                │              │
│ Top-K based  │                │ Top-K based  │
│ on semantic  │                │ on keywords  │
│ similarity   │                │              │
└──────────────┘                └──────────────┘
        │                               │
        ▼                               ▼
   [doc_a, doc_b, doc_c]        [doc_b, doc_d, doc_e]
        │                               │
        └───────────────┬───────────────┘
                        ▼
            ┌─────────────────────────┐
            │ RECIPROCAL RANK FUSION  │
            │ (RRF)                   │
            │                         │
            │ Combine rankings with   │
            │ weights:                │
            │ - Dense weight: 0.5     │
            │ - Sparse weight: 0.5    │
            └─────────────────────────┘
                        │
                        ▼
            Final ranking: [doc_b, doc_a, doc_d, doc_c, doc_e]

The doc that appears HIGH in BOTH rankings (doc_b)
gets the BEST combined score.
```

### Hands-On: Building Hybrid Search

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Setup sample documents
documents = [
    Document(
        page_content="Refund policy allows returns within 30 days of purchase.",
        metadata={"id": "1"}
    ),
    Document(
        page_content="Product SKU-12345 has been discontinued effective Q3 2024.",
        metadata={"id": "2"}
    ),
    Document(
        page_content="Customer service is available Monday-Friday 9-5 EST.",
        metadata={"id": "3"}
    ),
    Document(
        page_content="To install version 3.2.1 of our software, run the installer.",
        metadata={"id": "4"}
    ),
    Document(
        page_content="Returns require original receipt and undamaged packaging.",
        metadata={"id": "5"}
    ),
    Document(
        page_content="Error code 0x80004005 indicates a network connectivity issue.",
        metadata={"id": "6"}
    ),
    Document(
        page_content="Money-back guarantee covers all purchases for full refund.",
        metadata={"id": "7"}
    ),
    Document(
        page_content="Technical support contact: support@company.com",
        metadata={"id": "8"}
    ),
]

# ─────────────────────────────────────────────────────
# 1. Setup Dense (Semantic) Retriever
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="hybrid_demo",
    persist_directory="./hybrid_demo_db",
)
dense_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ─────────────────────────────────────────────────────
# 2. Setup Sparse (Keyword) Retriever
# ─────────────────────────────────────────────────────
sparse_retriever = BM25Retriever.from_documents(documents)
sparse_retriever.k = 3

# ─────────────────────────────────────────────────────
# 3. Combine into Ensemble Retriever
# ─────────────────────────────────────────────────────
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5],  # Equal weight - tune for your use case!
)

# ─────────────────────────────────────────────────────
# 4. Compare All Three on Different Query Types
# ─────────────────────────────────────────────────────

def compare_retrievers(query: str):
    print(f"\n{'═'*70}")
    print(f"🔍 Query: {query}")
    print('═' * 70)

    # Dense only
    print(f"\n📊 DENSE (Semantic) Results:")
    dense_results = dense_retriever.invoke(query)
    for i, doc in enumerate(dense_results):
        print(f"  {i+1}. [{doc.metadata['id']}] {doc.page_content[:70]}")

    # Sparse only
    print(f"\n🔤 SPARSE (BM25/Keyword) Results:")
    sparse_results = sparse_retriever.invoke(query)
    for i, doc in enumerate(sparse_results):
        print(f"  {i+1}. [{doc.metadata['id']}] {doc.page_content[:70]}")

    # Ensemble
    print(f"\n🎯 ENSEMBLE (Hybrid) Results:")
    ensemble_results = ensemble_retriever.invoke(query)
    for i, doc in enumerate(ensemble_results):
        print(f"  {i+1}. [{doc.metadata['id']}] {doc.page_content[:70]}")


# Test 1: Conceptual query (semantic wins)
compare_retrievers("How can I get my money back?")
# Dense will excel: "refund" ~ "money back" semantically

# Test 2: Exact term query (keyword wins)
compare_retrievers("software version 3.2.1")
# Sparse will excel: exact match on "3.2.1"

# Test 3: Specific code (keyword wins)
compare_retrievers("Error code 0x80004005")
# Sparse will excel: exact technical code

# Test 4: Mixed query (hybrid wins)
compare_retrievers("How do I install version 3.2.1?")
# Dense gets "install" docs, sparse gets "3.2.1" doc
# Hybrid combines for best results
```

---

## 16. Weighting Strategies

```
THE ENSEMBLE WEIGHT TRADEOFF:

weights = [dense_weight, sparse_weight]
Should sum to 1.0 for clarity

[1.0, 0.0]  → 100% semantic (pure dense, like similarity_search)
[0.7, 0.3]  → Semantic-leaning (most use cases)
[0.5, 0.5]  → Equal (good general default)
[0.3, 0.7]  → Keyword-leaning (technical/code-heavy domains)
[0.0, 1.0]  → 100% keyword (pure BM25)

WHEN TO LEAN SEMANTIC (higher dense weight):
✅ Natural language queries
✅ Conceptual questions
✅ User-facing chatbots
✅ Multilingual content
✅ Domain with lots of synonyms

WHEN TO LEAN KEYWORD (higher sparse weight):
✅ Technical documentation
✅ Code search
✅ Legal/medical with specific terminology
✅ Product catalogs (SKUs, model numbers)
✅ Acronym-heavy content

ALWAYS EVALUATE WITH YOUR DATA:
- Build test set of 20-30 queries
- Test different weight combinations
- Measure: which combo gets right docs in top-k?
```

---

## 17. When Hybrid Outperforms Pure Semantic

```
RESEARCH FINDING (multiple studies):
Hybrid search outperforms pure semantic search by 10-25%
on real-world benchmarks.

THE EVIDENCE-BASED RECOMMENDATION:
For production RAG systems, USE HYBRID SEARCH AS DEFAULT.

The only exception:
- Pure conceptual chatbots with no specific terminology
- Even then, hybrid usually doesn't hurt

Cost considerations:
- Hybrid is 2x the retrieval cost (running both)
- But quality improvement usually worth it
- BM25 is very fast (CPU-only, no embeddings)
- Negligible latency increase
```

---

# PART 5: Lab — Compare Semantic vs Keyword vs Hybrid

## 18. Complete Comparison Lab

```python
"""
COMPREHENSIVE RETRIEVAL COMPARISON LAB
═══════════════════════════════════════
Compare semantic, keyword, and hybrid search
on a realistic document collection.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
import json

# ═══════════════════════════════════════════════════════════
# REALISTIC TEST DOCUMENT COLLECTION
# ═══════════════════════════════════════════════════════════

documents = [
    # Customer service docs
    Document(
        page_content="Our refund policy allows full refunds within 30 days of purchase with original receipt.",
        metadata={"category": "refunds", "id": "doc-1"}
    ),
    Document(
        page_content="To request a return authorization, contact support@company.com with your order number.",
        metadata={"category": "returns", "id": "doc-2"}
    ),
    Document(
        page_content="Money-back guarantee applies to all subscription plans within first 14 days.",
        metadata={"category": "refunds", "id": "doc-3"}
    ),

    # Technical docs
    Document(
        page_content="Error code 0x80004005 typically indicates a network connectivity issue during installation.",
        metadata={"category": "technical", "id": "doc-4"}
    ),
    Document(
        page_content="To install version 3.2.1 of QuantumSync, download the installer from the customer portal.",
        metadata={"category": "technical", "id": "doc-5"}
    ),
    Document(
        page_content="SKU-78901 (QuantumSync Pro) has been replaced by SKU-78902 (QuantumSync Pro 2024).",
        metadata={"category": "product", "id": "doc-6"}
    ),

    # Account management
    Document(
        page_content="To reset your password, navigate to the login page and click 'Forgot Password'.",
        metadata={"category": "account", "id": "doc-7"}
    ),
    Document(
        page_content="Your account dashboard displays subscription status, billing history, and active services.",
        metadata={"category": "account", "id": "doc-8"}
    ),

    # Business information
    Document(
        page_content="Customer service hours are Monday through Friday 9 AM to 6 PM EST.",
        metadata={"category": "support", "id": "doc-9"}
    ),
    Document(
        page_content="Enterprise customers can contact their dedicated account manager directly via Slack.",
        metadata={"category": "support", "id": "doc-10"}
    ),

    # Policy documents
    Document(
        page_content="Data retention policy requires us to keep customer records for 7 years after account closure.",
        metadata={"category": "policy", "id": "doc-11"}
    ),
    Document(
        page_content="Cancellation requires 30-day written notice for annual subscriptions.",
        metadata={"category": "policy", "id": "doc-12"}
    ),
]

# ═══════════════════════════════════════════════════════════
# SETUP THREE RETRIEVERS
# ═══════════════════════════════════════════════════════════

print("Setting up retrievers...")

# 1. SEMANTIC (Dense) Retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="lab_demo",
    persist_directory="./lab_demo_db",
)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. KEYWORD (Sparse/BM25) Retriever
keyword_retriever = BM25Retriever.from_documents(documents)
keyword_retriever.k = 3

# 3. HYBRID (Ensemble) Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.5, 0.5],
)

print("✅ All retrievers ready\n")

# ═══════════════════════════════════════════════════════════
# TEST SUITE WITH GROUND TRUTH
# ═══════════════════════════════════════════════════════════

# Test queries with expected relevant docs (ground truth)
test_cases = [
    {
        "query": "How can I get my money back?",
        "expected_ids": ["doc-1", "doc-3"],
        "query_type": "Conceptual - paraphrased refund query",
    },
    {
        "query": "Error code 0x80004005",
        "expected_ids": ["doc-4"],
        "query_type": "Exact technical code",
    },
    {
        "query": "SKU-78901",
        "expected_ids": ["doc-6"],
        "query_type": "Exact product ID",
    },
    {
        "query": "How to install software version 3.2.1?",
        "expected_ids": ["doc-5"],
        "query_type": "Mixed - install (conceptual) + 3.2.1 (specific)",
    },
    {
        "query": "I forgot my password",
        "expected_ids": ["doc-7"],
        "query_type": "Conceptual account help",
    },
    {
        "query": "QuantumSync Pro 2024",
        "expected_ids": ["doc-6"],
        "query_type": "Brand name with year",
    },
    {
        "query": "Can I cancel my annual subscription?",
        "expected_ids": ["doc-12"],
        "query_type": "Conceptual policy query",
    },
]

# ═══════════════════════════════════════════════════════════
# EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════

def evaluate_retriever(name, retriever, test_cases) -> dict:
    """Calculate precision, recall, and hit rate for a retriever."""

    total_precision = 0
    total_recall    = 0
    total_hit_rate  = 0
    detailed_results = []

    for test in test_cases:
        query = test["query"]
        expected_ids = set(test["expected_ids"])

        # Run retrieval
        retrieved_docs = retriever.invoke(query)
        retrieved_ids = set(doc.metadata["id"] for doc in retrieved_docs)

        # Calculate metrics
        true_positives  = retrieved_ids & expected_ids

        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0
        recall    = len(true_positives) / len(expected_ids) if expected_ids else 0
        hit_rate  = 1.0 if true_positives else 0.0

        total_precision += precision
        total_recall    += recall
        total_hit_rate  += hit_rate

        detailed_results.append({
            "query":     query,
            "type":      test["query_type"],
            "expected":  list(expected_ids),
            "retrieved": list(retrieved_ids),
            "precision": precision,
            "recall":    recall,
            "hit":       bool(true_positives),
        })

    n = len(test_cases)
    return {
        "name":             name,
        "avg_precision":    total_precision / n,
        "avg_recall":       total_recall    / n,
        "avg_hit_rate":     total_hit_rate  / n,
        "detailed_results": detailed_results,
    }

# ═══════════════════════════════════════════════════════════
# RUN THE COMPARISON
# ═══════════════════════════════════════════════════════════

print("═" * 70)
print("RUNNING RETRIEVAL COMPARISON")
print("═" * 70)

results = {
    "Semantic": evaluate_retriever("Semantic", semantic_retriever, test_cases),
    "Keyword":  evaluate_retriever("Keyword",  keyword_retriever,  test_cases),
    "Hybrid":   evaluate_retriever("Hybrid",   hybrid_retriever,   test_cases),
}

# ─── Summary Table ────────────────────────────────────────────
print("\n" + "═" * 70)
print("SUMMARY RESULTS")
print("═" * 70)
print(f"\n{'Retriever':<12} {'Precision':<12} {'Recall':<12} {'Hit Rate':<12}")
print("─" * 50)

for name, data in results.items():
    print(f"{name:<12} "
          f"{data['avg_precision']:.3f}        "
          f"{data['avg_recall']:.3f}        "
          f"{data['avg_hit_rate']:.3f}")

# ─── Detailed Per-Query Analysis ──────────────────────────────
print("\n" + "═" * 70)
print("DETAILED PER-QUERY ANALYSIS")
print("═" * 70)

for i, test in enumerate(test_cases):
    print(f"\n{i+1}. Query: '{test['query']}'")
    print(f"   Type:  {test['query_type']}")
    print(f"   Expected: {test['expected_ids']}")

    for name in ["Semantic", "Keyword", "Hybrid"]:
        result = results[name]["detailed_results"][i]
        status = "✅" if result["hit"] else "❌"
        print(f"   {name:8s} {status} Retrieved: {result['retrieved']}")

# ─── Recommendations ──────────────────────────────────────────
print("\n" + "═" * 70)
print("RECOMMENDATIONS")
print("═" * 70)

best_overall = max(results.items(), key=lambda x: x[1]["avg_hit_rate"])
print(f"\n🏆 Best overall: {best_overall[0]} "
      f"(hit rate: {best_overall[1]['avg_hit_rate']:.3f})")

print("\n💡 Key insights:")
print("- Semantic excels at: conceptual queries, paraphrased queries")
print("- Keyword excels at:  exact terms, codes, IDs, product names")
print("- Hybrid combines both strengths for production use")
print("\nFor most production RAG systems, hybrid is the recommended default.")
```

---

# PART 6: Failure Modes and Interview Prep

---

## 19. Failure Modes 🚨

```
FAILURE 1: Always Returning Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   similarity_search always returns k results
           even for completely off-topic queries
Result:    LLM hallucinates from irrelevant context
Fix:       Use similarity_score_threshold
           Always have "I don't know" fallback

FAILURE 2: Score Direction Confusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Using > threshold when store uses distance (< is better)
Result:    Code keeps WORST matches, drops best matches
Fix:       Test direction with known query first
           Document the direction in code comments

FAILURE 3: Wrong k Value
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   k=20 → LLM gets too much noise, ignores middle docs
           k=1  → Missing complementary context
Result:    Poor answer quality, lost-in-the-middle problem
Fix:       Most use cases: k=3-5
           Test different k with your queries

FAILURE 4: MMR Without Enough Candidates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   fetch_k = k (no extra candidates for diversity)
Result:    MMR gives same results as similarity_search
Fix:       fetch_k should be 3-5x larger than k
           For k=5, use fetch_k=15-25

FAILURE 5: Pure Semantic for Technical Domains
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Embeddings struggle with codes, IDs, technical terms
Result:    Can't find docs about "SKU-12345" or "version 3.2.1"
Fix:       Use hybrid search for technical content
           Weight sparse higher (0.3 dense, 0.7 sparse)

FAILURE 6: Ignoring Score Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Top score = 0.95, but next 4 are < 0.4
Result:    Returning 5 docs but only 1 is actually relevant
Fix:       Monitor score distributions
           Use dynamic k based on score gaps

FAILURE 7: Same Threshold for All Use Cases
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Customer support and research tools use same threshold
Result:    Wrong tradeoff for different use cases
Fix:       Customer support → strict (high precision)
           Research tool → lenient (high recall)
```

---

## 20. Interview Questions and Answers

### Q1: "How does similarity search work in a vector store?"

**Strong Answer:**
> "Similarity search converts the user's query into a vector using the same embedding model used for documents, then finds the k nearest neighbors in vector space using cosine similarity. Modern vector stores use approximate nearest neighbor algorithms like HNSW to achieve this in O(log N) time instead of brute-force O(N), making search feasible at scale. The critical thing to understand is that similarity_search will ALWAYS return k results, regardless of whether any are actually relevant — this is why production systems also use similarity_score_threshold to filter out low-quality matches."

---

### Q2: "What's the difference between similarity_search and MMR?"

**Strong Answer:**
> "similarity_search returns the top-k most similar documents by cosine similarity — but the results may be near-duplicates. MMR — Maximal Marginal Relevance — balances relevance with diversity. It fetches a larger candidate pool (fetch_k), then iteratively selects documents using a formula: λ × relevance(query, doc) - (1-λ) × similarity(doc, already_selected). The lambda parameter controls the tradeoff: λ=1 is pure similarity_search, λ=0 is pure diversity. I use MMR for summarization, multi-faceted queries, and knowledge bases with redundant content. For specific factual queries, pure similarity_search is usually better."

---

### Q3: "When would you use hybrid search over pure semantic search?"

**Strong Answer:**
> "Pure semantic search is great for conceptual queries — finding 'refund policy' when user asks 'how do I get my money back?' But embeddings struggle with exact terms like product codes, version numbers, technical IDs, and rare jargon. Hybrid search combines semantic search (dense vectors) with BM25 keyword search (sparse vectors) using ensemble retrieval. The semantic component handles paraphrasing and synonyms, while the keyword component handles exact-match requirements. Research consistently shows hybrid outperforms pure semantic by 10-25% on real-world benchmarks. My production default is hybrid with 0.5/0.5 weights, then tune based on domain — technical docs lean keyword-heavy (0.3/0.7), customer chatbots lean semantic-heavy (0.7/0.3)."

---

### Q4: "How do you set a similarity threshold for RAG?"

**Strong Answer:**
> "Thresholds require calibration, not guessing. My process: First, collect 50 representative queries — mix of in-domain and out-of-domain. Second, run similarity search and record the top score for each. Third, manually label each result as relevant or not relevant. Fourth, plot precision-recall at different thresholds. Fifth, choose based on use case priorities — customer support tools need high precision (95%+) because wrong answers destroy trust, while research tools can tolerate lower precision for better recall. Sixth, implement two-tier fallback: try strict threshold first, fall back to lenient if no results, and finally return 'I don't have that information' if neither works. This builds user trust because the system knows when to admit uncertainty."

---

## 21. Common Mistakes ⚠️

```
MISTAKE 1: "More k always equals better answers"
TRUTH:     k=10+ causes lost-in-the-middle. LLMs ignore
           middle results. Sweet spot is k=3-5 for most cases.

MISTAKE 2: "MMR is always better than similarity"
TRUTH:     MMR sacrifices some relevance for diversity.
           For specific factual queries, this hurts performance.
           Use MMR only when redundancy is a real problem.

MISTAKE 3: "Just use semantic search, embeddings are smart"
TRUTH:     Embeddings fail on exact terms, codes, IDs, rare words.
           Hybrid search is the production-grade default.

MISTAKE 4: "Threshold 0.7 works for everything"
TRUTH:     Thresholds depend on: embedding model, domain,
           score direction. Always calibrate with real queries.

MISTAKE 5: "Higher BM25 weight is always better for technical docs"
TRUTH:     Even technical docs have conceptual queries.
           Balanced hybrid usually beats keyword-only.
           Always evaluate, don't assume.

MISTAKE 6: "Lambda=0.5 is the right MMR setting"
TRUTH:     Default isn't optimal for every domain.
           Test 0.3, 0.5, 0.7 with your data and pick best.

MISTAKE 7: "fetch_k doesn't matter for MMR"
TRUTH:     fetch_k = k = no diversity benefit
           Need fetch_k > k for MMR to actually work.
```

---

## 22. Engineer Mindset 🔧

```
WHAT A REAL RAG ENGINEER THINKS ABOUT:

1. "What kind of queries will users actually ask?"
   → Collect real query logs before tuning retrieval
   → Don't optimize for queries that don't exist
   → Match retrieval strategy to query distribution

2. "Build evaluation BEFORE optimization"
   → Create test set with ground truth
   → Measure baseline performance
   → Tune with metrics, not intuition

3. "Hybrid is the safe default for production"
   → Marginal cost (BM25 is cheap)
   → Significant quality improvement
   → Handles diverse query types

4. "Thresholds enable graceful failure"
   → Production systems must know when to say no
   → Better to fail gracefully than hallucinate
   → Build fallback chains: strict → lenient → "no answer"

5. "Monitor retrieval quality in production"
   → Log: query, retrieved docs, scores
   → Sample user feedback (thumbs up/down)
   → Watch for score distribution drift over time

6. "Different strategies for different query types"
   → Detect query type with classifier
   → Route to appropriate retriever
   → Factual → similarity_search
   → Analytical → MMR
   → Technical → hybrid (keyword-heavy)

7. "Always log scores"
   → Don't just retrieve, monitor what was retrieved
   → Track precision over time
   → Identify when re-tuning is needed

8. "Retrieval is the bottleneck, not the LLM"
   → Better retrieval > bigger LLM
   → Invest time here, not in prompt engineering
   → Garbage in = garbage out, always
```

---

## 23. Exercises 📝

### Beginner

```
1. Setup a Chroma vector store with 20 documents.
   Run similarity_search with k=3, 5, 10.
   Compare results: which docs appear in all three?

2. For your knowledge base, find the score of:
   - A document that should match perfectly
   - A document that's somewhat related
   - A completely unrelated query
   Determine: what threshold separates good from bad?

3. Implement threshold-based retrieval with fallback:
   - If strict threshold returns ≥ 1 doc → use those
   - Else if lenient threshold returns ≥ 1 doc → use those
   - Else return "I don't have that information"

4. Compare similarity_search vs MMR on the same query:
   - Use k=5 for both
   - Document the diversity difference visually
```

### Intermediate

```
5. Build a test set of 20 queries with expected relevant doc IDs.
   Calculate hit rate, precision, recall for:
   - similarity_search
   - MMR (lambda=0.5)
   - hybrid (50/50 weights)
   Plot results. Which is best for your data?

6. Implement a query classifier that routes queries:
   - "What is..." → similarity_search (factoid)
   - "Tell me about..." → MMR (broad coverage)
   - Contains numbers/codes → hybrid (keyword-heavy)
   Measure quality improvement vs single strategy.

7. Tune MMR lambda parameter:
   - Test lambda = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
   - Measure diversity (unique topics) and relevance
   - Plot the tradeoff curve

8. Build a dynamic k retriever:
   - Fetch k=10 candidates
   - Look at score gaps
   - Return only docs above a relative threshold
     (e.g., within 20% of top score)
   - This adapts k to query difficulty
```

### Advanced

```
9. Build a production-grade retrieval pipeline:
   - Hybrid search (semantic + BM25)
   - Threshold filtering
   - MMR re-ranking for diversity
   - Comprehensive logging
   - A/B testing framework

10. Compare hybrid search weights:
    - Test weights = [(1,0), (0.7,0.3), (0.5,0.5),
                      (0.3,0.7), (0,1)]
    - For each, measure on your test set
    - Plot quality vs weight ratio
    - Find optimal weights for your domain

11. Build query-aware retrieval:
    - Detect query intent (factual vs analytical vs lookup)
    - Choose strategy based on intent:
      * Factual → similarity, k=3, strict threshold
      * Analytical → MMR, k=7, lenient threshold
      * Lookup → hybrid keyword-heavy, k=3
    - Measure improvement over fixed strategy

12. Implement reciprocal rank fusion (RRF) manually:
    - Run multiple retrievers
    - Combine using RRF formula:
      score(d) = Σ 1/(k + rank(d in retriever_i))
    - Compare to EnsembleRetriever default
```

---

## 24. Resources 📚

```
LANGCHAIN DOCS:
🔗 Retrievers Overview
   https://python.langchain.com/docs/concepts/retrievers/

🔗 Vector Store as Retriever
   https://python.langchain.com/docs/how_to/vectorstore_retriever/

🔗 Ensemble Retriever
   https://python.langchain.com/docs/how_to/ensemble_retriever/

🔗 BM25 Retriever
   https://python.langchain.com/docs/integrations/retrievers/bm25/

PAPERS:
📄 "The Use of MMR, Diversity-Based Reranking
    for Reordering Documents and Producing Summaries"
   (Original MMR paper, Carbonell & Goldstein, 1998)
   https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

📄 "BM25 and Beyond" — Modern BM25 explanations
   https://en.wikipedia.org/wiki/Okapi_BM25

📄 "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods"
   https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

📄 "Hybrid Search: A Method for Better Information Retrieval"
   https://weaviate.io/blog/hybrid-search-explained

TUTORIALS:
🔗 LangChain Hybrid Search Tutorial
   https://python.langchain.com/docs/how_to/hybrid/

🔗 Pinecone: Hybrid Search Guide
   https://docs.pinecone.io/guides/data/understanding-hybrid-search

EVALUATION:
🔗 BEIR Benchmark (Retrieval Evaluation)
   https://github.com/beir-cellar/beir

🔗 RAGAS (RAG Evaluation Framework)
   https://docs.ragas.io/
```

---

## Module 5 Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 5 MENTAL MODEL                          │
│                                                                 │
│  RETRIEVAL = THE BOTTLENECK OF RAG                              │
│  Bad retrieval → Bad answers, no matter how good the LLM       │
│                                                                 │
│  SIMILARITY_SEARCH: The default                                 │
│    - Always returns k results (even if irrelevant!)             │
│    - Use k=3-5 for most cases                                   │
│    - Always check score direction (distance vs similarity)      │
│                                                                 │
│  SCORE THRESHOLD: Critical for production                       │
│    - Enables "I don't know" responses                           │
│    - Calibrate with real queries, not guesses                   │
│    - Two-tier fallback: strict → lenient → no answer            │
│                                                                 │
│  MMR: When diversity matters                                    │
│    - Balance relevance and diversity                            │
│    - λ=0.5 default, tune for your domain                        │
│    - fetch_k > k (else no diversity benefit)                    │
│    - Use for: summarization, multi-faceted queries             │
│                                                                 │
│  HYBRID SEARCH: Production default                              │
│    - Combines semantic (concepts) + keyword (exact terms)       │
│    - Outperforms pure semantic by 10-25% in research            │
│    - Use EnsembleRetriever in LangChain                         │
│    - Default weights 0.5/0.5, tune for domain                   │
│                                                                 │
│  GOLDEN RULES:                                                  │
│  1. Always evaluate with real queries and ground truth         │
│  2. Build threshold-based fallbacks                            │
│  3. Use hybrid search as production default                    │
│  4. k=3-5 is the sweet spot for most queries                   │
│  5. Different query types may need different strategies        │
└─────────────────────────────────────────────────────────────────┘
```