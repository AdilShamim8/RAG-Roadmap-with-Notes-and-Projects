# Module 7: Advanced RAG Patterns 

## The Cutting Edge of Retrieval-Augmented Generation

---

> **Before I explain — let me set the stage:**
>
> *"What's the difference between basic RAG and advanced RAG?"*
>
> A beginner might say: *"Advanced RAG uses more techniques."*
>
> The deeper truth: **Advanced RAG patterns don't just add techniques — they fundamentally rethink how retrieval and generation interact.**
>
> Basic RAG: Query → Retrieve → Generate (linear, one-shot)
> Advanced RAG: Query → Think → Retrieve → Evaluate → Maybe Re-retrieve → Generate (iterative, self-correcting)
>
> **These patterns make RAG systems that can reason about their own retrieval quality.**

---

# The Big Picture: The Evolution of RAG Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  GENERATION 1: Naive RAG                                          │
│  Query → Retrieve → Generate                                     │
│  Problem: One shot. No error correction.                         │
│                                                                   │
│  GENERATION 2: Advanced Retrieval (Module 6)                      │
│  Query → Multi-Query/Compress/Re-rank → Generate                 │
│  Problem: Still doesn't verify retrieval quality.                │
│                                                                   │
│  GENERATION 3: Advanced Patterns (THIS MODULE)                    │
│  Query → Retrieve → EVALUATE → Correct → Generate → VERIFY       │
│  Innovation: Self-correcting, multi-strategy, knowledge graphs   │
│                                                                   │
│  RAG Fusion:     Multiple queries + smart ranking                │
│  HyDE:           Bridge query-document semantic gap              │
│  Corrective RAG: Grade retrieval, fallback to web search         │
│  Self-RAG:       Reflect on own generation quality               │
│  Graph RAG:      Structured knowledge + multi-hop reasoning      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

# PART 1: RAG Fusion

---

## 1. What is RAG Fusion?

### The Problem It Solves

```
STANDARD RAG:
User query: "impact of climate change on agriculture"
              │
              ▼
    [single query embedding search]
              │
              ▼
    Top-k results (limited perspective)

PROBLEM:
A single query captures only ONE interpretation.
But "impact of climate change on agriculture" could mean:
- Crop yield changes due to global warming
- Water scarcity effects on farming
- Economic impact on agricultural markets
- Food security implications
- Adaptation strategies for farmers

Single query = single angle = missing relevant docs
```

### How RAG Fusion Works

```
RAG FUSION = Multi-Query + Reciprocal Rank Fusion (RRF)

USER QUERY: "impact of climate change on agriculture"
                           │
                           ▼
           ┌───────────────────────────────┐
           │    LLM GENERATES VARIATIONS   │
           │                               │
           │  Q1: "How does global warming │
           │       affect crop yields?"    │
           │  Q2: "Water scarcity impact   │
           │       on farming"             │
           │  Q3: "Climate change and food │
           │       security"               │
           │  Q4: "Agricultural adaptation │
           │       to changing climate"    │
           └───────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Search Q1    Search Q2    Search Q3 ...
              │            │            │
              ▼            ▼            ▼
         [rank list]  [rank list]  [rank list]
              │            │            │
              └────────────┼────────────┘
                           ▼
             ┌────────────────────────────┐
             │  RECIPROCAL RANK FUSION    │
             │                            │
             │  Merge all rank lists      │
             │  Docs appearing in MANY    │
             │  lists get boosted         │
             └────────────────────────────┘
                           │
                           ▼
              Final ranked results (best of all queries)
```

### What Makes RAG Fusion Different from Multi-Query?

```
MULTI-QUERY RETRIEVER (Module 6):
- Generate variations
- Take UNION of all results (just merge, deduplicate)
- No intelligent ranking across queries

RAG FUSION:
- Generate variations
- Each query produces a RANKED list
- Use RRF to INTELLIGENTLY combine rankings
- Documents ranked high across MULTIPLE queries get boosted

KEY DIFFERENCE:
Multi-Query = "find all relevant docs"
RAG Fusion  = "find and RANK the most consistently relevant docs"
```

---

## 2. Reciprocal Rank Fusion (RRF)

### The Math (Simple)

```
RRF FORMULA:

For each document d, across all queries:

  RRF_score(d) = Σ  1 / (k + rank(d, query_i))
                 i

Where:
  k = constant (typically 60)
  rank(d, query_i) = position of document d in query i's results
                     (1 = top result, 2 = second, etc.)
  If d not in query i's results, that term = 0
```

### Step-by-Step Example

```
Three queries each return ranked results:

Query 1 results:  [Doc_A(rank=1), Doc_B(rank=2), Doc_C(rank=3)]
Query 2 results:  [Doc_B(rank=1), Doc_D(rank=2), Doc_A(rank=3)]
Query 3 results:  [Doc_A(rank=1), Doc_E(rank=2), Doc_B(rank=3)]

RRF Scores (with k=60):

Doc_A: 1/(60+1) + 1/(60+3) + 1/(60+1)
     = 0.0164  + 0.0159  + 0.0164
     = 0.0487  ← HIGHEST (ranked #1 in 2 queries, #3 in 1)

Doc_B: 1/(60+2) + 1/(60+1) + 1/(60+3)
     = 0.0161  + 0.0164  + 0.0159
     = 0.0484  ← SECOND (ranked high in all 3 queries)

Doc_C: 1/(60+3) + 0       + 0
     = 0.0159
     = 0.0159  ← LOW (only appeared in 1 query)

Doc_D: 0       + 1/(60+2) + 0
     = 0.0161  ← LOW (only in 1 query)

Doc_E: 0       + 0       + 1/(60+2)
     = 0.0161  ← LOW (only in 1 query)

FINAL RRF RANKING:
1. Doc_A (0.0487)  ← Appears in ALL queries, mostly high rank
2. Doc_B (0.0484)  ← Also in all queries
3. Doc_D (0.0161)
4. Doc_E (0.0161)
5. Doc_C (0.0159)

KEY INSIGHT:
Docs that are consistently relevant across MANY queries
get the highest RRF score.
This is more robust than any single query's ranking.
```

---

## 3. Hands-On: Build a RAG Fusion Pipeline

```python
"""
RAG FUSION IMPLEMENTATION
Combines multi-query generation with Reciprocal Rank Fusion.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from collections import defaultdict

# ─────────────────────────────────────────────────────
# Setup knowledge base
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

docs = [
    Document(page_content="Climate change is causing rising global temperatures, "
                         "leading to longer growing seasons in some regions but "
                         "severe droughts in others."),
    Document(page_content="Water scarcity due to changing precipitation patterns "
                         "is a major threat to irrigated agriculture worldwide."),
    Document(page_content="Crop yields for wheat and corn are projected to decline "
                         "by 10-25% by 2050 due to heat stress."),
    Document(page_content="Agricultural adaptation strategies include drought-resistant "
                         "crop varieties and precision irrigation technologies."),
    Document(page_content="Food security is threatened as climate change disrupts "
                         "global supply chains and increases food prices."),
    Document(page_content="Carbon sequestration through regenerative agriculture "
                         "can help mitigate climate change while improving soil."),
    Document(page_content="Economic impact on farming communities includes "
                         "increased insurance costs and crop failure risks."),
    Document(page_content="Livestock production contributes to greenhouse gas "
                         "emissions through methane from cattle."),
    Document(page_content="Coastal flooding from sea level rise threatens "
                         "low-lying agricultural regions."),
    Document(page_content="Organic farming practices can improve soil health "
                         "and biodiversity in agricultural systems."),
]

vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="rag_fusion_demo"
)

# ─────────────────────────────────────────────────────
# Step 1: Query Generation
# ─────────────────────────────────────────────────────
query_generation_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant helping to generate multiple search queries.
Given the user's question, generate 4 different search queries that
explore different aspects of the topic.

Each query should approach the topic from a different angle to ensure
comprehensive document retrieval.

Return ONLY the queries, one per line.

Original question: {question}

Alternative queries:""")

def generate_queries(question: str) -> List[str]:
    """Generate multiple search queries using LLM."""
    chain = query_generation_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    # Parse generated queries
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]

    # Include the original query too
    all_queries = [question] + queries
    return all_queries

# ─────────────────────────────────────────────────────
# Step 2: Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Combine multiple ranked lists using RRF.

    Args:
        ranked_lists: List of ranked document lists
                     (one per query)
        k: RRF constant (default 60)

    Returns:
        Single merged ranked list of documents
    """
    # Track RRF scores for each document
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            # Use page_content as unique identifier
            doc_id = doc.page_content

            # Accumulate RRF score
            rrf_scores[doc_id] += 1 / (k + rank)
            doc_map[doc_id] = doc

    # Sort by RRF score (highest first)
    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Return documents in RRF order
    return [
        (doc_map[doc_id], score)
        for doc_id, score in sorted_docs
    ]

# ─────────────────────────────────────────────────────
# Step 3: Complete RAG Fusion Pipeline
# ─────────────────────────────────────────────────────
def rag_fusion(
    question: str,
    k_per_query: int = 5,
    final_k: int = 5,
) -> List[tuple]:
    """
    Complete RAG Fusion pipeline.
    """
    print(f"\n{'═'*70}")
    print(f"RAG FUSION PIPELINE")
    print(f"{'═'*70}")
    print(f"Original query: {question}\n")

    # Step 1: Generate query variations
    queries = generate_queries(question)
    print(f"📝 Generated {len(queries)} queries:")
    for i, q in enumerate(queries):
        print(f"   {i+1}. {q}")

    # Step 2: Retrieve for each query
    all_ranked_lists = []
    for i, query in enumerate(queries):
        results = vectorstore.similarity_search(query, k=k_per_query)
        all_ranked_lists.append(results)
        print(f"\n🔍 Query {i+1} returned {len(results)} docs")

    # Step 3: Apply RRF
    fused_results = reciprocal_rank_fusion(all_ranked_lists)

    # Step 4: Return top-k fused results
    final_results = fused_results[:final_k]

    print(f"\n{'─'*70}")
    print(f"🎯 RAG FUSION FINAL RESULTS (top {final_k}):")
    print(f"{'─'*70}")
    for i, (doc, score) in enumerate(final_results):
        print(f"\n  Rank {i+1} (RRF score: {score:.4f}):")
        print(f"    {doc.page_content}")

    return final_results

# ─────────────────────────────────────────────────────
# Run it!
# ─────────────────────────────────────────────────────
results = rag_fusion("impact of climate change on agriculture")

# Compare with basic retrieval
print(f"\n{'═'*70}")
print(f"COMPARISON: Basic vs RAG Fusion")
print(f"{'═'*70}")

basic_results = vectorstore.similarity_search(
    "impact of climate change on agriculture", k=5
)

print("\n📊 Basic Retrieval:")
for i, doc in enumerate(basic_results):
    print(f"  {i+1}. {doc.page_content[:80]}...")

print("\n🎯 RAG Fusion:")
for i, (doc, score) in enumerate(results):
    print(f"  {i+1}. [{score:.4f}] {doc.page_content[:80]}...")
```

### RAG Fusion Trade-offs

```
BENEFITS:
✅ Higher recall (multiple queries find more)
✅ More robust ranking (RRF is noise-resistant)
✅ Covers multiple angles of a topic
✅ Simple to implement

COSTS:
❌ N× retrieval latency (one search per query)
❌ LLM call to generate queries
❌ Higher overall cost

WHEN TO USE:
✅ Complex or ambiguous queries
✅ Exploratory research questions
✅ When recall matters more than speed
✅ Summarization tasks

WHEN NOT TO USE:
❌ Simple factoid queries ("What is the CEO's name?")
❌ Real-time applications where latency is critical
❌ Very small knowledge bases
```

---

# PART 2: HyDE (Hypothetical Document Embeddings)

---

## 4. The Fundamental Problem HyDE Solves

### The Query-Document Semantic Gap

```
THE INSIGHT:
Queries and documents live in DIFFERENT "zones" of embedding space.

Query:    "What causes cancer?"
          → Short, question-like, uncertain
          → Embedding captures "asking about cancer causes"

Document: "Mutations in oncogenes and tumor suppressor genes
           lead to uncontrolled cell growth..."
          → Long, declarative, detailed, authoritative
          → Embedding captures "explaining cancer biology"

These two embeddings may NOT be close in vector space
even though the document ANSWERS the query!

WHY?
- Different sentence structures (question vs statement)
- Different level of detail
- Different vocabulary ("causes" vs "mutations in oncogenes")
- Different length

This gap is called the QUERY-DOCUMENT ASYMMETRY.
```

### How HyDE Bridges the Gap

```
CLEVER IDEA:
Instead of embedding the QUERY to search...
Generate a HYPOTHETICAL ANSWER to the query...
Then embed THAT hypothetical answer to search!

Why? The hypothetical answer is DOCUMENT-LIKE:
- Declarative, not interrogative
- Detailed, not terse
- Uses document-like vocabulary
- Lives closer to actual documents in embedding space!

FLOW:
Query: "What causes cancer?"
            │
            ▼
     ┌──────────────┐
     │ LLM generates │
     │ hypothetical  │
     │ answer        │
     └──────────────┘
            │
            ▼
"Cancer is caused by mutations in genes that control
 cell growth, such as oncogenes and tumor suppressors..."
            │
            ▼
  [Embed this hypothetical answer]
            │
            ▼
  [Search vector store with this embedding]
            │
            ▼
  Finds REAL documents that are similar to the
  hypothetical answer — which are the ACTUAL answers!
```

### Important Nuance: The Hypothetical Answer Can Be Wrong

```
CRITICAL INSIGHT:
The hypothetical answer doesn't need to be FACTUALLY CORRECT!
It just needs to be SEMANTICALLY SIMILAR to real documents.

Why? We're using it for EMBEDDING SEARCH, not for answering.
The real answer comes from retrieved documents.

Example:
Query: "When was the Eiffel Tower built?"

Hypothetical: "The Eiffel Tower was constructed in 1892
               as a centerpiece for the World Exhibition."

              → WRONG DATE! (real answer is 1887-1889)
              → But the embedding captures "Eiffel Tower + construction + year"
              → This embedding is CLOSER to real docs than
                "When was the Eiffel Tower built?"

The REAL documents retrieved will have the correct facts.
HyDE just helps FIND them better.
```

---

## 5. Hands-On: Implement HyDE

```python
"""
HyDE (Hypothetical Document Embeddings) Implementation
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import numpy as np

# ─────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Technical knowledge base
docs = [
    Document(page_content="Transformers use self-attention mechanisms to process "
                         "sequences in parallel, enabling faster training than "
                         "recurrent neural networks."),
    Document(page_content="BERT (Bidirectional Encoder Representations from "
                         "Transformers) introduced masked language modeling "
                         "for pre-training language understanding."),
    Document(page_content="GPT models use autoregressive language modeling with "
                         "decoder-only transformer architecture for text generation."),
    Document(page_content="Attention mechanism computes weighted sums of value "
                         "vectors based on query-key similarity scores."),
    Document(page_content="Fine-tuning adapts pre-trained models to downstream "
                         "tasks using task-specific labeled data with "
                         "smaller learning rates."),
    Document(page_content="LoRA (Low-Rank Adaptation) freezes pre-trained weights "
                         "and adds trainable low-rank matrices for efficient "
                         "fine-tuning."),
    Document(page_content="Retrieval-Augmented Generation combines information "
                         "retrieval with language generation to reduce "
                         "hallucination."),
    Document(page_content="Vector embeddings represent text as dense numerical "
                         "vectors capturing semantic meaning in high-dimensional "
                         "space."),
]

vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="hyde_demo"
)

# ─────────────────────────────────────────────────────
# HyDE Implementation
# ─────────────────────────────────────────────────────

hyde_prompt = ChatPromptTemplate.from_template("""
Please write a short, technical passage that would answer the
following question. Write as if it's from a textbook or
documentation. The answer doesn't need to be perfectly accurate;
focus on using appropriate technical terminology and style.

Question: {question}

Passage:""")

class HyDERetriever:
    """HyDE-based retriever with comparison capability."""

    def __init__(self, vectorstore, llm, embeddings):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.hyde_chain = hyde_prompt | llm | StrOutputParser()

    def generate_hypothetical(self, question: str) -> str:
        """Generate a hypothetical answer document."""
        return self.hyde_chain.invoke({"question": question})

    def retrieve_standard(self, question: str, k: int = 3):
        """Standard retrieval using query embedding."""
        return self.vectorstore.similarity_search(question, k=k)

    def retrieve_hyde(self, question: str, k: int = 3):
        """HyDE retrieval using hypothetical document embedding."""
        # Step 1: Generate hypothetical answer
        hypothetical = self.generate_hypothetical(question)

        # Step 2: Search using hypothetical (not original query!)
        results = self.vectorstore.similarity_search(hypothetical, k=k)

        return results, hypothetical

    def compare(self, question: str, k: int = 3):
        """Compare standard vs HyDE retrieval."""

        print(f"\n{'═'*70}")
        print(f"QUERY: {question}")
        print(f"{'═'*70}")

        # Standard retrieval
        standard_results = self.retrieve_standard(question, k)

        # HyDE retrieval
        hyde_results, hypothetical = self.retrieve_hyde(question, k)

        # Show hypothetical document
        print(f"\n📝 HyDE Hypothetical Document:")
        print(f"   {hypothetical[:200]}...")

        # Compare results
        print(f"\n📊 STANDARD Retrieval:")
        for i, doc in enumerate(standard_results):
            print(f"   {i+1}. {doc.page_content[:80]}...")

        print(f"\n🎯 HyDE Retrieval:")
        for i, doc in enumerate(hyde_results):
            print(f"   {i+1}. {doc.page_content[:80]}...")

        # Show embedding similarity comparison
        query_vec = self.embeddings.embed_query(question)
        hyde_vec = self.embeddings.embed_query(hypothetical)

        def cosine_sim(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        # Check similarity of each approach's vectors to actual docs
        doc_vecs = self.embeddings.embed_documents(
            [doc.page_content for doc in docs]
        )

        print(f"\n📐 Vector Space Analysis:")
        print(f"   Query ↔ HyDE cosine: {cosine_sim(query_vec, hyde_vec):.4f}")

        return {
            "standard": standard_results,
            "hyde": hyde_results,
            "hypothetical": hypothetical,
        }


# ─────────────────────────────────────────────────────
# Test HyDE
# ─────────────────────────────────────────────────────
hyde_retriever = HyDERetriever(vectorstore, llm, embeddings)

# Test 1: Short question (big semantic gap)
hyde_retriever.compare("What is attention?")

# Test 2: Natural language question
hyde_retriever.compare("How can I fine-tune a model cheaply?")

# Test 3: Technical concept
hyde_retriever.compare("Explain how RAG reduces hallucination")
```

### When HyDE Helps vs Hurts

```
HyDE HELPS MOST WHEN:
✅ Short, terse queries: "what is attention?"
   → HyDE generates rich, document-like text
   → Much better embedding for search

✅ Questions vs declarative documents
   → Bridges the question-statement gap

✅ Domain-specific vocabulary
   → HyDE generates domain terms user didn't use

✅ Abstract or conceptual queries
   → HyDE adds concrete details for matching

HyDE HURTS WHEN:
❌ Query already uses document-like language
   → Extra LLM call with no benefit

❌ Factoid queries with specific terms
   → "What is the CEO's name?" doesn't benefit

❌ LLM generates hallucinated hypothetical far from truth
   → Searches in wrong area of vector space

❌ Latency-critical applications
   → Adds ~500ms for hypothetical generation
```

---

# PART 3: Corrective RAG (CRAG)

---

## 6. What is Corrective RAG?

### The Core Problem

```
STANDARD RAG BLINDLY TRUSTS RETRIEVAL:
1. Retrieve documents
2. Send to LLM
3. Generate answer

What if retrieved documents are WRONG or INSUFFICIENT?
→ LLM generates bad answer based on bad context
→ User gets hallucination or incomplete answer
→ No self-correction happens

CORRECTIVE RAG ADDS JUDGMENT:
1. Retrieve documents
2. GRADE each document for relevance     ← NEW
3. If documents are relevant → proceed
4. If documents are IRRELEVANT:          ← NEW
   a. Rewrite the query                 ← NEW
   b. Search the web as fallback        ← NEW
5. Generate answer with verified context
```

### CRAG Architecture

```
                    User Query
                        │
                        ▼
               ┌────────────────┐
               │   RETRIEVE     │
               │   (top-k docs) │
               └────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │ GRADE RELEVANCE│ ← LLM judges each doc
               │                │
               │  Relevant?     │
               └────────────────┘
                  /          \
                 /            \
            YES (all)       NO (any)
               │               │
               ▼               ▼
        ┌──────────┐   ┌──────────────────┐
        │  USE      │   │ REWRITE QUERY    │
        │  AS-IS    │   │ + WEB SEARCH     │
        └──────────┘   └──────────────────┘
               │               │
               └───────┬───────┘
                       ▼
              ┌────────────────┐
              │   GENERATE     │
              │   (with good   │
              │    context)    │
              └────────────────┘
                       │
                       ▼
                    Answer
```

---

## 7. Hands-On: Build CRAG with LangGraph

```python
"""
CORRECTIVE RAG (CRAG) with LangGraph
─────────────────────────────────────
A self-correcting RAG pipeline that:
1. Retrieves documents
2. Grades their relevance
3. Falls back to web search if needed
4. Rewrites queries for better retrieval
"""

from typing import TypedDict, List, Annotated
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END

# ─────────────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────────────
class CRAGState(TypedDict):
    question: str
    documents: List[Document]
    web_results: List[Document]
    generation: str
    relevance_scores: List[str]      # "relevant" or "irrelevant"
    rewritten_query: str
    retrieval_source: str            # "vectorstore" or "web"

# ─────────────────────────────────────────────────────
# Setup Components
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Knowledge base
docs = [
    Document(page_content="Machine learning models can overfit when trained "
                         "on too little data or with too many parameters."),
    Document(page_content="Regularization techniques like dropout and L2 "
                         "penalty help prevent overfitting."),
    Document(page_content="Cross-validation splits data into training and "
                         "validation sets to estimate model performance."),
    Document(page_content="Neural networks consist of layers of interconnected "
                         "neurons that transform inputs to outputs."),
    Document(page_content="Gradient descent optimizes model parameters by "
                         "following the negative gradient of the loss function."),
]

vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="crag_demo"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Web search tool
web_search = TavilySearchResults(max_results=3)

# ─────────────────────────────────────────────────────
# Node 1: RETRIEVE
# ─────────────────────────────────────────────────────
def retrieve(state: CRAGState) -> CRAGState:
    """Retrieve documents from vector store."""
    print("📚 RETRIEVING from vector store...")

    question = state["question"]
    documents = retriever.invoke(question)

    print(f"   Retrieved {len(documents)} documents")

    return {
        **state,
        "documents": documents,
        "retrieval_source": "vectorstore",
    }

# ─────────────────────────────────────────────────────
# Node 2: GRADE DOCUMENTS
# ─────────────────────────────────────────────────────

grading_prompt = ChatPromptTemplate.from_template("""
You are a grader assessing relevance of a retrieved document
to a user question.

Here is the retrieved document:
{document}

Here is the user question:
{question}

If the document contains information related to the question,
grade it as 'relevant'. Otherwise grade it as 'irrelevant'.

Provide a single word: 'relevant' or 'irrelevant'.
""")

grading_chain = grading_prompt | llm | StrOutputParser()

def grade_documents(state: CRAGState) -> CRAGState:
    """Grade each retrieved document for relevance."""
    print("📝 GRADING document relevance...")

    question = state["question"]
    documents = state["documents"]

    relevance_scores = []
    filtered_docs = []

    for i, doc in enumerate(documents):
        score = grading_chain.invoke({
            "document": doc.page_content,
            "question": question,
        }).strip().lower()

        relevance_scores.append(score)
        print(f"   Doc {i+1}: {score} — {doc.page_content[:60]}...")

        if score == "relevant":
            filtered_docs.append(doc)

    print(f"   Kept {len(filtered_docs)}/{len(documents)} docs")

    return {
        **state,
        "documents": filtered_docs,
        "relevance_scores": relevance_scores,
    }

# ─────────────────────────────────────────────────────
# Decision: Need web search?
# ─────────────────────────────────────────────────────
def decide_retrieval_action(state: CRAGState) -> str:
    """Decide whether to proceed with docs or fallback to web."""
    relevant_docs = state["documents"]

    if len(relevant_docs) > 0:
        print("✅ DECISION: Sufficient relevant documents found")
        return "generate"
    else:
        print("⚠️  DECISION: No relevant documents → Web search fallback")
        return "web_search"

# ─────────────────────────────────────────────────────
# Node 3: REWRITE QUERY (for web search)
# ─────────────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query rewriter. The original query did not return
relevant results from our knowledge base.

Rewrite this query to be better for web search — more specific,
using different keywords that might find better results.

Original query: {question}

Rewritten query:""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

def rewrite_query(state: CRAGState) -> CRAGState:
    """Rewrite the query for better web search results."""
    print("✏️  REWRITING query for web search...")

    rewritten = rewrite_chain.invoke({
        "question": state["question"]
    })

    print(f"   Original:  {state['question']}")
    print(f"   Rewritten: {rewritten}")

    return {
        **state,
        "rewritten_query": rewritten,
    }

# ─────────────────────────────────────────────────────
# Node 4: WEB SEARCH
# ─────────────────────────────────────────────────────
def web_search_node(state: CRAGState) -> CRAGState:
    """Search the web as fallback."""
    print("🌐 SEARCHING the web...")

    query = state.get("rewritten_query", state["question"])

    try:
        web_results = web_search.invoke(query)
        web_docs = [
            Document(
                page_content=result.get("content", ""),
                metadata={"source": result.get("url", "web"), "type": "web"}
            )
            for result in web_results
        ]
    except Exception as e:
        print(f"   Web search failed: {e}")
        web_docs = []

    print(f"   Found {len(web_docs)} web results")

    # Combine any existing relevant docs with web results
    all_docs = state.get("documents", []) + web_docs

    return {
        **state,
        "documents": all_docs,
        "web_results": web_docs,
        "retrieval_source": "web",
    }

# ─────────────────────────────────────────────────────
# Node 5: GENERATE
# ─────────────────────────────────────────────────────
generate_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you cannot answer from the context, say so.

Context:
{context}

Question: {question}

Answer:""")

generate_chain = generate_prompt | llm | StrOutputParser()

def generate(state: CRAGState) -> CRAGState:
    """Generate final answer using verified context."""
    print("🤖 GENERATING answer...")

    context = "\n\n".join([
        doc.page_content for doc in state["documents"]
    ])

    answer = generate_chain.invoke({
        "context": context,
        "question": state["question"],
    })

    print(f"   Source: {state.get('retrieval_source', 'unknown')}")
    print(f"   Answer: {answer[:150]}...")

    return {
        **state,
        "generation": answer,
    }

# ─────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────
workflow = StateGraph(CRAGState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate)

# Add edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional edge: grade → generate OR grade → rewrite
workflow.add_conditional_edges(
    "grade_documents",
    decide_retrieval_action,
    {
        "generate": "generate",
        "web_search": "rewrite_query",
    }
)

workflow.add_edge("rewrite_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
crag_app = workflow.compile()

# ─────────────────────────────────────────────────────
# Test the CRAG Pipeline
# ─────────────────────────────────────────────────────

# Test 1: In-domain query (should use vector store)
print("\n" + "═"*70)
print("TEST 1: In-domain query")
print("═"*70)
result1 = crag_app.invoke({
    "question": "How can I prevent overfitting?",
    "documents": [],
    "web_results": [],
    "generation": "",
    "relevance_scores": [],
    "rewritten_query": "",
    "retrieval_source": "",
})
print(f"\n📋 Final Answer: {result1['generation']}")

# Test 2: Out-of-domain query (should trigger web search)
print("\n" + "═"*70)
print("TEST 2: Out-of-domain query")
print("═"*70)
result2 = crag_app.invoke({
    "question": "What are the latest developments in quantum computing?",
    "documents": [],
    "web_results": [],
    "generation": "",
    "relevance_scores": [],
    "rewritten_query": "",
    "retrieval_source": "",
})
print(f"\n📋 Final Answer: {result2['generation']}")
```

---

# PART 4: Self-RAG

---

## 8. What is Self-RAG?

### The Problem: RAG Without Self-Awareness

```
STANDARD RAG generates an answer and hopes it's good.
No verification. No self-reflection. No correction.

SELF-RAG adds REFLECTION TOKENS — special decision points
where the system evaluates its own output:

1. SHOULD I RETRIEVE?
   Not all questions need retrieval.
   "What is 2+2?" → No retrieval needed
   "What is our company's revenue?" → Retrieval needed

2. IS THIS DOCUMENT RELEVANT?
   After retrieval, grade each document.
   Similar to CRAG but built into the generation loop.

3. IS MY ANSWER SUPPORTED?
   Check if the generated answer is actually grounded
   in the retrieved documents.

4. IS MY ANSWER USEFUL?
   Does the answer actually address the user's question?
```

### Self-RAG Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│ RETRIEVAL NEEDED? │ ← Reflection token 1
│ (classify query)  │
└──────────────────┘
    │          │
  YES          NO
    │          │
    ▼          ▼
┌────────┐  ┌──────────┐
│RETRIEVE│  │GENERATE  │
│ docs   │  │ directly │
└────────┘  └──────────┘
    │
    ▼
┌──────────────────┐
│ IS DOC RELEVANT? │ ← Reflection token 2
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ GENERATE ANSWER  │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ IS ANSWER        │ ← Reflection token 3
│ SUPPORTED BY     │
│ DOCUMENTS?       │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ IS ANSWER        │ ← Reflection token 4
│ USEFUL?          │
└──────────────────┘
    │
    ▼
Final Answer (verified)
```

---

## 9. Self-RAG Implementation Concept

```python
"""
SELF-RAG Conceptual Implementation
Shows the reflection token pattern.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─────────────────────────────────────────────────────
# Reflection Token 1: Does query need retrieval?
# ─────────────────────────────────────────────────────
needs_retrieval_prompt = ChatPromptTemplate.from_template("""
Determine if the following query needs to search a knowledge base
or can be answered from general knowledge.

Query: {question}

Answer with exactly one word: 'retrieve' or 'direct'

- 'retrieve': if the question asks about specific documents,
  company data, policies, or domain-specific facts
- 'direct': if it's general knowledge, math, greetings, etc.
""")

# ─────────────────────────────────────────────────────
# Reflection Token 2: Is document relevant?
# ─────────────────────────────────────────────────────
relevance_prompt = ChatPromptTemplate.from_template("""
Is this document relevant to answering the question?

Document: {document}
Question: {question}

Answer 'relevant' or 'irrelevant':
""")

# ─────────────────────────────────────────────────────
# Reflection Token 3: Is answer supported by documents?
# ─────────────────────────────────────────────────────
support_prompt = ChatPromptTemplate.from_template("""
Check if the following answer is fully supported by the provided documents.

Documents:
{documents}

Answer: {answer}

Rate as:
- 'fully_supported': Every claim is backed by the documents
- 'partially_supported': Some claims are supported, some are not
- 'not_supported': The answer makes claims not in the documents

Rating:
""")

# ─────────────────────────────────────────────────────
# Reflection Token 4: Is answer useful?
# ─────────────────────────────────────────────────────
utility_prompt = ChatPromptTemplate.from_template("""
Rate how useful this answer is for the user's question.

Question: {question}
Answer: {answer}

Rate from 1-5:
5: Perfectly answers the question
4: Mostly answers, minor gaps
3: Partially answers
2: Barely related
1: Not useful

Rating (number only):
""")

class SelfRAG:
    """Self-RAG with reflection tokens for quality control."""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def should_retrieve(self, question: str) -> bool:
        """Reflection 1: Does this query need retrieval?"""
        chain = needs_retrieval_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question}).strip().lower()
        decision = "retrieve" in result

        print(f"  🔍 Need retrieval? {'YES' if decision else 'NO'}")
        return decision

    def grade_relevance(self, question: str, doc) -> bool:
        """Reflection 2: Is this document relevant?"""
        chain = relevance_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "document": doc.page_content,
            "question": question,
        }).strip().lower()

        return "relevant" in result

    def check_support(self, answer: str, documents: list) -> str:
        """Reflection 3: Is the answer supported?"""
        chain = support_prompt | self.llm | StrOutputParser()
        docs_text = "\n".join([d.page_content for d in documents])
        result = chain.invoke({
            "documents": docs_text,
            "answer": answer,
        }).strip().lower()

        return result

    def check_utility(self, question: str, answer: str) -> int:
        """Reflection 4: Is the answer useful?"""
        chain = utility_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "question": question,
            "answer": answer,
        }).strip()

        try:
            return int(result[0])
        except:
            return 3

    def query(self, question: str) -> dict:
        """Complete Self-RAG pipeline."""

        print(f"\n{'═'*60}")
        print(f"SELF-RAG: {question}")
        print(f"{'═'*60}")

        # Reflection 1: Need retrieval?
        if not self.should_retrieve(question):
            # Generate directly without retrieval
            answer = self.llm.invoke(question).content
            return {
                "answer": answer,
                "source": "direct_generation",
                "support": "n/a",
                "utility": self.check_utility(question, answer),
            }

        # Retrieve
        documents = self.retriever.invoke(question)

        # Reflection 2: Grade relevance
        relevant_docs = [
            doc for doc in documents
            if self.grade_relevance(question, doc)
        ]
        print(f"  📝 Relevant docs: {len(relevant_docs)}/{len(documents)}")

        if not relevant_docs:
            return {
                "answer": "I don't have relevant information to answer.",
                "source": "no_relevant_docs",
                "support": "n/a",
                "utility": 1,
            }

        # Generate answer
        context = "\n".join([d.page_content for d in relevant_docs])
        generate_prompt = f"Based on this context:\n{context}\n\nAnswer: {question}"
        answer = self.llm.invoke(generate_prompt).content

        # Reflection 3: Check support
        support = self.check_support(answer, relevant_docs)
        print(f"  📋 Support level: {support}")

        # Reflection 4: Check utility
        utility = self.check_utility(question, answer)
        print(f"  ⭐ Utility score: {utility}/5")

        # If poorly supported, regenerate with stricter prompt
        if "not_supported" in support:
            print(f"  🔄 Re-generating with stricter grounding...")
            strict_prompt = (
                f"Answer ONLY using these facts:\n{context}\n\n"
                f"If you cannot answer from these facts alone, say so.\n\n"
                f"Question: {question}"
            )
            answer = self.llm.invoke(strict_prompt).content
            support = "re-generated_with_strict_grounding"

        return {
            "answer": answer,
            "source": "retrieval",
            "support": support,
            "utility": utility,
            "num_docs_used": len(relevant_docs),
        }


# Usage
self_rag = SelfRAG(vectorstore, llm)

# Test 1: Query needing retrieval
result = self_rag.query("How do I prevent overfitting in neural networks?")
print(f"\n📋 Answer: {result['answer']}")

# Test 2: Query NOT needing retrieval
result = self_rag.query("What is 2 + 2?")
print(f"\n📋 Answer: {result['answer']}")
```

---

# PART 5: Graph RAG

---

## 10. Why Graphs for RAG?

### The Limitation of Vector-Only RAG

```
VECTOR RAG'S WEAKNESS: Multi-hop reasoning

Question: "Which drugs interact with the medication prescribed
           by Dr. Smith to the patient who had surgery in March?"

This requires MULTIPLE HOPS of reasoning:
1. Find patient who had surgery in March → Patient A
2. Find Dr. Smith's prescription for Patient A → Drug X
3. Find Drug X's interactions → Drugs Y and Z

Vector similarity can't chain these relationships!
Each hop is a different query with different context.

GRAPH RAG EXCELS HERE because:
- Relationships between entities are EXPLICIT edges
- Multi-hop traversal is a native graph operation
- Knowledge structure is preserved, not flattened

Patient_A --[had_surgery]--> March
Patient_A --[prescribed_by]--> Dr_Smith
Dr_Smith  --[prescribed]--> Drug_X
Drug_X    --[interacts_with]--> Drug_Y
Drug_X    --[interacts_with]--> Drug_Z
```

### Knowledge Graph Structure

```
KNOWLEDGE GRAPH = Entities + Relationships

Entities (nodes):
  ● People:    "Dr. Smith", "Patient A"
  ● Things:    "Drug X", "Aspirin"
  ● Concepts:  "Hypertension", "Surgery"
  ● Events:    "Surgery on March 5th"

Relationships (edges):
  → PRESCRIBED
  → TREATS
  → INTERACTS_WITH
  → DIAGNOSED_WITH
  → PERFORMED_BY

Example:
  (Dr. Smith) --[PRESCRIBED]--> (Aspirin)
  (Aspirin)   --[TREATS]------> (Headache)
  (Aspirin)   --[INTERACTS]---> (Warfarin)
  (Patient A) --[DIAGNOSED]---> (Hypertension)
```

---

## 11. Neo4j GraphRAG

### Setting Up Neo4j

```python
"""
Graph RAG with Neo4j
─────────────────────
Store knowledge as a graph, query with natural language.
"""

# First: Install and run Neo4j
# docker run -p 7474:7474 -p 7687:7687 neo4j:latest

from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain

# Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password",
)

# ─────────────────────────────────────────────────────
# Step 1: Add data to graph using Cypher queries
# ─────────────────────────────────────────────────────

# Create nodes and relationships
graph.query("""
CREATE (smith:Doctor {name: 'Dr. Smith', specialty: 'Cardiology'})
CREATE (jones:Doctor {name: 'Dr. Jones', specialty: 'Neurology'})

CREATE (patientA:Patient {name: 'Patient A', age: 65})
CREATE (patientB:Patient {name: 'Patient B', age: 45})

CREATE (aspirin:Drug {name: 'Aspirin', type: 'NSAID'})
CREATE (warfarin:Drug {name: 'Warfarin', type: 'Anticoagulant'})
CREATE (metformin:Drug {name: 'Metformin', type: 'Antidiabetic'})

CREATE (hypertension:Condition {name: 'Hypertension'})
CREATE (diabetes:Condition {name: 'Type 2 Diabetes'})
CREATE (surgery:Event {name: 'Cardiac Surgery', date: '2024-03-15'})

// Relationships
CREATE (smith)-[:PRESCRIBED {date: '2024-03-20'}]->(aspirin)
CREATE (smith)-[:PRESCRIBED {date: '2024-03-20'}]->(warfarin)
CREATE (jones)-[:PRESCRIBED {date: '2024-04-01'}]->(metformin)

CREATE (smith)-[:TREATED]->(patientA)
CREATE (jones)-[:TREATED]->(patientB)

CREATE (patientA)-[:DIAGNOSED_WITH]->(hypertension)
CREATE (patientB)-[:DIAGNOSED_WITH]->(diabetes)

CREATE (patientA)-[:HAD_PROCEDURE]->(surgery)
CREATE (smith)-[:PERFORMED]->(surgery)

CREATE (aspirin)-[:INTERACTS_WITH {severity: 'HIGH'}]->(warfarin)
CREATE (aspirin)-[:TREATS]->(hypertension)
CREATE (metformin)-[:TREATS]->(diabetes)
""")

print("✅ Graph populated with nodes and relationships")

# ─────────────────────────────────────────────────────
# Step 2: Query with natural language
# ─────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# GraphCypherQAChain converts natural language → Cypher query
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,        # Show generated Cypher query
    allow_dangerous_requests=True,
)

# Test multi-hop queries
queries = [
    # Single hop
    "What drugs did Dr. Smith prescribe?",

    # Two hops
    "What conditions do Dr. Smith's patients have?",

    # Multi-hop with interaction
    "Which drugs prescribed by Dr. Smith interact with each other?",

    # Complex multi-hop
    "Which patient had cardiac surgery, and what drugs were they prescribed?",
]

for query in queries:
    print(f"\n{'═'*60}")
    print(f"QUERY: {query}")
    print(f"{'═'*60}")
    result = chain.invoke({"query": query})
    print(f"ANSWER: {result['result']}")
```

---

## 12. Adding PDF Data to Graph Database

```python
"""
Extract entities and relationships from PDFs
and add them to Neo4j for Graph RAG.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.graphs import Neo4jGraph
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─────────────────────────────────────────────────────
# Step 1: Load PDF
# ─────────────────────────────────────────────────────
loader = PyPDFLoader("documents/medical_records.pdf")
pages = loader.load()

# ─────────────────────────────────────────────────────
# Step 2: Extract entities and relationships using LLM
# ─────────────────────────────────────────────────────
extraction_prompt = ChatPromptTemplate.from_template("""
Extract entities and relationships from the following text.

Return a JSON object with:
- "entities": list of objects with "name", "type", "properties"
- "relationships": list of objects with "source", "target",
  "type", "properties"

Entity types: Person, Drug, Condition, Procedure, Organization
Relationship types: PRESCRIBED, TREATS, DIAGNOSED_WITH,
                   INTERACTS_WITH, WORKS_AT, PERFORMED

Text:
{text}

JSON output:""")

extraction_chain = extraction_prompt | llm | JsonOutputParser()

def extract_knowledge(text: str) -> dict:
    """Extract structured knowledge from text."""
    try:
        return extraction_chain.invoke({"text": text})
    except Exception as e:
        print(f"Extraction failed: {e}")
        return {"entities": [], "relationships": []}

# Extract from each page
all_entities = []
all_relationships = []

for page in pages[:5]:  # First 5 pages for demo
    knowledge = extract_knowledge(page.page_content)
    all_entities.extend(knowledge.get("entities", []))
    all_relationships.extend(knowledge.get("relationships", []))

print(f"Extracted {len(all_entities)} entities")
print(f"Extracted {len(all_relationships)} relationships")

# ─────────────────────────────────────────────────────
# Step 3: Add to Neo4j
# ─────────────────────────────────────────────────────
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password",
)

def add_to_graph(entities: list, relationships: list):
    """Add extracted knowledge to Neo4j."""

    # Add entities
    for entity in entities:
        props = entity.get("properties", {})
        props_str = ", ".join([
            f"{k}: '{v}'" for k, v in props.items()
        ])

        cypher = f"""
        MERGE (n:{entity['type']} {{name: '{entity['name']}'}})
        SET n += {{{props_str}}}
        """
        try:
            graph.query(cypher)
        except Exception as e:
            print(f"Entity insert failed: {e}")

    # Add relationships
    for rel in relationships:
        cypher = f"""
        MATCH (s {{name: '{rel['source']}'}})
        MATCH (t {{name: '{rel['target']}'}})
        MERGE (s)-[r:{rel['type']}]->(t)
        """
        try:
            graph.query(cypher)
        except Exception as e:
            print(f"Relationship insert failed: {e}")

add_to_graph(all_entities, all_relationships)
print("✅ Knowledge added to graph database")
```

---

## 13. When Graph RAG Outperforms Vector RAG

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│ Scenario         │ Vector RAG           │ Graph RAG            │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Factoid lookup   │ ✅ Excellent          │ 🟡 Overkill          │
│ Multi-hop query  │ ❌ Fails              │ ✅ Excels             │
│ Relationship Q&A │ ❌ Can't traverse     │ ✅ Native edges       │
│ Summarization    │ ✅ Good               │ 🟡 Okay              │
│ Exploratory      │ ✅ Good               │ ✅ Great              │
│ Setup cost       │ ✅ Low                │ ❌ High               │
│ Maintenance      │ ✅ Easy               │ ❌ Complex            │
│ Scale (docs)     │ ✅ Millions           │ 🟡 Depends            │
│ Unstructured text│ ✅ Native             │ ❌ Needs extraction   │
│ Structured data  │ 🟡 Metadata only      │ ✅ Native             │
└──────────────────┴──────────────────────┴──────────────────────┘

USE GRAPH RAG WHEN:
✅ Queries involve relationships between entities
✅ Multi-hop reasoning is required
✅ Data is naturally structured (medical, legal, financial)
✅ "Who prescribed what to whom?" type queries
✅ Network analysis (social, organizational)

USE VECTOR RAG WHEN:
✅ Queries are about content/meaning
✅ Documents are unstructured text
✅ Simple Q&A without relationship traversal
✅ Fast setup, minimal infrastructure

BEST APPROACH: COMBINE BOTH
Use vector RAG for content retrieval
Use graph RAG for relationship queries
Route queries to appropriate system
```

---

# PART 6: Failure Modes and Interview Prep

---

## 14. Failure Modes 🚨

```
RAG FUSION FAILURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem: LLM generates off-topic query variations
Result:  Irrelevant docs dilute rankings
Fix:     Better prompt engineering, validate queries

HyDE FAILURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem: Hypothetical answer completely wrong direction
Result:  Embedding searches in wrong area
Fix:     Use for broad queries, not specific factoids
         Test on edge cases before deploying

CRAG FAILURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem: Grader is too strict, marks everything irrelevant
Result:  Always falls back to web, ignoring knowledge base
Fix:     Calibrate grading prompt, test with known-good pairs

SELF-RAG FAILURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem: Multiple LLM calls per query (expensive, slow)
Result:  High latency, high cost
Fix:     Cache reflection decisions
         Use smaller model for reflection checks

GRAPH RAG FAILURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem: LLM generates invalid Cypher queries
Result:  Database errors, empty results
Fix:     Validate Cypher syntax before execution
         Show graph schema to LLM for context
         Use few-shot examples in prompt
```

---

## 15. Interview Questions and Answers

### Q1: "Explain RAG Fusion and how it improves over basic multi-query."

**Strong Answer:**
> "RAG Fusion generates multiple query variations like multi-query retrieval, but it differs in how it combines results. Multi-query simply takes the union of all results and deduplicates. RAG Fusion uses Reciprocal Rank Fusion — each query produces a ranked list, and documents are scored based on their rank across all lists using the formula: score = sum(1/(k+rank)). Documents that consistently appear high across multiple queries get the highest fused scores. This is more robust than any single query's ranking because it captures multiple perspectives. The tradeoff is N times the retrieval latency. I use it for complex, ambiguous queries where comprehensiveness matters more than speed."

---

### Q2: "How does HyDE work and when would you use it?"

**Strong Answer:**
> "HyDE bridges the query-document semantic gap. Instead of embedding the user's short question, it asks an LLM to generate a hypothetical answer — a document-like passage. This hypothetical is then embedded and used for vector search. The key insight: the hypothetical doesn't need to be factually correct — it just needs to be semantically similar to real documents. A question like 'what is attention?' produces a weak embedding, but a hypothetical passage about attention mechanisms produces an embedding closer to actual technical documents in vector space. I use HyDE for short or abstract queries where the gap between question style and document style is large. I avoid it for specific factoid queries or when latency is critical."

---

### Q3: "What is Corrective RAG and how does it differ from standard RAG?"

**Strong Answer:**
> "Corrective RAG adds a quality-control loop after retrieval. Standard RAG blindly trusts whatever the retriever returns. CRAG grades each retrieved document for relevance using an LLM. If documents are relevant, it proceeds normally. If documents are irrelevant, it triggers two corrective actions: first, it rewrites the query to better capture user intent; second, it falls back to web search as an alternative information source. This is implemented as a graph in LangGraph with conditional edges — the grading step determines whether to proceed to generation or divert to the web search path. It's critical for production systems because it prevents the silent failure mode where the LLM confidently generates answers from irrelevant context."

---

### Q4: "When would you use Graph RAG over Vector RAG?"

**Strong Answer:**
> "Graph RAG excels when queries require traversing relationships between entities — what I call multi-hop reasoning. A question like 'which drugs prescribed by Dr. Smith interact with each other?' requires three hops: find Dr. Smith, find his prescriptions, find drug interactions. Vector RAG can't chain these relationships because each chunk is independently embedded. Graph RAG with Neo4j stores entities as nodes and relationships as edges, enabling native traversal. I use Graph RAG for medical records, legal case relationships, organizational hierarchies, and supply chain queries. However, it requires significant upfront investment: entity extraction from documents, graph schema design, and Cypher query generation. For pure content-based Q&A over unstructured text, Vector RAG is simpler and often sufficient. The ideal production system combines both: Vector RAG for content, Graph RAG for relationships, with a query router deciding which to use."

---

## 16. Common Mistakes ⚠️

```
MISTAKE 1: "Use all advanced patterns simultaneously"
TRUTH:     Each pattern adds cost and complexity.
           Choose based on diagnosed failure modes.
           RAG Fusion alone adds 4-5x retrieval latency.

MISTAKE 2: "HyDE always improves retrieval"
TRUTH:     HyDE can hurt when the LLM generates a
           misleading hypothetical far from actual docs.
           Test carefully on your domain.

MISTAKE 3: "CRAG web fallback always works"
TRUTH:     Web search may return unreliable sources.
           Need to grade web results too.

MISTAKE 4: "Self-RAG adds minimal latency"
TRUTH:     4 reflection tokens = 4 LLM calls per query.
           Can 5x your response time and cost.
           Use smaller/faster models for reflection.

MISTAKE 5: "Graph RAG replaces Vector RAG"
TRUTH:     They're complementary, not competitive.
           Graph = relationships. Vector = content.
           Best systems use both with routing.

MISTAKE 6: "Knowledge graph extraction is reliable"
TRUTH:     LLM extraction has errors. Wrong entities,
           wrong relationships, missed connections.
           Always validate with domain expert.
```

---

## 17. Engineer Mindset 🔧

```
WHAT A REAL RAG ENGINEER THINKS ABOUT:

1. "Match the pattern to the problem"
   → Ambiguous queries? → RAG Fusion
   → Short question vs long docs? → HyDE
   → Unreliable retrieval? → CRAG
   → Need quality assurance? → Self-RAG
   → Relationship queries? → Graph RAG

2. "Cost-benefit analysis before implementing"
   → RAG Fusion: 4x retrieval cost for 15% quality boost
   → HyDE: 1 LLM call + marginal benefit = worth it?
   → CRAG: Web search fallback is powerful for coverage
   → Self-RAG: 4 LLM calls per query = expensive
   → Graph RAG: High setup, transformative for right use case

3. "Start simple, add complexity when measured"
   → Base: Hybrid search + threshold
   → If recall low: Add multi-query or RAG Fusion
   → If irrelevant results: Add CRAG grading
   → If context too noisy: Add compression
   → If multi-hop queries: Add Graph RAG

4. "Production stability over cutting-edge"
   → CRAG is most production-ready pattern
   → Self-RAG is most academically interesting
   → Choose based on maturity, not novelty

5. "Combine vector + graph for maximum coverage"
   → Build query classifier
   → Route content queries → Vector RAG
   → Route relationship queries → Graph RAG
   → Merge results for comprehensive answers
```

---

## 18. Exercises 📝

### Beginner

```
1. Implement RAG Fusion manually:
   - Generate 3 query variations
   - Retrieve for each
   - Implement RRF formula
   - Compare to single-query results

2. Build a simple HyDE retriever:
   - Generate hypothetical answer
   - Embed and search with it
   - Compare to direct query search

3. Implement a document grader:
   - Retrieve 5 docs
   - Grade each for relevance using LLM
   - Filter out irrelevant ones
   - This is the core of CRAG

4. Setup Neo4j and create a simple graph:
   - 5 entities (people, places)
   - 5 relationships
   - Query with Cypher
```

### Intermediate

```
5. Build complete CRAG with LangGraph:
   - Retrieve → Grade → Decide → Web Fallback → Generate
   - Test with in-domain and out-of-domain queries
   - Log which path was taken for each query

6. Implement Self-RAG reflection pipeline:
   - Add all 4 reflection tokens
   - Test: does it correctly identify when not to retrieve?
   - Test: does it catch unsupported claims?

7. Build a PDF-to-Graph pipeline:
   - Extract entities from a PDF
   - Create Neo4j graph
   - Query with natural language
   - Compare results to Vector RAG on same PDF

8. Evaluate RAG Fusion vs HyDE:
   - Create test set with 20 queries
   - Run basic, RAG Fusion, and HyDE
   - Measure recall@5 for each
   - Which is best for your domain?
```

### Advanced

```
9. Build a hybrid Vector + Graph RAG system:
   - Query classifier routes to appropriate system
   - Content queries → Vector RAG
   - Relationship queries → Graph RAG
   - Evaluate on mixed query set

10. Implement adaptive Self-RAG:
    - Use lightweight model for reflection tokens
    - Cache reflection decisions for similar queries
    - Measure latency reduction vs full Self-RAG

11. Production CRAG with monitoring:
    - Track: % queries needing web fallback
    - Track: grading accuracy over time
    - Alert when web fallback rate exceeds threshold
    - Dashboard showing retrieval quality metrics

12. End-to-end comparison of ALL Module 7 patterns:
    - Same knowledge base, same test queries
    - Basic RAG vs RAG Fusion vs HyDE vs CRAG
    - Metrics: accuracy, latency, cost per query
    - Recommend best pattern for different query types
```

---

## 19. Resources 📚

```
PAPERS:
📄 "RAG-Fusion: A New Take on Retrieval-Augmented Generation"
   https://arxiv.org/abs/2402.03367

📄 "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)
   https://arxiv.org/abs/2212.10496

📄 "Corrective Retrieval Augmented Generation" (CRAG)
   https://arxiv.org/abs/2401.15884

📄 "Self-RAG: Learning to Retrieve, Generate, and Critique"
   https://arxiv.org/abs/2310.11511

📄 "From Local to Global: A Graph RAG Approach"
   (Microsoft Graph RAG)
   https://arxiv.org/abs/2404.16130

TOOLS & DOCS:
🔗 LangGraph Documentation
   https://langchain-ai.github.io/langgraph/

🔗 Neo4j Graph Database
   https://neo4j.com/docs/

🔗 LangChain Graph RAG
   https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

🔗 Tavily Web Search API
   https://tavily.com/

TUTORIALS:
🔗 LangGraph CRAG Tutorial
   https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/

🔗 LangGraph Self-RAG
   https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/

🔗 Microsoft Graph RAG
   https://github.com/microsoft/graphrag
```

---

## Module 7 Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 7 MENTAL MODEL                          │
│                                                                 │
│  RAG FUSION: Multiple queries + smart rank merging             │
│    When: Ambiguous/complex queries needing comprehensive recall│
│    Cost: N× retrieval latency                                   │
│                                                                 │
│  HyDE: Generate fake answer → embed → search                   │
│    When: Short queries with big query-document gap             │
│    Insight: Hypothetical needn't be correct, just similar      │
│                                                                 │
│  CORRECTIVE RAG: Grade retrieval → fallback if bad             │
│    When: Must not answer from irrelevant context               │
│    Power: Self-correcting, web fallback                        │
│    Most production-ready advanced pattern                      │
│                                                                 │
│  SELF-RAG: Reflect at every step                               │
│    When: Quality assurance is paramount                        │
│    Cost: 4+ LLM calls per query (expensive!)                   │
│                                                                 │
│  GRAPH RAG: Knowledge graphs for relationships                 │
│    When: Multi-hop reasoning, entity relationships             │
│    Complement: Use WITH vector RAG, not instead of             │
│                                                                 │
│  GOLDEN RULES:                                                  │
│  1. Match pattern to specific failure mode                     │
│  2. Each pattern trades latency/cost for quality               │
│  3. CRAG is the safest first advanced pattern to add           │
│  4. Graph RAG for relationships, Vector RAG for content        │
│  5. Measure improvement before keeping any pattern             │
└─────────────────────────────────────────────────────────────────┘
```