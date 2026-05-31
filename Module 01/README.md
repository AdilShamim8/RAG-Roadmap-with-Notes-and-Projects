# Module 1: RAG Fundamentals and Architecture 

## 1.1 Introduction to RAG — Complete Deep Dive

---

# PART 1: What is RAG?

---

## 1. What is This?

> **RAG = Retrieval-Augmented Generation**

In the simplest possible words:

**RAG is a technique where an AI, before answering your question, first goes and searches for relevant information from a knowledge base — and then uses that retrieved information to generate a better, more accurate answer.**

Think of it as giving the AI **the ability to "look things up" before speaking.**

---

## 2. Why Do We Need It?

### The Core Problem: LLMs Are Frozen in Time 🧊

When a Large Language Model (LLM) like GPT-4 is trained, it learns from a massive dataset. But after training ends — **it knows nothing new.**

Imagine a brilliant professor who:
- Read every book ever written... **but only until December 2023**
- Has **no internet access**
- Cannot access **your company's private documents**
- Sometimes **confidently makes things up** when unsure

That professor is your LLM.

RAG solves **4 critical problems** that LLMs face:

---

### Problem 1: Hallucination 🌀

#### What is hallucination?

When an LLM doesn't know the answer, it doesn't say *"I don't know."*
It **invents a confident-sounding answer** that is completely wrong.

```
User: "What did CEO John Smith say in the Q3 2024 earnings call?"

LLM without RAG:
"John Smith mentioned a 23% revenue increase and plans
to expand into Asian markets..." ← COMPLETELY FABRICATED
```

#### Why does hallucination happen?

LLMs are trained to **predict the next most likely word**.
They are optimized to sound fluent and confident — not to be factually correct.

```
Training goal:  "Sound like the next word is correct"
Ideal goal:     "Only say things that are true"
```

These two goals are NOT the same. That gap = hallucination.

#### How RAG reduces hallucination:

```
Without RAG:          LLM guesses from memory (dangerous)
With RAG:             LLM reads actual documents first, then answers
```

RAG gives the LLM **ground truth to work from** instead of relying purely on memory.

---

### Problem 2: Knowledge Cutoff Limitations 📅

Every LLM has a **training cutoff date**. After that date — it knows nothing.

```
GPT-4 cutoff:     Early 2024
Today's date:     2025
Gap:              ~1+ year of unknown events
```

Real examples of what the LLM cannot know:
- New laws passed after the cutoff
- Latest research papers
- Recent company announcements
- Current stock prices
- New product launches

#### How RAG solves this:

RAG connects the LLM to a **live, updatable knowledge base**.
When new information exists → update the knowledge base → LLM now has access to it.

```
LLM weights:         Static (expensive to change)
Knowledge Base:      Dynamic (cheap to update)
```

---

### Problem 3: Domain-Specific Knowledge Injection 🏢

LLMs are trained on **public internet data**.
Your company's private knowledge does NOT exist on the internet.

```
Company documents the LLM has NEVER seen:
- Internal HR policies
- Proprietary research
- Customer contracts
- Medical records
- Legal case files
- Financial reports
```

#### How RAG solves this:

You **inject your private documents** into the knowledge base.
Now the LLM can answer questions using YOUR data — without ever training on it.

```
Before RAG:  "I have no information about your company policy"
After RAG:   "According to your HR policy document, page 3..."
```

This is **enormously valuable** for enterprise AI applications.

---

### Problem 4: Source Attribution and Transparency 📎

Without RAG:
```
User:  "Is it safe to take ibuprofen with blood thinners?"
LLM:   "Yes, it is generally considered safe..." ← Where did this come from?
```

This is dangerous. You cannot verify the source.

With RAG:
```
User:  "Is it safe to take ibuprofen with blood thinners?"
LLM:   "According to the Mayo Clinic document (source: mayo_clinic_drug_interactions.pdf,
        page 12): Ibuprofen may increase bleeding risk when combined with..."
```

Now you can:
- ✅ Verify the claim
- ✅ Trust the source
- ✅ Audit the answer
- ✅ Meet compliance requirements

**This is critical for healthcare, legal, and financial applications.**

---

# PART 2: Core Components of RAG

---

## 3. First-Principles Breakdown

### The 3 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   KNOWLEDGE BASE  →→→  RETRIEVER  →→→  GENERATOR           │
│                                                             │
│   (stores info)       (finds info)    (uses info to answer) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Let's build understanding of each:

---

### Component 1: Knowledge Base 📚

**What it is:**
A structured collection of documents, text, or data that the system can search through.

**What it contains:**
```
- PDF documents
- Word files
- Web pages
- Database records
- Emails
- Code files
- Any text-based information
```

**How it's stored:**
The documents are converted into **vector embeddings** (numerical representations) and stored in a **vector database** — so they can be searched semantically (by meaning, not just keywords).

> We'll go deep on this in Modules 3 and 4. For now: think of it as a **smart searchable library.**

---

### Component 2: Retriever 🔍

**What it is:**
The system that takes the user's question and **finds the most relevant documents** from the knowledge base.

**How it works (simplified):**
```
1. User asks: "What is our refund policy?"
2. Retriever converts question → mathematical vector
3. Retriever searches knowledge base for similar vectors
4. Returns top 3-5 most relevant document chunks
```

**The key insight:**
The retriever doesn't just match keywords.
It understands **meaning and context.**

```
Query:   "How do I get my money back?"
Finds:   Document about "Refund Policy"  ✅

These mean the same thing even though the words are different!
```

---

### Component 3: Generator 🤖

**What it is:**
The LLM (e.g., GPT-4, Claude, Llama) that receives:
1. The original user question
2. The retrieved relevant documents

And generates a **grounded, accurate answer.**

**The key constraint:**
The generator is instructed: **"Only answer using the provided documents."**

```
System prompt:
"You are a helpful assistant. Answer the user's question
ONLY using the following retrieved documents. If the answer
is not in the documents, say 'I don't know.'"

[Retrieved Documents injected here]

User Question: [question here]
```

This constraint is what **prevents hallucination.**

---

# PART 3: RAG Data Flow — End to End

---

## 4. Intuition / Analogy 🎭

### Analogy: The Research Assistant

Imagine you hire a brilliant research assistant named **Alex**.

**Without RAG (pure LLM):**
```
You:   "What does our latest contract with Company X say?"
Alex:  "I believe it says... [makes something up from general knowledge]"
```

**With RAG:**
```
You:   "What does our latest contract with Company X say?"
Alex:  1. Goes to the filing cabinet (Knowledge Base)
        2. Searches for "Company X contract" (Retriever)
        3. Pulls out the relevant pages (Retrieved chunks)
        4. Reads them carefully
        5. Gives you an accurate, sourced answer (Generator)
```

Alex didn't memorize the contract. Alex **looked it up** and then responded.

**RAG = giving the LLM the ability to "look things up" before answering.**

---

## 5. Step-by-Step RAG Data Flow

Let's trace a complete RAG interaction from start to finish:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    OFFLINE PHASE (one-time setup)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: DOCUMENT INGESTION
        Raw documents (PDFs, Word, Web pages)
                    ↓
Step 2: CHUNKING
        Break documents into smaller pieces
        "HR Policy" PDF → [chunk1, chunk2, chunk3...]
                    ↓
Step 3: EMBEDDING
        Convert each chunk into a vector (list of numbers)
        chunk1 → [0.23, 0.87, 0.12, 0.65, ...]
                    ↓
Step 4: VECTOR STORAGE
        Store all vectors in a Vector Database
        (Pinecone, ChromaDB, FAISS, Weaviate...)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ONLINE PHASE (every user query)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 5: USER QUERY
        "What is the maximum sick leave allowed per year?"
                    ↓
Step 6: QUERY EMBEDDING
        Convert query → vector
        "sick leave query" → [0.21, 0.85, 0.14, 0.61, ...]
                    ↓
Step 7: SIMILARITY SEARCH
        Compare query vector against all stored vectors
        Find top-k most similar chunks
        Returns: [chunk_7, chunk_12, chunk_3]
                    ↓
Step 8: CONTEXT INJECTION
        Build prompt:
        ┌────────────────────────────────────────┐
        │ System: Answer using only these docs:  │
        │ [chunk_7 text]                         │
        │ [chunk_12 text]                        │
        │ [chunk_3 text]                         │
        │                                        │
        │ User: What is the max sick leave/year? │
        └────────────────────────────────────────┘
                    ↓
Step 9: LLM GENERATION
        LLM reads context + question
        Generates grounded answer
                    ↓
Step 10: RESPONSE WITH SOURCES
        "According to the HR Policy Document (Section 4.2),
         employees are entitled to 12 sick days per year."
         [Source: hr_policy_2024.pdf, page 8]
```

---

## 6. Math Intuition (Simplified)

The mathematical core of RAG is **similarity search**.

### How does the retriever find relevant documents?

#### Step 1: Represent text as numbers (vectors)

```
"sick leave policy"    → [0.21, 0.85, 0.14, 0.61, 0.33]
"annual leave rules"   → [0.19, 0.82, 0.18, 0.59, 0.31]  ← similar!
"company picnic food"  → [0.91, 0.12, 0.77, 0.22, 0.88]  ← different!
```

#### Step 2: Measure similarity using Cosine Similarity

```
                  A · B
cos(θ) =  ─────────────────
           ||A|| × ||B||
```

**In plain English:**
- If two vectors point in the **same direction** → cosine similarity ≈ 1 → **very similar**
- If two vectors point in **different directions** → cosine similarity ≈ 0 → **very different**

```
similarity("sick leave", "annual leave rules") = 0.95  ✅ HIGH → retrieved
similarity("sick leave", "company picnic")     = 0.12  ❌ LOW → not retrieved
```

> We'll go very deep on this math in Module 3. For now: **higher similarity = more relevant = gets retrieved.**

---

# PART 4: RAG vs Fine-Tuning vs Prompt Engineering

---

## 7. The Three Approaches — Intuition First

Think of these three approaches as three different ways to make your LLM smarter:

```
┌─────────────────────────────────────────────────────────────────┐
│  Approach          │  What you're doing                         │
├─────────────────────────────────────────────────────────────────┤
│  Prompt Engineering│  Giving better instructions               │
│  RAG               │  Giving relevant documents at runtime     │
│  Fine-Tuning       │  Teaching new knowledge through training  │
└─────────────────────────────────────────────────────────────────┘
```

### Analogy: Teaching an Employee

```
Prompt Engineering:
"Here are your instructions for this call. Be professional.
 Focus on returns. Mention our 30-day policy."
→ You're coaching before the task

RAG:
"Here is the customer's order history and our return policy
 document. Now handle their request."
→ You're providing reference material at task time

Fine-Tuning:
"You went through 6 months of intensive training on all
 company policies, products, and procedures."
→ You've trained the knowledge INTO the employee's brain
```

---

## 8. Comparison Matrix

```
┌──────────────────┬────────────────┬──────────────┬─────────────────┐
│   Dimension      │ Prompt Eng.    │    RAG       │  Fine-Tuning    │
├──────────────────┼────────────────┼──────────────┼─────────────────┤
│ Cost             │ 💚 Very Low    │ 🟡 Medium    │ 🔴 High         │
│ Setup Time       │ 💚 Hours       │ 🟡 Days      │ 🔴 Weeks        │
│ Maintenance      │ 💚 Easy        │ 🟡 Moderate  │ 🔴 Complex      │
│ Latency          │ 💚 Fastest     │ 🟡 Medium    │ 💚 Fast         │
│ Knowledge Update │ 💚 Instant     │ 💚 Easy      │ 🔴 Retrain      │
│ Private Data     │ 🔴 Risky       │ 💚 Safe      │ 🟡 Possible     │
│ Accuracy         │ 🔴 Low         │ 🟡 High      │ 💚 Very High    │
│ Hallucination    │ 🔴 High risk   │ 💚 Reduced   │ 🟡 Some risk    │
│ Source Citation  │ 🔴 None        │ 💚 Yes       │ 🔴 No           │
│ Scalability      │ 💚 Easy        │ 💚 Good      │ 🔴 Expensive    │
│ Custom Behavior  │ 🟡 Limited     │ 🔴 Limited   │ 💚 Excellent    │
│ Context Limit    │ 🔴 Limited     │ 🟡 Managed   │ 💚 In weights   │
└──────────────────┴────────────────┴──────────────┴─────────────────┘

💚 = Advantage   🟡 = Neutral   🔴 = Disadvantage
```

---

## 9. Decision Framework: When to Use Each?

```
START
  │
  ▼
Is your task solvable just by giving
better instructions or examples?
  │
  ├── YES ──→ Use PROMPT ENGINEERING
  │           (Cheapest, fastest, try first always)
  │
  ▼
Do you need the LLM to access specific documents,
databases, or frequently updated information?
  │
  ├── YES ──→ Use RAG
  │           (Best for knowledge-heavy Q&A,
  │            enterprise search, chatbots with docs)
  │
  ▼
Do you need the model to behave differently,
use a custom style/tone, or deeply internalize
specialized skills (like medical coding)?
  │
  ├── YES ──→ Use FINE-TUNING
  │           (Best for style adaptation,
  │            specialized domain behavior,
  │            consistent format output)
  │
  ▼
Can you combine approaches?
  │
  └── YES ──→ RAG + Fine-Tuning combined
              (Most powerful but most expensive)
```

### When to Use Each — Real Examples

```
PROMPT ENGINEERING:
✅ "Always respond in bullet points"
✅ "You are a customer service agent named Sam"
✅ "Summarize this text in 3 sentences"
✅ "Translate this to French"

RAG:
✅ "Answer questions about our 500-page product manual"
✅ "Chat with our legal documents database"
✅ "Answer based on today's news articles"
✅ "Search our internal company wiki"
✅ "Medical Q&A from research papers"

FINE-TUNING:
✅ "Write code in our company's specific coding style"
✅ "Generate medical billing codes (ICD-10)"
✅ "Respond like our brand voice consistently"
✅ "Understand industry-specific jargon deeply"
✅ "Classify emails in our custom taxonomy"

RAG + FINE-TUNING:
✅ Medical assistant that speaks like a doctor (fine-tune)
   AND retrieves from medical literature (RAG)
✅ Legal assistant with legal reasoning style (fine-tune)
   AND retrieves from case law database (RAG)
```

---

## 10. Failure Modes 🚨

### RAG Failure Modes

```
FAILURE 1: Retrieval Failure (Wrong chunks retrieved)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Retriever fetches irrelevant documents
Why:       Poor embeddings, bad chunking, vague query
Result:    LLM hallucinates because context is wrong
Example:   Query about "bank account" retrieves
           documents about "river bank"

FAILURE 2: Context Window Overflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Too many retrieved chunks exceed LLM limit
Why:       k (number of results) set too high
Result:    LLM ignores important parts or errors out

FAILURE 3: Lost in the Middle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   LLM ignores information in the middle of context
Why:       LLMs pay more attention to beginning and end
Result:    Key information missed even when retrieved correctly

FAILURE 4: Outdated Knowledge Base
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Knowledge base not updated regularly
Why:       No pipeline to refresh documents
Result:    RAG answers become stale over time

FAILURE 5: Hallucination Despite Retrieval
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   LLM mixes retrieved info with training data
Why:       LLM doesn't strictly follow "use only context"
Result:    Partial hallucination in the final answer
```

### Fine-Tuning Failure Modes

```
FAILURE 1: Catastrophic Forgetting
Problem:   Fine-tuning on new data → LLM forgets old knowledge
Why:       Model weights overwritten

FAILURE 2: Overfitting
Problem:   Model memorizes training examples, doesn't generalize
Why:       Too many epochs, too little data

FAILURE 3: Data Leakage
Problem:   Private training data can be extracted
Why:       Model memorizes sensitive examples
```

---

## 11. Interview Questions + Strong Answers

### Q1: "What is RAG and why is it needed?"

**Strong Answer:**
> "RAG stands for Retrieval-Augmented Generation. It solves four core limitations of LLMs: hallucination, knowledge cutoff, lack of domain-specific data, and inability to cite sources. The approach works in two phases: offline, where documents are chunked, embedded, and stored in a vector database; and online, where a user query is embedded, similar documents are retrieved via semantic search, and those documents are injected into the LLM's context before generation. The key insight is that LLM weights remain static while the knowledge base remains dynamic and updatable."

---

### Q2: "When would you choose RAG over fine-tuning?"

**Strong Answer:**
> "I choose RAG when the core challenge is knowledge access — specifically when information changes frequently, comes from private documents, or requires source citation for auditability. I choose fine-tuning when the challenge is behavioral — teaching the model a specific style, tone, format, or deeply specialized skill that doesn't change often. In practice, RAG is almost always the first thing I try because it's faster, cheaper, and easier to update. Fine-tuning adds significant cost and complexity and should only be added when RAG alone is insufficient."

---

### Q3: "What are the failure modes of RAG?"

**Strong Answer:**
> "The most critical failure mode is retrieval failure — when the wrong chunks are retrieved, the LLM hallucinates because it's grounding on bad context. This is often caused by poor chunking strategy or weak embeddings. Other failure modes include: context window overflow when too many chunks are retrieved; the 'lost in the middle' problem where LLMs ignore relevant information in the center of long contexts; and stale knowledge bases that haven't been updated. The most dangerous failure is when the LLM mixes retrieved content with its own training data, producing partially hallucinated answers that appear grounded."

---

### Q4: "Explain the end-to-end RAG data flow."

**Strong Answer:**
> "RAG has two phases. The offline ingestion phase: documents are loaded, chunked into smaller pieces, converted to vector embeddings using an embedding model, and stored in a vector database. The online query phase: the user's query is embedded using the same embedding model, a similarity search (typically cosine similarity) is performed against stored vectors, the top-k most relevant chunks are retrieved, these chunks are injected into the LLM prompt as context, and the LLM generates a grounded response — ideally citing the source documents."

---

## 12. Common Mistakes and Misconceptions ⚠️

```
MISTAKE 1: "RAG eliminates hallucination completely"
TRUTH:     RAG reduces hallucination significantly but doesn't
           eliminate it. LLMs can still hallucinate by mixing
           retrieved content with training data.

MISTAKE 2: "Fine-tuning is always better than RAG"
TRUTH:     Fine-tuning is expensive, slow to update, and doesn't
           provide source attribution. RAG is preferred for most
           knowledge-intensive use cases.

MISTAKE 3: "RAG is just keyword search"
TRUTH:     RAG uses semantic/vector search — it understands
           meaning, not just keywords. "car" and "automobile"
           will match even though the words are different.

MISTAKE 4: "Just throw all documents into the knowledge base"
TRUTH:     Chunking strategy, document quality, and metadata
           filtering are critical. Garbage in = garbage out.

MISTAKE 5: "RAG is slow and expensive"
TRUTH:     The retrieval step adds only milliseconds. The main
           cost is the LLM generation, which is the same as
           without RAG. Vector search is extremely fast.

MISTAKE 6: "RAG and fine-tuning are mutually exclusive"
TRUTH:     The most powerful systems use BOTH:
           Fine-tune for behavior + RAG for knowledge.
```

---

## 13. Practical ML Usage

### Real-World RAG Applications

```
Industry          Use Case                      Knowledge Base
─────────────────────────────────────────────────────────────
Legal Tech        Contract Q&A                  Legal documents
Healthcare        Clinical decision support      Medical literature
Finance           Earnings call analysis        Financial reports
Customer Service  Support chatbot               Product manuals
Education         Textbook Q&A system           Course materials
HR Tech           Policy assistant              HR documents
Software Dev      Codebase Q&A                  Code + documentation
Government        Regulation lookup             Policy documents
```

### Production RAG Stack (Common)

```
Document Loading:    LangChain document loaders
Chunking:           RecursiveCharacterTextSplitter
Embedding Model:    OpenAI text-embedding-3-small / BGE / E5
Vector Database:    Pinecone / ChromaDB / Weaviate / FAISS
LLM:               GPT-4o / Claude 3.5 / Llama 3
Orchestration:      LangChain / LlamaIndex / LangGraph
Evaluation:         RAGAS framework
```

---

## 14. Engineer Mindset 🔧

**What a real RAG engineer thinks about:**

```
Question 1: "Is RAG even the right solution here?"
→ Try prompt engineering first. Add RAG only if needed.

Question 2: "What is my retrieval quality?"
→ Measure precision and recall of retrieved chunks.
  A bad retriever ruins the whole system.

Question 3: "What chunking strategy should I use?"
→ Chunk size matters enormously. Too small = missing context.
  Too large = irrelevant noise injected into prompt.

Question 4: "How do I keep my knowledge base fresh?"
→ Build an ingestion pipeline with scheduled updates.
  Stale knowledge base = stale answers.

Question 5: "How do I evaluate end-to-end quality?"
→ Evaluate retrieval AND generation separately.
  Use RAGAS for automated RAG evaluation.

Question 6: "What happens when nothing is retrieved?"
→ Build fallback behavior. Never let the LLM answer
  from pure memory if the system requires grounding.

Question 7: "What are my latency requirements?"
→ RAG adds retrieval latency. Optimize chunk count (k),
  use faster vector stores, consider caching.
```

---

## 15. Exercises 📝

### Beginner Level

```
1. In your own words, explain why a doctor would trust a RAG-powered
   medical assistant more than a pure LLM assistant.

2. List 3 real-world scenarios where knowledge cutoff would be
   a serious problem for an LLM.

3. Draw the RAG data flow on paper — offline phase and online phase.
   Label every step without looking at notes.

4. For each scenario below, decide: Prompt Engineering, RAG, or Fine-Tuning?
   a) Chatbot that answers from a 200-page employee handbook
   b) Email classifier that uses your company's 12 custom categories
   c) Assistant that always responds in a formal, legal writing style
   d) Q&A system over daily news articles
```

### Intermediate Level

```
5. Why is cosine similarity used instead of Euclidean distance
   for comparing text embeddings? (Hint: think about magnitude vs direction)

6. If your RAG system keeps retrieving the wrong documents,
   what are 5 things you would investigate and fix?

7. A healthcare company says "We want zero hallucination."
   Explain why RAG reduces but doesn't eliminate hallucination,
   and what additional safeguards you would add.

8. Design a RAG system for a law firm that needs to query
   10,000 legal case documents. What components would you choose
   and why?
```

### Advanced Level

```
9. How would you evaluate whether your RAG system is actually
   better than a pure LLM for your specific use case?
   Define metrics and evaluation strategy.

10. When would you use BOTH RAG + fine-tuning? Design a system
    architecture that combines both approaches for a medical
    diagnosis assistant.
```

---

## 16. Resources 📚

```
PAPERS:
📄 Original RAG Paper: "Retrieval-Augmented Generation for
   Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
   https://arxiv.org/abs/2005.11401

📄 "Lost in the Middle: How Language Models Use Long Contexts"
   https://arxiv.org/abs/2307.03172

COURSES & DOCS:
🔗 LangChain RAG Documentation
   https://python.langchain.com/docs/tutorials/rag/

🔗 LlamaIndex RAG Guide
   https://docs.llamaindex.ai/

EVALUATION:
🔗 RAGAS (RAG Evaluation Framework)
   https://docs.ragas.io/

PRACTICAL:
🔗 DeepLearning.AI - "Building and Evaluating Advanced RAG"
   https://www.deeplearning.ai/short-courses/
```

---

## Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG in One Mental Model                  │
│                                                             │
│  LLMs are brilliant but frozen.                            │
│  RAG gives them a library to look things up in real-time.  │
│                                                             │
│  OFFLINE:  Documents → Chunks → Vectors → Store            │
│  ONLINE:   Query → Retrieve → Inject → Generate            │
│                                                             │
│  The retriever is the heart of RAG.                        │
│  Bad retrieval = bad answers, always.                      │
│                                                             │
│  Use RAG for KNOWLEDGE problems.                           │
│  Use Fine-Tuning for BEHAVIOR problems.                    │
│  Use Prompt Engineering for INSTRUCTION problems.          │
└─────────────────────────────────────────────────────────────┘
```