# Module 2: Document Processing and Chunking 

## The Foundation That Makes or Breaks Your RAG System

---

> **First, let's build intuition before diving into code and concepts.**

**Before I explain — what do you think happens when you feed a 500-page PDF to a RAG system?**

A beginner might think: *"Just upload the PDF and the AI reads it."*

The reality is far more nuanced — and getting this wrong is the #1 reason RAG systems fail in production.

---

# The Big Picture: Why Document Processing Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   GARBAGE IN  →  GARBAGE RETRIEVED  →  GARBAGE ANSWER          │
│                                                                 │
│   The retriever can only find what was properly stored.        │
│   The LLM can only answer from what the retriever finds.       │
│                                                                 │
│   Document Processing = The Foundation of Everything in RAG    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**A perfect LLM + perfect vector store + bad chunking = broken RAG system.**

This module is about getting the foundation right.

---

# PART 1: Document Loaders in LangChain

---

## 1. What is a Document Loader?

### Simple Definition

A **Document Loader** is a tool that reads raw files (PDFs, web pages, CSVs, etc.) and converts them into a **standardized format** that the RAG pipeline can work with.

### Why Do We Need Loaders?

```
Reality of real-world data:
┌─────────────────────────────────────────────────┐
│  PDF files      → Binary format, complex layout │
│  Web pages      → HTML tags, JavaScript noise   │
│  CSV files      → Tabular rows and columns      │
│  JSON files     → Nested key-value structures   │
│  Word docs      → Proprietary Microsoft format  │
│  Code files     → Syntax-specific structure     │
└─────────────────────────────────────────────────┘

All different formats → Need ONE standard output
```

### LangChain's Standard Document Object

Every loader outputs a **Document object** with exactly two fields:

```python
Document(
    page_content = "The actual text content here...",
    metadata     = {
        "source": "hr_policy.pdf",
        "page":   3,
        "author": "HR Department",
        # any other useful information
    }
)
```

**This is the universal currency of LangChain's RAG pipeline.**
Everything upstream produces Documents. Everything downstream consumes Documents.

---

## 2. PDF Loaders — Deep Dive

### Why PDF is Hard 🧩

PDF (Portable Document Format) was designed for **printing**, not for text extraction. This creates serious challenges:

```
PDF Challenges:
┌────────────────────────────────────────────────────────┐
│ 1. Multi-column layouts    → Text order gets scrambled │
│ 2. Tables                  → Lose row/column structure │
│ 3. Headers and footers     → Repeated noise on pages   │
│ 4. Scanned PDFs            → Images, no real text      │
│ 5. Mathematical formulas   → Garbled symbols           │
│ 6. Embedded images         → Text inside images missed │
│ 7. Footnotes               → Break reading flow        │
└────────────────────────────────────────────────────────┘
```

### The 4 PDF Loaders Compared

```
┌─────────────────┬──────────────┬──────────────┬────────────────┬──────────────┐
│   Loader        │   Speed      │  Accuracy    │  Tables        │  Best For    │
├─────────────────┼──────────────┼──────────────┼────────────────┼──────────────┤
│ PyPDFLoader     │ 💚 Fastest   │ 🟡 Basic     │ 🔴 Poor        │ Simple PDFs  │
│ PDFMiner        │ 🟡 Medium    │ 🟡 Good      │ 🟡 Partial     │ Text-heavy   │
│ PDFPlumber      │ 🟡 Medium    │ 💚 Best      │ 💚 Excellent   │ Tables/data  │
│ Unstructured    │ 🔴 Slowest   │ 💚 Best      │ 💚 Excellent   │ Complex docs │
└─────────────────┴──────────────┴──────────────┴────────────────┴──────────────┘
```

---

### PyPDFLoader — The Fast Default

```python
from langchain_community.document_loaders import PyPDFLoader

# Basic usage
loader = PyPDFLoader("documents/hr_policy.pdf")
pages = loader.load()

# What you get back:
# pages = [Document, Document, Document, ...]
# One Document per page

print(f"Total pages: {len(pages)}")
print(f"First page content: {pages[0].page_content[:200]}")
print(f"First page metadata: {pages[0].metadata}")

# Output:
# Total pages: 47
# First page content: "HR Policy Manual 2024\nSection 1: Employment..."
# First page metadata: {'source': 'documents/hr_policy.pdf', 'page': 0}
```

**When to use PyPDFLoader:**
```
✅ Simple text-based PDFs
✅ Speed is priority
✅ No complex tables needed
✅ Quick prototyping
❌ Avoid for scanned PDFs
❌ Avoid when table extraction matters
```

---

### PDFPlumber — The Table Expert

```python
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("documents/financial_report.pdf")
docs = loader.load()

# PDFPlumber extracts tables WITH structure preserved
# A table like:
# | Quarter | Revenue | Growth |
# | Q1 2024 | $2.3M   | 12%    |
# | Q2 2024 | $2.8M   | 22%    |
#
# PyPDFLoader would give you:
# "Quarter Revenue Growth Q1 2024 $2.3M 12% Q2 2024 $2.8M 22%"
# (jumbled, loses structure)
#
# PDFPlumber gives you structured text preserving the table format
```

**When to use PDFPlumber:**
```
✅ Financial reports with tables
✅ Scientific papers with data tables
✅ Any document where table structure matters
✅ When accuracy > speed
```

---

### Unstructured — The Swiss Army Knife

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# Basic mode
loader = UnstructuredPDFLoader("documents/complex_report.pdf")

# High-resolution mode (slower but better for scanned PDFs)
loader = UnstructuredPDFLoader(
    "documents/scanned_report.pdf",
    mode="elements",           # Returns individual elements
    strategy="hi_res",        # Uses OCR for scanned docs
)

docs = loader.load()

# With mode="elements", each element has type metadata:
# element_type: "Title", "NarrativeText", "Table", "ListItem"
# This lets you process different parts differently!

for doc in docs[:5]:
    print(f"Type: {doc.metadata.get('category')}")
    print(f"Content: {doc.page_content[:100]}")
    print("---")
```

**When to use Unstructured:**
```
✅ Scanned PDFs (needs OCR)
✅ Complex mixed-layout documents
✅ When you need element-level classification
✅ Production systems where accuracy is critical
✅ Supports 30+ file types beyond PDF
❌ Slower - not for quick prototyping
❌ May need additional dependencies / API key
```

---

### PDFMiner — The Text Extraction Specialist

```python
from langchain_community.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("documents/research_paper.pdf")
docs = loader.load()

# PDFMiner excels at:
# - Maintaining reading order in complex layouts
# - Handling text with special characters
# - Better font and position information
# Good middle ground between speed and accuracy
```

---

## 3. Web Loaders

### WebBaseLoader — Single Page Loading

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Simple usage
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Machine_learning")
docs = loader.load()

# With BeautifulSoup filtering (remove nav, ads, footers)
loader = WebBaseLoader(
    web_paths=["https://docs.python.org/3/tutorial/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
            # Only extract these CSS classes - ignore navigation, ads etc.
        )
    ),
)
docs = loader.load()

# Load multiple pages at once
loader = WebBaseLoader([
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3",
])
docs = loader.load()
print(f"Loaded {len(docs)} documents from web")
```

---

### RecursiveUrlLoader — Crawl Entire Websites

```python
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup

# Custom extractor function - cleans HTML noise
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator="\n", strip=True)

loader = RecursiveUrlLoader(
    url="https://docs.langchain.com/",  # Starting URL
    max_depth=3,                         # How deep to crawl
    extractor=bs4_extractor,            # Clean the HTML
    prevent_outside=True,               # Stay on same domain
)

docs = loader.load()
print(f"Crawled {len(docs)} pages")

# Perfect for:
# - Ingesting entire documentation sites
# - Crawling product catalogs
# - Building knowledge bases from websites
```

**Web Loading Challenges:**
```
⚠️  JavaScript-rendered content (React/Vue sites)
    → WebBaseLoader won't see JS-rendered text
    → Need Selenium or Playwright-based loaders

⚠️  Rate limiting and robots.txt
    → Respect website's crawl rules
    → Add delays between requests

⚠️  Login-protected pages
    → Need authentication handling

⚠️  Dynamic content that changes
    → Need scheduled re-ingestion pipeline
```

---

## 4. Structured Data Loaders

### CSV Loader

```python
from langchain_community.document_loaders import CSVLoader

# Basic CSV loading
loader = CSVLoader("data/products.csv")
docs = loader.load()

# Each ROW becomes one Document
# CSV:
# product_id, name,        price, category
# 001,        Widget A,    29.99, Electronics
# 002,        Gadget B,    49.99, Electronics

# Produces Documents like:
# Document(
#   page_content="product_id: 001\nname: Widget A\nprice: 29.99\ncategory: Electronics",
#   metadata={"source": "data/products.csv", "row": 0}
# )

# Custom configuration
loader = CSVLoader(
    file_path="data/products.csv",
    source_column="product_id",    # Use this column as source in metadata
    csv_args={
        "delimiter": ",",
        "fieldnames": ["id", "name", "price", "category"]
    }
)
```

### JSON Loader

```python
from langchain_community.document_loaders import JSONLoader
import json

# JSON file structure:
# {
#   "articles": [
#     {"title": "...", "content": "...", "author": "..."},
#     {"title": "...", "content": "...", "author": "..."}
#   ]
# }

loader = JSONLoader(
    file_path="data/articles.json",
    jq_schema=".articles[]",           # Navigate JSON with jq syntax
    content_key="content",             # Which field is the main text
    metadata_func=lambda record, meta: {  # Custom metadata extraction
        **meta,
        "title": record.get("title"),
        "author": record.get("author")
    }
)
docs = loader.load()

# Each article becomes one Document with title and author in metadata
```

---

## 5. TextLoader — The Simple Default

```python
from langchain_community.document_loaders import TextLoader

# Simple text file
loader = TextLoader(
    "documents/notes.txt",
    encoding="utf-8"    # Always specify encoding to avoid issues
)
docs = loader.load()

# Entire file = One Document
# Good for: plain text, markdown files, simple notes

# For multiple files in a directory:
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "documents/",           # Directory path
    glob="**/*.txt",        # Pattern to match files
    loader_cls=TextLoader,  # Which loader to use
    show_progress=True,     # Progress bar
    use_multithreading=True # Load files in parallel
)
docs = loader.load()
print(f"Loaded {len(docs)} text files")
```

---

## 6. Hands-On: Load from 5 Different Sources

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    JSONLoader,
    TextLoader,
)
from langchain_core.documents import Document
from typing import List

def load_all_sources() -> List[Document]:
    all_docs = []

    # ── Source 1: PDF ──────────────────────────────────────
    print("Loading PDF...")
    pdf_loader = PyPDFLoader("docs/annual_report.pdf")
    pdf_docs = pdf_loader.load()
    print(f"  ✅ PDF: {len(pdf_docs)} pages loaded")
    all_docs.extend(pdf_docs)

    # ── Source 2: Web Page ─────────────────────────────────
    print("Loading Web Page...")
    web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/RAG")
    web_docs = web_loader.load()
    print(f"  ✅ Web: {len(web_docs)} pages loaded")
    all_docs.extend(web_docs)

    # ── Source 3: CSV ──────────────────────────────────────
    print("Loading CSV...")
    csv_loader = CSVLoader("data/products.csv")
    csv_docs = csv_loader.load()
    print(f"  ✅ CSV: {len(csv_docs)} rows loaded")
    all_docs.extend(csv_docs)

    # ── Source 4: JSON ─────────────────────────────────────
    print("Loading JSON...")
    json_loader = JSONLoader(
        file_path="data/articles.json",
        jq_schema=".articles[]",
        content_key="content"
    )
    json_docs = json_loader.load()
    print(f"  ✅ JSON: {len(json_docs)} records loaded")
    all_docs.extend(json_docs)

    # ── Source 5: Text File ────────────────────────────────
    print("Loading Text...")
    txt_loader = TextLoader("docs/notes.txt", encoding="utf-8")
    txt_docs = txt_loader.load()
    print(f"  ✅ Text: {len(txt_docs)} documents loaded")
    all_docs.extend(txt_docs)

    # ── Summary ────────────────────────────────────────────
    print(f"\n📚 Total documents loaded: {len(all_docs)}")

    # Show source distribution
    sources = {}
    for doc in all_docs:
        src = doc.metadata.get("source", "unknown")
        src_type = src.split(".")[-1] if "." in src else "web"
        sources[src_type] = sources.get(src_type, 0) + 1
    print(f"📊 Source breakdown: {sources}")

    return all_docs

docs = load_all_sources()
```

---

# PART 2: Text Splitting Strategies

---

## 7. Why Do We Need to Split? The Core Problem

### Analogy: The Textbook Problem 📖

Imagine someone asks you:
*"What does Chapter 7 say about neural networks?"*

You can't hand them the entire 800-page textbook and say "it's in there somewhere."

You need to:
1. Find the relevant section (Chapter 7 on neural networks)
2. Give them just that section — not the whole book

**This is exactly what chunking does for RAG.**

### The Technical Constraints

```
Why we can't just use the whole document:

1. CONTEXT WINDOW LIMIT
   LLMs have a maximum input size
   GPT-4:     128,000 tokens max
   Claude:    200,000 tokens max
   But filling the entire context = expensive + slow

2. RETRIEVAL PRECISION
   If chunks are too large → Retrieved chunk contains
   lots of irrelevant text → LLM gets confused

3. EMBEDDING QUALITY
   Embedding models work best on focused text
   A 10,000-word document → one blurry embedding
   A 300-word chunk → one sharp, precise embedding

4. COST
   More tokens in prompt = higher API cost
   Retrieve 3 focused chunks << Inject entire document
```

### The Goldilocks Problem of Chunking

```
Too Small (< 50 tokens):
   ❌ Missing context — chunk alone is meaningless
   ❌ Poor retrieval — embedding doesn't capture full idea
   Example: "The patient should take..." ← take WHAT?

Too Large (> 2000 tokens):
   ❌ Retrieval noise — chunk covers many topics
   ❌ Diluted embeddings — embedding becomes unfocused
   ❌ Higher cost — more tokens in every prompt

Just Right (128-512 tokens for most cases):
   ✅ Complete thought captured
   ✅ Sharp, focused embedding
   ✅ Manageable cost
   ✅ Good retrieval precision
```

---

## 8. RecursiveCharacterTextSplitter — The Recommended Default

### First-Principles Understanding

**The core idea:** Split text by trying a hierarchy of separators, from largest natural break to smallest, until chunks are small enough.

```
Hierarchy of natural text breaks:
┌────────────────────────────────────────────────────┐
│  Level 1: "\n\n"  → Paragraph break (best split)  │
│  Level 2: "\n"    → Line break                     │
│  Level 3: " "     → Word break                     │
│  Level 4: ""      → Character break (last resort)  │
└────────────────────────────────────────────────────┘

Logic: "Split at the most natural boundary possible
        that keeps chunks under chunk_size"
```

### Hierarchical Separator Logic — Step by Step

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence.
It enables computers to learn from data.

Types of Machine Learning

There are three main types:
Supervised learning uses labeled data.
Unsupervised learning finds hidden patterns.
Reinforcement learning learns through rewards.

Applications

Machine learning is used in healthcare, finance, and more.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,          # Maximum characters per chunk
    chunk_overlap=20,        # Characters shared between consecutive chunks
    separators=["\n\n", "\n", " ", ""],  # Hierarchy of separators
    length_function=len,     # How to measure chunk size (characters here)
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:80]}...")
    print("---")
```

**Step-by-step trace of how RecursiveCharacterTextSplitter works:**

```
STEP 1: Try splitting by "\n\n" (paragraph breaks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Text splits into paragraphs:
  Para 1: "Introduction to Machine Learning\n\nMachine learning is..."
  Para 2: "Types of Machine Learning\n\nThere are three main types:..."
  Para 3: "Applications\n\nMachine learning is used in healthcare..."

Check: Is Para 1 under chunk_size (100 chars)?
  Para 1 length = 87 chars → ✅ FITS → Keep as one chunk

Check: Is Para 2 under chunk_size?
  Para 2 length = 156 chars → ❌ TOO BIG → Go deeper

STEP 2: Try splitting Para 2 by "\n" (line breaks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Line 1: "Types of Machine Learning"
  Line 2: "There are three main types:"
  Line 3: "Supervised learning uses labeled data."
  ...

Check sizes → All fit under 100 chars ✅

STEP 3: Combine lines that fit together under chunk_size
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Chunk: "Types of Machine Learning\nThere are three main types:" = 52 chars ✅
  Add next line: "...Supervised learning..." → 91 chars ✅
  Add next line: "...Unsupervised learning..." → 134 chars ❌ too big
  → Finalize previous chunk, start new one

RESULT: Natural, semantically coherent chunks!
```

---

## 9. Parameter Tuning: chunk_size and chunk_overlap

### Understanding chunk_size

```python
# chunk_size measured in CHARACTERS by default
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# chunk_size measured in TOKENS (more accurate for LLMs)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Using tiktoken to count tokens like GPT-4 does
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,                    # 256 TOKENS
    chunk_overlap=25,                  # 25 tokens overlap
    length_function=lambda x: len(enc.encode(x)),  # Count tokens, not chars
)

# Why tokens matter:
# "AI" = 1 token
# "Artificial Intelligence" = 2 tokens
# "supercalifragilisticexpialidocious" = 6 tokens
# 500 characters ≠ 500 tokens (usually ~100-200 tokens)
```

### Understanding chunk_overlap

**The core problem overlap solves:**

```
WITHOUT OVERLAP:
Chunk 1: "...The patient was administered Drug X at 10mg dosage."
Chunk 2: "The side effects were observed within 24 hours..."

Query: "What were the side effects of Drug X?"
Retriever finds Chunk 2: "side effects were observed within 24 hours"
But Chunk 2 has NO MENTION of Drug X!
LLM: "Which drug are we talking about?" 🤷

WITH OVERLAP (chunk_overlap = 50 chars):
Chunk 1: "...The patient was administered Drug X at 10mg dosage."
Chunk 2: "Drug X at 10mg dosage. The side effects were observed within 24 hours..."
          ←─── This part repeated from Chunk 1 ───→

Query: "What were the side effects of Drug X?"
Retriever finds Chunk 2 — now contains "Drug X" AND "side effects"
LLM gives perfect answer ✅
```

**Visualizing overlap:**

```
Document: [===================================]
                   positions 0 to 1000

chunk_size=200, chunk_overlap=50:

Chunk 1: [0    ────────────────── 200]
Chunk 2:           [150 ─────────────────── 350]
Chunk 3:                     [300 ──────────────────── 500]
                  ↑                    ↑
                 50 char            50 char
                 overlap             overlap
```

---

## 10. Other Text Splitters

### CharacterTextSplitter — Simple Single Separator

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",    # Split ONLY on this one separator
    chunk_size=500,
    chunk_overlap=50,
    is_separator_regex=False,
)

# Difference from Recursive:
# CharacterTextSplitter:   Splits on ONE separator only
# RecursiveCharacterTextSplitter: Tries multiple separators hierarchically

# When to use:
# ✅ When you know exactly where splits should happen
# ✅ Documents with clear, consistent delimiters
# ❌ Most cases — RecursiveCharacterTextSplitter is better default
```

### TokenTextSplitter — Split by Token Count

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=256,      # 256 TOKENS (not characters)
    chunk_overlap=25,
)

# Why tokens?
# LLMs think in tokens, not characters
# This gives EXACT control over how much context is in each chunk
# Critical for staying within model context limits

# Relationship: 1 token ≈ 4 characters (rough average for English)
# 256 tokens ≈ 1024 characters ≈ ~180 words
```

---

## 11. Semantic Chunking — The Intelligent Approach

### The Problem with Fixed-Size Splitting

```
Traditional chunking:
Document: "...Einstein developed the theory of relativity. | This changed
           physics forever. | The speed of light is constant..."

Split at character 500:
Chunk 1: "...Einstein developed the theory of relativity. This changed"
Chunk 2: "physics forever. The speed of light is constant..."

❌ "This changed" is in Chunk 1 but "physics" — what changed? — is in Chunk 2
   Arbitrary split broke a complete idea!
```

### How Semantic Chunking Works

**Core idea:** Instead of splitting at fixed sizes, detect when the **meaning/topic shifts** and split there.

```
Semantic Chunking Process:
━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Split text into sentences
  s1: "Einstein developed relativity."
  s2: "This revolutionized physics."
  s3: "The speed of light is constant."
  s4: "Stock markets rose today."       ← TOPIC SHIFT!
  s5: "The NASDAQ gained 2.3%."

Step 2: Embed each sentence
  s1 → [0.82, 0.31, 0.64, ...]  ← physics topic
  s2 → [0.79, 0.33, 0.61, ...]  ← physics topic
  s3 → [0.85, 0.29, 0.67, ...]  ← physics topic
  s4 → [0.12, 0.88, 0.23, ...]  ← finance topic (very different!)
  s5 → [0.09, 0.91, 0.19, ...]  ← finance topic

Step 3: Measure similarity between consecutive sentences
  sim(s1, s2) = 0.94  → same topic, keep together
  sim(s2, s3) = 0.91  → same topic, keep together
  sim(s3, s4) = 0.18  → TOPIC SHIFT DETECTED! Split here!
  sim(s4, s5) = 0.96  → same topic, keep together

Step 4: Split at detected boundaries
  Chunk 1: s1 + s2 + s3  (physics content)
  Chunk 2: s4 + s5       (finance content)
```

### Breakpoint Threshold Types

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Method 1: PERCENTILE
# Split when similarity drops below the Nth percentile
# More splits overall — good for varied documents
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,  # Split at bottom 30% of similarities
)

# Method 2: STANDARD DEVIATION
# Split when similarity is more than N std deviations below mean
# More conservative — fewer, larger chunks
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0,  # 1 std dev below mean = split
)

# Method 3: GRADIENT
# Uses rate of change in similarity (derivative)
# Best for detecting sudden topic shifts
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="gradient",
    breakpoint_threshold_amount=95,  # Top 5% of gradient = split
)

# Usage
docs = splitter.create_documents([long_text])
```

**Comparison of threshold types:**

```
PERCENTILE:
  Behavior:  Fixed % of sentences will be split points
  Best for:  Documents with many topic changes
  Risk:      Can over-split or under-split

STANDARD DEVIATION:
  Behavior:  Only splits on statistically unusual drops
  Best for:  More uniform documents
  Risk:      May miss subtle topic shifts

GRADIENT:
  Behavior:  Detects rapid changes in similarity
  Best for:  Documents with sudden topic shifts
  Risk:      Can be sensitive to noise
```

---

## 12. MarkdownTextSplitter — Structure-Aware Splitting

```python
from langchain_text_splitters import MarkdownTextSplitter

markdown_text = """
# Introduction
This document explains our product.

## Features
Our product has amazing features.

### Feature 1: Speed
It is very fast.

### Feature 2: Accuracy
It is very accurate.

## Pricing
The basic plan costs $10/month.
"""

splitter = MarkdownTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

docs = splitter.create_documents([markdown_text])

# Smart: Splits at heading boundaries, not arbitrary positions
# Respects the document's inherent structure

# Chunk 1: Everything under "# Introduction"
# Chunk 2: "## Features" + "### Feature 1: Speed"
# Chunk 3: "### Feature 2: Accuracy"
# Chunk 4: "## Pricing"

# Separators used:
# ["#", "##", "###", "####", "#####", "######", "`\n\n", "\n\n", "\n", " "]
```

---

## 13. Code Splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python code splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
)

python_code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    def load_data(self, filepath):
        \"\"\"Load data from file.\"\"\"
        with open(filepath) as f:
            self.data = json.load(f)
        return self

    def process(self):
        \"\"\"Process the loaded data.\"\"\"
        return [self.transform(item) for item in self.data]

def standalone_function(x, y):
    \"\"\"A simple utility function.\"\"\"
    return x + y
"""

chunks = python_splitter.split_text(python_code)
# Splits at: class definitions, function definitions, methods
# NOT in the middle of a function!

# JavaScript splitter
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=500,
    chunk_overlap=50,
)

# Supported languages:
# Language.PYTHON, Language.JS, Language.TS, Language.JAVA,
# Language.C, Language.CPP, Language.GO, Language.RUBY,
# Language.RUST, Language.SCALA, Language.SWIFT, Language.MARKDOWN

# Why code splitters matter:
# ❌ Bad: Splits inside a function → chunk is syntactically broken
# ✅ Good: Splits between functions → each chunk is valid code unit
```

---

## 14. LLM-Based Text Splitting

```python
# Concept: Use an LLM itself to determine where splits should occur
# The LLM understands semantic meaning better than any algorithm

# This is the most accurate but most expensive approach

from langchain.text_splitter import TextSplitter
from langchain_openai import ChatOpenAI

# Conceptual implementation:
class LLMTextSplitter:
    def __init__(self, llm, chunk_size_target=500):
        self.llm = llm
        self.chunk_size_target = chunk_size_target

    def split_text(self, text: str) -> list[str]:
        # Ask LLM to identify natural split points
        prompt = f"""
        Analyze the following text and identify the best points
        to split it into coherent chunks of approximately
        {self.chunk_size_target} tokens each.

        Identify split points that preserve:
        1. Complete ideas and arguments
        2. Context necessary to understand each chunk
        3. Natural topic boundaries

        Return a list of indices where splits should occur.

        TEXT:
        {text}
        """
        # LLM returns split points
        # We then split at those points

        # In practice: Use agentic chunking approaches
        # where LLM proposes chunk boundaries

# When to use LLM-based splitting:
# ✅ Critical applications where quality >> cost
# ✅ Complex technical documents where meaning is crucial
# ✅ Legal documents requiring precise boundary detection
# ❌ High-volume ingestion (too expensive, too slow)
# ❌ Real-time applications
```

---

## 15. Hands-On: Compare Chunking Strategies

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import statistics

def compare_chunking_strategies(text: str):
    """Compare all chunking strategies on the same document."""

    strategies = {
        "RecursiveCharacter": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ),
        "Character": CharacterTextSplitter(
            separator="\n\n", chunk_size=500, chunk_overlap=50
        ),
        "Token": TokenTextSplitter(
            chunk_size=128, chunk_overlap=12
        ),
    }

    results = {}

    for name, splitter in strategies.items():
        chunks = splitter.split_text(text)
        chunk_sizes = [len(chunk) for chunk in chunks]

        results[name] = {
            "num_chunks": len(chunks),
            "avg_size": statistics.mean(chunk_sizes),
            "min_size": min(chunk_sizes),
            "max_size": max(chunk_sizes),
            "std_dev": statistics.stdev(chunk_sizes) if len(chunks) > 1 else 0,
            "sample_chunk": chunks[0][:150] + "..." if chunks else "",
        }

    # Print comparison table
    print("\n" + "="*70)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*70)

    for name, stats in results.items():
        print(f"\n📦 {name}:")
        print(f"   Chunks:   {stats['num_chunks']}")
        print(f"   Avg Size: {stats['avg_size']:.0f} chars")
        print(f"   Min/Max:  {stats['min_size']} / {stats['max_size']} chars")
        print(f"   Std Dev:  {stats['std_dev']:.1f}")
        print(f"   Sample:   {stats['sample_chunk']}")

    return results

# Test it
with open("technical_document.txt") as f:
    document_text = f.read()

results = compare_chunking_strategies(document_text)

# Key insight from this comparison:
# Lower std_dev = more uniform chunks = more predictable retrieval
# RecursiveCharacterTextSplitter usually has best balance
```

---

# PART 3: Chunking Best Practices

---

## 16. Optimal Chunk Sizes by Use Case

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CHUNK SIZE SELECTION GUIDE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Use Case: FACTOID QUERIES                                         │
│  Example:  "What is the boiling point of water?"                   │
│  Size:     128–256 tokens                                          │
│  Why:      Answer is a single fact → small focused chunk           │
│            Large chunk = diluted embedding = poor retrieval        │
│                                                                     │
│  Use Case: GENERAL RAG (most common)                               │
│  Example:  "Explain our refund policy"                             │
│  Size:     256–512 tokens                                          │
│  Why:      Need enough context to give complete answer             │
│            Not so large that retrieval quality drops               │
│                                                                     │
│  Use Case: COMPLEX ANALYSIS                                        │
│  Example:  "Summarize the methodology of this research"            │
│  Size:     512–1024 tokens                                         │
│  Why:      Complex ideas need more context                         │
│            Trade-off: larger chunk = less precise embedding        │
│                                                                     │
│  Use Case: DOCUMENT SUMMARIES                                      │
│  Example:  "What is this entire report about?"                     │
│  Size:     Use parent-child chunking (discussed in Module 7)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 17. Overlap Strategies

```
RECOMMENDED: 10–20% overlap of chunk_size

For chunk_size=512:
  chunk_overlap = 51–102 tokens  (10%–20%)
  Practical choice: 50–100 tokens

Too little overlap (< 5%):
  ❌ Context breaks at boundaries
  ❌ Queries spanning two chunks fail

Too much overlap (> 30%):
  ❌ Redundant content in multiple chunks
  ❌ More chunks = more storage, slower retrieval
  ❌ Same info retrieved multiple times wastes context window

The 10-20% sweet spot balances:
  ✅ Context continuity
  ✅ Storage efficiency
  ✅ Retrieval precision
```

---

## 18. How Chunk Size Affects Retrieval Quality

```
THE PRECISION-RECALL TRADEOFF IN CHUNKING:

SMALL CHUNKS (128 tokens):
┌─────────────────────────────────────────┐
│ + High precision — focused embeddings   │
│ + Exact fact retrieval                  │
│ - Low recall — misses surrounding ctx  │
│ - Answer may lack completeness          │
│ - Need more chunks to cover full topic  │
└─────────────────────────────────────────┘

LARGE CHUNKS (1024 tokens):
┌─────────────────────────────────────────┐
│ + High recall — captures full context  │
│ + Better for multi-part questions       │
│ - Low precision — diluted embeddings   │
│ - Irrelevant text retrieved             │
│ - Higher cost per retrieval             │
└─────────────────────────────────────────┘

SOLUTION: HIERARCHICAL CHUNKING
  Store BOTH small chunks (for retrieval precision)
  AND large chunks (for context completeness)
  Retrieve by small, return the large parent
  → Called "Parent-Child Chunking" (Module 7)
```

---

## 19. Handling Tables and Structured Data

```
TABLES ARE THE HARDEST CHALLENGE IN CHUNKING

Problem: A table split across chunks loses meaning
┌────────┬──────────┬────────┐
│ Month  │ Revenue  │ Growth │  ← Header in Chunk 1
├────────┼──────────┼────────┤
│ Jan    │ $100K    │  5%    │  ← Data rows might be split
│ Feb    │ $120K    │ 20%    │    across Chunk 1 and Chunk 2
└────────┴──────────┴────────┘  ← Chunk 2 has data, no headers!

Solutions:
━━━━━━━━━

1. KEEP TABLES TOGETHER (preferred)
   Use Unstructured loader to identify tables
   Ensure entire table becomes one chunk
   Even if it exceeds chunk_size

2. CONVERT TO NATURAL LANGUAGE
   "In January, revenue was $100K with 5% growth.
    In February, revenue was $120K with 20% growth."
   → Now table works like normal text chunks

3. SEPARATE PIPELINE FOR STRUCTURED DATA
   Tables → dedicated SQL/CSV search pipeline
   Not vector search — use exact/SQL queries
   Only join with RAG for final response

4. USE PDFPLUMBER OR UNSTRUCTURED
   These preserve table structure better
   Add table-type metadata: {"element_type": "table"}
   Handle tables differently in retrieval
```

---

## 20. Preserving Context Across Chunks

```python
# Problem: Each chunk loses its surrounding context

# Solution 1: Add contextual headers to each chunk
def add_context_to_chunks(docs, document_title, section_name):
    """Prepend document context to each chunk."""
    contextualized = []
    for doc in docs:
        context_prefix = f"Document: {document_title}\nSection: {section_name}\n\n"
        doc.page_content = context_prefix + doc.page_content
        contextualized.append(doc)
    return contextualized

# Solution 2: Sliding window approach
# Instead of overlap, use a wider window that shifts
# Captures more surrounding context

# Solution 3: Contextual Retrieval (Anthropic's approach)
# Use LLM to generate a context summary for each chunk:
from langchain_openai import ChatOpenAI

def add_contextual_summary(chunk_text: str, full_document: str, llm) -> str:
    """Generate contextual description for a chunk."""
    prompt = f"""
    Here is a document:
    <document>
    {full_document[:2000]}  # First 2000 chars as context
    </document>

    Here is a chunk from this document:
    <chunk>
    {chunk_text}
    </chunk>

    Provide a brief (2-3 sentence) context that situates this chunk
    within the overall document. Focus on what this chunk is about
    and how it relates to the document's main topic.
    """
    context = llm.invoke(prompt).content
    return f"{context}\n\n{chunk_text}"

# This is "Contextual Retrieval" — improves retrieval by 49% (Anthropic)
```

---

# PART 4: Metadata Management

---

## 21. Why Metadata is Critical

### The Library Analogy 📚

Without metadata, your knowledge base is like a library where:
- All books are in one giant pile
- No titles on spines
- No catalog system
- No author information
- No subject classification

With metadata, you can:
- *"Show me only documents from 2024"*
- *"Search only in the HR department's documents"*
- *"Find information only from approved sources"*
- *"Retrieve only from the legal team's contracts"*

**Metadata enables FILTERED retrieval — finding the right answer AND from the right source.**

---

## 22. Adding Metadata to Documents

```python
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime

# Method 1: Add metadata during loading
loader = PyPDFLoader("docs/q3_2024_report.pdf")
docs = loader.load()

# Enrich with additional metadata
for doc in docs:
    doc.metadata.update({
        "document_type":  "financial_report",
        "department":     "finance",
        "quarter":        "Q3",
        "year":           2024,
        "confidential":   True,
        "author":         "CFO Office",
        "ingested_at":    datetime.now().isoformat(),
        "version":        "1.0",
    })

# Method 2: Create Documents with metadata from scratch
docs = [
    Document(
        page_content="The refund policy allows returns within 30 days...",
        metadata={
            "source":         "hr_manual.pdf",
            "page":           12,
            "section":        "Returns Policy",
            "document_type":  "policy",
            "department":     "hr",
            "effective_date": "2024-01-01",
            "version":        "3.2",
            "last_reviewed":  "2024-06-15",
        }
    )
]
```

---

## 23. Building a Metadata Schema

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class DocumentMetadata:
    """
    Standardized metadata schema for your RAG pipeline.
    Define this once — use everywhere.
    """
    # Source tracking
    source: str              # Original file path or URL
    source_type: str         # "pdf", "web", "csv", "json"
    document_id: str         # Unique identifier

    # Content classification
    document_type: str       # "policy", "report", "manual", "contract"
    department: str          # "hr", "legal", "finance", "engineering"
    topic: str               # Main topic/category

    # Temporal information
    created_at: str          # Document creation date
    updated_at: str          # Last update date
    ingested_at: str         # When added to knowledge base
    version: str             # Document version

    # Location within document
    page: Optional[int]      # Page number
    chunk_index: int         # Index of this chunk in the document
    total_chunks: int        # Total chunks from this document

    # Access control
    access_level: str        # "public", "internal", "confidential"
    allowed_roles: list      # ["hr", "management", "all"]

    # Quality signals
    confidence_score: float  # How confident we are in extraction quality

def create_metadata(
    doc,
    source_file: str,
    department: str,
    doc_type: str,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    """Create standardized metadata for a document chunk."""
    return {
        "source":         source_file,
        "source_type":    source_file.split(".")[-1],
        "document_id":    f"{source_file}_{chunk_index}",
        "document_type":  doc_type,
        "department":     department,
        "page":           doc.metadata.get("page", 0),
        "chunk_index":    chunk_index,
        "total_chunks":   total_chunks,
        "ingested_at":    datetime.now().isoformat(),
        "access_level":   "internal",
        "char_count":     len(doc.page_content),
    }
```

---

## 24. Metadata Filtering in Retrieval

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Filter retrieval by metadata — only search specific documents

# Example 1: Only retrieve from HR department
hr_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"department": "hr"}
    }
)

# Example 2: Only retrieve from 2024 financial reports
finance_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "department":     "finance",
            "year":           2024,
            "document_type":  "financial_report"
        }
    }
)

# Example 3: Dynamic filtering based on user role
def get_retriever_for_user(user_role: str):
    """Return appropriate retriever based on user's role."""
    if user_role == "executive":
        # Executives can access everything
        filter_dict = {}
    elif user_role == "hr_staff":
        # HR staff sees only HR documents
        filter_dict = {"department": "hr", "access_level": {"$ne": "confidential"}}
    else:
        # Regular employees see only public docs
        filter_dict = {"access_level": "public"}

    return vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": filter_dict}
    )
```

---

## 25. Document Versioning Strategies

```python
from datetime import datetime
import hashlib

def compute_document_hash(content: str) -> str:
    """Generate unique hash for document content."""
    return hashlib.md5(content.encode()).hexdigest()

class DocumentVersionManager:
    """
    Manages document versions in the knowledge base.
    Ensures old versions are replaced, not duplicated.
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def upsert_document(self, new_docs: list, source_id: str):
        """
        Add or update documents.
        Replaces old version with new version.
        """
        # Step 1: Delete old version
        self.vectorstore.delete(
            where={"source_id": source_id}
        )
        print(f"🗑️  Deleted old version of: {source_id}")

        # Step 2: Add new version with version metadata
        for i, doc in enumerate(new_docs):
            doc.metadata.update({
                "source_id":     source_id,
                "version":       datetime.now().isoformat(),
                "content_hash":  compute_document_hash(doc.page_content),
                "is_latest":     True,
            })

        self.vectorstore.add_documents(new_docs)
        print(f"✅ Added new version of: {source_id} ({len(new_docs)} chunks)")

    def check_if_updated(self, source_id: str, new_content: str) -> bool:
        """Check if document has changed since last ingestion."""
        results = self.vectorstore.get(
            where={"source_id": source_id},
            limit=1
        )
        if not results:
            return True  # New document, needs ingestion

        old_hash = results[0].metadata.get("content_hash")
        new_hash = compute_document_hash(new_content)
        return old_hash != new_hash  # True if content changed
```

---

## 26. Hands-On: Complete Metadata-Enriched Pipeline

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
import hashlib
from pathlib import Path

class MetadataEnrichedPipeline:
    """
    Production-ready document ingestion pipeline
    with comprehensive metadata management.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=51,  # 10% overlap
            length_function=len,
        )

    def ingest_pdf(
        self,
        filepath: str,
        department: str,
        document_type: str,
        access_level: str = "internal",
    ):
        """Load, chunk, enrich, and store a PDF document."""

        print(f"\n📄 Processing: {filepath}")

        # Step 1: Load
        loader = PyPDFLoader(filepath)
        raw_docs = loader.load()
        print(f"  Loaded {len(raw_docs)} pages")

        # Step 2: Split
        chunks = self.splitter.split_documents(raw_docs)
        total_chunks = len(chunks)
        print(f"  Created {total_chunks} chunks")

        # Step 3: Enrich metadata
        source_id = Path(filepath).stem  # filename without extension
        ingestion_time = datetime.now().isoformat()

        for i, chunk in enumerate(chunks):
            content_hash = hashlib.md5(
                chunk.page_content.encode()
            ).hexdigest()[:8]

            chunk.metadata.update({
                # Source tracking
                "source_id":     source_id,
                "source_type":   "pdf",
                "filepath":      filepath,

                # Classification
                "department":    department,
                "document_type": document_type,
                "access_level":  access_level,

                # Chunk info
                "chunk_index":   i,
                "total_chunks":  total_chunks,
                "char_count":    len(chunk.page_content),
                "content_hash":  content_hash,

                # Temporal
                "ingested_at":   ingestion_time,

                # Quality
                "is_latest":     True,
            })

        # Step 4: Store in vector database
        self.vectorstore.add_documents(chunks)
        print(f"  ✅ Stored {total_chunks} chunks with metadata")

        return {
            "source_id":    source_id,
            "total_chunks": total_chunks,
            "metadata_keys": list(chunks[0].metadata.keys()),
        }

    def query_with_filter(
        self,
        query: str,
        department: str = None,
        document_type: str = None,
        k: int = 5,
    ):
        """Query with optional metadata filtering."""

        # Build filter dynamically
        filter_dict = {}
        if department:
            filter_dict["department"] = department
        if document_type:
            filter_dict["document_type"] = document_type

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": filter_dict if filter_dict else None,
            }
        )

        results = retriever.invoke(query)

        print(f"\n🔍 Query: {query}")
        print(f"   Filter: {filter_dict}")
        print(f"   Found {len(results)} relevant chunks:\n")

        for i, doc in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"  Source:  {doc.metadata.get('source_id')}")
            print(f"  Page:    {doc.metadata.get('page')}")
            print(f"  Dept:    {doc.metadata.get('department')}")
            print(f"  Preview: {doc.page_content[:100]}...")
            print()

        return results

# Usage
pipeline = MetadataEnrichedPipeline()

# Ingest multiple documents
pipeline.ingest_pdf(
    "docs/hr_policy_2024.pdf",
    department="hr",
    document_type="policy"
)

pipeline.ingest_pdf(
    "docs/q3_financial_report.pdf",
    department="finance",
    document_type="financial_report",
    access_level="confidential"
)

# Query with filtering
results = pipeline.query_with_filter(
    query="What is the vacation policy?",
    department="hr",
)

results = pipeline.query_with_filter(
    query="What was the revenue in Q3?",
    department="finance",
    document_type="financial_report",
)
```

---

# PART 5: Failure Modes and Interview Prep

---

## 27. Failure Modes of Document Processing 🚨

```
FAILURE 1: Wrong PDF Loader Choice
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Using PyPDFLoader on a scanned PDF
Result:    Empty or garbled page_content (no real text)
Fix:       Use UnstructuredPDFLoader with strategy="hi_res"
Detection: Check if page_content is empty or < 50 chars

FAILURE 2: Chunk Size Too Small
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   chunk_size=50 creates meaningless fragments
Result:    "The patient should..." (should WHAT?)
Fix:       Minimum 128 tokens for meaningful chunks
Detection: Retrieve chunks and manually read them

FAILURE 3: Zero Overlap on Continuous Text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   chunk_overlap=0 breaks ideas at boundaries
Result:    Queries spanning two chunks fail completely
Fix:       Always use 10-20% overlap

FAILURE 4: Encoding Errors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   TextLoader crashes on non-UTF-8 files
Result:    Pipeline fails on certain documents
Fix:       Always specify encoding, use try/except
Code:      TextLoader("file.txt", encoding="utf-8",
                      autodetect_encoding=True)

FAILURE 5: Metadata Inconsistency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Different documents have different metadata keys
Result:    Filtering breaks, lineage tracking fails
Fix:       Define metadata schema upfront, validate on ingest

FAILURE 6: Not Handling Empty Documents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Empty PDF pages create empty chunks
Result:    Empty chunks stored → retrieved for any query
Fix:       Filter out documents with short/empty content

Code fix:
chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

FAILURE 7: Table Splitting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Table split across chunk boundary
Result:    Data rows without headers → meaningless
Fix:       Use PDFPlumber, detect table elements,
           ensure table fits in one chunk
```

---

## 28. Interview Questions and Answers

### Q1: "Why is chunking important in RAG, and what strategy would you use?"

**Strong Answer:**
> "Chunking is critical because LLMs have context window limits, embedding models work best on focused text, and retrieval precision depends on how well each chunk represents a single idea. My default is RecursiveCharacterTextSplitter because it respects natural text boundaries — it tries paragraph breaks before line breaks before word breaks, ensuring semantically coherent chunks. For chunk size, I follow the use case: 128-256 tokens for fact retrieval, 256-512 for general Q&A, 512-1024 for analysis tasks. I always use 10-20% overlap to prevent context loss at boundaries."

---

### Q2: "When would you use SemanticChunker over RecursiveCharacterTextSplitter?"

**Strong Answer:**
> "RecursiveCharacterTextSplitter is my default because it's fast, deterministic, and works well for most documents. I switch to SemanticChunker when documents have variable-length sections covering distinct topics — research papers, long articles, policy documents. SemanticChunker embeds consecutive sentences and detects similarity drops to find natural topic boundaries, creating chunks that are semantically coherent rather than size-bounded. The tradeoff is cost and speed — SemanticChunker makes embedding API calls during ingestion, which is expensive at scale."

---

### Q3: "How does metadata improve RAG retrieval?"

**Strong Answer:**
> "Metadata enables two critical capabilities: filtered retrieval and source attribution. Without metadata filtering, a query about 'vacation policy' might retrieve from any department's documents. With metadata, I can filter to HR department documents only — dramatically improving precision. For source attribution, metadata lets the system cite exactly which document, page, and version the answer came from — essential for compliance in healthcare, legal, and finance. In production, I define a strict metadata schema upfront and validate every document against it during ingestion."

---

### Q4: "How would you handle tables in a PDF for RAG?"

**Strong Answer:**
> "Tables are the hardest challenge in document processing for RAG. Standard text splitters destroy table structure. My approach has three parts: First, use PDFPlumber or UnstructuredPDFLoader which preserve table structure better. Second, detect table elements using Unstructured's element-type metadata and keep entire tables as single chunks even if they exceed normal chunk_size limits. Third, for structured data like financial tables, consider a separate retrieval pipeline — convert tables to SQL or use exact-match search rather than semantic search. For smaller tables that need to stay in the RAG pipeline, I convert them to natural language: 'In Q1 2024, revenue was $2.3M, representing 12% growth.'"

---

## 29. Common Mistakes ⚠️

```
MISTAKE 1: "One chunk size fits all"
TRUTH:     Different query types need different chunk sizes.
           Build multiple indices or tune per document type.

MISTAKE 2: "More overlap is always better"
TRUTH:     > 30% overlap creates redundancy, higher costs,
           and the same info retrieved multiple times.
           10-20% is the sweet spot.

MISTAKE 3: "Just use PyPDFLoader for everything"
TRUTH:     PyPDFLoader is basic. For tables, complex layouts,
           or scanned docs, you need PDFPlumber or Unstructured.

MISTAKE 4: "Metadata is optional"
TRUTH:     In production, metadata is essential for filtering,
           access control, versioning, and debugging.
           Add it during ingestion, not later.

MISTAKE 5: "CharacterTextSplitter is the same as RecursiveCharacterTextSplitter"
TRUTH:     Character uses ONE separator. Recursive uses a
           HIERARCHY of separators. Recursive almost always
           produces better chunks.

MISTAKE 6: "Chunk once, never revisit"
TRUTH:     Chunk strategy should be evaluated with retrieval
           metrics. Wrong chunk size = poor retrieval = poor answers.
           Always evaluate with real queries.
```

---

## 30. Engineer Mindset 🔧

```
WHAT A REAL RAG ENGINEER THINKS ABOUT:

1. "What does my document corpus look like?"
   → PDFs with tables? → Use PDFPlumber + Unstructured
   → Technical docs? → Use Markdown/Code splitters
   → Web content? → Use WebBaseLoader with BS4 cleaning
   → Mixed? → Build loader per document type

2. "How will users query this system?"
   → Short factual queries → smaller chunks (128-256)
   → Complex analytical → larger chunks (512-1024)
   → Unknown → start with 256-512, measure and adjust

3. "What metadata do I need for filtering?"
   → Map out user stories first
   → "Users should only see their department's docs"
   → → Need department metadata
   → Design schema before building pipeline

4. "How do I keep the knowledge base fresh?"
   → Build scheduled ingestion pipeline
   → Implement version management
   → Use content hashing to detect changes
   → Only re-ingest changed documents (cost optimization)

5. "How do I validate ingestion quality?"
   → After ingestion, retrieve sample queries
   → Manually inspect chunks — do they make sense?
   → Check for empty chunks, encoding errors, table splitting
   → Log chunk statistics: avg size, min/max, count

6. "What's my fallback when loading fails?"
   → Build try/except around each loader
   → Log failures with document ID
   → Alert on high failure rates
   → Never let one bad document crash the pipeline
```

---

## 31. Exercises 📝

### Beginner

```
1. Load a PDF of your choice using all 4 PDF loaders.
   Compare the first page content from each.
   Which preserved the most information?

2. Take any article (500+ words) and chunk it with:
   - chunk_size=100, overlap=0
   - chunk_size=100, overlap=20
   - chunk_size=500, overlap=50
   Read the chunks manually. Which feels most natural?

3. Define a metadata schema for a company's HR documents.
   List all metadata fields you would track and why.

4. Load a CSV file using CSVLoader.
   What does each row look like as a Document?
   What metadata is automatically added?
```

### Intermediate

```
5. Build a pipeline that loads 3 PDFs and adds:
   - Department metadata
   - Ingestion timestamp
   - Content hash
   - Chunk index and total chunk count
   Store in ChromaDB and verify metadata is queryable.

6. Compare RecursiveCharacterTextSplitter vs SemanticChunker
   on the same document. Count chunks, measure average size,
   and manually evaluate 5 chunks from each.
   Which would you choose and why?

7. Build a document version management system that:
   - Detects when a document has changed (using hash)
   - Deletes old version from vector store
   - Ingests new version with updated metadata
```

### Advanced

```
8. Design and implement a production ingestion pipeline that:
   - Handles PDF, web, CSV, and JSON sources
   - Applies source-specific chunking strategies
   - Enforces a consistent metadata schema
   - Handles failures gracefully with logging
   - Supports document versioning
   - Filters out empty/low-quality chunks

9. Implement a table extraction pipeline using PDFPlumber:
   - Detect tables in a financial PDF
   - Keep tables as single chunks (don't split)
   - Convert table data to natural language for embedding
   - Add table-specific metadata

10. Build an evaluation harness for chunking strategy:
    - Define 10 test queries for your document
    - For each query, retrieve top-3 chunks
    - Manually score each chunk (1-3: relevant/partial/irrelevant)
    - Compare scores across 3 different chunk strategies
    - Report which strategy performs best and why
```

---

## 32. Resources 📚

```
DOCUMENTATION:
🔗 LangChain Document Loaders
   https://python.langchain.com/docs/integrations/document_loaders/

🔗 LangChain Text Splitters
   https://python.langchain.com/docs/concepts/text_splitters/

🔗 Unstructured.io Documentation
   https://docs.unstructured.io/

PAPERS:
📄 "Contextual Retrieval" (Anthropic, 2024)
   https://www.anthropic.com/news/contextual-retrieval

📄 "Evaluating Chunking Strategies for RAG"
   https://research.trychroma.com/evaluating-chunking

PRACTICAL:
🔗 PDFPlumber GitHub
   https://github.com/jsvine/pdfplumber

🔗 LangChain Semantic Chunker
   https://python.langchain.com/docs/how_to/semantic-chunker/

TOOLS:
🔗 Chroma Vector Store
   https://docs.trychroma.com/
```

---

## Module 2 Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 2 MENTAL MODEL                          │
│                                                                 │
│  LOADERS:  Match loader to document type.                      │
│  PyPDF=simple, PDFPlumber=tables, Unstructured=complex.        │
│                                                                 │
│  CHUNKING: RecursiveCharacterTextSplitter is your default.     │
│  It respects natural boundaries. Always use 10-20% overlap.   │
│                                                                 │
│  SIZE:     Factoid=128-256, General=256-512, Complex=512-1024 │
│  Wrong size = wrong embeddings = wrong retrieval = wrong answer│
│                                                                 │
│  METADATA: Define schema first. Add during ingestion.          │
│  Enables filtering, access control, versioning, debugging.     │
│                                                                 │
│  TABLES:   The hardest problem. Use PDFPlumber or Unstructured.│
│  Don't split tables. Convert to natural language when needed.  │
│                                                                 │
│  GOLDEN RULE: Garbage in = Garbage retrieved = Garbage answer  │
│  Document processing quality = RAG system quality              │
└─────────────────────────────────────────────────────────────────┘
```