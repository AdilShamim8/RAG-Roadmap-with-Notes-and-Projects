# Module 9: RAG Evaluation using RAGAS 

## Measuring What Actually Matters

---

> **Before I explain — let me ask:**
>
> *"How do you know your RAG system is actually good?"*
>
> A beginner might say: *"I ask it questions and the answers look right."*
>
> That's exactly the problem. **"Looks right" is not measurement.**
>
> Here's the brutal truth about RAG in production:
> - Without metrics, you can't compare two systems
> - Without metrics, you can't tell if your changes helped or hurt
> - Without metrics, you can't catch silent quality degradation
> - Without metrics, you're flying blind
>
> **Evaluation is what separates engineers from researchers.**
> Engineers measure. Researchers experiment. You need both.

---

# The Big Picture: Why RAG Evaluation is Hard

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  WHY EVALUATING RAG IS HARDER THAN ML CLASSIFICATION:           │
│                                                                  │
│  Classification:                                                  │
│  ━━━━━━━━━━━━━━━                                                 │
│  Prediction: "spam" or "not spam"                                │
│  Ground truth: "spam"                                            │
│  Metric: Accuracy = exact match                                  │
│  Easy! ✅                                                         │
│                                                                  │
│  RAG:                                                            │
│  ━━━━━                                                           │
│  Query: "What is our refund policy?"                            │
│  Generated: "Customers can return items within 30 days..."       │
│  Ground truth: "Returns accepted within 30 days of purchase"    │
│  Metric: ???                                                     │
│  - Exact match? No (different wording)                          │
│  - Word overlap? Too crude                                       │
│  - Manual review? Doesn't scale                                  │
│  Hard! ❌                                                         │
│                                                                  │
│  THE SOLUTION:                                                   │
│  Use LLMs as judges to evaluate quality across                  │
│  multiple dimensions automatically.                              │
│                                                                  │
│  This is what RAGAS does.                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

# PART 1: Why RAG Evaluation Matters

---

## 1. The Two Halves of RAG Quality

### RAG has TWO things that can go wrong

```
USER QUERY
    │
    ▼
┌─────────────────┐
│   RETRIEVAL     │  ← Can fail here
│ (find docs)     │     Wrong docs retrieved
└─────────────────┘     Missing relevant docs
    │                   Too much noise
    ▼
┌─────────────────┐
│  GENERATION     │  ← Can fail here
│ (write answer)  │     Hallucination
└─────────────────┘     Ignores context
    │                   Incomplete answer
    ▼
  ANSWER

YOU NEED METRICS FOR BOTH:
- Retrieval metrics: Did we find the right info?
- Generation metrics: Did we use it correctly?
```

### The Diagnostic Power of Separate Metrics

```
SCENARIO 1: Bad answer, but retrieval was perfect
→ Generation problem
→ Fix: Better prompt, better LLM, better chunking

SCENARIO 2: Good answer, but retrieval was bad
→ Lucky (LLM filled gaps from training data)
→ Risk: Will fail when LLM doesn't know
→ Fix: Improve retrieval

SCENARIO 3: Bad answer, bad retrieval
→ Root cause is retrieval
→ Fix retrieval first, generation will likely improve

SCENARIO 4: Good answer, good retrieval
→ System working as intended ✅

Without separate metrics, you can't diagnose which is broken!
```

---

# PART 2: Key Evaluation Metrics

---

## 2. The Five Core RAG Metrics

```
RETRIEVAL METRICS (Is what we found useful?)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONTEXT PRECISION  → Are retrieved chunks relevant?
2. CONTEXT RECALL     → Did we find all needed info?
3. NOISE SENSITIVITY  → How does irrelevant info affect quality?


GENERATION METRICS (Did we use it well?)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. FAITHFULNESS       → Is the answer grounded in context?
5. ANSWER RELEVANCE   → Does the answer address the question?


THE INPUTS NEEDED FOR EVALUATION:
- question:          User's query
- answer:            Generated response
- contexts:          Retrieved documents
- ground_truth:      Expected/ideal answer (for some metrics)
```

---

## 3. Metric 1: Faithfulness

### What It Measures

```
FAITHFULNESS = How much of the answer is supported by the context?

It asks: "Did the LLM make up things not in the retrieved docs?"

This is the #1 metric for catching HALLUCINATION.
```

### How It Works (Step by Step)

```
INPUT:
- question: "What is the refund policy?"
- answer:   "Customers can return items within 30 days with original
             receipt. The CEO personally approves all refunds."
- contexts: ["Returns accepted within 30 days with proof of purchase"]

STEP 1: Break answer into individual claims
- Claim 1: "Customers can return items within 30 days"
- Claim 2: "Receipt is required"
- Claim 3: "The CEO personally approves all refunds"

STEP 2: For each claim, check if supported by context
- Claim 1: ✅ Supported (30 days mentioned)
- Claim 2: ✅ Supported (proof of purchase = receipt)
- Claim 3: ❌ NOT supported (nothing about CEO!)

STEP 3: Calculate score
Supported claims / Total claims = 2/3 = 0.67

INTERPRETATION:
0.67 = 67% of answer is grounded in retrieved context
33% is HALLUCINATION (or knowledge from LLM training)

GOAL: 1.0 (100% faithful)
PRODUCTION TARGET: ≥ 0.85
```

### Why Faithfulness Matters Most

```
LOW FAITHFULNESS = HALLUCINATION

This is THE critical safety metric for production RAG:
- Medical: Wrong drug info can kill people
- Legal: False legal claims have consequences
- Financial: Made-up numbers cause bad decisions

If faithfulness is low, EVERYTHING else doesn't matter.
You're outputting potentially false information.
```

---

## 4. Metric 2: Answer Relevance

### What It Measures

```
ANSWER RELEVANCE = Does the answer actually address the question?

It catches answers that are:
- Off-topic
- Too generic
- Incomplete
- Talking around the question without answering it

NOT about correctness — about RELEVANCE TO THE QUERY.
```

### How It Works

```
INPUT:
- question: "What is the capital of France?"
- answer:   "France is a beautiful country with rich history and culture.
             Paris is its capital. The Eiffel Tower is famous."

STEP 1: LLM generates artificial questions from the answer
"What kind of country is France?"
"What is France's capital?"
"What is famous in Paris?"

STEP 2: Calculate similarity between original question and generated ones
- Original: "What is the capital of France?"
- Gen 1:    "What kind of country is France?"    → similarity 0.4
- Gen 2:    "What is France's capital?"          → similarity 0.95
- Gen 3:    "What is famous in Paris?"           → similarity 0.5

STEP 3: Average the similarities
(0.4 + 0.95 + 0.5) / 3 = 0.62

INTERPRETATION:
0.62 = Answer is somewhat relevant but discusses extra topics
A perfect 1.0 answer would generate ONLY questions like the original

GOAL: 1.0 (laser-focused answer)
PRODUCTION TARGET: ≥ 0.80
```

---

## 5. Metric 3: Context Precision

### What It Measures

```
CONTEXT PRECISION = Are retrieved chunks relevant AND in good ranked order?

This is the "did we retrieve junk?" metric.
It also rewards putting the most relevant chunk FIRST.
```

### How It Works

```
INPUT:
- question:   "What is the refund policy?"
- contexts:   [
                "Office hours are 9-5 EST",                  ← rank 1
                "Returns accepted within 30 days",           ← rank 2
                "Dress code is business casual"              ← rank 3
              ]
- ground_truth: "Refunds available within 30 days"

STEP 1: For each retrieved chunk, judge relevance
- Chunk 1: NOT relevant (about office hours)
- Chunk 2: RELEVANT (about returns/refunds)
- Chunk 3: NOT relevant (about dress code)

STEP 2: Calculate precision@k for each position
- @1: 0/1 = 0.0    (chunk 1 not relevant)
- @2: 1/2 = 0.5    (1 of 2 relevant)
- @3: 1/3 = 0.33   (1 of 3 relevant)

STEP 3: Calculate weighted average
Context Precision = sum(precision@k * relevant_at_k) / total_relevant
                  = (0*0 + 0.5*1 + 0.33*0) / 1
                  = 0.5

PROBLEM IDENTIFIED:
Relevant doc was at position 2, not 1.
Better retrieval should put relevant doc FIRST.

GOAL: 1.0 (all relevant docs, with most relevant first)
PRODUCTION TARGET: ≥ 0.75
```

---

## 6. Metric 4: Context Recall

### What It Measures

```
CONTEXT RECALL = Did we retrieve ALL the info needed to answer correctly?

This is the "did we miss anything?" metric.
Requires ground truth answer to know what info SHOULD have been retrieved.
```

### How It Works

```
INPUT:
- question:     "What are our leave policies?"
- contexts:     ["Vacation: 15 days per year",
                 "Sick leave: 10 days per year"]
- ground_truth: "Vacation leave is 15 days, sick leave is 10 days,
                 parental leave is 12 weeks, and bereavement is 5 days"

STEP 1: Break ground truth into claims
- Claim 1: "Vacation leave is 15 days"
- Claim 2: "Sick leave is 10 days"
- Claim 3: "Parental leave is 12 weeks"
- Claim 4: "Bereavement is 5 days"

STEP 2: For each claim, check if supported by retrieved contexts
- Claim 1: ✅ Supported (vacation 15 days)
- Claim 2: ✅ Supported (sick 10 days)
- Claim 3: ❌ NOT in retrieved contexts
- Claim 4: ❌ NOT in retrieved contexts

STEP 3: Calculate recall
Supported claims / Total claims = 2/4 = 0.5

INTERPRETATION:
We only retrieved 50% of the information needed.
The system MISSED parental leave and bereavement info.

GOAL: 1.0 (retrieved all needed info)
PRODUCTION TARGET: ≥ 0.85

WHEN RECALL IS LOW:
- Need to retrieve MORE chunks (increase k)
- Need better retrieval strategy
- Need to expand the query (multi-query)
```

---

## 7. Metric 5: Noise Sensitivity

### What It Measures

```
NOISE SENSITIVITY = How easily does the LLM get distracted by irrelevant info?

Adds intentional noise to context, sees how much it degrades the answer.

Lower score = better (system is robust to noise)
```

### How It Works

```
SCENARIO:
- question: "What is our refund policy?"
- relevant context: "Refunds within 30 days"

TEST 1: Only relevant context
- Answer: "Refunds within 30 days" ✅

TEST 2: Add irrelevant noise context
- Context: ["Refunds within 30 days",
            "Office hours are 9-5",
            "Dress code is casual"]
- Answer: "Refunds within 30 days with business casual dress code" ❌

The noise (dress code) confused the LLM!
Noise sensitivity captures how often this happens.

GOAL: 0.0 (no sensitivity, robust to noise)
PRODUCTION TARGET: ≤ 0.2

FIXING HIGH NOISE SENSITIVITY:
- Use compression to remove noise (Module 6)
- Better prompt engineering ("use ONLY relevant info")
- Better retrieval to reduce noise
- Re-ranking to prioritize relevant docs
```

---

## 8. Summary: All Five Metrics

```
┌─────────────────────┬──────────────┬─────────────┬────────────┐
│ Metric              │ What         │ Range       │ Target     │
├─────────────────────┼──────────────┼─────────────┼────────────┤
│ Faithfulness        │ Hallucination│ 0-1 (high)  │ ≥ 0.85     │
│ Answer Relevance    │ On-topic     │ 0-1 (high)  │ ≥ 0.80     │
│ Context Precision   │ Retrieval    │ 0-1 (high)  │ ≥ 0.75     │
│ Context Recall      │ Completeness │ 0-1 (high)  │ ≥ 0.85     │
│ Noise Sensitivity   │ Robustness   │ 0-1 (LOW)   │ ≤ 0.20     │
└─────────────────────┴──────────────┴─────────────┴────────────┘

THE DIAGNOSTIC CHEAT SHEET:
- Low faithfulness?     → Generation problem (LLM hallucinating)
- Low answer relevance? → Generation problem (off-topic)
- Low context precision?→ Retrieval problem (junk results)
- Low context recall?   → Retrieval problem (missing info)
- High noise sensitivity?→ Both (need compression + prompt fix)
```

---

# PART 3: RAGAS Framework

---

## 9. What is RAGAS?

### Simple Definition

```
RAGAS = "Retrieval Augmented Generation Assessment"

A Python framework that automates RAG evaluation by:
1. Implementing all the metrics we discussed
2. Using LLMs as judges (LLM-as-a-judge pattern)
3. Providing a standard interface for evaluation
4. Generating synthetic test datasets

GitHub: https://github.com/explodinggradients/ragas
Stars:  10,000+ (most popular RAG eval framework)
```

### Why RAGAS?

```
WITHOUT RAGAS:
❌ Hand-code each metric from scratch
❌ Reinvent the wheel for every project
❌ Inconsistent metric implementations
❌ No standard for comparison

WITH RAGAS:
✅ Standardized metric implementations
✅ Active research-backed development
✅ Easy integration with LangChain
✅ Can compare to industry benchmarks
✅ Continuous improvements from community

It's the de facto standard for RAG evaluation.
```

### The LLM-as-a-Judge Pattern

```
HOW RAGAS WORKS UNDER THE HOOD:

RAGAS uses LLMs to evaluate quality:

For faithfulness:
  "Extract all claims from this answer."
  "For each claim, is it supported by the context?"
  → LLM judges, RAGAS calculates score

For answer relevance:
  "Generate questions this answer could be answering."
  "Compare similarity to original question."
  → LLM generates, embeddings compare

LIMITATION: Quality depends on the judge LLM
- Use GPT-4 for highest quality evaluation
- Use GPT-3.5 for cost-effective evaluation
- Use local models for privacy
```

---

## 10. RAGAS Setup

```python
# Install RAGAS
# pip install ragas
# pip install datasets  # For HuggingFace datasets format

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    noise_sensitivity_relevant,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS needs:
# 1. An LLM (the judge)
# 2. An embedding model (for some metrics)
# 3. A dataset to evaluate

# Configure the judge LLM
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)
```

---

# PART 4: Creating Evaluation Datasets

---

## 11. The Dataset Format

### What RAGAS Needs

```python
# RAGAS expects a Dataset with these columns:

dataset_format = {
    "question":     "What is the refund policy?",    # required
    "answer":       "Customers can return...",       # required
    "contexts":     ["Returns accepted...", ...],    # required (list of strings)
    "ground_truth": "Refunds within 30 days",        # required for some metrics
}

# For multiple test cases:
dataset = {
    "question":     ["Q1?", "Q2?", "Q3?"],
    "answer":       ["A1", "A2", "A3"],
    "contexts":     [["c1a", "c1b"], ["c2a"], ["c3a", "c3b", "c3c"]],
    "ground_truth": ["GT1", "GT2", "GT3"],
}
```

---

## 12. Manual Dataset Creation

```python
"""
APPROACH 1: Manually create evaluation dataset
Best for: small, high-quality datasets with known correct answers
"""

from datasets import Dataset

# Define your test cases
test_data = {
    "question": [
        "What is the vacation policy?",
        "How does the refund process work?",
        "What are the office hours?",
        "Can I work remotely?",
        "What is the maternity leave policy?",
    ],
    "answer": [
        "Employees get 15 vacation days per year.",
        "Refunds are processed within 7-10 business days.",
        "Office hours are 9 AM to 5 PM EST.",
        "Remote work is allowed 3 days per week.",
        "Maternity leave is 12 weeks paid.",
    ],
    "contexts": [
        ["Employees are entitled to 15 vacation days annually."],
        ["Refund processing takes 7-10 business days after approval."],
        ["Office hours: Monday-Friday, 9 AM to 5 PM EST."],
        ["Remote work policy allows 3 days per week from home."],
        ["Primary caregivers receive 12 weeks of paid maternity leave."],
    ],
    "ground_truth": [
        "Employees receive 15 days of vacation per year.",
        "Refunds processed in 7-10 business days.",
        "Business hours are 9 AM - 5 PM EST, Monday through Friday.",
        "Up to 3 days of remote work per week is allowed.",
        "12 weeks of paid maternity leave for primary caregivers.",
    ],
}

# Convert to Dataset
eval_dataset = Dataset.from_dict(test_data)
print(eval_dataset)

# Output:
# Dataset({
#     features: ['question', 'answer', 'contexts', 'ground_truth'],
#     num_rows: 5
# })
```

---

## 13. Generating Dataset from Your RAG Pipeline

```python
"""
APPROACH 2: Generate evaluation data using YOUR RAG pipeline
This evaluates the actual system you've built.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset

# ─────────────────────────────────────────────────────
# Setup RAG pipeline
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

docs = [
    Document(page_content="Employees get 15 vacation days per year."),
    Document(page_content="Sick leave is 10 days annually with doctor's note."),
    Document(page_content="Maternity leave is 12 weeks paid."),
    Document(page_content="Remote work allowed 3 days per week."),
    Document(page_content="Office hours: 9 AM to 5 PM EST."),
]

vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="eval_demo"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context provided.

Context: {context}
Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─────────────────────────────────────────────────────
# Define test questions and expected answers
# ─────────────────────────────────────────────────────
test_questions = [
    "How many vacation days do employees get?",
    "What is the sick leave policy?",
    "How long is maternity leave?",
    "Can employees work from home?",
    "What are the office hours?",
]

ground_truths = [
    "Employees get 15 vacation days per year.",
    "Sick leave is 10 days annually with a doctor's note.",
    "Maternity leave is 12 weeks paid.",
    "Yes, remote work is allowed 3 days per week.",
    "Office hours are 9 AM to 5 PM EST.",
]

# ─────────────────────────────────────────────────────
# Generate dataset by running the RAG pipeline
# ─────────────────────────────────────────────────────
def generate_eval_dataset(questions, ground_truths, retriever, rag_chain):
    """Run RAG pipeline on test questions and collect data for evaluation."""

    answers = []
    contexts = []

    for question in questions:
        # Get answer from RAG chain
        answer = rag_chain.invoke(question)
        answers.append(answer)

        # Get contexts that were retrieved
        retrieved_docs = retriever.invoke(question)
        contexts.append([doc.page_content for doc in retrieved_docs])

        print(f"✓ Processed: {question[:50]}...")

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

# Generate the evaluation dataset
print("Generating evaluation dataset by running RAG pipeline...")
eval_dataset = generate_eval_dataset(
    test_questions,
    ground_truths,
    retriever,
    rag_chain,
)

print(f"\n✅ Dataset created with {len(eval_dataset)} test cases")
print(f"\nFirst example:")
print(f"  Q: {eval_dataset[0]['question']}")
print(f"  A: {eval_dataset[0]['answer']}")
print(f"  Contexts: {len(eval_dataset[0]['contexts'])} chunks")
print(f"  GT: {eval_dataset[0]['ground_truth']}")
```

---

## 14. Auto-Generating Test Questions with LLM

```python
"""
APPROACH 3: Use LLM to auto-generate diverse test questions
Best for: building large test sets quickly
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

question_gen_prompt = ChatPromptTemplate.from_template("""
Given the following document, generate {num_questions} diverse test questions
and their ground truth answers.

Generate different types:
- Direct factual questions
- Reasoning questions
- Comparison questions
- Hypothetical questions

Return as JSON array with objects containing "question" and "answer" keys.

Document:
{document}

JSON output:""")

def generate_test_questions(document: str, num_questions: int = 5):
    """Use LLM to generate test Q&A pairs from a document."""

    chain = question_gen_prompt | llm | JsonOutputParser()

    qa_pairs = chain.invoke({
        "document":      document,
        "num_questions": num_questions,
    })

    return qa_pairs

# Example usage
sample_doc = """
TechCorp was founded in 2015 by Jane Smith. The company is headquartered
in San Francisco, California. Our flagship product is QuantumSync,
launched in 2020. In Q3 2024, we reported revenue of $45 million.
We employ 500 people across 5 global offices. Our refund policy allows
full refunds within 30 days of purchase.
"""

qa_pairs = generate_test_questions(sample_doc, num_questions=5)

# Print generated questions
for i, qa in enumerate(qa_pairs):
    print(f"\n{i+1}. Q: {qa['question']}")
    print(f"   A: {qa['answer']}")
```

---

# PART 5: Hands-On Labs

---

## 15. Lab 1: Evaluate Each Metric Individually

```python
"""
LAB 1: Evaluate each RAGAS metric individually
This helps you understand what each metric measures.
"""

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    NoiseSensitivity,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

# ─────────────────────────────────────────────────────
# Setup evaluator
# ─────────────────────────────────────────────────────
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

# ─────────────────────────────────────────────────────
# Prepare test dataset (using same examples as before)
# ─────────────────────────────────────────────────────
test_data = {
    "question": [
        "How many vacation days do employees get?",
        "What is the sick leave policy?",
        "How long is maternity leave?",
    ],
    "answer": [
        "Employees get 15 vacation days per year.",
        "Sick leave allows 10 days per year. Employees also get free coffee.",
        "Maternity leave is 6 months paid.",  # Intentionally wrong!
    ],
    "contexts": [
        ["Employees are entitled to 15 vacation days annually."],
        ["Sick leave is 10 days per year with a doctor's note required."],
        ["Maternity leave is 12 weeks paid for primary caregivers."],
    ],
    "ground_truth": [
        "Employees receive 15 days of vacation per year.",
        "10 days of sick leave per year with doctor's note.",
        "12 weeks paid maternity leave.",
    ],
}

eval_dataset = Dataset.from_dict(test_data)

# ─────────────────────────────────────────────────────
# Test 1: Faithfulness (catches hallucination)
# ─────────────────────────────────────────────────────
print("═" * 70)
print("METRIC 1: FAITHFULNESS")
print("Detects hallucination (claims not in context)")
print("═" * 70)

result = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness()],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

print(f"\nFaithfulness scores per question:")
df = result.to_pandas()
for i, row in df.iterrows():
    print(f"\n  Q{i+1}: {row['question'][:60]}...")
    print(f"  Score: {row['faithfulness']:.3f}")
    if row['faithfulness'] < 0.85:
        print(f"  ⚠️  Low faithfulness detected (possible hallucination)")

print(f"\nOverall faithfulness: {df['faithfulness'].mean():.3f}")

# Expected:
# Q2 will score lower (claim about "free coffee" not in context)
# Q3 will score lower (says 6 months but context says 12 weeks)

# ─────────────────────────────────────────────────────
# Test 2: Answer Relevance
# ─────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("METRIC 2: ANSWER RELEVANCE")
print("Detects off-topic answers")
print("═" * 70)

result = evaluate(
    dataset=eval_dataset,
    metrics=[ResponseRelevancy()],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

df = result.to_pandas()
print(f"\nAnswer relevance scores:")
for i, row in df.iterrows():
    print(f"  Q{i+1}: {row['answer_relevancy']:.3f}")

# ─────────────────────────────────────────────────────
# Test 3: Context Precision (requires ground_truth)
# ─────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("METRIC 3: CONTEXT PRECISION")
print("Measures if retrieved contexts are relevant AND well-ranked")
print("═" * 70)

result = evaluate(
    dataset=eval_dataset,
    metrics=[LLMContextPrecisionWithReference()],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

df = result.to_pandas()
print(f"\nContext precision scores:")
for i, row in df.iterrows():
    print(f"  Q{i+1}: {row['llm_context_precision_with_reference']:.3f}")

# ─────────────────────────────────────────────────────
# Test 4: Context Recall
# ─────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("METRIC 4: CONTEXT RECALL")
print("Measures if we retrieved ALL info needed")
print("═" * 70)

result = evaluate(
    dataset=eval_dataset,
    metrics=[LLMContextRecall()],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

df = result.to_pandas()
print(f"\nContext recall scores:")
for i, row in df.iterrows():
    print(f"  Q{i+1}: {row['context_recall']:.3f}")

# ─────────────────────────────────────────────────────
# Test 5: Noise Sensitivity
# ─────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("METRIC 5: NOISE SENSITIVITY")
print("Measures robustness to irrelevant context")
print("LOWER is better (less affected by noise)")
print("═" * 70)

# Add noisy context to test
noisy_data = test_data.copy()
noisy_data["contexts"] = [
    contexts + ["Office cafeteria serves sushi on Wednesdays"]
    for contexts in noisy_data["contexts"]
]

noisy_dataset = Dataset.from_dict(noisy_data)

result = evaluate(
    dataset=noisy_dataset,
    metrics=[NoiseSensitivity()],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

df = result.to_pandas()
print(f"\nNoise sensitivity scores (lower is better):")
for i, row in df.iterrows():
    print(f"  Q{i+1}: {row['noise_sensitivity_relevant']:.3f}")
```

---

## 16. Lab 2: Evaluate Complete RAG Pipeline

```python
"""
LAB 2: Complete RAG Pipeline Evaluation
End-to-end evaluation of all metrics together.
"""

import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset

# ─────────────────────────────────────────────────────
# Build the RAG pipeline to evaluate
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Knowledge base
company_docs = [
    Document(page_content="TechCorp was founded in 2015 by Jane Smith in San Francisco."),
    Document(page_content="Our flagship product QuantumSync was launched in 2020."),
    Document(page_content="Q1 2024 revenue: $30M, Q2: $38M, Q3: $45M, Q4: $52M."),
    Document(page_content="We employ 500 people across 5 global offices."),
    Document(page_content="Refund policy: full refunds within 30 days of purchase."),
    Document(page_content="Customer support is available 24/7 via email and chat."),
    Document(page_content="QuantumSync Pro is priced at $99/month with annual discount."),
    Document(page_content="Our customer base grew 40% year-over-year in 2024."),
]

vectorstore = Chroma.from_documents(
    company_docs, embeddings, collection_name="full_eval"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based on the context.
If the context doesn't contain the answer, say "I don't have that information."

Context: {context}
Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─────────────────────────────────────────────────────
# Build evaluation dataset
# ─────────────────────────────────────────────────────
test_questions = [
    "Who founded TechCorp and when?",
    "Where is TechCorp headquartered?",
    "What was the total 2024 revenue?",
    "How many employees does TechCorp have?",
    "What is the refund policy?",
    "How much does QuantumSync Pro cost?",
    "When was QuantumSync launched?",
    "How can I contact customer support?",
]

ground_truths = [
    "TechCorp was founded by Jane Smith in 2015.",
    "TechCorp is headquartered in San Francisco.",
    "Total 2024 revenue was $165 million (sum of all quarters).",
    "TechCorp has 500 employees across 5 offices.",
    "Full refunds are available within 30 days of purchase.",
    "QuantumSync Pro is $99 per month with annual discount available.",
    "QuantumSync was launched in 2020.",
    "Customer support is available 24/7 via email and chat.",
]

# Generate evaluation data
print("🔄 Running RAG pipeline on test questions...")
answers = []
contexts = []

for q in test_questions:
    answer = rag_chain.invoke(q)
    retrieved = retriever.invoke(q)
    answers.append(answer)
    contexts.append([doc.page_content for doc in retrieved])

eval_dataset = Dataset.from_dict({
    "question":     test_questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

print(f"✅ Generated evaluation data for {len(eval_dataset)} questions")

# ─────────────────────────────────────────────────────
# Run full evaluation with all metrics
# ─────────────────────────────────────────────────────
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

print("\n📊 Running RAGAS evaluation with all metrics...")
print("(This may take a few minutes due to LLM calls)")

result = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

# ─────────────────────────────────────────────────────
# Analyze results
# ─────────────────────────────────────────────────────
df = result.to_pandas()

print("\n" + "═" * 70)
print("OVERALL RAG SYSTEM SCORES")
print("═" * 70)

metrics_summary = {
    "Faithfulness":          df['faithfulness'].mean(),
    "Answer Relevance":      df['answer_relevancy'].mean(),
    "Context Precision":     df['llm_context_precision_with_reference'].mean(),
    "Context Recall":        df['context_recall'].mean(),
}

for metric, score in metrics_summary.items():
    bar_length = int(score * 30)
    bar = "█" * bar_length + "░" * (30 - bar_length)
    print(f"\n{metric}:")
    print(f"  [{bar}] {score:.3f}")

    # Diagnose issues
    if metric == "Faithfulness" and score < 0.85:
        print(f"  ⚠️  Below target (0.85) - possible hallucination!")
    elif metric == "Answer Relevance" and score < 0.80:
        print(f"  ⚠️  Below target (0.80) - answers may be off-topic")
    elif metric == "Context Precision" and score < 0.75:
        print(f"  ⚠️  Below target (0.75) - retrieval has noise")
    elif metric == "Context Recall" and score < 0.85:
        print(f"  ⚠️  Below target (0.85) - missing important info")
    else:
        print(f"  ✅ Meets target")

# ─────────────────────────────────────────────────────
# Per-question analysis (find problem cases)
# ─────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("PER-QUESTION ANALYSIS")
print("═" * 70)

for i, row in df.iterrows():
    print(f"\nQ{i+1}: {row['question']}")
    print(f"   Answer: {row['answer'][:100]}...")
    print(f"   Faith: {row['faithfulness']:.2f} | "
          f"Rel: {row['answer_relevancy']:.2f} | "
          f"Prec: {row['llm_context_precision_with_reference']:.2f} | "
          f"Recall: {row['context_recall']:.2f}")

    # Flag problem cases
    issues = []
    if row['faithfulness'] < 0.85:
        issues.append("⚠️  Possible hallucination")
    if row['answer_relevancy'] < 0.80:
        issues.append("⚠️  Off-topic")
    if row['llm_context_precision_with_reference'] < 0.75:
        issues.append("⚠️  Bad retrieval order")
    if row['context_recall'] < 0.85:
        issues.append("⚠️  Missing info")

    if issues:
        for issue in issues:
            print(f"   {issue}")

# ─────────────────────────────────────────────────────
# Save results to CSV for further analysis
# ─────────────────────────────────────────────────────
df.to_csv("rag_evaluation_results.csv", index=False)
print(f"\n💾 Results saved to rag_evaluation_results.csv")
```

---

## 17. Lab 3: A/B Testing Two RAG Systems

```python
"""
LAB 3: Compare two RAG systems using RAGAS
This is how you justify improvements!
"""

# System A: Basic RAG with k=3
# System B: Hybrid RAG with k=5 + re-ranking

from ragas import evaluate
from ragas.metrics import (
    Faithfulness, ResponseRelevancy,
    LLMContextPrecisionWithReference, LLMContextRecall
)

# Same test questions, different retrieval strategies
test_questions = [...]  # Your test questions
ground_truths = [...]   # Your ground truths

# ─────────────────────────────────────────────────────
# Generate data from System A (baseline)
# ─────────────────────────────────────────────────────
system_a_data = generate_eval_dataset(
    test_questions, ground_truths,
    retriever_a, rag_chain_a,
)

# ─────────────────────────────────────────────────────
# Generate data from System B (improved)
# ─────────────────────────────────────────────────────
system_b_data = generate_eval_dataset(
    test_questions, ground_truths,
    retriever_b, rag_chain_b,
)

# ─────────────────────────────────────────────────────
# Evaluate both
# ─────────────────────────────────────────────────────
metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]

result_a = evaluate(system_a_data, metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)
result_b = evaluate(system_b_data, metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)

# ─────────────────────────────────────────────────────
# Compare results
# ─────────────────────────────────────────────────────
df_a = result_a.to_pandas()
df_b = result_b.to_pandas()

print("\n" + "═" * 70)
print("A/B TEST RESULTS")
print("═" * 70)

metrics_to_compare = [
    'faithfulness',
    'answer_relevancy',
    'llm_context_precision_with_reference',
    'context_recall',
]

print(f"\n{'Metric':<35} {'System A':<12} {'System B':<12} {'Δ':<10}")
print("-" * 70)

for metric in metrics_to_compare:
    score_a = df_a[metric].mean()
    score_b = df_b[metric].mean()
    delta = score_b - score_a
    arrow = "📈" if delta > 0 else "📉"

    print(f"{metric:<35} {score_a:.3f}       {score_b:.3f}       "
          f"{arrow} {delta:+.3f}")

# Statistical significance test (simple version)
from scipy import stats

print("\n" + "═" * 70)
print("STATISTICAL SIGNIFICANCE (paired t-test)")
print("═" * 70)

for metric in metrics_to_compare:
    t_stat, p_value = stats.ttest_rel(df_b[metric], df_a[metric])
    significant = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
    print(f"\n{metric}:")
    print(f"  p-value: {p_value:.4f} {significant}")
```

---

# PART 6: Failure Modes and Interview Prep

---

## 18. Failure Modes 🚨

```
FAILURE 1: Bad Judge LLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Using weak model (GPT-3.5) for evaluation
Result:    Unreliable scores, especially for faithfulness
Fix:       Use GPT-4 or Claude for judging
           At least for production evaluation runs

FAILURE 2: Test Set Doesn't Match Production
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Manual test questions don't reflect real user queries
Result:    Good eval scores, bad real-world performance
Fix:       Collect real production queries
           Build test set from actual user logs

FAILURE 3: Ground Truth Quality
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   LLM-generated ground truths may be wrong
Result:    Evaluating against incorrect "truth"
Fix:       Have human experts review ground truths
           Especially critical for context_recall

FAILURE 4: Too Small Test Set
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Only 5-10 test questions
Result:    Scores have high variance, can't detect real changes
Fix:       Minimum 50-100 test cases for reliable metrics
           More diversity = more reliable insights

FAILURE 5: Evaluating Once and Forgetting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Evaluate during development, never in production
Result:    Quality degrades silently over time
Fix:       Continuous evaluation in CI/CD
           Track metrics over time
           Alert on degradation

FAILURE 6: Cost Explosion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Each evaluation = many LLM calls
           100 questions × 4 metrics × 3 LLM calls each = 1,200 calls!
Result:    High evaluation costs
Fix:       Use cheaper judge models for development
           Cache evaluations of unchanged components
           Run full eval only before major releases

FAILURE 7: Ignoring Per-Question Failures
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Only looking at average metrics
Result:    Some questions catastrophically fail, but average looks fine
Fix:       Always analyze per-question results
           Flag and investigate queries with low scores
```

---

## 19. Interview Questions and Answers

### Q1: "How do you evaluate a RAG system?"

**Strong Answer:**
> "RAG evaluation needs to measure both retrieval and generation quality separately because they fail in different ways. I use the RAGAS framework with five core metrics: Faithfulness measures hallucination by checking if every claim in the answer is supported by retrieved context. Answer Relevance checks if the answer actually addresses the question. Context Precision evaluates whether retrieved chunks are relevant and well-ranked. Context Recall measures if we retrieved ALL the info needed. Noise Sensitivity tests robustness to irrelevant context. Production targets I aim for: faithfulness ≥ 0.85, relevance ≥ 0.80, precision ≥ 0.75, recall ≥ 0.85, noise sensitivity ≤ 0.20. I build evaluation datasets from real production queries when possible, use GPT-4 as the judge LLM for reliability, and run evaluations in CI/CD to catch regressions."

---

### Q2: "How does the Faithfulness metric work in RAGAS?"

**Strong Answer:**
> "Faithfulness is the most critical metric for catching hallucination. It works in three steps: First, the judge LLM breaks the generated answer into individual atomic claims — for example, 'Customers can return items within 30 days and the CEO personally approves all refunds' becomes two claims. Second, for each claim, the LLM checks if it's supported by the retrieved context. Third, the score is calculated as supported_claims / total_claims. A score of 0.67 means one-third of the answer is hallucinated. I always set 0.85 as the minimum production threshold because anything lower means significant unreliability. Faithfulness is what makes RAG safe for production — without it, you can't trust your system isn't making things up."

---

### Q3: "What's the difference between Context Precision and Context Recall?"

**Strong Answer:**
> "They measure opposite aspects of retrieval quality. Context Precision answers 'are the docs we retrieved relevant?' — it penalizes returning irrelevant chunks and rewards putting the most relevant chunks first. Low precision means your retriever returns junk. Context Recall answers 'did we retrieve all the info needed?' — it's calculated by checking each claim in the ground truth against retrieved contexts. Low recall means your retriever missed important information. These two metrics often trade off: lowering the relevance threshold improves recall but hurts precision, while raising it does the opposite. In production, I aim for precision ≥ 0.75 and recall ≥ 0.85. If recall is low, I increase k or use multi-query retrieval. If precision is low, I add re-ranking or use hybrid search."

---

### Q4: "How do you build a good evaluation dataset for RAG?"

**Strong Answer:**
> "There are three approaches with different tradeoffs. First, manual creation by domain experts — highest quality but doesn't scale, good for small critical test sets of 20-50 questions. Second, generating from your RAG pipeline using real or sample documents — captures realistic patterns but ground truth needs human review. Third, LLM-generated questions from your documents — fastest to scale but quality varies. My approach for production: start with 50 manually-crafted questions covering different query types — factoid, analytical, multi-hop. Add 100+ questions sampled from real user logs to capture distribution. Have domain experts verify all ground truth answers. Keep a separate 'golden set' of 20 critical questions that should ALWAYS work. Run full evaluation on every release, golden set evaluation on every PR. This catches both gradual degradation and breaking changes."

---

## 20. Common Mistakes ⚠️

```
MISTAKE 1: "Just look at the answers, they seem fine"
TRUTH:     Eyeballing doesn't scale. You can't compare versions
           without numbers. Always quantify.

MISTAKE 2: "One metric is enough"
TRUTH:     Each metric catches different failures.
           You need ALL of them for complete picture.

MISTAKE 3: "Test on synthetic questions only"
TRUTH:     Real users ask weird, ambiguous, edge-case questions.
           Real query distribution is essential.

MISTAKE 4: "Run evaluation once during development"
TRUTH:     RAG quality drifts as docs update and LLMs change.
           Continuous evaluation is essential.

MISTAKE 5: "High overall scores = good system"
TRUTH:     Average hides catastrophic failures on specific queries.
           Always examine per-question scores.

MISTAKE 6: "GPT-3.5 is fine for evaluation"
TRUTH:     Weaker judge models give unreliable scores.
           Use GPT-4 / Claude for production evaluation.

MISTAKE 7: "Skip ground truth, it takes too long"
TRUTH:     Without ground truth, you can't measure recall.
           This is the most important metric for retrieval quality.

MISTAKE 8: "Only evaluate the final answer"
TRUTH:     You need retrieval metrics to diagnose problems.
           Without them, you can't tell if retrieval or generation is broken.
```

---

## 21. Engineer Mindset 🔧

```
WHAT A REAL RAG ENGINEER THINKS ABOUT:

1. "Measure before optimizing"
   → Establish baseline with current system
   → Set targets based on use case requirements
   → Optimize specific weak metrics, not random changes

2. "Evaluation is infrastructure"
   → Build it into your CI/CD pipeline
   → Run on every PR (golden set at minimum)
   → Track metrics in dashboards over time

3. "Cost-aware evaluation"
   → Full eval: expensive but thorough (release time)
   → Golden set: cheap and fast (every PR)
   → Smoke tests: 5 queries (every commit)
   → Match evaluation depth to release stage

4. "Per-metric diagnosis"
   → Low faithfulness → fix prompts, lower temperature
   → Low relevance → better prompts, smaller k
   → Low precision → add re-ranking or hybrid
   → Low recall → increase k or multi-query
   → Match action to root cause

5. "Continuous improvement loop"
   → Production monitoring → identify failure cases
   → Add failures to test set
   → Iterate retrieval/generation
   → Measure improvement
   → Deploy if metrics improve

6. "Trust but verify automated evaluation"
   → Spot-check LLM judges with manual review
   → Calibrate judge prompts for your domain
   → Watch for evaluation drift over time

7. "Beyond RAGAS for production"
   → Add custom metrics for your domain
   → Latency budgets
   → Cost per query
   → User feedback signals
```

---

## 22. Exercises 📝

### Beginner

```
1. Setup RAGAS and run your first evaluation:
   - Install ragas and datasets
   - Create dataset with 5 test cases
   - Evaluate faithfulness only
   - Interpret the results

2. Test faithfulness with intentional hallucination:
   - Write answers that include facts not in context
   - Verify faithfulness score drops
   - Confirm RAGAS catches your hallucinations

3. Compare answer quality:
   - For one question, generate 3 different answers
   - Score each with all 5 metrics
   - Identify which has best balance

4. Build a 10-question test set:
   - For your knowledge base
   - Manually write ground truths
   - Run full evaluation
   - Identify weakest metric
```

### Intermediate

```
5. A/B test two RAG configurations:
   - System A: k=3, no re-ranking
   - System B: k=5, with re-ranking
   - Use same 50 test questions
   - Run statistical significance test

6. Build automatic test set generation:
   - Use LLM to generate 50 questions from documents
   - Have it generate corresponding ground truths
   - Manually verify 10% as sanity check
   - Use for evaluation

7. Diagnose a real RAG system:
   - Run RAGAS on existing system
   - Identify weakest metric
   - Make ONE change to address it
   - Re-evaluate to verify improvement

8. Build cost-aware evaluation pipeline:
   - Quick eval: 5 questions, cheap LLM (every commit)
   - Standard eval: 30 questions, GPT-4 (every PR)
   - Full eval: 100 questions, GPT-4 (every release)
```

### Advanced

```
9. Build CI/CD evaluation pipeline:
   - Triggered on PR
   - Runs golden set evaluation
   - Comments PR with results
   - Blocks merge if metrics regress

10. Production monitoring system:
    - Sample 1% of production queries
    - Run evaluation automatically
    - Alert on metric degradation
    - Dashboard showing trends

11. Custom metric development:
    - Identify domain-specific quality dimension
    - Build evaluator using LLM-as-a-judge
    - Validate against human ratings
    - Integrate into RAGAS evaluation

12. End-to-end evaluation framework:
    - Multiple test sets (regression, edge cases, real)
    - Multiple metrics (RAGAS + custom)
    - Cost tracking
    - Latency tracking
    - Per-document-type analysis
    - Auto-generated quality report
```

---

## 23. Resources 📚

```
RAGAS:
🔗 RAGAS Official Documentation
   https://docs.ragas.io/

🔗 RAGAS GitHub
   https://github.com/explodinggradients/ragas

🔗 RAGAS Tutorials
   https://docs.ragas.io/en/stable/getstarted/

PAPERS:
📄 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
   https://arxiv.org/abs/2309.15217

📄 "ARES: An Automated Evaluation Framework for RAG Systems"
   https://arxiv.org/abs/2311.09476

📄 "Evaluating Retrieval Quality in RAG"
   https://research.trychroma.com/evaluating-retrieval

OTHER EVALUATION TOOLS:
🔗 TruLens (Alternative RAG evaluation)
   https://www.trulens.org/

🔗 LangSmith Evaluation
   https://docs.smith.langchain.com/evaluation

🔗 DeepEval (Pytest for LLMs)
   https://github.com/confident-ai/deepeval

🔗 Phoenix by Arize
   https://docs.arize.com/phoenix

DATASETS:
🔗 BEIR Benchmark (Retrieval)
   https://github.com/beir-cellar/beir

🔗 MTEB Benchmark (Embeddings)
   https://huggingface.co/spaces/mteb/leaderboard

TUTORIALS:
🔗 RAGAS with LangChain
   https://docs.ragas.io/en/stable/howtos/integrations/langchain/

🔗 Production RAG Evaluation
   https://blog.langchain.com/evaluating-rag-pipelines/
```

---

## Module 9 Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 9 MENTAL MODEL                          │
│                                                                 │
│  WHY EVALUATE: "Looks right" doesn't scale.                    │
│  Without metrics, you fly blind. Without metrics,              │
│  you can't compare versions or detect degradation.             │
│                                                                 │
│  THE FIVE METRICS:                                              │
│                                                                 │
│  GENERATION METRICS:                                            │
│    Faithfulness     → Hallucination check (≥ 0.85)             │
│    Answer Relevance → On-topic check    (≥ 0.80)               │
│                                                                 │
│  RETRIEVAL METRICS:                                             │
│    Context Precision → Quality of retrieval (≥ 0.75)           │
│    Context Recall    → Completeness        (≥ 0.85)            │
│    Noise Sensitivity → Robustness         (≤ 0.20)             │
│                                                                 │
│  DIAGNOSTIC POWER:                                              │
│    Low faithfulness → Generation problem (hallucination)       │
│    Low relevance    → Generation problem (off-topic)           │
│    Low precision    → Retrieval problem (noise)                │
│    Low recall       → Retrieval problem (missing info)         │
│                                                                 │
│  RAGAS FRAMEWORK:                                               │
│    Industry-standard RAG evaluation library                    │
│    Uses LLMs as judges                                          │
│    Standardized metric implementations                         │
│                                                                 │
│  EVALUATION DATA:                                               │
│    Need: question, answer, contexts, ground_truth              │
│    Manual: small, high-quality                                 │
│    Pipeline-generated: realistic, scalable                     │
│    LLM-generated: fast, but verify quality                     │
│                                                                 │
│  GOLDEN RULES:                                                  │
│  1. Use GPT-4 or Claude as judge LLM                          │
│  2. Build test sets from REAL user queries                    │
│  3. Evaluate ALL metrics together, not just one               │
│  4. Look at per-question scores, not just averages            │
│  5. Run evaluation in CI/CD continuously                      │
│  6. Match action to specific metric failure                   │
└─────────────────────────────────────────────────────────────────┘
```