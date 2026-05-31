# Module 8: Agentic RAG with LangGraph 

## Where RAG Learns to Think, Decide, and Act

---

> **Before I explain — let me ask:**
>
> *"What's the biggest limitation of all the RAG patterns we've covered so far?"*
>
> A beginner might say: *"They sometimes return wrong results."*
>
> The deeper truth: **They follow a FIXED workflow.**
>
> Every previous RAG: Query → Retrieve → Generate (always)
> Even advanced CRAG: Always grades, always considers web fallback
>
> But what if the system could **decide for itself**?
> - "This is a simple math question — I don't need to retrieve"
> - "I need to search 3 different knowledge bases"
> - "Let me retrieve, think, then retrieve again with a better query"
> - "I should verify my answer by retrieving more"
>
> **Agentic RAG = RAG with autonomous decision-making.**
>
> This is the frontier of production RAG systems.

---

# The Big Picture: Why Agentic RAG Matters

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  THE EVOLUTION:                                                   │
│                                                                   │
│  STATIC RAG (Modules 1-7):                                       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━                                       │
│  Fixed workflow. Same path for every query.                      │
│  Limited by upfront design.                                      │
│                                                                   │
│  AGENTIC RAG (Module 8):                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━                                       │
│  Dynamic workflow. Adapts to each query.                         │
│  Reasons about what to do next.                                  │
│  Uses tools strategically.                                       │
│  Can loop, branch, retry, refine.                                │
│                                                                   │
│  REAL-WORLD IMPACT:                                              │
│  - Customer support: Routes to right knowledge base              │
│  - Research assistants: Multi-step investigation                 │
│  - Code agents: Search docs, write code, test, debug             │
│  - Healthcare: Verify diagnoses with multiple sources            │
│                                                                   │
│  THIS IS WHERE MODERN AI IS HEADED.                              │
└───────────────────────────────────────────────────────────────────┘
```

---

# PART 1: Introduction to Agentic RAG

---

## 1. What is Agentic RAG?

### Simple Definition

**Agentic RAG = RAG + Agent capabilities**

An agent is an AI system that can:
1. **Reason** about what to do
2. **Choose** tools to use
3. **Execute** multi-step plans
4. **Adapt** based on intermediate results

When you combine these with RAG, the system can decide:
- *Should I retrieve at all?*
- *Which knowledge base should I search?*
- *Do I need to refine my query?*
- *Is this answer good enough or should I search more?*

### The Core Mental Shift

```
TRADITIONAL RAG:
━━━━━━━━━━━━━━━━━
You design the workflow:
  1. ALWAYS retrieve
  2. ALWAYS use top-k
  3. ALWAYS generate

The system is a PROGRAM that runs the same way.

AGENTIC RAG:
━━━━━━━━━━━━━
You give the LLM tools and goals:
  - "Here's a retriever tool"
  - "Here's a web search tool"
  - "Here's a calculator"
  - "Answer the user's question"

The system is an AGENT that DECIDES what to do.
```

---

## 2. Traditional RAG vs Agentic RAG

### Side-by-Side Comparison

```
┌──────────────────────┬─────────────────────┬─────────────────────┐
│ Aspect               │ Traditional RAG     │ Agentic RAG         │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Workflow             │ Fixed, linear       │ Dynamic, branching  │
│ Decision-making      │ None (hard-coded)   │ LLM-driven          │
│ Number of retrievals │ Always 1            │ 0 to many           │
│ Adaptation           │ None                │ Per-query           │
│ Tool use             │ Just retrieval      │ Multiple tools      │
│ Error correction     │ None                │ Self-correcting     │
│ Complex queries      │ Often fails         │ Handles well        │
│ Latency              │ Predictable, fast   │ Variable, often slow│
│ Cost                 │ Lower               │ Higher              │
│ Debugging            │ Easier              │ Harder              │
│ Best for             │ Simple Q&A          │ Complex tasks       │
└──────────────────────┴─────────────────────┴─────────────────────┘
```

### Concrete Example

```
QUERY: "What were our company's Q3 revenue,
        and how did it compare to industry average?"

TRADITIONAL RAG:
1. Embed the entire question
2. Retrieve top-5 chunks from knowledge base
3. Hope chunks contain both pieces of information
4. Generate answer

Problem: The question has TWO parts:
- Q3 revenue (internal data)
- Industry average (likely external data)
A single retrieval won't get both!

AGENTIC RAG:
1. LLM analyzes: "I need two pieces of information"
2. LLM calls retriever_tool("company Q3 revenue 2024")
   → Gets internal financial data
3. LLM analyzes: "Now I need industry comparison"
4. LLM calls web_search_tool("industry average Q3 2024 revenue")
   → Gets external industry data
5. LLM synthesizes both into a complete answer

This adaptive multi-step reasoning is impossible with traditional RAG.
```

---

## 3. Key Capabilities

### 1. Reasoning

```
The LLM thinks before acting:

User: "What's our refund policy for international customers?"

LLM reasoning:
"This question has two aspects:
1. General refund policy
2. International-specific rules

Let me search for both."
```

### 2. Tool Use

```
Agentic systems have a TOOLBOX:

🔍 retriever_tool: Search knowledge base
🌐 web_search:     Search the internet
🔢 calculator:     Math operations
📅 date_tool:      Current date/time
✉️  email_tool:    Send emails
📊 sql_tool:       Query databases

LLM picks the RIGHT tool for each step.
```

### 3. Multi-Step Execution

```
Step 1: Retrieve general policy
Step 2: Read it, identify international section reference
Step 3: Retrieve international section specifically
Step 4: Synthesize complete answer

Each step's output informs the next step's action.
```

### 4. Self-Correction

```
"My first retrieval got irrelevant results.
 Let me reformulate the query and try again."

The system can recognize failure and adapt.
```

---

## 4. Market Significance and Real-World Impact

```
PRODUCTION USE CASES OF AGENTIC RAG:

🏥 HEALTHCARE
   - Diagnose with multiple symptom databases
   - Verify drug interactions across sources
   - Multi-hop medical reasoning

⚖️ LEGAL
   - Research case law across multiple databases
   - Cross-reference statutes and precedents
   - Generate briefs with citations

💰 FINANCE
   - Analyze quarterly reports + market data + news
   - Compliance checking across multiple regulations
   - Investment research with multiple data sources

🛒 E-COMMERCE
   - Customer support that routes to right knowledge
   - Product recommendations with inventory check
   - Order tracking across systems

🎓 EDUCATION
   - Personalized tutoring with multiple textbooks
   - Research assistant for academic work
   - Adaptive learning paths

The TAM (Total Addressable Market) for agentic AI
is estimated in HUNDREDS OF BILLIONS by 2030.
```

---

# PART 2: RAG as a Tool for Agents

---

## 5. The Tool-Calling Pattern

### Core Concept

```
In agentic RAG, retrieval is NOT a step in a pipeline.
Retrieval is a TOOL the LLM can choose to call.

Traditional:
  pipeline: query → retriever → llm → response

Agentic:
  agent_loop:
    LLM thinks
    LLM picks tool (or none)
    Tool executes
    LLM thinks again
    ... loop until LLM is satisfied
    LLM gives final response

The LLM is the "brain" that orchestrates everything.
```

### Why This Matters

```
WITH RETRIEVAL AS A TOOL:

✅ LLM can SKIP retrieval for simple queries
   "What is 2+2?" → No retrieval needed
   "Hello!" → No retrieval needed
   Saves cost and latency

✅ LLM can call retrieval MULTIPLE TIMES
   Complex query needs multiple searches
   Each search uses learnings from previous

✅ LLM can choose AMONG MULTIPLE retrievers
   Search legal docs OR financial docs OR HR docs
   Right tool for the right question

✅ LLM can COMBINE tools
   Retrieve + Calculate + Search web + Synthesize
```

---

## 6. Creating Retriever Tools

### Method 1: Using create_retriever_tool

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────
# Setup knowledge base
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

hr_docs = [
    Document(page_content="Employees get 15 vacation days per year."),
    Document(page_content="Sick leave allows 10 days annually with doctor's note."),
    Document(page_content="Remote work is allowed 3 days per week."),
    Document(page_content="Parental leave is 12 weeks paid for primary caregiver."),
]

vectorstore = Chroma.from_documents(
    hr_docs, embeddings, collection_name="hr_kb"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ─────────────────────────────────────────────────────
# Convert retriever to a TOOL
# ─────────────────────────────────────────────────────
hr_tool = create_retriever_tool(
    retriever=retriever,
    name="search_hr_policies",        # Tool name (LLM sees this)
    description=(                      # Tool description (LLM reads this!)
        "Search the company HR knowledge base for information about "
        "employee policies including vacation, sick leave, remote work, "
        "and parental leave. Use this for any HR-related questions."
    ),
)

# The LLM uses the DESCRIPTION to decide when to use this tool!
# A good description is CRITICAL.
```

### What Makes a Good Tool Description

```
GOOD DESCRIPTIONS:
✅ Specific about what's in the knowledge base
✅ Lists example topics covered
✅ Indicates when to use the tool
✅ Indicates what kind of queries it handles best

BAD DESCRIPTION:
❌ "A search tool"
❌ "Use this to find information"
❌ Too vague, LLM won't know when to use it

EXAMPLE BAD:
hr_tool = create_retriever_tool(
    retriever=retriever,
    name="search",
    description="Search documents.",
)

EXAMPLE GOOD:
hr_tool = create_retriever_tool(
    retriever=retriever,
    name="search_hr_policies",
    description=(
        "Search the HR knowledge base. Contains information about: "
        "vacation policy (15 days/year), sick leave (10 days/year), "
        "remote work policy (3 days/week), parental leave (12 weeks). "
        "Use this for ANY question about employee benefits, time off, "
        "or workplace policies."
    ),
)

The LLM picks tools based on description quality!
```

---

## 7. Multiple Knowledge Base Tools

### Real-World Scenario: Different Departments

```python
"""
A company chatbot that needs to handle questions about
HR policies, legal matters, and technical documentation.
Each domain has its own knowledge base.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─────────────────────────────────────────────────────
# Three separate knowledge bases
# ─────────────────────────────────────────────────────

# HR Knowledge Base
hr_docs = [
    Document(page_content="Employees get 15 vacation days per year."),
    Document(page_content="Sick leave is 10 days annually."),
    Document(page_content="Remote work is allowed 3 days per week."),
]
hr_store = Chroma.from_documents(hr_docs, embeddings, collection_name="hr")
hr_tool = create_retriever_tool(
    retriever=hr_store.as_retriever(search_kwargs={"k": 3}),
    name="search_hr_policies",
    description=(
        "Search HR knowledge base for employee policies including "
        "vacation, sick leave, remote work, benefits, and workplace "
        "guidelines. Use for any HR or people-related questions."
    ),
)

# Legal Knowledge Base
legal_docs = [
    Document(page_content="All contracts must be reviewed by legal team before signing."),
    Document(page_content="Non-disclosure agreements have 5-year terms by default."),
    Document(page_content="Data privacy follows GDPR for EU customers."),
]
legal_store = Chroma.from_documents(legal_docs, embeddings, collection_name="legal")
legal_tool = create_retriever_tool(
    retriever=legal_store.as_retriever(search_kwargs={"k": 3}),
    name="search_legal_documents",
    description=(
        "Search the legal knowledge base for contracts, compliance, "
        "NDAs, data privacy regulations (GDPR, CCPA), intellectual "
        "property, and legal policies. Use for legal questions."
    ),
)

# Technical Knowledge Base
tech_docs = [
    Document(page_content="To deploy services, use kubectl apply -f deployment.yaml"),
    Document(page_content="All API endpoints require Bearer token authentication."),
    Document(page_content="Database backups run nightly at 2 AM UTC."),
]
tech_store = Chroma.from_documents(tech_docs, embeddings, collection_name="tech")
tech_tool = create_retriever_tool(
    retriever=tech_store.as_retriever(search_kwargs={"k": 3}),
    name="search_technical_docs",
    description=(
        "Search technical documentation for deployment instructions, "
        "API references, database operations, infrastructure setup, "
        "and engineering processes. Use for technical/engineering questions."
    ),
)

# Bundle all tools
tools = [hr_tool, legal_tool, tech_tool]
```

---

## 8. When Agents Should Retrieve vs Respond Directly

### The Decision Framework

```
LLM's INTERNAL DECISION:

User query received
        │
        ▼
"Can I answer this from general knowledge?"
        │
   YES  │  NO
        │
  ┌─────┴─────┐
  ▼           ▼
Direct      Use retrieval
response    tool

EXAMPLES OF DIRECT (no retrieval):
✅ "Hello!"
✅ "What is 5 + 3?"
✅ "Translate 'hello' to Spanish"
✅ "What is the capital of France?"
✅ "Write a haiku about coding"

EXAMPLES NEEDING RETRIEVAL:
✅ "What is OUR company's refund policy?"
✅ "When was Project X launched?"
✅ "What did the CEO say in the last earnings call?"
✅ "Show me Q3 sales figures for our region"
✅ "What are our security requirements?"

THE KEY INSIGHT:
Retrieval needed when answer requires PROPRIETARY
or SPECIFIC knowledge not in LLM training data.
```

---

## 9. Hands-On: Build an Agent with Retrieval Tool

```python
"""
Build a complete agent that uses retrieval as a tool.
Demonstrates: tool selection, multi-step reasoning, adaptation.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent

# ─────────────────────────────────────────────────────
# Setup knowledge base
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

company_docs = [
    Document(page_content="Our company TechCorp was founded in 2015 by Jane Smith."),
    Document(page_content="TechCorp's headquarters is in San Francisco, California."),
    Document(page_content="Our flagship product is the QuantumSync platform launched in 2020."),
    Document(page_content="In Q3 2024, TechCorp reported revenue of $45 million."),
    Document(page_content="Our refund policy allows full refunds within 30 days."),
    Document(page_content="Customer support is available 24/7 via email and chat."),
    Document(page_content="TechCorp employs 500 people across 5 global offices."),
]

vectorstore = Chroma.from_documents(
    company_docs, embeddings, collection_name="company_kb"
)

# ─────────────────────────────────────────────────────
# Create retriever tool
# ─────────────────────────────────────────────────────
retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    name="search_company_info",
    description=(
        "Search TechCorp's internal knowledge base for information about: "
        "company history, founders, headquarters, products, revenue, "
        "policies (refund, support), and employee information. "
        "Use this for ANY questions about TechCorp specifically."
    ),
)

# ─────────────────────────────────────────────────────
# Create the agent using LangGraph
# ─────────────────────────────────────────────────────
# create_react_agent is a pre-built agent that:
# 1. Receives user query
# 2. LLM decides if it needs tools
# 3. If yes, calls appropriate tool
# 4. Sees tool output
# 5. Decides next action (more tools? final answer?)
# 6. Returns final answer

agent = create_react_agent(
    model=llm,
    tools=[retriever_tool],
)

# ─────────────────────────────────────────────────────
# Test the agent
# ─────────────────────────────────────────────────────

def run_agent(question: str):
    """Run the agent and display its reasoning."""
    print(f"\n{'═'*70}")
    print(f"USER: {question}")
    print('═' * 70)

    # Agent processes the query
    result = agent.invoke({
        "messages": [("user", question)]
    })

    # Show all messages (reasoning trace)
    print("\n🤖 AGENT REASONING TRACE:")
    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "content") and msg.content:
            print(f"\n[{msg_type}]")
            print(f"  {msg.content[:300]}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[TOOL CALL]")
                print(f"  Tool: {tc['name']}")
                print(f"  Args: {tc['args']}")

    print(f"\n📋 FINAL ANSWER: {result['messages'][-1].content}")

# Test 1: Question needing retrieval
run_agent("Who founded TechCorp and when?")

# Test 2: Question NOT needing retrieval (general knowledge)
run_agent("What is 25 multiplied by 4?")

# Test 3: Complex question needing multiple retrievals
run_agent("Tell me about TechCorp's flagship product and recent revenue.")

# Test 4: Question requiring no retrieval
run_agent("Hello, how are you today?")
```

### What You'll Observe

```
EXPECTED BEHAVIOR:

Question 1: "Who founded TechCorp?"
→ Agent calls search_company_info("TechCorp founder")
→ Gets relevant docs
→ Answers: "TechCorp was founded by Jane Smith in 2015"

Question 2: "What is 25 × 4?"
→ Agent does NOT call retriever (general knowledge)
→ Answers directly: "100"

Question 3: "TechCorp's product and revenue?"
→ Agent may call retriever once with combined query
→ OR call retriever twice with separate queries
→ Synthesizes complete answer

Question 4: "Hello"
→ Agent responds directly without any tools
→ "Hello! How can I help you?"

The agent DECIDES when retrieval is needed!
```

---

# PART 3: LangGraph Fundamentals for RAG

---

## 10. Why LangGraph?

### The Limitation of LangChain Chains

```
LANGCHAIN CHAINS:
chain = prompt | llm | output_parser

✅ Simple, linear flows
❌ No branching logic
❌ No loops
❌ No conditional execution
❌ Hard to implement: "retry if quality is low"
❌ Hard to implement: "use different tool based on input"

THIS IS WHERE LANGGRAPH COMES IN.

LANGGRAPH:
- Define your workflow as a GRAPH
- Nodes = functions that do work
- Edges = how data flows between nodes
- Conditional edges = branching logic
- Loops = retry, refine, iterate
- State = shared memory across nodes
```

---

## 11. Core LangGraph Concepts

### State

```
STATE = a dictionary that flows through the graph.
Every node can READ and UPDATE the state.

Think of it as the "scratchpad" your graph uses.

Example state:
{
    "question": "What is the refund policy?",
    "retrieved_docs": [doc1, doc2, doc3],
    "answer": None,                # filled later
    "needs_more_info": False,      # filled later
}
```

### Nodes

```
NODES = Python functions that:
- Take state as input
- Do some work (call LLM, retrieve, etc.)
- Return updated state

def retrieve_node(state):
    docs = vectorstore.similarity_search(state["question"])
    return {"retrieved_docs": docs}

def generate_node(state):
    answer = llm.invoke(...)
    return {"answer": answer}
```

### Edges

```
EDGES = connections between nodes.

NORMAL EDGES:
"After node A, always go to node B"
workflow.add_edge("retrieve", "generate")

CONDITIONAL EDGES:
"After node A, decide where to go based on state"
workflow.add_conditional_edges(
    "grade_docs",
    decide_next_step,    # function returning next node name
    {
        "good": "generate",
        "bad":  "rewrite_query"
    }
)
```

### Putting It Together

```
USER QUERY
    │
    ▼
[START]
    │
    ▼
┌─────────────┐
│  retrieve   │ ← Node
└─────────────┘
    │
    ▼ Edge
┌─────────────┐
│ grade_docs  │ ← Node
└─────────────┘
    │
    ▼ Conditional Edge
   /  \
  /    \
"good" "bad"
  │      │
  ▼      ▼
generate rewrite
  │      │
  │      └──→ back to retrieve (loop!)
  │
  ▼
[END]
```

---

## 12. State Schemas with TypedDict

```python
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ─────────────────────────────────────────────────────
# Define the State Schema
# ─────────────────────────────────────────────────────

class RAGState(TypedDict):
    """
    State that flows through the RAG graph.

    Fields:
        question:        The user's question
        documents:       Retrieved documents
        generation:      LLM's generated answer
        web_search:      Whether to do web search
        iteration_count: How many times we've looped
    """
    question: str
    documents: List[Document]
    generation: str
    web_search: bool
    iteration_count: int

# For chat-based agents with message history
class AgentState(TypedDict):
    """
    State for conversational agents.

    The Annotated[..., add_messages] is special:
    It tells LangGraph to APPEND new messages
    rather than REPLACING the message list.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    context: str
```

---

## 13. Building RAG Graphs Step-by-Step

### Complete Example: Building from Scratch

```python
"""
COMPLETE AGENTIC RAG GRAPH FROM SCRATCH
─────────────────────────────────────────
A self-deciding RAG system that:
1. Decides if retrieval is needed
2. Retrieves if needed
3. Grades retrieved documents
4. Either generates or retrieves again
5. Returns final answer
"""

from typing import TypedDict, List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# ─────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

docs = [
    Document(page_content="TechCorp was founded in 2015 by Jane Smith."),
    Document(page_content="TechCorp is headquartered in San Francisco."),
    Document(page_content="QuantumSync is our flagship product, launched 2020."),
    Document(page_content="Q3 2024 revenue was $45 million."),
    Document(page_content="Refund policy: full refunds within 30 days."),
]

vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="agent_demo"
)

# ─────────────────────────────────────────────────────
# Step 1: Define the State
# ─────────────────────────────────────────────────────
class AgenticRAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    decision: str           # "needs_retrieval" or "direct_answer"
    relevance: str          # "relevant" or "irrelevant"
    iteration: int

# ─────────────────────────────────────────────────────
# Step 2: Define Nodes
# ─────────────────────────────────────────────────────

# Node 1: Decide if retrieval is needed
decision_prompt = ChatPromptTemplate.from_template("""
Determine if the following question needs to search a company knowledge base.

Knowledge base contains: TechCorp company info (founder, products, revenue, policies)

Question: {question}

Respond with EXACTLY one word:
- "retrieve" if the question is about TechCorp specifically
- "direct" if it's a general question or greeting

Decision:""")

def decide_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """First node: decide whether to retrieve."""
    print(f"\n🤔 NODE: decide_retrieval")

    chain = decision_prompt | llm | StrOutputParser()
    decision = chain.invoke({"question": state["question"]}).strip().lower()

    decision = "needs_retrieval" if "retrieve" in decision else "direct_answer"
    print(f"   Decision: {decision}")

    return {**state, "decision": decision}

# Node 2: Retrieve documents
def retrieve(state: AgenticRAGState) -> AgenticRAGState:
    """Retrieve documents from knowledge base."""
    print(f"\n📚 NODE: retrieve")

    results = vectorstore.similarity_search(state["question"], k=3)
    print(f"   Retrieved {len(results)} documents")

    return {**state, "documents": results}

# Node 3: Grade documents
grading_prompt = ChatPromptTemplate.from_template("""
Are these documents relevant to answering the question?

Documents:
{documents}

Question: {question}

Respond with EXACTLY one word: "relevant" or "irrelevant"
""")

def grade_documents(state: AgenticRAGState) -> AgenticRAGState:
    """Check if retrieved documents are useful."""
    print(f"\n📝 NODE: grade_documents")

    docs_text = "\n".join([d.page_content for d in state["documents"]])
    chain = grading_prompt | llm | StrOutputParser()
    relevance = chain.invoke({
        "documents": docs_text,
        "question": state["question"],
    }).strip().lower()

    relevance = "relevant" if "relevant" in relevance and "irrelevant" not in relevance else "irrelevant"
    print(f"   Relevance: {relevance}")

    return {**state, "relevance": relevance}

# Node 4: Generate with retrieval context
generate_with_context_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:""")

def generate_with_context(state: AgenticRAGState) -> AgenticRAGState:
    """Generate answer using retrieved context."""
    print(f"\n🤖 NODE: generate_with_context")

    context = "\n".join([d.page_content for d in state["documents"]])
    chain = generate_with_context_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["question"],
    })

    print(f"   Generated answer (length: {len(answer)})")
    return {**state, "generation": answer}

# Node 5: Generate without retrieval
def generate_direct(state: AgenticRAGState) -> AgenticRAGState:
    """Generate answer directly (no retrieval needed)."""
    print(f"\n🤖 NODE: generate_direct")

    answer = llm.invoke(state["question"]).content
    print(f"   Generated direct answer")

    return {**state, "generation": answer}

# Node 6: Handle no relevant docs
def handle_no_relevant_docs(state: AgenticRAGState) -> AgenticRAGState:
    """When retrieval didn't find relevant info."""
    print(f"\n⚠️  NODE: handle_no_relevant_docs")

    return {
        **state,
        "generation": (
            "I couldn't find relevant information in the knowledge base "
            "to answer your question. Please try rephrasing or asking "
            "about TechCorp-specific topics."
        )
    }

# ─────────────────────────────────────────────────────
# Step 3: Define Routing Functions
# ─────────────────────────────────────────────────────

def route_after_decision(state: AgenticRAGState) -> str:
    """Route based on retrieval decision."""
    if state["decision"] == "needs_retrieval":
        return "retrieve"
    else:
        return "generate_direct"

def route_after_grading(state: AgenticRAGState) -> str:
    """Route based on document relevance."""
    if state["relevance"] == "relevant":
        return "generate_with_context"
    else:
        return "handle_no_relevant_docs"

# ─────────────────────────────────────────────────────
# Step 4: Build the Graph
# ─────────────────────────────────────────────────────
workflow = StateGraph(AgenticRAGState)

# Add nodes
workflow.add_node("decide_retrieval", decide_retrieval)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_with_context", generate_with_context)
workflow.add_node("generate_direct", generate_direct)
workflow.add_node("handle_no_relevant_docs", handle_no_relevant_docs)

# Add edges
workflow.add_edge(START, "decide_retrieval")

# Conditional edge after decision
workflow.add_conditional_edges(
    "decide_retrieval",
    route_after_decision,
    {
        "retrieve": "retrieve",
        "generate_direct": "generate_direct",
    }
)

# After retrieve, always grade
workflow.add_edge("retrieve", "grade_documents")

# Conditional edge after grading
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate_with_context": "generate_with_context",
        "handle_no_relevant_docs": "handle_no_relevant_docs",
    }
)

# All generation nodes lead to END
workflow.add_edge("generate_with_context", END)
workflow.add_edge("generate_direct", END)
workflow.add_edge("handle_no_relevant_docs", END)

# Compile
app = workflow.compile()

# ─────────────────────────────────────────────────────
# Step 5: Test the Graph
# ─────────────────────────────────────────────────────

def test_agentic_rag(question: str):
    """Test the agentic RAG system."""
    print(f"\n{'═'*70}")
    print(f"USER QUESTION: {question}")
    print('═' * 70)

    result = app.invoke({
        "question": question,
        "documents": [],
        "generation": "",
        "decision": "",
        "relevance": "",
        "iteration": 0,
    })

    print(f"\n{'─'*70}")
    print(f"📋 FINAL ANSWER:")
    print(f"{'─'*70}")
    print(result["generation"])

# Test cases
test_agentic_rag("Who founded TechCorp?")
test_agentic_rag("What is 100 divided by 4?")
test_agentic_rag("What was the weather like on Mars yesterday?")
```

### Visualizing the Graph

```python
# LangGraph can visualize your graph!

from IPython.display import Image

# Display graph as PNG
display(Image(app.get_graph().draw_mermaid_png()))

# Or get mermaid code
print(app.get_graph().draw_mermaid())
```

---

# PART 4: Agentic RAG Design Patterns

---

## 14. The ReAct Pattern

### What is ReAct?

```
ReAct = Reasoning + Acting

The LLM alternates between:
- THINKING: "What should I do next?"
- ACTING: Calling a tool
- OBSERVING: Looking at tool output
- THINKING: "Now what?"

This loop continues until the LLM decides it has enough info.

EXAMPLE:
Thought: "I need to find the company's Q3 revenue"
Action:  call_tool("search_company", "Q3 revenue")
Observation: "Q3 2024 revenue was $45 million"
Thought: "Now I need to compare with industry"
Action:  call_tool("web_search", "industry Q3 2024 revenue")
Observation: "Industry average was $40 million"
Thought: "I have all the info I need"
Final Answer: "TechCorp's Q3 revenue of $45M exceeded the industry average of $40M."
```

### Hands-On: Implement ReAct RAG Agent

```python
"""
ReAct AGENT WITH RAG
─────────────────────
The agent uses reasoning + tools to solve complex queries.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ─────────────────────────────────────────────────────
# Setup knowledge bases
# ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Internal company KB
company_docs = [
    Document(page_content="TechCorp Q1 2024 revenue: $30M"),
    Document(page_content="TechCorp Q2 2024 revenue: $38M"),
    Document(page_content="TechCorp Q3 2024 revenue: $45M"),
    Document(page_content="TechCorp Q4 2024 revenue: $52M"),
    Document(page_content="TechCorp's customer count grew 40% in 2024"),
]
company_store = Chroma.from_documents(
    company_docs, embeddings, collection_name="company"
)

company_tool = create_retriever_tool(
    retriever=company_store.as_retriever(search_kwargs={"k": 3}),
    name="search_company_data",
    description=(
        "Search internal TechCorp data including: quarterly revenue, "
        "customer growth, financial metrics, and company performance. "
        "Use for ANY questions about TechCorp's financials or metrics."
    ),
)

# ─────────────────────────────────────────────────────
# Add a calculator tool
# ─────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like "100 + 50" or "45 * 1.15"

    Returns:
        The result as a string
    """
    try:
        # Safe eval for simple math
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# ─────────────────────────────────────────────────────
# Create ReAct Agent
# ─────────────────────────────────────────────────────
tools = [company_tool, calculator]

# Custom system message guiding the agent
system_message = """You are a helpful assistant for TechCorp.
You have access to:
1. search_company_data - for company-specific information
2. calculator - for mathematical calculations

For each question:
1. Think about what information you need
2. Use the appropriate tool to get it
3. If you need more info, use more tools
4. Once you have everything, provide a clear answer

Always cite sources when using retrieved data.
"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_message,
)

# ─────────────────────────────────────────────────────
# Test the ReAct Agent
# ─────────────────────────────────────────────────────

def run_react_agent(question: str):
    """Run agent and show the reasoning trace."""
    print(f"\n{'═'*70}")
    print(f"QUESTION: {question}")
    print('═' * 70)

    # Stream the agent's execution to see step-by-step
    for chunk in agent.stream(
        {"messages": [("user", question)]},
        stream_mode="values",
    ):
        latest = chunk["messages"][-1]

        if hasattr(latest, "tool_calls") and latest.tool_calls:
            print(f"\n💭 AGENT THINKING...")
            for tc in latest.tool_calls:
                print(f"   🔧 Calling tool: {tc['name']}")
                print(f"      Arguments: {tc['args']}")

        elif hasattr(latest, "content") and latest.content:
            msg_type = type(latest).__name__
            if msg_type == "ToolMessage":
                print(f"\n📊 TOOL RESULT:")
                print(f"   {latest.content[:200]}")
            elif msg_type == "AIMessage":
                print(f"\n🤖 AGENT RESPONSE:")
                print(f"   {latest.content}")

# Test 1: Simple retrieval
run_react_agent("What was TechCorp's Q3 2024 revenue?")

# Test 2: Calculation requiring retrieval first
run_react_agent("What was TechCorp's total revenue for all of 2024?")

# Test 3: Complex multi-step reasoning
run_react_agent(
    "If TechCorp's revenue grew at the same rate from Q3 to Q4 2024, "
    "what would Q1 2025 revenue be?"
)
```

### What You'll See

```
For "What was TechCorp's total revenue for all of 2024?":

💭 AGENT THINKING...
   🔧 Calling tool: search_company_data
      Arguments: {'query': 'TechCorp quarterly revenue 2024'}

📊 TOOL RESULT:
   Q1 2024 revenue: $30M, Q2 2024 revenue: $38M,
   Q3 2024 revenue: $45M, Q4 2024 revenue: $52M

💭 AGENT THINKING...
   🔧 Calling tool: calculator
      Arguments: {'expression': '30 + 38 + 45 + 52'}

📊 TOOL RESULT:
   165

🤖 AGENT RESPONSE:
   TechCorp's total revenue for 2024 was $165 million,
   broken down as: Q1 $30M, Q2 $38M, Q3 $45M, Q4 $52M.

THE AGENT FIGURED OUT ON ITS OWN:
1. It needed to retrieve data first
2. It needed to do math after
3. It chose the right tool for each step
4. It synthesized everything into a clear answer
```

---

## 15. Plan-and-Execute Pattern

### How It Differs from ReAct

```
ReAct:
  Think → Act → Observe → Think → Act → ...
  (one step at a time)

Plan-and-Execute:
  PLAN: Make a multi-step plan upfront
  EXECUTE: Run each step
  REPLAN: Adjust if needed

BENEFITS OF PLAN-AND-EXECUTE:
✅ Better for complex queries with clear structure
✅ More efficient (less back-and-forth)
✅ Easier to debug and reason about
✅ Plan visible upfront for review

WHEN TO USE:
✅ Multi-step research tasks
✅ Complex data analysis queries
✅ Tasks with clear logical structure

WHEN ReAct IS BETTER:
✅ Exploratory queries
✅ When next step depends heavily on previous result
✅ Simpler queries that don't need planning
```

### Conceptual Implementation

```python
"""
PLAN-AND-EXECUTE PATTERN (Conceptual)
"""

from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

class PlanExecuteState(TypedDict):
    question: str
    plan: List[str]
    past_steps: List[tuple]
    response: str

# Planner: Creates the plan
planner_prompt = ChatPromptTemplate.from_template("""
Create a step-by-step plan to answer this question.

Each step should be specific and actionable.
Use tools: search_company_data, calculator

Question: {question}

Plan (numbered steps):
""")

def create_plan(state: PlanExecuteState) -> PlanExecuteState:
    """Generate a multi-step plan."""
    chain = planner_prompt | llm
    plan_text = chain.invoke({"question": state["question"]}).content

    # Parse into steps
    steps = [s.strip() for s in plan_text.split("\n") if s.strip()]
    return {**state, "plan": steps}

# Executor: Runs each step
def execute_step(state: PlanExecuteState) -> PlanExecuteState:
    """Execute one step of the plan."""
    if not state["plan"]:
        return state

    current_step = state["plan"][0]
    # ... execute the step using tools ...
    result = "step result here"

    return {
        **state,
        "plan": state["plan"][1:],   # Remove completed step
        "past_steps": state["past_steps"] + [(current_step, result)]
    }

# Decision: More steps or done?
def should_continue(state: PlanExecuteState) -> str:
    if state["plan"]:
        return "execute"
    else:
        return "respond"
```

---

## 16. Reflection / Self-Correction Pattern

### The Pattern

```
GENERATE → REFLECT → IMPROVE → REPEAT

Step 1: Generate initial answer
Step 2: Reflect on quality:
        - Is it accurate?
        - Is it complete?
        - Are there errors?
Step 3: If issues found:
        - Identify what's wrong
        - Generate improved version
Step 4: Repeat until satisfied (or max iterations)

This is like having a built-in editor that reviews
and improves the LLM's own work.
```

### Implementation Sketch

```python
"""
REFLECTION PATTERN
"""

class ReflectionState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    critique: str
    iteration: int
    max_iterations: int

def generate_answer(state):
    """Generate or regenerate answer."""
    answer = llm.invoke(...).content
    return {**state, "answer": answer}

def reflect(state):
    """Critique the current answer."""
    critique_prompt = f"""
    Question: {state['question']}
    Answer: {state['answer']}
    Sources: {state['documents']}

    Critique this answer:
    1. Is it accurate based on sources?
    2. Is it complete?
    3. Are there errors or omissions?

    If perfect, say "APPROVED".
    Otherwise, provide specific improvement suggestions.
    """
    critique = llm.invoke(critique_prompt).content
    return {**state, "critique": critique, "iteration": state["iteration"] + 1}

def should_continue(state) -> str:
    if "APPROVED" in state["critique"]:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "regenerate"

# Graph:
# START → generate → reflect → [decide] → generate (loop) OR END
```

---

# PART 5: Failure Modes and Interview Prep

---

## 17. Failure Modes 🚨

```
FAILURE 1: Tool Description Too Vague
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Description like "search documents"
Result:    Agent picks wrong tool or doesn't pick tool at all
Fix:       Specific descriptions with examples of what's covered

FAILURE 2: Infinite Loops
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Agent keeps calling tools without making progress
Result:    Burns API budget, never gives answer
Fix:       Add iteration counter, max_iterations limit
           Add explicit "give up" condition

FAILURE 3: Too Many Tool Calls
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Agent over-retrieves on simple questions
Result:    Slow response, high cost
Fix:       Better system prompt about when NOT to use tools

FAILURE 4: Tool Confusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Multiple similar tools confuse the LLM
Result:    Random tool selection
Fix:       Distinct, non-overlapping tool descriptions
           Maybe consolidate similar tools

FAILURE 5: State Loss Between Nodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Node forgets to pass through state fields
Result:    Downstream nodes get None values
Fix:       ALWAYS return full state with updates
           Use {**state, "new_field": value}

FAILURE 6: Bad Conditional Routing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   Routing function returns invalid node name
Result:    Graph errors out
Fix:       Test routing functions independently
           Validate returns match graph edges

FAILURE 7: LLM Doesn't Call Tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem:   LLM tries to answer without using available tools
Result:    Hallucinated answers
Fix:       Explicit system prompt encouraging tool use
           Few-shot examples of good tool usage
```

---

## 18. Interview Questions and Answers

### Q1: "What is Agentic RAG and how does it differ from traditional RAG?"

**Strong Answer:**
> "Traditional RAG follows a fixed pipeline — query, retrieve, generate — the same way for every query. Agentic RAG uses an LLM as an orchestrator that decides what to do based on the query. The retriever becomes a tool the LLM can choose to call, skip, or call multiple times. This enables capabilities impossible with static RAG: skipping retrieval for general knowledge questions, calling different retrievers for different parts of a query, looping back to refine retrieval based on intermediate results, and combining retrieval with other tools like calculators or web search. The tradeoff is variable latency, higher cost from extra LLM calls, and more complex debugging. But for complex queries requiring multi-step reasoning, agentic RAG is often the only approach that works."

---

### Q2: "When would you use LangGraph instead of LangChain chains?"

**Strong Answer:**
> "LangChain chains are great for linear pipelines: prompt → LLM → parser. They break down when you need branching logic, loops, or conditional execution. LangGraph treats your workflow as a state graph where nodes are functions, edges define flow, and conditional edges enable branching based on state. Use cases that demand LangGraph: agentic systems with tool selection, CRAG-style self-correcting RAG with retrieval grading and fallback, multi-step plan-and-execute patterns, reflection loops where the LLM critiques its own output and regenerates. The state is the shared scratchpad that flows through nodes — each node reads state, does work, and returns updates. This stateful, graph-based execution is essential for sophisticated agent behavior."

---

### Q3: "How do you create a retriever as a tool for an agent?"

**Strong Answer:**
> "I use create_retriever_tool from langchain.tools.retriever. It takes three key parameters: the retriever instance, a name like 'search_hr_policies', and most critically, a description. The description is what the LLM reads to decide when to use this tool. A bad description like 'search documents' leads to inconsistent tool selection. A good description specifies what's in the knowledge base, what types of questions it answers, and when the agent should use it. For example: 'Search HR knowledge base for vacation policy, sick leave, remote work, and benefits. Use this for ANY questions about employee policies.' For multi-domain systems, I create separate tools per knowledge base — hr_tool, legal_tool, tech_tool — letting the LLM route queries to the right one based on descriptions."

---

### Q4: "Explain the ReAct pattern with a concrete example."

**Strong Answer:**
> "ReAct stands for Reasoning + Acting. The LLM alternates between thinking about what to do and taking actions through tools. Concrete example: User asks 'What was TechCorp's total 2024 revenue?' The agent thinks: 'I need quarterly data first.' Acts: calls search_company_data. Observes: gets Q1 $30M, Q2 $38M, Q3 $45M, Q4 $52M. Thinks: 'Now I need to sum these.' Acts: calls calculator with '30+38+45+52'. Observes: gets 165. Thinks: 'I have my answer.' Responds: 'Total 2024 revenue was $165M.' This loop continues until the LLM decides it has enough information. LangGraph's create_react_agent provides this out of the box. The pattern shines for tasks requiring multiple tool calls where each step's output informs the next."

---

## 19. Common Mistakes ⚠️

```
MISTAKE 1: "Agentic RAG is always better"
TRUTH:     For simple Q&A, static RAG is faster, cheaper,
           and easier to debug. Use agentic for complex cases.

MISTAKE 2: "Just add more tools to make agent more capable"
TRUTH:     More tools = more confusion. Each tool should
           have distinct, non-overlapping responsibilities.

MISTAKE 3: "Tool descriptions don't matter much"
TRUTH:     Descriptions are the LLM's only guide for tool selection.
           Bad descriptions = bad tool selection = broken agent.

MISTAKE 4: "Forgetting to handle infinite loops"
TRUTH:     Always set max_iterations and termination conditions.
           Otherwise an agent can burn through your budget.

MISTAKE 5: "Returning partial state from nodes"
TRUTH:     Each node should return full state with updates.
           Use {**state, "new_field": value} pattern.

MISTAKE 6: "Testing only happy paths"
TRUTH:     Test what happens when tools fail, return empty,
           or LLM picks wrong tool. Build error handling.

MISTAKE 7: "Not monitoring agent decisions in production"
TRUTH:     Log every tool call, every decision, every step.
           Critical for debugging and improvement.
```

---

## 20. Engineer Mindset 🔧

```
WHAT A REAL AGENTIC RAG ENGINEER THINKS ABOUT:

1. "Start simple, add agency only when needed"
   → Try basic RAG first
   → Add agent capabilities only for proven complex queries
   → Don't over-engineer

2. "Tool descriptions are critical infrastructure"
   → Treat them like API documentation
   → Update as knowledge bases change
   → Test tool selection with edge cases

3. "Limit agent autonomy with guardrails"
   → max_iterations to prevent infinite loops
   → Cost limits per query
   → Allowed action lists
   → Human-in-the-loop for sensitive operations

4. "Design for observability"
   → Log every tool call
   → Track decision points
   → Visualize agent reasoning traces
   → Use LangSmith for production monitoring

5. "Plan for graceful failure"
   → Tool calls can fail
   → LLM can give up
   → Network issues happen
   → Always have a fallback response

6. "State design matters as much as code"
   → Keep state minimal and well-typed
   → Document each field's purpose
   → Don't pass huge documents around state

7. "Test the graph as a system"
   → Unit test each node
   → Integration test the full graph
   → Run with realistic queries
   → Measure cost per query type

8. "Production agents need cost controls"
   → Each query = multiple LLM calls
   → Monitor cost per user/query type
   → Set budgets and alerts
   → Cache where possible
```

---

## 21. Exercises 📝

### Beginner

```
1. Create a retriever tool from a vector store:
   - Set up Chroma with 10 documents
   - Create retriever
   - Convert to tool with create_retriever_tool
   - Write a clear, specific description

2. Build a simple agent with create_react_agent:
   - Add one retriever tool
   - Test with: in-domain query, off-domain query, math query
   - Observe which queries trigger retrieval

3. Create a simple LangGraph with 3 nodes:
   - Node 1: receive query
   - Node 2: process query
   - Node 3: return result
   - Connect with edges, compile, run

4. Define a TypedDict state:
   - 5 fields with different types
   - One node that updates state
   - Verify state flows correctly
```

### Intermediate

```
5. Build a multi-tool agent:
   - 3 different retriever tools (HR, Legal, Tech)
   - 1 calculator tool
   - Test that agent picks the right tool for different queries
   - Test cross-domain queries that need multiple tools

6. Build CRAG using LangGraph from scratch:
   - retrieve → grade → decide → generate OR web_fallback
   - All conditional routing
   - Test with in-domain and out-of-domain queries

7. Implement an agent with iteration limits:
   - Track iteration count in state
   - Force termination at max 5 iterations
   - Test with query that might loop forever

8. Build a reflection-based agent:
   - generate → reflect → improve → repeat
   - Max 3 reflection cycles
   - Compare quality of initial vs final answer
```

### Advanced

```
9. Build a production agentic RAG system:
   - Multiple knowledge bases (5+ tools)
   - Cost tracking per query
   - Comprehensive error handling
   - Monitoring and logging
   - Human-in-the-loop for sensitive actions

10. Implement plan-and-execute agent:
    - Planner generates step-by-step plan
    - Executor runs each step
    - Replan if step fails
    - Compare to ReAct on complex queries

11. Build a hybrid agent:
    - Some queries → simple RAG (fast path)
    - Complex queries → agentic RAG (smart path)
    - Query classifier routes between them
    - Measure cost/latency improvement

12. Implement multi-agent system:
    - Specialist agents (HR agent, Legal agent, Tech agent)
    - Orchestrator agent routes queries
    - Agents can collaborate on complex queries
    - Build with LangGraph subgraphs
```

---

## 22. Resources 📚

```
LANGRAPH DOCS:
🔗 LangGraph Documentation
   https://langchain-ai.github.io/langgraph/

🔗 LangGraph Tutorials
   https://langchain-ai.github.io/langgraph/tutorials/

🔗 LangGraph RAG Tutorial
   https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/

🔗 create_react_agent Reference
   https://langchain-ai.github.io/langgraph/reference/prebuilt/

LANGCHAIN AGENTS:
🔗 Tool Calling Guide
   https://python.langchain.com/docs/concepts/tool_calling/

🔗 Retriever Tools
   https://python.langchain.com/docs/how_to/tools_retrieval/

🔗 LangSmith (Agent Observability)
   https://www.langchain.com/langsmith

PAPERS:
📄 "ReAct: Synergizing Reasoning and Acting in Language Models"
   https://arxiv.org/abs/2210.03629

📄 "Plan-and-Solve Prompting"
   https://arxiv.org/abs/2305.04091

📄 "Reflexion: Language Agents with Verbal Reinforcement Learning"
   https://arxiv.org/abs/2303.11366

EXAMPLES:
🔗 LangGraph Agent Examples
   https://github.com/langchain-ai/langgraph/tree/main/examples

🔗 Production Agent Patterns
   https://blog.langchain.com/tag/agents/
```

---

## Module 8 Summary: The 20% That Gives 80% Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 8 MENTAL MODEL                          │
│                                                                 │
│  AGENTIC RAG = RAG + autonomous decision-making                │
│  The LLM becomes the orchestrator, not just the generator      │
│                                                                 │
│  TOOLS: Retrievers become tools the LLM can choose             │
│    - create_retriever_tool(retriever, name, description)       │
│    - Description quality determines tool selection quality     │
│    - Multiple tools for multi-domain knowledge                 │
│                                                                 │
│  LANGGRAPH: Graph-based workflows for complex flows            │
│    - State: Shared scratchpad flowing through nodes           │
│    - Nodes: Functions that do work and update state            │
│    - Edges: Connect nodes (normal or conditional)              │
│    - Loops, branches, retries — all possible                   │
│                                                                 │
│  PATTERNS:                                                      │
│    ReAct:              Think → Act → Observe → repeat          │
│    Plan-and-Execute:   Plan upfront → Execute steps            │
│    Reflection:         Generate → Critique → Improve           │
│                                                                 │
│  KEY DECISIONS:                                                 │
│    - Use agentic when query complexity demands it              │
│    - Tool descriptions are critical infrastructure             │
│    - Always add iteration limits and cost controls             │
│    - Log everything for production observability               │
│                                                                 │
│  GOLDEN RULES:                                                  │
│  1. Start with simple RAG, add agency only when needed         │
│  2. Tool descriptions = LLM's only guide → make them great     │
│  3. State design matters — keep it minimal and typed           │
│  4. Always add max_iterations to prevent infinite loops        │
│  5. Production agentic RAG needs cost monitoring               │
│  6. Test failure modes, not just happy paths                   │
└─────────────────────────────────────────────────────────────────┘
```