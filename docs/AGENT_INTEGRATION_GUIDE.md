# Konte Agent Integration Guide

This guide shows how to integrate Konte's contextual RAG into AI agent frameworks.

## Overview

Konte provides both **retrieval** and **full RAG** (retrieval + answer generation):

```
# Retrieval only
User Query → Konte (retrieval) → Chunks → Your LLM → Answer

# Full RAG (built-in)
User Query → Konte (retrieval + LLM) → Answer
```

## Quick Start

### Option 1: Full RAG with Built-in Answer Generation

```python
import asyncio
from konte import Project

project = Project.open("my_project")

async def main():
    # Full RAG: retrieval + LLM answer generation
    response, answer = await project.query_with_answer(
        query="What is the HS code for RAM chips?",
        mode="hybrid",
        max_chunks=5,
    )

    print(answer.answer)        # LLM-generated answer
    print(answer.model)         # e.g., "Qwen3-VL-8B-Instruct"
    print(answer.sources_used)  # Number of chunks used

asyncio.run(main())
```

### Option 2: Retrieval Only (BYO LLM)

```python
from konte import Project

project = Project.open("my_project")

# Query returns RetrievalResponse with chunks
response = project.query("What is the HS code for RAM chips?", mode="hybrid", top_k=10)

# Use results with your own LLM
for result in response.results:
    print(f"[{result.score:.2f}] {result.content[:100]}...")
```

### CLI

```bash
# Full RAG with answer
konte ask my_project "What is the HS code for DDR5 RAM?"
konte ask my_project "What is the HS code for DDR5 RAM?" --show-sources --mode hybrid

# Retrieval only
konte query my_project "DDR5 RAM" --mode hybrid --top-k 10
```

## RetrievalResponse Structure

```python
response = project.query("query text")

response.results          # List[RetrievalResult] - ranked chunks
response.query            # Original query string
response.total_found      # Number of results
response.top_score        # Highest relevance score (0-1)
response.score_spread     # Difference between top and bottom scores
response.has_high_confidence  # True if top_score >= 0.7
response.suggested_action # "deliver", "query_more", or "refine_query"

# Each result contains:
result.content    # Chunk text
result.context    # LLM-generated context (if built with context)
result.score      # Relevance score
result.source     # Source document path
result.chunk_id   # Unique identifier
result.metadata   # Additional metadata dict
```

## GeneratedAnswer Structure

```python
response, answer = await project.query_with_answer(query)

answer.answer        # LLM-generated answer text
answer.model         # Model used (e.g., "Qwen3-VL-8B-Instruct")
answer.sources_used  # Number of chunks used for generation
```

---

## Built-in RAG Answer Generation

Konte includes built-in LLM answer generation using BackendAI (Qwen3-VL-8B-Instruct by default):

### Basic Usage

```python
import asyncio
from konte import Project

async def main():
    project = Project.open("tariff_codes")

    # Full RAG with default settings
    response, answer = await project.query_with_answer(
        query="What is the HS code for DDR5 RAM?",
    )
    print(answer.answer)

asyncio.run(main())
```

### Advanced Options

```python
response, answer = await project.query_with_answer(
    query="What is the HS code for DDR5 RAM?",
    mode="hybrid",           # "hybrid", "semantic", or "lexical"
    top_k=20,                # Chunks to retrieve
    max_chunks=10,           # Chunks to use for answer generation
    timeout=60.0,            # LLM timeout in seconds
    prompt_template="""      # Custom prompt (optional)
Context:
{context}

Question: {question}

Answer based only on the context above:"""
)
```

### Standalone Answer Generation

```python
from konte import generate_answer, Project

project = Project.open("tariff_codes")

# Get retrieval results first
response = project.query("DDR5 RAM", mode="hybrid", top_k=10)

# Generate answer separately
answer = await generate_answer(
    question="What is the HS code for DDR5 RAM?",
    retrieval_response=response,
    max_chunks=5,
)
print(answer.answer)
```

---

## LangChain Integration

### Basic RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from konte import Project

# Load Konte project
project = Project.open("tariff_codes")

# Create LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a tariff classification expert.
Use the provided context to answer questions about HS codes.
If the context doesn't contain enough information, say so.

Context:
{context}"""),
    ("human", "{question}")
])

# Create chain
chain = prompt | llm | StrOutputParser()


def query_with_rag(question: str, top_k: int = 10) -> str:
    """Query Konte and generate answer with LLM."""
    # Retrieve relevant chunks
    response = project.query(question, mode="hybrid", top_k=top_k)

    # Format context from chunks
    context = "\n\n---\n\n".join([
        f"[Score: {r.score:.2f}] {r.content}"
        for r in response.results
    ])

    # Generate answer
    return chain.invoke({"context": context, "question": question})


# Usage
answer = query_with_rag("What HS code applies to DDR4 RAM memory chips?")
print(answer)
```

### As LangChain Retriever

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from konte import Project


class KonteRetriever(BaseRetriever):
    """LangChain-compatible retriever using Konte."""

    project: Project
    mode: str = "hybrid"
    top_k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        response = self.project.query(query, mode=self.mode, top_k=self.top_k)

        return [
            Document(
                page_content=r.content,
                metadata={
                    "score": r.score,
                    "source": r.source,
                    "chunk_id": r.chunk_id,
                    "context": r.context,
                    **r.metadata,
                }
            )
            for r in response.results
        ]


# Usage with LangChain RetrievalQA
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

project = Project.open("tariff_codes")
retriever = KonteRetriever(project=project, mode="hybrid", top_k=10)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    chain_type="stuff",
    retriever=retriever,
)

answer = qa_chain.invoke("What is the HS code for semiconductor memory?")
```

### With LCEL (LangChain Expression Language)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from konte import Project

project = Project.open("tariff_codes")
llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template("""
Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:""")


def retrieve(query: str) -> str:
    """Retrieve and format context from Konte."""
    response = project.query(query, mode="hybrid", top_k=10)
    return "\n\n".join([r.content for r in response.results])


# LCEL chain
chain = (
    {"context": retrieve, "question": RunnablePassthrough()}
    | prompt
    | llm
)

answer = chain.invoke("What HS code for Intel processors?")
```

---

## Agno Integration

### Basic Agno Agent with Konte Tool

```python
from agno import Agent, tool
from konte import Project

# Load project
project = Project.open("tariff_codes")


@tool
def search_tariff_codes(query: str, top_k: int = 10) -> str:
    """Search tariff code database for relevant information.

    Args:
        query: Search query about products or HS codes.
        top_k: Number of results to return.

    Returns:
        Relevant tariff code information.
    """
    response = project.query(query, mode="hybrid", top_k=top_k)

    if not response.results:
        return "No relevant information found."

    results = []
    for r in response.results:
        results.append(f"[Relevance: {r.score:.2f}]\n{r.content}")

    return "\n\n---\n\n".join(results)


# Create agent
agent = Agent(
    model="gpt-4o",
    tools=[search_tariff_codes],
    instructions="""You are a tariff classification expert.
Use the search_tariff_codes tool to find relevant HS codes.
Always cite the specific HS code (e.g., 8542.32) in your answers.""",
)

# Run agent
response = agent.run("Classify Samsung DDR4 RAM memory chips")
print(response.content)
```

### Multi-Project Agent

```python
from agno import Agent, tool
from konte import Project

# Load multiple specialized projects
hs_codes = Project.open("hs_codes")
gri_rules = Project.open("gri_rules")
precedents = Project.open("precedents")


@tool
def search_hs_codes(query: str) -> str:
    """Search HS code definitions and descriptions."""
    response = hs_codes.query(query, mode="hybrid", top_k=5)
    return "\n\n".join([f"[{r.score:.2f}] {r.content}" for r in response.results])


@tool
def search_gri_rules(query: str) -> str:
    """Search General Rules for Interpretation."""
    response = gri_rules.query(query, mode="hybrid", top_k=5)
    return "\n\n".join([f"[{r.score:.2f}] {r.content}" for r in response.results])


@tool
def search_precedents(query: str) -> str:
    """Search classification precedents and examples."""
    response = precedents.query(query, mode="hybrid", top_k=5)
    return "\n\n".join([f"[{r.score:.2f}] {r.content}" for r in response.results])


agent = Agent(
    model="gpt-4o",
    tools=[search_hs_codes, search_gri_rules, search_precedents],
    instructions="""You are a tariff classification expert.

For classification tasks:
1. First search_hs_codes for relevant headings
2. Check search_gri_rules for interpretation guidance
3. Look up search_precedents for similar cases

Provide the HS code with reasoning.""",
)

response = agent.run("What is the correct classification for a USB flash drive?")
```

### Using suggested_action for Agent Decisions

```python
from agno import Agent, tool
from konte import Project

project = Project.open("tariff_codes")


@tool
def search_with_confidence(query: str) -> dict:
    """Search tariff database and return results with confidence indicator.

    Returns dict with:
    - results: Retrieved information
    - action: "deliver" (confident), "query_more" (need more info), "refine_query" (low match)
    - confidence: Score from 0-1
    """
    response = project.query(query, mode="hybrid", top_k=10)

    results = "\n\n".join([r.content for r in response.results[:5]])

    return {
        "results": results,
        "action": response.suggested_action,
        "confidence": response.top_score,
        "has_high_confidence": response.has_high_confidence,
    }


agent = Agent(
    model="gpt-4o",
    tools=[search_with_confidence],
    instructions="""You are a tariff classification assistant.

When searching:
- If action="deliver" and has_high_confidence=True: Provide the answer
- If action="query_more": Try alternative search terms
- If action="refine_query": Ask user for more product details

Always explain your confidence level.""",
)
```

---

## Advanced Patterns

### Parallel Multi-Project Retrieval

```python
from konte import Project, RetrievalResponse

def query_all_projects(
    projects: list[Project],
    query: str,
    top_k: int = 5,
) -> dict[str, RetrievalResponse]:
    """Query multiple projects and collect results."""
    results = {}
    for project in projects:
        response = project.query(query, mode="hybrid", top_k=top_k)
        results[project._config.name] = response
    return results


def merge_results(responses: dict[str, RetrievalResponse], top_k: int = 10):
    """Merge and rank results from multiple projects."""
    all_results = []
    for project_name, response in responses.items():
        for result in response.results:
            all_results.append((project_name, result))

    # Sort by score
    all_results.sort(key=lambda x: x[1].score, reverse=True)
    return all_results[:top_k]


# Usage
projects = [
    Project.open("hs_codes"),
    Project.open("gri_rules"),
    Project.open("precedents"),
]

responses = query_all_projects(projects, "memory chip classification")
merged = merge_results(responses, top_k=10)

for project_name, result in merged:
    print(f"[{result.score:.2f}] ({project_name}) {result.content[:80]}...")
```

### Retrieval with Re-ranking

```python
from konte import Project
from langchain_openai import ChatOpenAI

project = Project.open("tariff_codes")
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def retrieve_and_rerank(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve chunks and re-rank with LLM."""
    # Get more results than needed
    response = project.query(query, mode="hybrid", top_k=top_k * 3)

    # Re-rank with LLM
    reranked = []
    for result in response.results:
        relevance_prompt = f"""Rate the relevance of this text to the query on a scale of 0-10.
Query: {query}
Text: {result.content[:500]}

Return only a number."""

        score = llm.invoke(relevance_prompt).content.strip()
        try:
            reranked.append({
                "content": result.content,
                "original_score": result.score,
                "rerank_score": float(score) / 10,
                "source": result.source,
            })
        except ValueError:
            continue

    # Sort by rerank score
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
```

### Streaming Responses

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from konte import Project

project = Project.open("tariff_codes")
llm = ChatOpenAI(model="gpt-4o", streaming=True)

prompt = ChatPromptTemplate.from_template("""
Context: {context}

Question: {question}

Answer:""")


async def stream_rag_response(question: str):
    """Stream RAG response token by token."""
    # Retrieve context
    response = project.query(question, mode="hybrid", top_k=10)
    context = "\n\n".join([r.content for r in response.results])

    # Stream response
    chain = prompt | llm
    async for chunk in chain.astream({"context": context, "question": question}):
        print(chunk.content, end="", flush=True)


# Usage
import asyncio
asyncio.run(stream_rag_response("What HS code for DDR4 RAM?"))
```

---

## Best Practices

### 1. Choose the Right Retrieval Mode

```python
# Exact keyword matching (HS codes, product names)
response = project.query("8542.32", mode="lexical")

# Semantic similarity (natural language questions)
response = project.query("chips for computer memory", mode="semantic")

# Best overall (combines both)
response = project.query("DDR4 RAM classification", mode="hybrid")
```

### 2. Use suggested_action for Agent Flow

```python
response = project.query(query)

if response.suggested_action == "deliver":
    # High confidence - provide answer
    return generate_answer(response.results)
elif response.suggested_action == "query_more":
    # Medium confidence - try related queries
    return expand_search(query)
else:  # "refine_query"
    # Low confidence - ask user for more info
    return ask_for_clarification()
```

### 3. Include Context in Prompts

```python
# Good: Include relevance scores
context = "\n\n".join([
    f"[Relevance: {r.score:.2f}]\n{r.content}"
    for r in response.results
])

# Better: Include source for citations
context = "\n\n".join([
    f"[Source: {r.source}, Score: {r.score:.2f}]\n{r.content}"
    for r in response.results
])
```

### 4. Handle Low Confidence

```python
if not response.has_high_confidence:
    # Add disclaimer to prompt
    prompt += "\nNote: Search confidence is low. Acknowledge uncertainty in your answer."
```

### 5. Optimize top_k

```python
# For focused answers: fewer, more relevant chunks
response = project.query(query, top_k=5)

# For comprehensive coverage: more chunks
response = project.query(query, top_k=20)

# For agent with multiple tool calls: balance
response = project.query(query, top_k=10)
```

---

## Example: Complete Classification Agent

```python
"""Complete HS Code Classification Agent using Konte + Agno."""

from agno import Agent, tool
from konte import Project

# Load projects
hs_project = Project.open("hs_codes")
rules_project = Project.open("gri_rules")


@tool
def search_hs_codes(query: str, top_k: int = 10) -> str:
    """Search HS code database."""
    response = hs_project.query(query, mode="hybrid", top_k=top_k)

    if response.suggested_action == "refine_query":
        return f"Low confidence results. Consider more specific terms.\n\n" + \
               "\n\n".join([r.content for r in response.results[:3]])

    return "\n\n---\n\n".join([
        f"[{r.score:.2f}] {r.content}"
        for r in response.results
    ])


@tool
def search_classification_rules(query: str) -> str:
    """Search GRI rules for interpretation guidance."""
    response = rules_project.query(query, mode="hybrid", top_k=5)
    return "\n\n".join([r.content for r in response.results])


@tool
def get_hs_code_details(hs_code: str) -> str:
    """Get details for a specific HS code like 8542.32."""
    response = hs_project.query(hs_code, mode="lexical", top_k=3)
    if response.results:
        return response.results[0].content
    return f"No details found for {hs_code}"


# Create agent
classification_agent = Agent(
    model="gpt-4o",
    tools=[search_hs_codes, search_classification_rules, get_hs_code_details],
    instructions="""You are an expert customs tariff classification agent.

Your task is to classify products to the correct HS code (Harmonized System).

Process:
1. Use search_hs_codes to find relevant headings for the product
2. Use search_classification_rules if interpretation guidance is needed
3. Use get_hs_code_details to verify the specific code

Always provide:
- The HS code (e.g., 8542.32)
- Brief reasoning for the classification
- Confidence level (high/medium/low)

If uncertain, explain what additional information would help.""",
)


# Run classification
if __name__ == "__main__":
    products = [
        "Samsung 16GB DDR4 RAM memory module",
        "Intel Core i9 desktop processor",
        "64GB USB flash drive",
        "Bluetooth wireless earbuds",
    ]

    for product in products:
        print(f"\n{'='*60}")
        print(f"Product: {product}")
        print("="*60)
        response = classification_agent.run(f"Classify: {product}")
        print(response.content)
```

---

## Summary

| Method | API | Best For |
|--------|-----|----------|
| **Full RAG** | `project.query_with_answer()` | Complete RAG with built-in LLM (BackendAI/OpenAI) |
| **Retrieval Only** | `project.query()` | Custom LLM pipelines, agent tools |
| **LangChain** | Custom Retriever class | Existing LangChain apps, LCEL chains |
| **Agno** | `@tool` decorated functions | Autonomous agents, multi-tool workflows |
| **CLI** | `konte ask` / `konte query` | Quick testing, scripts |

### Configuration

```bash
# BackendAI (default for answer generation)
BACKENDAI_ENDPOINT=https://qwen3vl.asia03.app.backend.ai/v1
BACKENDAI_MODEL_NAME=Qwen3-VL-8B-Instruct

# Falls back to OpenAI if BackendAI not configured
OPENAI_API_KEY=sk-...
```

---

## Combined Project with Metadata Filtering

When you have multiple knowledge bases, you can create a combined project that includes all chunks with `document_name` metadata for filtering:

```python
from konte import Project

# Query combined project
project = Project.open("all_tariff_documents")
response = project.query("DDR5 RAM classification", mode="hybrid", top_k=20)

# Filter results by document source
for result in response.results:
    doc_name = result.metadata.get("document_name", "unknown")
    print(f"[{result.score:.2f}] ({doc_name}) {result.content[:100]}...")

# Filter to specific documents only
hs_results = [r for r in response.results if r.metadata.get("document_name") == "hs_machinery_electronics_guide"]
```

### Available Metadata Fields

When using combined projects, each result includes:

```python
result.metadata = {
    "document_name": "hs_machinery_electronics_guide",  # Source project name
    "original_chunk_id": "doc_s0_c1",                   # Original chunk ID
    "original_segment_idx": 0,                           # Original segment index
    "original_source": "doc.md",                         # Original source file
}
```

Konte provides both retrieval and full RAG capabilities out of the box.
