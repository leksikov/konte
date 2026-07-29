# Architecture Overview

High-level view of Konte's pipeline: documents are segmented and chunked, each chunk is contextualized by an LLM that sees the full surrounding segment, and the contextualized chunks feed both a FAISS semantic index and a BM25 lexical index. Queries run against either or both indexes, with hybrid results combined via reciprocal rank fusion.

```mermaid
flowchart TB
    subgraph Ingestion["📄 Document Ingestion"]
        A[Documents<br/>PDF, TXT, MD] --> B[Segmenter<br/>~8000 tokens]
        B --> C[Chunker<br/>800 tokens]
    end

    subgraph Context["🧠 Context Generation"]
        C --> D{Skip Context?}
        D -->|No| E[LLM generates<br/>100-200 token context<br/>per chunk]
        D -->|Yes| F[Raw chunks]
        E --> G[Contextualized Chunks<br/>context + content]
        F --> G
    end

    subgraph Indexing["📚 Index Building"]
        G --> H{FAISS enabled?}
        G --> I{BM25 enabled?}
        H -->|Yes| J[Embed chunks]
        J --> K[(FAISS Index)]
        I -->|Yes| L[(BM25 Index)]
    end

    subgraph Retrieval["🔍 Retrieval"]
        M[Query] --> N{Mode?}
        N -->|Semantic| K
        N -->|Lexical| L
        N -->|Hybrid| O[Both indexes]
        K --> P[Top-K results]
        L --> P
        O --> Q[Reciprocal Rank<br/>Fusion]
        Q --> P
    end

    subgraph Response["📤 Response"]
        P --> R[RetrievalResponse]
        R --> S[suggested_action<br/>deliver / query_more / refine_query]
        S --> T[Agent / Application]
    end

    style Ingestion fill:#e1f5fe
    style Context fill:#fff3e0
    style Indexing fill:#e8f5e9
    style Retrieval fill:#fce4ec
    style Response fill:#f3e5f5
```

For the detailed implementation flow (token counting, retries, score calculation), see [architecture_detailed.md](architecture_detailed.md).
