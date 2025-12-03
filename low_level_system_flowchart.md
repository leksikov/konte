flowchart TB
    subgraph Input["📄 Input"]
        A[Document Files<br/>PDF, TXT, MD]
    end

    subgraph Loader["📂 Document Loader"]
        A --> B{File Type?}
        B -->|PDF| C[pypdf extract text]
        B -->|TXT| D[Read UTF-8]
        B -->|MD| E[Read UTF-8]
        C --> F[Raw Text + Metadata]
        D --> F
        E --> F
        F --> G[metadata: filename,<br/>file_type, path]
    end

    subgraph Segmenter["✂️ Segmenter"]
        G --> H[Count tokens<br/>tiktoken]
        H --> I{tokens > 8000?}
        I -->|No| J[Single Segment]
        I -->|Yes| K[Split into segments]
        K --> L[Find word boundary<br/>near 8000 tokens]
        L --> M[Apply 800 token<br/>overlap]
        M --> N[Multiple Segments]
        J --> O[Segments List]
        N --> O
    end

    subgraph Chunker["🔪 Chunker"]
        O --> P[For each segment]
        P --> Q[Count tokens]
        Q --> R{tokens > 800?}
        R -->|No| S[Single Chunk]
        R -->|Yes| T[Split into chunks]
        T --> U[Find word boundary<br/>near 800 tokens]
        U --> V[Apply 80 token<br/>overlap]
        V --> W[Multiple Chunks]
        S --> X[Chunks List]
        W --> X
        X --> Y[Assign chunk_id<br/>source, segment_idx,<br/>chunk_idx]
    end

    subgraph ContextGen["🧠 Context Generation"]
        Y --> Z{Skip Context?}
        Z -->|Yes| AA[Empty context]
        Z -->|No| AB[Load prompt template<br/>from prompts/]
        AB --> AC[For each segment's chunks]
        AC --> AD[Build prompt:<br/>1. Segment text ~8000 tokens<br/>2. Chunk text ~800 tokens]
        AD --> AE[Async semaphore<br/>max 10 parallel]
        AE --> AF[Call OpenAI gpt-4.1]
        AF --> AG{Rate limited?}
        AG -->|Yes| AH[Exponential backoff<br/>retry]
        AH --> AF
        AG -->|No| AI{Timeout?}
        AI -->|Yes| AJ[Retry with<br/>increased timeout]
        AJ --> AF
        AI -->|No| AK[Context text<br/>100-200 tokens]
        AA --> AL[ContextualizedChunk]
        AK --> AL
        AL --> AM[contextualized_content =<br/>context + content]
    end

    subgraph FAISSBuild["🔷 FAISS Index Building"]
        AM --> AN{FAISS enabled?}
        AN -->|No| AO[Skip FAISS]
        AN -->|Yes| AP[Batch chunks<br/>for embedding]
        AP --> AQ[Call OpenAI<br/>text-embedding-3-small]
        AQ --> AR[Embedding vectors]
        AR --> AS[Create FAISS<br/>IndexFlatIP]
        AS --> AT[Add vectors<br/>to index]
        AT --> AU[Build ID mapping<br/>chunk_id ↔ faiss_idx]
        AU --> AV[Save faiss.index]
        AU --> AW[Save faiss_ids.json]
    end

    subgraph BM25Build["🔶 BM25 Index Building"]
        AM --> AX{BM25 enabled?}
        AX -->|No| AY[Skip BM25]
        AX -->|Yes| AZ[Tokenize<br/>contextualized_content]
        AZ --> BA[Build BM25Okapi<br/>index]
        BA --> BB[Save bm25.pkl<br/>pickle]
    end

    subgraph Storage["💾 Storage"]
        AV --> BC[Project Directory]
        AW --> BC
        BB --> BC
        AM --> BD[Save chunks.json]
        BD --> BC
        BC --> BE[config.json<br/>ProjectConfig]
    end

    subgraph Query["❓ Query Input"]
        BF[User Query] --> BG[Embed query<br/>text-embedding-3-small]
    end

    subgraph RetrievalProcess["🔍 Retrieval Process"]
        BG --> BH{Mode?}
        
        BH -->|Semantic| BI[FAISS search]
        BI --> BJ[Get top-N vectors<br/>by cosine similarity]
        BJ --> BK[Map faiss_idx<br/>to chunk_id]
        BK --> BL[Semantic Results<br/>+ scores]
        
        BH -->|Lexical| BM[Tokenize query]
        BM --> BN[BM25 search]
        BN --> BO[Get top-N chunks<br/>by BM25 score]
        BO --> BP[Lexical Results<br/>+ scores]
        
        BH -->|Hybrid| BQ[Both searches]
        BQ --> BI
        BQ --> BM
        BL --> BR[Reciprocal Rank Fusion]
        BP --> BR
        BR --> BS[For each result:<br/>RRF_score = Σ 1/k+rank]
        BS --> BT[Merge by chunk_id]
        BT --> BU[Sort by RRF_score<br/>descending]
        BU --> BV[Apply fusion weights<br/>semantic: 0.6, lexical: 0.4]
    end

    subgraph Fallback["⚠️ Fallback Logic"]
        BH -->|Hybrid but<br/>only FAISS| BW[Log warning]
        BW --> BI
        BH -->|Hybrid but<br/>only BM25| BX[Log warning]
        BX --> BM
    end

    subgraph ScoreCalc["📊 Score Calculation"]
        BL --> BY[Normalize scores<br/>to 0-1 range]
        BP --> BY
        BV --> BY
        BY --> BZ[top_score =<br/>max score]
        BY --> CA[score_spread =<br/>top - bottom]
    end

    subgraph ActionLogic["🎯 Suggested Action"]
        BZ --> CB{top_score >= 0.7?}
        CB -->|Yes| CC[suggested_action =<br/>'deliver']
        CB -->|No| CD{top_score >= 0.4?}
        CD -->|Yes| CE[suggested_action =<br/>'query_more']
        CD -->|No| CF[suggested_action =<br/>'refine_query']
        
        CC --> CG[has_high_confidence<br/>= true]
        CE --> CH[has_high_confidence<br/>= false]
        CF --> CH
    end

    subgraph Response["📤 RetrievalResponse"]
        BY --> CI[results: List]
        CI --> CJ[RetrievalResult<br/>content, context,<br/>score, source,<br/>chunk_id, metadata]
        BZ --> CK[top_score: float]
        CA --> CL[score_spread: float]
        CG --> CM[has_high_confidence: bool]
        CH --> CM
        CC --> CN[suggested_action: str]
        CE --> CN
        CF --> CN
        
        CJ --> CO[RetrievalResponse]
        CK --> CO
        CL --> CO
        CM --> CO
        CN --> CO
        BF --> CP[query: str]
        CP --> CO
        CI --> CQ[total_found: int]
        CQ --> CO
    end

    subgraph AgnoIntegration["🤖 Agno Integration"]
        CO --> CR[as_retriever]
        CR --> CS[Callable:<br/>query → RetrievalResponse]
        CS --> CT[Agno Agent<br/>Tool]
    end

    %% Styling
    style Input fill:#e3f2fd
    style Loader fill:#e1f5fe
    style Segmenter fill:#b3e5fc
    style Chunker fill:#81d4fa
    style ContextGen fill:#fff3e0
    style FAISSBuild fill:#e8f5e9
    style BM25Build fill:#c8e6c9
    style Storage fill:#f5f5f5
    style Query fill:#fce4ec
    style RetrievalProcess fill:#f8bbd9
    style Fallback fill:#ffcdd2
    style ScoreCalc fill:#e1bee7
    style ActionLogic fill:#d1c4e9
    style Response fill:#c5cae9
    style AgnoIntegration fill:#b2dfdb
