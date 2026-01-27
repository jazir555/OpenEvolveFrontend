flowchart TB
    %% Deterministic LLM Systems v2.0 - Complete Architecture

    U[User / App Request] --> APP[Application Layer]

    %% ============================================
    %% LAYER 0: Pre-Generation Filtering
    %% ============================================

    APP --> L0_LM[Lagrange Mapper - Attractor Detection]
    L0_LM --> L0_MIT[Two Phase Mitigation - Rephrase and Regenerate]

    %% ============================================
    %% LAYER 1: Task Decomposition
    %% ============================================

    L0_MIT --> L1_ROMA[ROMA - MECE Atomizer and DAG Planner]
    L1_ROMA --> L1_MAKER[MDAP MAKER - One Step Agents and Voting]
    L1_ROMA --> L1_RPG[RPG ZeroRepo - Codebase Planning Graph]

    %% ============================================
    %% LAYER 5: Context Management
    %% ============================================

    APP --> L5_SIZE[Document Size Assessment]
    L5_SIZE -->|less than 10MB| L5_RAG[Traditional RAG with Vector DB]
    L5_SIZE -->|10MB or more| L5_MAT[Matryoshka Code Based Exploration]

    %% ============================================
    %% LAYER 6: Temporal Knowledge
    %% ============================================

    L5_RAG --> L6_KE[Knowledge Engine - Bi temporal KG]
    L5_MAT --> L6_KE
    L6_KE --> L6_MEM[Agent Memory Systems - Episode Based Storage]
    L6_MEM --> L6_TQ[Point in Time Queries - Valid and Transaction Time]
    L6_TQ --> L6_HS[Hybrid Search - Semantic and Keyword and Graph]
    L6_HS --> L6_CD[Contradiction Detection and Resolution]

    %% ============================================
    %% LAYER 2: Constrained Generation
    %% ============================================

    L1_MAKER --> L2_DSPy[DSPy Modules - Signatures and Typed IO]
    L1_RPG --> L2_DSPy
    L6_CD --> L2_DSPy
    L2_DSPy --> L2_ROUTE[Generator Router]

    L2_LMQ[LMQL Constraints] --> L2_ROUTE
    L2_OUT[Outlines Logit Masking] --> L2_ROUTE
    L2_JF[Jsonformer Template] --> L2_ROUTE

    %% ============================================
    %% LAYER 3: Verification
    %% ============================================

    L2_ROUTE --> L3_STEER[Steer - Local Judges and Rule Injection]
    L3_STEER -->|pass| L3_GR[Guardrails - Enterprise Validators]
    L3_STEER -->|fail| L3_REASK[Reask With Feedback]
    L3_GR -->|pass| L3_CUST[Custom Domain Validators]
    L3_GR -->|fail| L3_REASK
    L3_CUST -->|fail| L3_REASK
    L3_REASK --> L2_ROUTE

    %% ============================================
    %% LAYER 4: Learning
    %% ============================================

    L3_CUST -->|pass| L4_ACE[ACE - Agent and Reflector]
    L4_ACE --> L4_LOOP[Learning Loop - Feedback to Skillbook]
    L4_DSPyOPT[DSPy Teleprompters] --> L4_LOOP

    %% ============================================
    %% LCoT Integration
    %% ============================================

    L4_LOOP --> LCOT_INV[Inverse Search Over Reasoning Chains]
    LCOT_INV --> LCOT_SP[SciencePedia - 3M Questions]
    LCOT_SP --> LCOT_PLATO[Plato Synthesis Agent]

    %% ============================================
    %% LAYER 7: Formal Verification
    %% ============================================

    LCOT_PLATO --> L7_LEAN[Lean 4 Theorem Prover]
    LCOT_PLATO --> L7_Z3[Z3 SMT Solver]
    L7_LEAN --> L7_PROPS[Proposition Extractor]
    L7_Z3 --> L7_PROPS
    L7_PROPS --> L7_FORM[Formal Guarantees and Proofs]

    %% ============================================
    %% LAYER 8: Multi-Modal Generation
    %% ============================================

    L7_FORM --> MM_DETECT[Multi Modal Mode Detection]
    MM_DETECT -->|text| MM_TEXT[Text Generator - LMQL Outlines]
    MM_DETECT -->|image| MM_IMG[ControlNet - Image Generation]
    MM_DETECT -->|code| MM_CODE[CodeT5 StarCoder - Code Generation]
    MM_TEXT --> MM_CONS[Cross Modal Consistency Verifier]
    MM_IMG --> MM_CONS
    MM_CODE --> MM_CONS

    %% ============================================
    %% Final Output
    %% ============================================

    MM_CONS --> OUTOK[Final Verified Output]

    %% ============================================
    %% Observability
    %% ============================================

    OUTOK --> OBS_DEP[API Gateway to Services]
    OBS_DEP --> OBS_SHARED[Redis Postgres Qdrant Neo4j MLflow]
    OBS_TEL[OpenTelemetry Traces] --> OBS_GRAF[Grafana Dashboards]
    OBS_MET[Prometheus Metrics] --> OBS_GRAF
    OBS_LOG[Structured Logging] --> OBS_GRAF
    OBS_MLF[MLflow Tracking] --> OBS_GRAF
    OBS_GRAF --> OBS_ALERT[Alerting Rules]

    %% ============================================
    %% Distributed Coordination
    %% ============================================

    OUTOK --> DIST_KAFKA[Kafka NATS Task Distribution]
    OUTOK --> DIST_ETCD[etcd Raft Consensus State]
    DIST_KAFKA --> DIST_COORD[Distributed Coordinator]
    DIST_ETCD --> DIST_COORD
    DIST_COORD --> DIST_SEED[Hash of Prompt to Seed]
    DIST_SEED --> DIST_VOTE[Consensus Check]
    DIST_VOTE -->|identical| DIST_OUT[Single Result]
    DIST_VOTE -->|different| DIST_VOTE2[Fallback Voting]
    DIST_VOTE2 --> DIST_OUT

    %% ============================================
    %% Knowledge Graph of Thought
    %% ============================================

    subgraph KGOT[Knowledge Graph of Thought]
        direction TB
        KGOT_MEM[Temporal Knowledge Storage]
        KGOT_REL[Relationship Graph]
        KGOT_EPIS[Episode Based Episodes]
        KGOT_EVOL[Knowledge Evolution]
        KGOT_MEM --> KGOT_REL
        KGOT_REL --> KGOT_EPIS
        KGOT_EPIS --> KGOT_EVOL
    end

    L6_KE -.-> KGOT

    %% ============================================
    %% Styles
    %% ============================================

    classDef layer0 fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    classDef layer1 fill:#4ecdc4,stroke:#0891b2,stroke-width:3px
    classDef layer2 fill:#45b7d1,stroke:#0c4a6e,stroke-width:3px
    classDef layer3 fill:#96ceb4,stroke:#2d6a4f,stroke-width:3px
    classDef layer4 fill:#ffeaa7,stroke:#d63031,stroke-width:3px
    classDef layer5 fill:#dfe6e9,stroke:#2d3436,stroke-width:3px
    classDef layer6 fill:#a29bfe,stroke:#6c5ce7,stroke-width:3px
    classDef layer7 fill:#fd79a8,stroke:#e84393,stroke-width:3px
    classDef layer8 fill:#fdcb6e,stroke:#e17055,stroke-width:3px
    classDef lcot fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    classDef observability fill:#55a3ff,stroke:#0056b3,stroke-width:2px,stroke-dasharray: 5 5
    classDef distributed fill:#fab1a0,stroke:#e17055,stroke-width:2px,stroke-dasharray: 5 5

    class L0_LM,L0_MIT layer0
    class L1_ROMA,L1_MAKER,L1_RPG layer1
    class L2_DSPy,L2_ROUTE,L2_LMQ,L2_OUT,L2_JF layer2
    class L3_STEER,L3_GR,L3_CUST,L3_REASK layer3
    class L4_ACE,L4_LOOP,L4_DSPyOPT layer4
    class L5_SIZE,L5_RAG,L5_MAT layer5
    class L6_KE,L6_MEM,L6_TQ,L6_HS,L6_CD layer6
    class L7_LEAN,L7_Z3,L7_PROPS,L7_FORM layer7
    class LCOT_INV,LCOT_SP,LCOT_PLATO lcot
    class MM_DETECT,MM_TEXT,MM_IMG,MM_CODE,MM_CONS layer8
    class OBS_DEP,OBS_SHARED,OBS_TEL,OBS_MET,OBS_LOG,OBS_MLF,OBS_GRAF,OBS_ALERT observability
    class DIST_KAFKA,DIST_ETCD,DIST_COORD,DIST_SEED,DIST_VOTE,DIST_VOTE2,DIST_OUT distributed
