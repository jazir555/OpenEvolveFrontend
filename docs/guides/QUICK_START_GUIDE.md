# OpenEvolve Federation - Quick Start Guide

**Last Updated**: 2026-02-03
**Status**: Production Ready

---

## 🚀 Getting Started in 5 Minutes

### Prerequisites

- Node.js 20+
- npm or pnpm
- Access to all core services (ICR, OpenEvolve, RAGBits, Graphiti, Vector DB, Z3, LeanAide)

### Step 1: Environment Configuration

```bash
# Create .env file
cat > .env << 'EOF'
# ICR Configuration
OPENEVOLVE_ICR_API_URL=http://localhost:8080
TIMEOUT_MS=5000

# OpenEvolve Configuration
OPENEVOLVE_API_URL=http://localhost:8002

# Knowledge Systems
RAGBITS_API_URL=http://localhost:8082
GRAPHITI_API_URL=http://localhost:8084
VECTORDB_URL=http://localhost:8083

# Formal Verification
Z3_API_URL=http://localhost:8080
LEANAIDE_API_URL=http://localhost:8081

# BubbleLab
BUBBLELAB_API_URL=http://localhost:3000
EOF
```

### Step 2: Install Dependencies

```bash
# Install all adapter dependencies
cd glue/adapters/icr-adapter && npm install
cd glue/lib/evolved-code-capture && npm install
cd glue/lib/unified-knowledge-query && npm install
cd glue/adapters/ragbits-graphiti-sync && npm install
cd glue/orchestration/unified-verification && npm install
cd glue/lib/proof-knowledge-base && npm install
```

### Step 3: Verify Services

```bash
# Run all probe scripts to verify services are accessible
bash glue/adapters/icr-adapter/probes/check_api.sh
bash glue/lib/evolved-code-capture/probes/check_storage.sh
bash glue/lib/unified-knowledge-query/probes/check_unified.sh
```

### Step 4: Build Everything

```bash
# Build all TypeScript
npm run build --workspaces --if-present
```

### Step 5: Start Using!

```typescript
// Example: ICR with memory
import { icrAdapter } from '@openevolve/icr-adapter';

const result = await icrAdapter.createContextualRequestWithMemory(
  "Refine this React component",
  { context_window: 5 }
);

console.log(result.result.content);
```

---

## 📚 Component Usage Guide

### 1. ICR Adapter

**Location**: `glue/adapters/icr-adapter/`

**7 Modes Available**:
- Refine: Traditional iterative refinement
- React: React application development
- Deepthink: Strategic problem-solving
- Adaptive Deepthink: Full deepthink access
- Agentic: Tool-based manipulation
- Contextual: Multi-agent collaboration (with memory!)
- Generative UI: Interactive UI generation

### 2. Evolved Code Capture

**Location**: `glue/lib/evolved-code-capture/`

Store OpenEvolve's evolved code for semantic search and reuse.

### 3. Unified Knowledge Query

**Location**: `glue/lib/unified-knowledge-query/`

Query RAGBits, Graphiti, and Vector DB simultaneously with intelligent result fusion.

### 4. Unified Verification Orchestrator

**Location**: `glue/orchestration/unified-verification/`

Cross-validate proofs with Z3 and LeanAide with confidence aggregation.

### 5. BubbleLab Evolution Workflows

**Location**: `core-projects/BubbleLab/apps/bubble-studio/src/bubbles/`

Trigger and manage OpenEvolve evolutions from BubbleLab workflows.

### 6. Proof Knowledge Base

**Location**: `glue/lib/proof-knowledge-base/`

Store and search formal proofs with semantic similarity and lineage tracking.

---

## 🎯 Common Workflows

### Workflow 1: Knowledge-Augmented Refinement

1. Retrieve relevant knowledge from all systems
2. Use knowledge to inform ICR refinement
3. Store learnings for future use

### Workflow 2: Evolution with Verification

1. Trigger OpenEvolve evolution
2. Validate with Z3 and LeanAide
3. Apply if confidence > 0.9

### Workflow 3: Continuous Learning Loop

1. Execute workflow with knowledge
2. System automatically learns from outcomes
3. Future executions improve automatically

---

## 📖 Additional Resources

- [Full Implementation Report](./INTEGRATION_IMPLEMENTATION_COMPLETE.md)
- [Gap Analysis Report](./INTEGRATION_GAP_ANALYSIS_REPORT.md)
- [Federation Constitution](./CLAUDE.md)

---

**Status**: Ready for Production
EOF