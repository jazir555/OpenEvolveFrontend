# RAGBits Integration Plan for BubbleLab

## 1. Executive Summary
This document outlines the step-by-step plan to integrate **RAGBits** (Retrieval-Augmented Generation library) into **BubbleLab** (Agentic Platform). The integration will enable BubbleLab users to perform semantic search, ingest documents, and leverage knowledge retrieval capabilities directly within their workflows.

## 2. Architecture Overview
The integration follows a **Client-Server** model to bridge the TypeScript-based BubbleLab frontend with the Python-based RAGBits backend.

*   **Frontend (BubbleLab)**:
    *   Uses `bubblelabs-ragbits-plugin` (React/TypeScript).
    *   Provides UI for configuration, search, and ingestion.
    *   Communicates with the backend via REST API.
*   **Backend (Python Service)**:
    *   Wraps `ragbits-core` functionality using FastAPI.
    *   Exposes endpoints for search, ingestion, and management.
    *   Handles vector database interactions and LLM processing.

## 3. Prerequisites
*   **BubbleLab Repository**: Checked out and `pnpm install` completed.
*   **RAGBits Repository**: Checked out and accessible.
*   **Python Environment**: Python 3.10+ with `ragbits` dependencies installed.
*   **Node.js Environment**: Node.js 18+ for BubbleLab.

## 4. Implementation Steps

### Phase 1: Frontend Integration (Plugin Setup)

The `bubblelabs-ragbits-plugin` already exists. We need to register and activate it within BubbleLab.

1.  **Build the Plugin**:
    ```bash
    cd bubblelabs-ragbits-plugin
    npm install
    npm run build
    ```

2.  **Register Integration in BubbleLab**:
    *   **File**: `BubbleLab/apps/bubble-studio/src/lib/integrations.ts`
    *   **Action**: Add RAGBits to `SERVICE_LOGOS` and `OPENEVOLVE_INTEGRATIONS`.
    *   **Code**:
        ```typescript
        // In SERVICE_LOGOS
        RAGBits: '/integrations/ragbits.svg',

        // In OPENEVOLVE_INTEGRATIONS
        { name: 'RAGBits', file: SERVICE_LOGOS['RAGBits'] },
        ```

3.  **Add Visual Assets**:
    *   Create or copy a `ragbits.svg` logo to `BubbleLab/apps/bubble-studio/public/integrations/`.

4.  **Install Plugin in BubbleLab**:
    *   Update `BubbleLab/package.json` or `pnpm-workspace.yaml` to include `bubblelabs-ragbits-plugin`.
    *   Run `pnpm install` in the BubbleLab root.

### Phase 2: Backend Service Implementation

We need to serve `ragbits` functionality via HTTP.

1.  **Create API Server**:
    *   **File**: `ragbits_server.py` (New file)
    *   **Framework**: FastAPI
    *   **Endpoints**:
        *   `GET /health`: Health check.
        *   `POST /search`: Semantic search with filters.
        *   `POST /ingest`: Document ingestion.
        *   `POST /ingest/batch`: Batch ingestion.
        *   `GET /stats`: Index statistics.

2.  **Integration Logic**:
    *   Initialize `ragbits` components (VectorStore, Embedder) in `ragbits_server.py`.
    *   Map API requests to `ragbits` method calls.

3.  **Run Server**:
    *   Command: `python ragbits_server.py` (hosting on port 8002 or similar).

### Phase 3: Configuration & Verification

1.  **Configure Plugin**:
    *   In BubbleLab UI, navigate to RAGBits settings.
    *   Set `Server URL` to `http://localhost:8002` (or wherever the Python backend is running).

2.  **Testing**:
    *   **Ingest Test**: Upload a sample document via the UI.
    *   **Search Test**: Query for terms related to the uploaded document.
    *   **Verification**: Ensure results are returned with correct relevance scores.

## 5. Timeline & Milestones
*   **Milestone 1**: Plugin built and registered in BubbleLab UI (Visual integration).
*   **Milestone 2**: `ragbits_server.py` running and responding to health checks.
*   **Milestone 3**: End-to-end flow (Ingest -> Search) working.

## 6. Resources
*   `bubblelabs-ragbits-plugin/README.md`: Plugin documentation.
*   `ragbits/packages/ragbits-core`: Core library documentation.
