# Matryoshka Implementation Plan

## 1. Overview
Matryoshka is a **Recursive Language Model (RLM)** system designed to process documents significantly larger than an LLM's context window (100x larger). It achieves this by allowing the LLM to write code to explore the document iteratively, rather than relying on traditional chunking or vector databases.

This project handles the integration of Matryoshka into the OpenEvolve/Frontend system as **Layer 5: Context Management** of the Deterministic Pipeline.

## 2. Current Status
- **Location**: `core-projects/Matryoshka`
- **Type**: TypeScript / Node.js application
- **Interfaces**: CLI (`rlm`) and MCP Server (`rlm-mcp`)
- **Status in OpenEvolve**: "Experimental" / "Planned"
- **Build Status**: **Success** (Fixed TypeScript errors and built locally).

## 3. Integration Goals
The primary goal is to enable the **ContextManager** in the Deterministic Pipeline to handle large contexts (Documents, raw text, URLs) by offloading the analysis to Matryoshka.

**Specific Objectives:**
1.  **Build**: (Completed) Build Matryoshka locally.
2.  **Adapter**: (Completed) Create a Python adapter (`MatryoshkaClient`) to interface with the Node.js application.
3.  **Pipeline Integration**: (Completed) Implement the `ContextManager` class to route large document queries to Matryoshka.
4.  **Verification**: (Completed) Verify the integration with a test suite.
5.  **Generalization**: (Completed) Adapt the system for general use beyond local documents (Text, URLs).

## 4. Implementation Steps

### Step 1: Build & Setup (Completed)
*   **Action**: Navigate to `core-projects/Matryoshka`.
*   **Fixes**: Resolved TS errors (implicit any, missing deps, casts).
*   **Verification**: `node dist/index.js --help` runs successfully.

### Step 2: Develop Python Adapter (Completed)
*   **Location**: `glue/adapters/matryoshka_adapter.py`
*   **Design**: Implemented `MatryoshkaClient` with `analyze()`, `analyze_text()`, and `analyze_url()`.
*   **Verification**: `tests/test_matryoshka_adapter.py` passes.

### Step 3: Implement Context Manager (Completed)
*   **Location**: `knowledge_engine/context_manager.py`
*   **Logic**:
    *   `process_input(query, data, type)` handles 'file', 'text', 'url'.
    *   Large inputs are automatically routed to Matryoshka.
*   **Verification**: `tests/test_context_manager.py` passes.

### Step 4: Testing (Completed)
*   **Unit Tests**: Verified construction, cleanup, and routing logic across all input types.

### Step 5: Generalization (Completed)
*   **Feature**: Matryoshka integration now supports direct string input and URL downloading via temporary file orchestration in the adapter.
*   **Benefit**: Can analyze large web pages or large raw data strings without manual file management.

## 6. Next Actions
*   **Configuration**: Set up `OPENAI_API_KEY` or Ollama for actual usage.
*   **Integration**: Wire `ContextManager` into the main application logic where document analysis is required.