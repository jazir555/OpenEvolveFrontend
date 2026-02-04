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
The primary goal is to enable the **ContextManager** in the Deterministic Pipeline to handle large documents (e.g., >10MB) by offloading the analysis to Matryoshka.

**Specific Objectives:**
1.  **Build**: (Completed) Build Matryoshka locally.
2.  **Adapter**: Create a Python adapter (`MatryoshkaClient`) to interface with the Node.js application.
3.  **Pipeline Integration**: Implement the `ContextManager` class to route large document queries to Matryoshka.
4.  **Verification**: Verify the integration with a test suite.

## 4. Implementation Steps

### Step 1: Build & Setup (Completed)
*   **Action**: Navigate to `core-projects/Matryoshka`.
*   **Fixes**: Resolved TS errors (implicit any, missing deps, casts).
*   **Verification**: `node dist/index.js --help` runs successfully.

### Step 2: Develop Python Adapter (Completed)
*   **Location**: `glue/adapters/matryoshka_adapter.py`
*   **Design**: Implemented `MatryoshkaClient` using `subprocess` to call `dist/index.js`.
*   **Verification**: Unit tests in `tests/test_matryoshka_adapter.py` pass.

### Step 3: Implement Context Manager (Completed)
*   **Location**: `knowledge_engine/context_manager.py`
*   **Logic**:
    *   Checks document size (Threshold: 10MB).
    *   Routes >10MB to Matryoshka.
    *   Routes <10MB to standard RAG (with graceful fallback).
*   **Verification**: Unit tests in `tests/test_context_manager.py` pass.

### Step 4: Testing (Completed)
*   **Unit Tests**: Created and verified `tests/test_matryoshka_adapter.py` and `tests/test_context_manager.py`.
*   **Status**: All tests passed.

## 6. Next Actions
*   **Configuration**: Set up `OPENAI_API_KEY` or Ollama for actual usage.
*   **Integration**: Wire `ContextManager` into the main application logic where document analysis is required.
