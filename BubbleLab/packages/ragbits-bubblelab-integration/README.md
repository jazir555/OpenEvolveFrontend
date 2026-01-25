# RAGBits BubbleLab Integration

A comprehensive integration between RAGBits (Retrieval-Augmented Generation framework) and BubbleLab (workflow automation platform).

## Overview

This package provides seamless integration between RAGBits and BubbleLab, enabling users to:

- Visually configure RAG workflows using BubbleLab's interface
- Leverage RAGBits' document processing and search capabilities
- Monitor and debug RAG workflows through BubbleLab's observability features
- Export RAG workflows as production-ready TypeScript code

## Installation

```bash
npm install ragbits-bubblelab-integration
```

## Features

- **RAG Bubble Components**: Specialized bubbles for document ingestion, semantic search, response generation, and index management
- **Configuration Mapping**: Automatic conversion between BubbleLab workflow definitions and RAGBits configurations
- **Workflow Execution Engine**: Execute RAG workflows defined in BubbleLab
- **Monitoring & Debugging**: Real-time metrics, performance tracking, and debugging tools
- **Configuration Generation**: Generate deployment-ready configurations from visual workflows

## Quick Start

```typescript
import { RagbitsBubbleLabIntegration } from 'ragbits-bubblelab-integration';

// Initialize the integration
const integration = RagbitsBubbleLabIntegration.getInstance();

// Create a workflow engine
const workflowEngine = integration.createWorkflowEngine(workflowConfig);

// Execute a workflow
const result = await workflowEngine.executeWorkflow();
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Configuration Guide](docs/CONFIGURATION_GUIDE.md)

## License

MIT