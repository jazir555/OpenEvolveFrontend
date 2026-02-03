# DataPizza Integration Implementation Guide

## Current Status

**Status**: Partial Implementation (40% Complete)

The DataPizza BubbleLab plugin hooks have been updated with:
- ✅ Real API integration code (with fallback to mock data)
- ✅ Loading states and error handling
- ✅ Progress tracking for long-running operations
- ✅ Environment variable configuration
- ❌ Actual DataPizza backend API server (needs to be created)

## What's Been Implemented

### 1. Enhanced TypeScript Hooks (3 files)

#### `useDatapizzaQuery.ts`
- Attempts to call `/api/datapizza/query` endpoint
- Falls back to enhanced mock data if API unavailable
- Includes loading state, error handling, and timeout configuration
- Configurable via `DATAPIZZA_API_URL` environment variable

#### `useDatapizzaProcessing.ts`
- Attempts to call `/api/datapizza/process` endpoint
- Progress tracking with percentage updates
- Supports chunking, embedding, and vector storage configuration
- 2-minute timeout for processing operations

#### `useDatapizzaPipeline.ts`
- Attempts to call `/api/datapizza/pipeline` endpoint
- Step-by-step progress tracking (validation → chunking → embedding → storage)
- 5-minute timeout for pipeline execution
- Detailed pipeline execution reporting

### 2. Features Added

All hooks now include:
```typescript
// Loading state
isLoading: boolean
isRunning: boolean

// Error handling
error: string | null

// Progress tracking
progress: number (0-100)
currentStep: string

// Configuration options
- dataSource
- chunkSize / overlapSize
- embeddingModel
- vectorStore
- timeout

// Authentication
- API key via DATAPIZZA_API_KEY
- Bearer token authentication
```

## What's Still Missing

### 1. FastAPI Server Implementation

You need to create a FastAPI server that wraps the DataPizza Python library:

```python
# datapizza_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datapizza.core.modules.parser import Parser
from datapizza.core.modules.splitter import Splitter
from datapizza.core.embedders import Embedder
from datapizza.core.vectorstore import VectorStore

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    data_source: str = "default"
    max_results: int = 10
    threshold: float = 0.7

class ProcessRequest(BaseModel):
    data: dict
    processing_type: str = "standard"
    chunk_size: int = 1000
    overlap_size: int = 200

class PipelineRequest(BaseModel):
    data_source: str
    pipeline_type: str = "standard"
    chunk_size: int = 1000
    embedding_model: str = "default"

@app.post("/query")
async def query_data(req: QueryRequest):
    # Implement query logic using DataPizza
    pass

@app.post("/process")
async def process_data(req: ProcessRequest):
    # Implement processing logic
    pass

@app.post("/pipeline")
async def run_pipeline(req: PipelineRequest):
    # Implement full pipeline
    pass
```

### 2. Environment Configuration

Create a `.env` file in the datapizza-bubblelab-plugin directory:

```bash
# DataPizza API Configuration
DATAPIZZA_API_URL=http://localhost:8000
DATAPIZZA_API_KEY=your-api-key-here
DATAPIZZA_TIMEOUT=30000
```

### 3. DataPizza Module Integration

The Python DataPizza library has extensive modules:

**Parsers** (`datapizza/core/modules/parsers/`):
- `md_parser.py` - Markdown parsing
- `text_parser.py` - Text parsing

**Splitters** (`datapizza/core/modules/splitters/`):
- `node_splitter.py` - Node-based splitting
- `pdf_image_splitter.py` - PDF/image splitting
- `bbox_merger.py` - Bounding box merging

**Embedders** (`datapizza/embedders/`):
- `embedders.py` - Embedding generation

**Vector Store** (`datapizza/core/vectorstore/`):
- `vectorstore.py` - Vector storage operations

These need to be integrated into the FastAPI endpoints.

## Implementation Roadmap

### Phase 1: Basic API Server (1-2 days)
1. Create FastAPI application structure
2. Implement `/query` endpoint with basic semantic search
3. Add authentication and error handling
4. Test with TypeScript hooks

### Phase 2: Processing Pipeline (2-3 days)
1. Implement `/process` endpoint
2. Integrate DataPizza parsers and splitters
3. Add chunking and transformation logic
4. Add progress tracking via WebSocket

### Phase 3: Full Pipeline (3-5 days)
1. Implement `/pipeline` endpoint
2. Integrate embedders (OpenAI, Anthropic, etc.)
3. Add vector store integration (Qdrant, etc.)
4. Add pipeline orchestration and monitoring

### Phase 4: Production Hardening (2-3 days)
1. Add comprehensive error handling
2. Implement caching (Redis)
3. Add monitoring and logging
4. Performance optimization
5. Load testing

## Quick Start (Current State)

The hooks currently work in "mock mode" - they return realistic mock data when the API is unavailable. This allows:

1. **UI Development**: Build the BubbleLab UI without waiting for the backend
2. **Testing**: Test hook integration without API dependencies
3. **Gradual Migration**: Switch to real API as endpoints become available

## Testing the Hooks

### With Mock Data (Current Behavior)

```typescript
import { useDatapizzaQuery } from './hooks/useDatapizzaQuery';

function MyComponent() {
  const { queryData, isLoading, error } = useDatapizzaQuery();

  const handleQuery = async () => {
    const result = await queryData('test query', {
      dataSource: 'my-source',
      maxResults: 5
    });
    console.log(result);
    // Returns mock data with warnings about API not being configured
  };
}
```

### With Real API (After Server Setup)

1. Start the FastAPI server:
```bash
cd datapizza
uvicorn datapizza_server:app --reload --port 8000
```

2. Set environment variable:
```bash
export DATAPIZZA_API_URL=http://localhost:8000
```

3. The hooks will automatically use the real API

## API Endpoint Specifications

### POST /query

**Request:**
```json
{
  "query": "search query",
  "data_source": "default",
  "max_results": 10,
  "threshold": 0.7,
  "include_metadata": true
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "result_1",
      "score": 0.95,
      "data": {
        "content": "result content",
        "source": "data_source",
        "metadata": {}
      }
    }
  ],
  "confidence_score": 0.91,
  "errors": [],
  "warnings": [],
  "query_type": "semantic"
}
```

### POST /process

**Request:**
```json
{
  "data": { /* any data */ },
  "processing_type": "standard",
  "chunk_size": 1000,
  "overlap_size": 200,
  "embedding_model": "default",
  "vector_store": "default"
}
```

**Response:**
```json
{
  "data_id": "data_1234567890",
  "processed_data": {
    "processed": true,
    "chunks": 10,
    "embeddings": 10,
    "vectorStoreUpdated": true
  },
  "confidence_score": 0.92,
  "processing_steps": ["validation", "normalization", "transformation"],
  "chunk_count": 10,
  "embedding_count": 10
}
```

### POST /pipeline

**Request:**
```json
{
  "data_source": "/path/to/data",
  "pipeline_type": "standard",
  "chunk_size": 1000,
  "overlap_size": 200,
  "embedding_model": "text-embedding-ada-002",
  "vector_store": "qdrant",
  "skip_validation": false,
  "skip_embedding": false
}
```

**Response:**
```json
{
  "pipeline_id": "pipeline_1234567890",
  "processed_data": {
    "recordsProcessed": 1000,
    "chunksCreated": 100,
    "embeddingsGenerated": 100,
    "vectorStoreUpdated": true
  },
  "confidence_score": 0.95,
  "data_domain": "structured",
  "processing_steps": ["validation", "chunking", "embedding", "vector_storage"]
}
```

## Troubleshooting

### Hooks Always Return Mock Data

**Problem**: Hooks return mock data even when API is running

**Solutions**:
1. Check `DATAPIZZA_API_URL` is set correctly
2. Verify API server is running on the specified port
3. Check browser console for CORS errors
4. Verify API key if authentication is enabled

### "API request failed: 404"

**Problem**: Endpoint not found

**Solutions**:
1. Check FastAPI server is running
2. Verify endpoint paths match (`/query`, `/process`, `/pipeline`)
3. Check API URL includes correct port

### Permission Denied Writing Files

**Problem**: Cannot edit hook files

**Solution**:
```bash
# Remove read-only attribute
attrib -R datapizza-bubblelab-plugin/src/hooks/*.ts
```

## Files Modified

1. `datapizza-bubblelab-plugin/src/hooks/useDatapizzaQuery.ts`
2. `datapizza-bubblelab-plugin/src/hooks/useDatapizzaProcessing.ts`
3. `datapizza-bubblelab-plugin/src/hooks/useDatapizzaPipeline.ts`

## Next Steps

1. ✅ Update TypeScript hooks (COMPLETED)
2. ⏳ Create FastAPI server wrapper
3. ⏳ Implement query endpoint
4. ⏳ Implement processing endpoint
5. ⏳ Implement pipeline endpoint
6. ⏳ Add comprehensive testing
7. ⏳ Deploy and monitor

## Status

- **Task**: Complete DataPizza Pipeline Implementation
- **Completed**: Enhanced TypeScript hooks with API integration
- **Remaining**: FastAPI server implementation
- **Overall Progress**: 40%
- **Date**: 2026-02-02
