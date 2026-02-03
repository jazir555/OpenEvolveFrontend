#!/usr/bin/env python3
"""
Test script to verify RAGBits integration with BubbleLab.
"""

import asyncio
import tempfile
import os
from typing import Dict, Any, List

async def test_ragbits_integration():
    """Test the complete RAGBits integration with BubbleLab."""
    print("Testing RAGBits integration with BubbleLab...")
    
    # Test 1: Check if RAGBits document processor can be imported and initialized
    print("\n1. Testing RAGBitsDocumentProcessor...")
    try:
        from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor, RAGBitsProcessorConfig
        
        # Create a basic config
        config = RAGBitsProcessorConfig()
        print(f"   + RAGBitsProcessorConfig created with: {config.vector_store_type} store")
        
        # Create processor
        processor = RAGBitsDocumentProcessor(config)
        print("   + RAGBitsDocumentProcessor created")
        
        # Try to initialize (this may fail if RAGBits is not installed)
        success = await processor.initialize()
        print(f"   + Initialization successful: {success}")
        
        if success:
            stats = await processor.get_statistics()
            print(f"   + Statistics: {stats}")
        
    except ImportError as e:
        print(f"   - RAGBitsDocumentProcessor import failed: {e}")
    except Exception as e:
        print(f"   - Error in RAGBitsDocumentProcessor test: {e}")
    
    # Test 2: Check if RAGBits retriever can be imported and used
    print("\n2. Testing RAGBitsEnhancedRetriever...")
    try:
        from knowledge_engine.ragbits_retriever import get_ragbits_retriever
        
        retriever = get_ragbits_retriever()
        print("   + RAGBitsEnhancedRetriever retrieved via singleton")
        
        # Check if it's available
        stats = await retriever.get_statistics()
        print(f"   + Retriever stats: {stats}")
        
    except ImportError as e:
        print(f"   - RAGBitsEnhancedRetriever import failed: {e}")
    except Exception as e:
        print(f"   - Error in RAGBitsEnhancedRetriever test: {e}")
    
    # Test 3: Check if RAGBits safety functions exist
    print("\n3. Testing RAGBits safety functions...")
    try:
        from knowledge_engine.ragbits_safety import (
            validate_query,
            validate_top_k,
            validate_filters,
            safe_execute,
            create_safe_wrapper
        )
        
        print("   + All RAGBits safety functions imported successfully")
        
        # Test validation functions
        is_valid = validate_query("test query")
        print(f"   + Query validation works: {is_valid}")
        
        top_k = validate_top_k(5)
        print(f"   + Top-k validation works: {top_k}")
        
        filters = validate_filters({"test": "value"})
        print(f"   + Filters validation works: {filters}")
        
    except ImportError as e:
        print(f"   - RAGBits safety functions import failed: {e}")
    except Exception as e:
        print(f"   - Error in RAGBits safety functions test: {e}")
    
    # Test 4: Check if API endpoint would work (without actually calling it)
    print("\n4. Testing API endpoint availability...")
    try:
        from api_server import app
        # Check if the route exists by looking at the routes
        routes = [route.path for route in app.routes]
        ragbits_routes = [route for route in routes if 'ragbits' in route.lower()]
        if ragbits_routes:
            print(f"   + RAGBits API endpoints available: {ragbits_routes}")
        else:
            print("   - RAGBits API endpoints not found")
    except Exception as e:
        print(f"   - Error checking API endpoints: {e}")
    
    # Test 5: Check if ragbits server exists
    print("\n5. Testing ragbits server existence...")
    try:
        import os
        server_exists = os.path.exists("ragbits_server.py")
        print(f"   + RAGBits server file exists: {server_exists}")
        
        if server_exists:
            print("   + RAGBits server available for standalone operation")
    except Exception as e:
        print(f"   - Error checking ragbits server: {e}")
    
    print("\n" + "="*60)
    print("RAGBits-BubbleLab Integration Test Complete")
    print("="*60)
    print("\nSummary:")
    print("- RAGBits should be available for document processing and semantic search")
    print("- API endpoints /openevolve/ragbits/search, /ingest, /stats should be accessible")
    print("- BubbleLab can request document ingestion and search through the API")
    print("- Safety and validation functions should be available")
    print("- Fallback mechanisms should handle missing RAGBits gracefully")


if __name__ == "__main__":
    asyncio.run(test_ragbits_integration())