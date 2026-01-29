#!/usr/bin/env python3

"""
Simple test to verify the tripartite integration works.
"""

import tempfile
import os
from pathlib import Path

# Test imports
print("Testing imports...")

try:
    import chromadb
    print("ChromaDB imported successfully")
except Exception as e:  # TODO: Catch specific exception instead of Exception
    print(f"ChromaDB import failed: {e}")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformers imported successfully")
except Exception as e:  # TODO: Catch specific exception instead of Exception
    print(f"SentenceTransformers import failed: {e}")
    exit(1)

try:
    from ace_steer_integration import AceSteerBridge
    print("ACE+Steer integration imported successfully")
except Exception as e:  # TODO: Catch specific exception instead of Exception
    print(f"ACE+Steer import failed: {e}")
    exit(1)

# Test ChromaDB functionality
print("\nTesting ChromaDB functionality...")

with tempfile.TemporaryDirectory() as temp_dir:
    try:
        # Create ChromaDB client
        client = chromadb.PersistentClient(path=temp_dir)
        
        # Create collection
        collection = client.create_collection("test_collection")
        
        # Add some documents
        collection.add(
            documents=["This is a test document", "Another test document"],
            metadatas=[{"source": "test1"}, {"source": "test2"}],
            ids=["doc1", "doc2"]
        )
        
        # Query the collection
        results = collection.query(
            query_texts=["test document"],
            n_results=2
        )
        
        print(f"ChromaDB works - retrieved {len(results['documents'][0])} documents")
        
        # Clean up
        client.delete_collection("test_collection")
        
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"ChromaDB test failed: {e}")
        exit(1)

# Test ACE+Steer bridge
print("\nTesting ACE+Steer bridge...")

try:
    bridge = AceSteerBridge("test_agent")
    
    # Test prompt preparation
    prompt = bridge.prepare_prompt("Test task", "Test context")
    assert "TASK:" in prompt
    print("ACE+Steer bridge works - prompt preparation successful")
    
    # Test verification
    verification = bridge.verify_and_learn(
        query="Test query",
        output={"result": "test"},
        verifications=["json"]
    )
    assert "all_passed" in verification
    print("ACE+Steer bridge works - verification successful")
    
except Exception as e:  # TODO: Catch specific exception instead of Exception
    print(f"ACE+Steer test failed: {e}")
    exit(1)

# Test our custom components
print("\nTesting custom tripartite components...")

try:
    # Import our custom modules
    from langchain_chroma_integration import (
        KnowledgeBaseConfig,
        KnowledgeBaseManager,
        Document,
        RecursiveCharacterTextSplitter
    )
    print("Custom components imported successfully")
    
    # Test Document class
    doc = Document("Test content", {"source": "test"})
    assert doc.page_content == "Test content"
    assert doc.metadata["source"] == "test"
    print("Document class works")
    
    # Test text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    docs = splitter.split_documents([doc])
    assert len(docs) >= 1
    print("Text splitter works")
    
    # Test knowledge base manager
    with tempfile.TemporaryDirectory() as temp_dir:
        config = KnowledgeBaseConfig()
        config.persist_directory = temp_dir
        
        kb = KnowledgeBaseManager(config)
        stats = kb.get_knowledge_stats()
        assert stats["document_count"] == 0
        print("Knowledge base manager initialized")
        
        # Add knowledge
        doc_ids = kb.add_knowledge("Test knowledge", source="test")
        assert len(doc_ids) > 0
        print("Knowledge added successfully")
        
        # Retrieve knowledge
        results = kb.retrieve_knowledge("knowledge", k=1)
        assert len(results) > 0
        print("Knowledge retrieval works")
        
        kb.close()
        
except Exception as e:  # TODO: Catch specific exception instead of Exception
    print(f"Custom components test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nAll tests passed! The tripartite integration is working correctly.")
print("\nComponents verified:")
print("  ChromaDB - Vector database")
print("  SentenceTransformers - Embeddings")
print("  ACE+Steer - Self-improving + verification")
print("  Custom Integration - Knowledge management")
print("\nThe tripartite system is ready for use!")