"""
LangChain + ChromaDB Integration for OpenEvolve Tripartite System

This module provides the third component for the ACE + Steer + LangChain tripartite system:
- ACE: Self-improving capabilities through skill learning
- Steer: Reliability verification through deterministic output validation  
- LangChain + ChromaDB: Long-term memory and contextual knowledge retrieval

Key Features:
1. ChromaDB Vector Database: Persistent knowledge storage
2. LangChain Integration: Advanced RAG capabilities
3. Knowledge Retrieval: Semantic search for relevant context
4. Memory Management: Organize and retrieve learned knowledge
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import ChromaDB and embedding components
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple

# Create a simple Document class for compatibility
class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

# Simple text splitter for compatibility
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunked_docs = []
        for doc in documents:
            content = doc.page_content
            # Simple splitting by sentences/paragraphs
            sentences = content.split('. ')
            current_chunk = ""
            current_chunk_size = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_chunk_size + sentence_length <= self.chunk_size:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
                    current_chunk_size += sentence_length
                else:
                    if current_chunk:
                        chunked_docs.append(Document(current_chunk, doc.metadata.copy()))
                    current_chunk = sentence
                    current_chunk_size = sentence_length
            
            if current_chunk:
                chunked_docs.append(Document(current_chunk, doc.metadata.copy()))
        
        return chunked_docs

# Import existing ACE and Steer components
from ace_steer_integration import AceSteerBridge
from steer_crewai_bridge import SteerCrewAIWorkflowBridge

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class KnowledgeBaseConfig:
    """Configuration for the knowledge base system."""
    
    def __init__(self):
        self.persist_directory = "./knowledge_base"
        self.embedding_model = "all-MiniLM-L6-v2"  # Lightweight but effective
        self.collection_name = "openevolve_knowledge"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_knowledge_age_days = 30
        
        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

# ============================================================================
# KNOWLEDGE BASE MANAGER
# ============================================================================

class KnowledgeBaseManager:
    """
    Manages the ChromaDB vector database for persistent knowledge storage.
    """
    
    def __init__(self, config: Optional[KnowledgeBaseConfig] = None):
        self.config = config or KnowledgeBaseConfig()
        self.client = chromadb.PersistentClient(path=self.config.persist_directory)
        self.collection = None
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        self._initialize_collection()
        
    def _initialize_collection(self):
        """Initialize or load the ChromaDB collection."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing knowledge base with {self.collection.count()} documents")
        except (IOError, ValueError, RuntimeError) as e:
            logger.warning(f"Could not load existing knowledge base: {e}")
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info("Created new knowledge base")
    
    def add_knowledge(self, 
                     text: str, 
                     metadata: Optional[Dict[str, Any]] = None,
                     source: str = "unknown") -> List[str]:
        """
        Add knowledge to the vector database.
        
        Args:
            text: The text content to add
            metadata: Additional metadata
            source: Source of the knowledge
            
        Returns:
            List of document IDs that were added
        """
        if metadata is None:
            metadata = {}
            
        # Add source to metadata
        metadata["source"] = source
        
        # Create document
        doc = Document(
            page_content=text,
            metadata=metadata
        )
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        docs = text_splitter.split_documents([doc])
        
        # Prepare data for ChromaDB
        documents = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [f"doc_{i}" for i in range(len(docs))]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(ids)} knowledge chunks from source: {source}")
        return ids
    
    def retrieve_knowledge(self, 
                          query: str, 
                          k: int = 5,
                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve relevant knowledge based on semantic similarity.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        if filter_metadata is None:
            filter_metadata = {}
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            # Convert to Document objects
            documents = []
            for i in range(len(results['documents'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                documents.append(doc)
                
            logger.info(f"Retrieved {len(documents)} knowledge documents for query: '{query[:50]}...'")
            return documents
        except (IOError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []
    
    def search_with_scores(self, 
                          query: str, 
                          k: int = 5) -> List[tuple[Document, float]]:
        """
        Search knowledge with similarity scores.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to Document objects with scores
            documents_with_scores = []
            for i in range(len(results['documents'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                # Convert distance to similarity score (1 - normalized distance)
                distance = results['distances'][0][i]
                similarity = 1 - (distance / (1 + distance))  # Simple conversion
                documents_with_scores.append((doc, similarity))
                
            return documents_with_scores
        except (IOError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to search with scores: {e}")
            return []
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            return {
                "document_count": self.collection.count(),
                "collection_name": self.config.collection_name,
                "persist_directory": self.config.persist_directory,
                "embedding_model": self.config.embedding_model
            }
        except (IOError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {
                "error": str(e),
                "document_count": 0
            }
    
    def clear_knowledge(self) -> bool:
        """Clear all knowledge from the database."""
        try:
            self.collection.delete()
            logger.info("Cleared all knowledge from database")
            return True
        except (IOError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to clear knowledge: {e}")
            return False
    
    def close(self):
        """Close the vector store connection."""
        if hasattr(self.client, 'close'):
            self.client.close()
        logger.info("Knowledge base connection closed")

# ============================================================================
# LANGCHAIN INTEGRATION LAYER
# ============================================================================

class LangChainIntegration:
    """
    LangChain integration layer that connects knowledge base with ACE + Steer.
    """
    
    def __init__(self):
        self.knowledge_base = KnowledgeBaseManager()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def enhance_context_with_knowledge(self, 
                                      query: str, 
                                      existing_context: str = "") -> str:
        """
        Enhance agent context with relevant knowledge from the knowledge base.
        
        Args:
            query: The agent's query/task
            existing_context: Any existing context
            
        Returns:
            Enhanced context string
        """
        # Retrieve relevant knowledge
        knowledge_docs = self.knowledge_base.retrieve_knowledge(query, k=3)
        
        if not knowledge_docs:
            return existing_context
            
        # Format knowledge for context
        knowledge_context = "RELEVANT KNOWLEDGE:\n"
        for i, doc in enumerate(knowledge_docs, 1):
            knowledge_context += f"\n{i}. [Source: {doc.metadata.get('source', 'unknown')}]\n"
            knowledge_context += f"{doc.page_content}\n"
        
        # Combine with existing context
        if existing_context:
            enhanced_context = f"{knowledge_context}\n\nORIGINAL CONTEXT:\n{existing_context}"
        else:
            enhanced_context = knowledge_context
            
        return enhanced_context
    
    def store_learning_experience(self, 
                                 query: str, 
                                 response: str, 
                                 verification_result: Dict[str, Any],
                                 source: str = "ace_steer_execution"):
        """
        Store learning experiences in the knowledge base.
        
        Args:
            query: The original query
            response: The agent's response
            verification_result: Steer verification result
            source: Source of the learning experience
        """
        # Create knowledge entry
        knowledge_text = f"""
QUERY: {query}

RESPONSE: {response}

VERIFICATION STATUS: {'PASS' if verification_result.get('all_passed') else 'FAIL'}

VERIFICATION DETAILS:
"""
        
        if not verification_result.get('all_passed'):
            for result in verification_result.get('results', []):
                if not result.get('passed'):
                    knowledge_text += f"- {result['judge']}: {result.get('reason', 'No reason')}\n"
                    for fix in result.get('suggested_fixes', []):
                        knowledge_text += f"  - Fix: {fix.get('title', 'Unknown')}\n"
        
        # Add metadata
        metadata = {
            "type": "learning_experience",
            "verification_status": "pass" if verification_result.get('all_passed') else "fail",
            "query_length": len(query),
            "response_length": len(response),
            "timestamp": "",  # Could add actual timestamp
        }
        
        # Store in knowledge base
        self.knowledge_base.add_knowledge(
            text=knowledge_text,
            metadata=metadata,
            source=source
        )
    
    def get_rag_context(self, query: str) -> str:
        """
        Get Retrieval-Augmented Generation context for a query.
        
        Args:
            query: The query to enhance
            
        Returns:
            RAG-enhanced context string
        """
        return self.enhance_context_with_knowledge(query)

# ============================================================================
# TRIPARTITE SYSTEM INTEGRATION
# ============================================================================

class TripartiteAgentSystem:
    """
    Complete tripartite system integrating ACE + Steer + LangChain.
    
    Components:
    - ACE: Self-improving agent capabilities
    - Steer: Reliability verification
    - LangChain: Knowledge retrieval and memory
    """
    
    def __init__(self, 
                 agent_id: str = "tripartite_agent",
                 skillbook_path: Optional[str] = None):
        # Initialize ACE + Steer bridge
        self.ace_steer = AceSteerBridge(
            ace_agent_id=agent_id,
            skillbook_path=skillbook_path
        )
        
        # Initialize LangChain knowledge integration
        self.langchain = LangChainIntegration()
        
        # Initialize Steer workflow bridge
        self.steer_workflow = SteerCrewAIWorkflowBridge()
        
        logger.info(f"Tripartite Agent System initialized: {agent_id}")
    
    def execute_with_knowledge(self, 
                              task: str, 
                              verifications: List[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the complete tripartite system.
        
        Args:
            task: The task to execute
            verifications: List of Steer verifications to run
            **kwargs: Additional arguments for execution
            
        Returns:
            Dictionary with execution results and metadata
        """
        if verifications is None:
            verifications = ["json", "slop"]
            
        # Step 1: Retrieve relevant knowledge
        knowledge_context = self.langchain.get_rag_context(task)
        logger.info(f"Retrieved knowledge context for task: '{task[:50]}...'")
        
        # Step 2: Prepare prompt with ACE skills and knowledge
        enhanced_prompt = self.ace_steer.prepare_prompt(
            task=task,
            context=knowledge_context
        )
        
        # Step 3: Execute task (this would be replaced with actual agent execution)
        # For now, we'll simulate execution with the enhanced prompt
        execution_result = {
            "task": task,
            "prompt": enhanced_prompt,
            "knowledge_used": len(knowledge_context) > 0,
            "response": f"[SIMULATED RESPONSE] Based on knowledge: {knowledge_context[:100]}..."
        }
        
        # Step 4: Verify with Steer
        verification_result = self.ace_steer.verify_and_learn(
            query=task,
            output=execution_result["response"],
            verifications=verifications,
            reasoning="Tripartite system execution"
        )
        
        # Step 5: Store learning experience
        self.langchain.store_learning_experience(
            query=task,
            response=execution_result["response"],
            verification_result=verification_result
        )
        
        # Step 6: Compile final result
        final_result = {
            "success": verification_result["all_passed"],
            "task": task,
            "response": execution_result["response"],
            "knowledge_context": knowledge_context,
            "verification": verification_result,
            "stats": {
                "knowledge_documents_retrieved": len(self.langchain.knowledge_base.retrieve_knowledge(task, k=3)),
                "verification_passed": verification_result["all_passed"],
                "failed_verifications": verification_result["failed_verifications"]
            }
        }
        
        return final_result
    
    def add_knowledge(self, 
                     text: str, 
                     source: str = "manual",
                     metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add knowledge to the system's knowledge base.
        
        Args:
            text: Knowledge text to add
            source: Source of the knowledge
            metadata: Additional metadata
            
        Returns:
            List of document IDs that were added
        """
        return self.langchain.knowledge_base.add_knowledge(text, metadata, source)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return self.langchain.knowledge_base.get_knowledge_stats()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "ace_steer": {
                "agent_id": self.ace_steer.ace_agent_id,
                "skillbook_path": self.ace_steer.skillbook_path,
                "steer_status": self.ace_steer.steer_status
            },
            "knowledge_base": self.get_knowledge_stats(),
            "steer_workflow": {
                "available_verifications": self.steer_workflow.list_available_verifications()
            }
        }

# ============================================================================
# DECORATOR FOR EASY INTEGRATION
# ============================================================================

def tripartite_agent_capture(
    agent_id: str = "tripartite_agent",
    verifications: List[str] = None,
    skillbook_path: Optional[str] = None
):
    """
    Decorator to easily create tripartite agents.
    
    Args:
        agent_id: Unique identifier for the agent
        verifications: List of Steer verifications to run
        skillbook_path: Optional path to ACE skillbook
        
    Returns:
        Decorator function
    """
    if verifications is None:
        verifications = ["json", "slop"]
        
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create tripartite system
            system = TripartiteAgentSystem(
                agent_id=agent_id,
                skillbook_path=skillbook_path
            )
            
            # Get task from function arguments
            task = kwargs.get("task") or args[0] if args else "Unknown Task"
            
            # Execute with tripartite system
            result = system.execute_with_knowledge(
                task=str(task),
                verifications=verifications
            )
            
            # Call original function with enhanced context
            if "knowledge_context" in result:
                kwargs["knowledge_context"] = result["knowledge_context"]
            
            original_result = func(*args, **kwargs)
            
            # Attach tripartite results
            if isinstance(original_result, dict):
                original_result["_tripartite_results"] = result
            
            return original_result
            
        return wrapper
        
    return decorator

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_tripartite_system() -> TripartiteAgentSystem:
    """Initialize the complete tripartite system."""
    logger.info("Initializing OpenEvolve Tripartite System...")
    
    system = TripartiteAgentSystem()
    
    # Log system status
    status = system.get_system_status()
    logger.info(f"System initialized with {status['knowledge_base']['document_count']} knowledge documents")
    logger.info(f"Available verifications: {status['steer_workflow']['available_verifications']}")
    
    return system

# Auto-initialize on import
_tripartite_system = initialize_tripartite_system()

if __name__ == "__main__":
    print("🚀 OpenEvolve Tripartite System (ACE + Steer + LangChain)")
    print("=" * 60)
    
    # Test the system
    system = TripartiteAgentSystem()
    
    # Add some test knowledge
    system.add_knowledge(
        "The ACE system learns from agent executions and stores skills in a skillbook.",
        source="system_docs",
        metadata={"type": "system", "category": "ace"}
    )
    
    system.add_knowledge(
        "Steer provides deterministic verification of LLM outputs through reality locks.",
        source="system_docs", 
        metadata={"type": "system", "category": "steer"}
    )
    
    # Test execution
    result = system.execute_with_knowledge(
        "How does the ACE system work with Steer?"
    )
    
    print(f"✅ Task completed: {result['success']}")
    print(f"📚 Knowledge used: {result['stats']['knowledge_documents_retrieved']} documents")
    print(f"🔍 Verification passed: {result['stats']['verification_passed']}")
    print(f"💬 Response: {result['response'][:100]}...")