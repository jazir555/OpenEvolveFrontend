"""
External Knowledge Integration Module

This module provides integration with external knowledge sources to enhance
workflow execution with external data and expertise.
"""

import requests
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class KnowledgeSourceType(Enum):
    """Types of knowledge sources."""
    DATABASE = "database"
    API = "api"
    DOCUMENT = "document"
    WEB = "web"


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge from an external source."""
    source: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata
        }


@dataclass
class KnowledgeSourceConfig:
    """Configuration for a knowledge source."""
    name: str
    source_type: KnowledgeSourceType
    endpoint: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    timeout: int = 30
    max_retries: int = 3
    fallback_enabled: bool = True
    cache_ttl: int = 3600  # seconds


class KnowledgeSourceConnector(ABC):
    """Base class for knowledge source integrations."""
    
    def __init__(self, config: KnowledgeSourceConfig):
        """
        Initialize knowledge source connector.
        
        Args:
            config: Configuration for the knowledge source
        """
        self.config = config
        self.is_available = True
        self.last_error: Optional[str] = None
    
    @abstractmethod
    def query(self, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """
        Query the knowledge source.
        
        Args:
            context: Query context including problem domain, keywords, etc.
            
        Returns:
            List of knowledge items
        """
        pass
    
    def validate_connection(self) -> bool:
        """
        Validate connection to the knowledge source.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Attempt a simple query
            test_context = {"query": "test", "limit": 1}
            self.query(test_context)
            self.is_available = True
            self.last_error = None
            return True
        except Exception as e:
            self.is_available = False
            self.last_error = str(e)
            logger.warning(f"Knowledge source {self.config.name} validation failed: {e}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the knowledge source.
        
        Returns:
            Metadata dictionary
        """
        return {
            "name": self.config.name,
            "type": self.config.source_type.value,
            "is_available": self.is_available,
            "last_error": self.last_error,
            "endpoint": self.config.endpoint
        }
    
    def _handle_error(self, error: Exception, context: str) -> List[KnowledgeItem]:
        """
        Handle errors with fallback behavior.
        
        Args:
            error: The exception that occurred
            context: Context description for logging
            
        Returns:
            Empty list or fallback results
        """
        self.last_error = str(error)
        logger.error(f"Error in {self.config.name} ({context}): {error}")
        
        if self.config.fallback_enabled:
            logger.info(f"Continuing with available information for {self.config.name}")
            return []
        else:
            raise


class KnowledgeSource(ABC):
    """Abstract base class for external knowledge sources (legacy compatibility)."""
    
    @abstractmethod
    def query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge source."""
        pass
    
    @abstractmethod
    def get_relevant_knowledge(self, problem_statement: str) -> List[Dict[str, Any]]:
        """Get relevant knowledge for a problem."""
        pass


class WikipediaKnowledgeSource(KnowledgeSource):
    """Wikipedia as an external knowledge source."""
    
    def __init__(self):
        """Initialize Wikipedia knowledge source."""
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    def query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query Wikipedia."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 5
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("query", {}).get("search", [])
            
            return {
                "source": "wikipedia",
                "query": query,
                "results": [
                    {
                        "title": r["title"],
                        "snippet": r["snippet"],
                        "pageid": r["pageid"]
                    }
                    for r in results
                ]
            }
        except Exception as e:
            return {"source": "wikipedia", "error": str(e), "results": []}
    
    def get_relevant_knowledge(self, problem_statement: str) -> List[Dict[str, Any]]:
        """Get relevant Wikipedia articles for a problem."""
        result = self.query(problem_statement, {})
        return result.get("results", [])


class WebSearchKnowledgeSource(KnowledgeSource):
    """Web search as an external knowledge source using DuckDuckGo."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search knowledge source.
        
        Args:
            api_key: API key for search service (optional, not needed for DuckDuckGo)
        """
        self.api_key = api_key
        self.search_url = "https://html.duckduckgo.com/html/"
    
    def query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query web search using DuckDuckGo."""
        try:
            max_results = context.get("max_results", 5)
            
            # Use DuckDuckGo HTML search (no API key required)
            params = {
                "q": query,
                "kl": "us-en"  # Language/region
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.post(
                self.search_url,
                data=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            # Parse HTML response
            from html.parser import HTMLParser
            
            class DuckDuckGoParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self.current_result = {}
                    self.in_result = False
                    self.in_title = False
                    self.in_snippet = False
                    
                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == "div" and attrs_dict.get("class") == "result":
                        self.in_result = True
                        self.current_result = {}
                    elif self.in_result and tag == "a" and attrs_dict.get("class") == "result__a":
                        self.in_title = True
                        self.current_result["url"] = attrs_dict.get("href", "")
                    elif self.in_result and tag == "a" and attrs_dict.get("class") == "result__snippet":
                        self.in_snippet = True
                
                def handle_data(self, data):
                    if self.in_title:
                        self.current_result["title"] = data.strip()
                    elif self.in_snippet:
                        self.current_result["snippet"] = data.strip()
                
                def handle_endtag(self, tag):
                    if tag == "a" and self.in_title:
                        self.in_title = False
                    elif tag == "a" and self.in_snippet:
                        self.in_snippet = False
                    elif tag == "div" and self.in_result:
                        if self.current_result:
                            self.results.append(self.current_result)
                        self.in_result = False
                        self.current_result = {}
            
            parser = DuckDuckGoParser()
            parser.feed(response.text)
            
            results = parser.results[:max_results]
            
            return {
                "source": "web_search",
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "source": "web_search",
                "query": query,
                "results": [],
                "error": str(e)
            }
    
    def get_relevant_knowledge(self, problem_statement: str) -> List[Dict[str, Any]]:
        """Get relevant web search results."""
        result = self.query(problem_statement, {})
        return result.get("results", [])


class CustomAPIKnowledgeSource(KnowledgeSource):
    """Custom API as an external knowledge source."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize custom API knowledge source.
        
        Args:
            api_url: Base URL for the API
            api_key: API key for authentication
        """
        self.api_url = api_url
        self.api_key = api_key
    
    def query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query custom API."""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                f"{self.api_url}/query",
                json={"query": query, "context": context},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            return {"source": "custom_api", "error": str(e), "results": []}
    
    def get_relevant_knowledge(self, problem_statement: str) -> List[Dict[str, Any]]:
        """Get relevant knowledge from custom API."""
        result = self.query(problem_statement, {})
        return result.get("results", [])


class DatabaseConnector(KnowledgeSourceConnector):
    """Connector for SQL/NoSQL databases."""
    
    def __init__(self, config: KnowledgeSourceConfig):
        """
        Initialize database connector.
        
        Args:
            config: Configuration including connection details
        """
        super().__init__(config)
        self.connection = None
        self.db_type = config.metadata.get("db_type", "postgresql")  # postgresql, mongodb, etc.
    
    def query(self, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """
        Query the database.
        
        Args:
            context: Query context
            
        Returns:
            List of knowledge items from database
        """
        try:
            query_str = context.get("query", "")
            domain = context.get("domain", "")
            limit = context.get("limit", 10)
            
            logger.info(f"Querying {self.db_type} database: {query_str}")
            
            results = []
            
            if self.db_type == "postgresql":
                results = self._query_postgresql(query_str, domain, limit)
            elif self.db_type == "mongodb":
                results = self._query_mongodb(query_str, domain, limit)
            else:
                logger.warning(f"Unsupported database type: {self.db_type}")
            
            return results
            
        except Exception as e:
            return self._handle_error(e, "database query")
    
    def _query_postgresql(self, query_str: str, domain: str, limit: int) -> List[KnowledgeItem]:
        """Query PostgreSQL database."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            if not self.connection:
                self.connect()
            
            if not self.connection:
                return []
            
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Build query based on context
            sql = """
                SELECT id, content, relevance_score, metadata
                FROM knowledge_base
                WHERE content ILIKE %s
            """
            params = [f"%{query_str}%"]
            
            if domain:
                sql += " AND domain = %s"
                params.append(domain)
            
            sql += " ORDER BY relevance_score DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                item = KnowledgeItem(
                    source=f"postgresql:{self.config.name}",
                    content=row["content"],
                    relevance_score=row.get("relevance_score", 0.5),
                    metadata=row.get("metadata", {})
                )
                results.append(item)
            
            cursor.close()
            return results
            
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            return []
        except Exception as e:
            logger.error(f"PostgreSQL query failed: {e}")
            return []
    
    def _query_mongodb(self, query_str: str, domain: str, limit: int) -> List[KnowledgeItem]:
        """Query MongoDB database."""
        try:
            import pymongo
            
            if not self.connection:
                self.connect()
            
            if not self.connection:
                return []
            
            # Get database and collection from config
            db_name = self.config.metadata.get("database", "knowledge")
            collection_name = self.config.metadata.get("collection", "items")
            
            db = self.connection[db_name]
            collection = db[collection_name]
            
            # Build query
            query = {
                "$text": {"$search": query_str}
            }
            
            if domain:
                query["domain"] = domain
            
            # Execute query with text search score
            cursor = collection.find(
                query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            results = []
            for doc in cursor:
                item = KnowledgeItem(
                    source=f"mongodb:{self.config.name}",
                    content=doc.get("content", ""),
                    relevance_score=doc.get("score", 0.5),
                    metadata={k: v for k, v in doc.items() if k not in ["_id", "content", "score"]}
                )
                results.append(item)
            
            return results
            
        except ImportError:
            logger.error("pymongo not installed. Install with: pip install pymongo")
            return []
        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
            return []
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful
        """
        try:
            if self.db_type == "postgresql":
                return self._connect_postgresql()
            elif self.db_type == "mongodb":
                return self._connect_mongodb()
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def _connect_postgresql(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            import psycopg2
            
            credentials = self.config.credentials or {}
            
            self.connection = psycopg2.connect(
                host=credentials.get("host", "localhost"),
                port=credentials.get("port", 5432),
                database=credentials.get("database", "knowledge"),
                user=credentials.get("user", "postgres"),
                password=credentials.get("password", ""),
                connect_timeout=self.config.timeout
            )
            
            logger.info(f"Connected to PostgreSQL database: {credentials.get('database')}")
            return True
            
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def _connect_mongodb(self) -> bool:
        """Connect to MongoDB database."""
        try:
            import pymongo
            
            credentials = self.config.credentials or {}
            
            # Build connection string
            if "connection_string" in credentials:
                connection_string = credentials["connection_string"]
            else:
                host = credentials.get("host", "localhost")
                port = credentials.get("port", 27017)
                user = credentials.get("user", "")
                password = credentials.get("password", "")
                
                if user and password:
                    connection_string = f"mongodb://{user}:{password}@{host}:{port}/"
                else:
                    connection_string = f"mongodb://{host}:{port}/"
            
            self.connection = pymongo.MongoClient(
                connection_string,
                serverSelectionTimeoutMS=self.config.timeout * 1000
            )
            
            # Test connection
            self.connection.server_info()
            
            logger.info(f"Connected to MongoDB database")
            return True
            
        except ImportError:
            logger.error("pymongo not installed. Install with: pip install pymongo")
            return False
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info(f"Disconnected from {self.db_type} database")
            except Exception as e:
                logger.error(f"Error disconnecting from database: {e}")


class APIConnector(KnowledgeSourceConnector):
    """Connector for REST APIs."""
    
    def __init__(self, config: KnowledgeSourceConfig):
        """
        Initialize API connector.
        
        Args:
            config: Configuration including API endpoint and credentials
        """
        super().__init__(config)
        self.session = requests.Session()
        
        # Set up authentication if provided
        if config.credentials:
            api_key = config.credentials.get("api_key")
            if api_key:
                self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def query(self, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """
        Query the REST API.
        
        Args:
            context: Query context
            
        Returns:
            List of knowledge items from API
        """
        try:
            query_str = context.get("query", "")
            domain = context.get("domain", "")
            
            if not self.config.endpoint:
                raise ValueError("API endpoint not configured")
            
            # Make API request
            response = self.session.post(
                f"{self.config.endpoint}/query",
                json={"query": query_str, "domain": domain, "context": context},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Convert API response to KnowledgeItems
            items = []
            for item in data.get("results", []):
                knowledge_item = KnowledgeItem(
                    source=self.config.name,
                    content=item.get("content", ""),
                    relevance_score=item.get("relevance", 0.5),
                    metadata=item.get("metadata", {})
                )
                items.append(knowledge_item)
            
            return items
            
        except requests.exceptions.Timeout:
            logger.warning(f"API request to {self.config.name} timed out")
            return self._handle_error(TimeoutError("API timeout"), "API query")
        except requests.exceptions.RequestException as e:
            return self._handle_error(e, "API request")
        except Exception as e:
            return self._handle_error(e, "API query")


class DocumentConnector(KnowledgeSourceConnector):
    """Connector for document repositories."""
    
    def __init__(self, config: KnowledgeSourceConfig):
        """
        Initialize document connector.
        
        Args:
            config: Configuration including repository details
        """
        super().__init__(config)
        self.doc_type = config.metadata.get("doc_type", "text")  # text, pdf, markdown, etc.
        self.repository_path = config.endpoint or ""
    
    def query(self, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """
        Query the document repository.
        
        Args:
            context: Query context
            
        Returns:
            List of knowledge items from documents
        """
        try:
            query_str = context.get("query", "")
            keywords = context.get("keywords", [])
            limit = context.get("limit", 10)
            
            logger.info(f"Querying document repository: {self.repository_path}")
            
            if not self.repository_path or not os.path.exists(self.repository_path):
                logger.warning(f"Document repository path not found: {self.repository_path}")
                return []
            
            items = []
            
            # Search through documents in the repository
            for root, dirs, files in os.walk(self.repository_path):
                for file in files:
                    if len(items) >= limit:
                        break
                    
                    file_path = os.path.join(root, file)
                    
                    # Process different document types
                    try:
                        content = self._extract_document_content(file_path)
                        
                        if not content:
                            continue
                        
                        # Calculate relevance score
                        relevance = self._calculate_relevance(content, query_str, keywords)
                        
                        if relevance > 0.1:  # Minimum relevance threshold
                            item = KnowledgeItem(
                                source=f"document:{file}",
                                content=content[:1000],  # Limit content length
                                relevance_score=relevance,
                                metadata={
                                    "file_path": file_path,
                                    "file_type": os.path.splitext(file)[1],
                                    "file_size": os.path.getsize(file_path)
                                }
                            )
                            items.append(item)
                    
                    except Exception as e:
                        logger.debug(f"Error processing document {file_path}: {e}")
                        continue
                
                if len(items) >= limit:
                    break
            
            # Sort by relevance
            items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return items[:limit]
            
        except Exception as e:
            return self._handle_error(e, "document query")
    
    def _extract_document_content(self, file_path: str) -> str:
        """Extract text content from a document."""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext in [".txt", ".md", ".py", ".js", ".java", ".cpp", ".c", ".h"]:
                # Plain text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif ext == ".pdf":
                # PDF files
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages[:10]:  # Limit to first 10 pages
                            text += page.extract_text()
                        return text
                except ImportError:
                    logger.debug("PyPDF2 not installed. Install with: pip install PyPDF2")
                    return ""
            
            elif ext in [".docx", ".doc"]:
                # Word documents
                try:
                    import docx
                    doc = docx.Document(file_path)
                    return "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.debug("python-docx not installed. Install with: pip install python-docx")
                    return ""
            
            elif ext == ".json":
                # JSON files
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            
            else:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        
        except Exception as e:
            logger.debug(f"Error extracting content from {file_path}: {e}")
            return ""
    
    def _calculate_relevance(self, content: str, query: str, keywords: List[str]) -> float:
        """Calculate relevance score for content."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Check query match
        if query_lower in content_lower:
            score += 0.5
        
        # Check keyword matches
        if keywords:
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            score += (keyword_matches / len(keywords)) * 0.5
        
        # Check query word matches
        query_words = query_lower.split()
        if query_words:
            word_matches = sum(1 for word in query_words if word in content_lower)
            score += (word_matches / len(query_words)) * 0.3
        
        return min(1.0, score)
    
    def index_documents(self, document_paths: List[str]) -> int:
        """
        Index documents for faster searching.
        
        Args:
            document_paths: List of paths to documents
            
        Returns:
            Number of documents indexed
        """
        try:
            logger.info(f"Indexing {len(document_paths)} documents")
            
            indexed_count = 0
            index_data = []
            
            for doc_path in document_paths:
                try:
                    if not os.path.exists(doc_path):
                        logger.warning(f"Document not found: {doc_path}")
                        continue
                    
                    # Extract content
                    content = self._extract_document_content(doc_path)
                    
                    if not content:
                        continue
                    
                    # Create index entry
                    index_entry = {
                        "path": doc_path,
                        "content": content,
                        "file_type": os.path.splitext(doc_path)[1],
                        "file_size": os.path.getsize(doc_path),
                        "indexed_at": time.time()
                    }
                    
                    index_data.append(index_entry)
                    indexed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error indexing document {doc_path}: {e}")
                    continue
            
            # Save index to file
            if index_data:
                index_file = os.path.join(self.repository_path, ".document_index.json")
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, indent=2)
                logger.info(f"Saved index to {index_file}")
            
            return indexed_count
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return 0


class KnowledgeCache:
    """Caches external knowledge queries to minimize redundant API calls."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize knowledge cache.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[List[KnowledgeItem], float]] = {}  # key -> (items, expiry_time)
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    def _generate_key(self, context: Dict[str, Any]) -> str:
        """
        Generate cache key from context.
        
        Args:
            context: Query context
            
        Returns:
            Cache key string
        """
        # Create a stable key from context
        import hashlib
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def get(self, query_key: str) -> Optional[List[KnowledgeItem]]:
        """
        Get cached knowledge items.
        
        Args:
            query_key: Cache key
            
        Returns:
            Cached knowledge items or None if not found/expired
        """
        import time
        
        if query_key not in self.cache:
            self.miss_count += 1
            return None
        
        items, expiry_time = self.cache[query_key]
        
        # Check if expired
        if time.time() > expiry_time:
            del self.cache[query_key]
            self.miss_count += 1
            return None
        
        self.hit_count += 1
        return items
    
    def set(self, query_key: str, results: List[KnowledgeItem], ttl: Optional[int] = None) -> None:
        """
        Cache knowledge items.
        
        Args:
            query_key: Cache key
            results: Knowledge items to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        import time
        
        # Evict oldest item if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        ttl = ttl or self.default_ttl
        expiry_time = time.time() + ttl
        self.cache[query_key] = (results, expiry_time)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item (oldest by expiry time)."""
        if not self.cache:
            return
        
        # Find item with earliest expiry time
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        del self.cache[oldest_key]
        self.eviction_count += 1
        logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match against cache keys
            
        Returns:
            Number of entries invalidated
        """
        import re
        
        pattern_re = re.compile(pattern)
        keys_to_remove = [k for k in self.cache.keys() if pattern_re.search(k)]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching '{pattern}'")
        return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        logger.info("Cleared knowledge cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count
        }


class KnowledgeIntegrationManager:
    """Manages integration with multiple external knowledge sources."""
    
    def __init__(self, cache_max_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize knowledge integration manager.
        
        Args:
            cache_max_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
        """
        self.sources: Dict[str, KnowledgeSource] = {}
        self.connectors: Dict[str, KnowledgeSourceConnector] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}  # Legacy cache
        self.knowledge_cache = KnowledgeCache(max_size=cache_max_size, default_ttl=cache_ttl)
        self.cache_enabled = True
    
    def register_source(self, name: str, source: KnowledgeSource):
        """
        Register an external knowledge source (legacy).
        
        Args:
            name: Name for the source
            source: Knowledge source instance
        """
        self.sources[name] = source
    
    def register_connector(self, connector: KnowledgeSourceConnector):
        """
        Register a knowledge source connector.
        
        Args:
            connector: Knowledge source connector instance
        """
        self.connectors[connector.config.name] = connector
        logger.info(f"Registered knowledge connector: {connector.config.name}")
    
    def validate_all_connections(self) -> Dict[str, bool]:
        """
        Validate connections to all registered connectors.
        
        Returns:
            Dictionary mapping connector names to validation status
        """
        results = {}
        for name, connector in self.connectors.items():
            results[name] = connector.validate_connection()
        return results
    
    def get_connector_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered connectors.
        
        Returns:
            Dictionary mapping connector names to metadata
        """
        return {
            name: connector.get_metadata()
            for name, connector in self.connectors.items()
        }
    
    def query_all_sources(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Query all registered knowledge sources (legacy).
        
        Args:
            query: Query string
            context: Query context
            
        Returns:
            Dictionary mapping source names to results
        """
        results = {}
        
        for name, source in self.sources.items():
            try:
                results[name] = source.query(query, context)
            except Exception as e:
                results[name] = {"error": str(e), "results": []}
        
        return results
    
    def query_all_connectors(self, context: Dict[str, Any]) -> Dict[str, List[KnowledgeItem]]:
        """
        Query all registered knowledge connectors with caching.
        
        Args:
            context: Query context
            
        Returns:
            Dictionary mapping connector names to knowledge items
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = self.knowledge_cache._generate_key(context)
            cached_results = self.knowledge_cache.get(cache_key)
            if cached_results is not None:
                logger.debug(f"Cache hit for query")
                # Reconstruct results dict from cached items
                # For now, return all items under a single key
                # In production, you'd want to track which connector each item came from
                return {"cached": cached_results}
        
        results = {}
        all_items = []
        
        for name, connector in self.connectors.items():
            if not connector.is_available:
                logger.warning(f"Skipping unavailable connector: {name}")
                results[name] = []
                continue
            
            try:
                items = connector.query(context)
                results[name] = items
                all_items.extend(items)
                logger.info(f"Retrieved {len(items)} items from {name}")
            except Exception as e:
                logger.error(f"Error querying {name}: {e}")
                results[name] = []
        
        # Cache all results
        if self.cache_enabled and all_items:
            cache_key = self.knowledge_cache._generate_key(context)
            self.knowledge_cache.set(cache_key, all_items)
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get knowledge cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return self.knowledge_cache.get_stats()
    
    def clear_knowledge_cache(self) -> None:
        """Clear the knowledge cache."""
        self.knowledge_cache.clear()
    
    def invalidate_cache_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Regex pattern to match
            
        Returns:
            Number of entries invalidated
        """
        return self.knowledge_cache.invalidate(pattern)
    
    def get_relevant_knowledge_for_problem(
        self,
        problem_statement: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get relevant knowledge from all sources for a problem.
        
        Args:
            problem_statement: Problem statement
            
        Returns:
            Dictionary mapping source names to knowledge items
        """
        # Check cache first
        cache_key = f"problem:{problem_statement}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = {}
        
        for name, source in self.sources.items():
            try:
                results[name] = source.get_relevant_knowledge(problem_statement)
            except Exception as e:
                results[name] = []
        
        # Cache results
        self.cache[cache_key] = results
        
        return results
    
    def enrich_context_with_external_knowledge(
        self,
        context: Dict[str, Any],
        problem_statement: str
    ) -> Dict[str, Any]:
        """
        Enrich workflow context with external knowledge.
        
        Args:
            context: Current context
            problem_statement: Problem statement
            
        Returns:
            Enriched context
        """
        external_knowledge = self.get_relevant_knowledge_for_problem(problem_statement)
        
        enriched_context = context.copy()
        enriched_context["external_knowledge"] = external_knowledge
        
        # Summarize external knowledge
        total_items = sum(len(items) for items in external_knowledge.values())
        enriched_context["external_knowledge_summary"] = {
            "total_sources": len(external_knowledge),
            "total_items": total_items,
            "sources": list(external_knowledge.keys())
        }
        
        return enriched_context
    
    def clear_cache(self):
        """Clear the knowledge cache."""
        self.cache.clear()


# Global knowledge integration manager
_global_manager: Optional[KnowledgeIntegrationManager] = None


def get_knowledge_integration_manager() -> KnowledgeIntegrationManager:
    """Get or create the global knowledge integration manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = KnowledgeIntegrationManager()
        # Register default sources
        _global_manager.register_source("wikipedia", WikipediaKnowledgeSource())
    return _global_manager


def enable_external_knowledge_integration():
    """Enable external knowledge integration with default sources."""
    manager = get_knowledge_integration_manager()
    return manager
