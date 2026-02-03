"""
Browser Research Agent

The "Live Web" interface - enables agents to browse the web in real-time.
When the Blue Team hits an error, they can search GitHub Issues, read docs, 
and ingest fresh knowledge rather than hallucinating fixes.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import re
from urllib.parse import urljoin, urlparse
import html

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from web search"""
    title: str
    url: str
    snippet: str
    source: str  # google, github, stackoverflow, etc.
    relevance_score: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'source': self.source,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp
        }


@dataclass
class ResearchSession:
    """A research session with multiple queries and results"""
    session_id: str
    query: str
    results: List[SearchResult]
    pages_visited: List[str]
    knowledge_extracted: List[Dict[str, Any]]
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'pages_visited': self.pages_visited,
            'knowledge_extracted': self.knowledge_extracted,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


@dataclass
class PageContent:
    """Extracted content from a web page"""
    url: str
    title: str
    text_content: str
    code_blocks: List[str]
    links: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'title': self.title,
            'text_content': self.text_content[:1000] + '...' if len(self.text_content) > 1000 else self.text_content,
            'code_blocks': self.code_blocks[:5],  # Limit code blocks
            'links': self.links[:10],  # Limit links
            'metadata': self.metadata
        }


class BrowserResearchAgent:
    """
    Headless browser agent for live web research.
    
    Capabilities:
    - Search multiple sources (Google, GitHub, StackOverflow, etc.)
    - Browse and extract content from web pages
    - Read GitHub issues and PRs
    - Extract knowledge for ingestion into Knowledge Engine
    - Monitor documentation changes
    
    Example:
        agent = BrowserResearchAgent()
        
        # Research an error
        session = await agent.research_error(
            error_message="Z3 solver timeout",
            context="Constraint solving failure"
        )
        
        # Ingest findings into knowledge graph
        await agent.ingest_to_knowledge_engine(session, knowledge_engine)
    """
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        github_token: Optional[str] = None,
        rate_limit_delay: float = 1.0,
        cache_dir: str = "./research_cache"
    ):
        """
        Initialize Browser Research Agent.
        
        Args:
            google_api_key: Google Custom Search API key
            google_cse_id: Google Custom Search Engine ID
            github_token: GitHub personal access token
            rate_limit_delay: Delay between requests (seconds)
            cache_dir: Directory to cache research results
        """
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.github_token = github_token
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.active_sessions: Dict[str, ResearchSession] = {}
        self.research_history: List[Dict[str, Any]] = []
        
        # HTTP client
        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    'User-Agent': 'OpenEvolve-Research-Agent/1.0'
                }
            )
        else:
            self.client = None
            logger.warning("httpx not available, web research will be limited")
        
        logger.info({
            'msg': 'BrowserResearchAgent initialized',
            'google_search': bool(google_api_key and google_cse_id),
            'github_auth': bool(github_token)
        })
    
    async def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            sources: List of sources to search ['google', 'github', 'stackoverflow']
            max_results: Maximum results per source
            
        Returns:
            List of search results
        """
        sources = sources or ['google', 'github']
        results = []
        
        logger.info({
            'msg': 'Starting web search',
            'query': query,
            'sources': sources
        })
        
        for source in sources:
            try:
                if source == 'google':
                    source_results = await self._search_google(query, max_results)
                elif source == 'github':
                    source_results = await self._search_github(query, max_results)
                elif source == 'stackoverflow':
                    source_results = await self._search_stackoverflow(query, max_results)
                else:
                    continue
                
                results.extend(source_results)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error({
                    'msg': f'Search failed for {source}',
                    'error': str(e)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:max_results]
    
    async def research_error(
        self,
        error_message: str,
        context: Optional[str] = None,
        search_github: bool = True,
        search_stackoverflow: bool = True
    ) -> ResearchSession:
        """
        Research a specific error message.
        
        Args:
            error_message: The error message to research
            context: Additional context about the error
            search_github: Search GitHub issues
            search_stackoverflow: Search StackOverflow
            
        Returns:
            ResearchSession with findings
        """
        session_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(error_message.encode()).hexdigest()[:8]}"
        
        # Build search query
        query = error_message
        if context:
            query = f"{context} {error_message}"
        
        logger.info({
            'msg': 'Starting error research',
            'session_id': session_id,
            'error': error_message[:100]
        })
        
        # Determine sources
        sources = []
        if search_github:
            sources.append('github')
        if search_stackoverflow:
            sources.append('stackoverflow')
        
        # Search
        results = await self.search(query, sources=sources, max_results=10)
        
        # Visit top results and extract knowledge
        pages_visited = []
        knowledge_extracted = []
        
        for result in results[:5]:  # Visit top 5
            try:
                page = await self.fetch_page(result.url)
                if page:
                    pages_visited.append(result.url)
                    
                    # Extract relevant knowledge
                    knowledge = self._extract_error_knowledge(page, error_message)
                    if knowledge:
                        knowledge_extracted.append({
                            'source': result.url,
                            'knowledge': knowledge
                        })
                
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Failed to fetch {result.url}: {e}")
        
        # Create session
        session = ResearchSession(
            session_id=session_id,
            query=query,
            results=results,
            pages_visited=pages_visited,
            knowledge_extracted=knowledge_extracted,
            end_time=datetime.now(timezone.utc).isoformat()
        )
        
        self.active_sessions[session_id] = session
        self._log_research(session)
        
        return session
    
    async def fetch_page(self, url: str) -> Optional[PageContent]:
        """
        Fetch and extract content from a web page.
        
        Args:
            url: URL to fetch
            
        Returns:
            PageContent or None if failed
        """
        if not self.client:
            return None
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            
            if BS4_AVAILABLE:
                return self._parse_with_bs4(url, html_content)
            else:
                return self._parse_basic(url, html_content)
                
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    async def search_github_issues(
        self,
        repo: str,
        query: str,
        state: str = 'all'
    ) -> List[SearchResult]:
        """
        Search GitHub issues in a specific repository.
        
        Args:
            repo: Repository (e.g., "Z3Prover/z3")
            query: Search query
            state: Issue state ('open', 'closed', 'all')
            
        Returns:
            List of issue results
        """
        if not self.client:
            return []
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        try:
            url = f'https://api.github.com/search/issues'
            params = {
                'q': f'{query} repo:{repo}',
                'sort': 'updated',
                'order': 'desc'
            }
            
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', [])[:10]:
                results.append(SearchResult(
                    title=item['title'],
                    url=item['html_url'],
                    snippet=item.get('body', '')[:200] + '...',
                    source='github',
                    relevance_score=self._calculate_relevance(item.get('body', ''), query)
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    async def ingest_to_knowledge_engine(
        self,
        session: ResearchSession,
        knowledge_engine: Any
    ) -> bool:
        """
        Ingest research findings into the Knowledge Engine.
        
        Args:
            session: Research session with findings
            knowledge_engine: KnowledgeEngine instance
            
        Returns:
            True if ingestion successful
        """
        try:
            from knowledge_engine.integrations.kggen_integration import KGGenIntegration
            
            # Combine all knowledge
            combined_text = f"Research Query: {session.query}\n\n"
            
            for item in session.knowledge_extracted:
                combined_text += f"Source: {item['source']}\n"
                combined_text += f"Knowledge: {json.dumps(item['knowledge'])}\n\n"
            
            # Extract knowledge graph
            kg_gen = KGGenIntegration()
            kg = await kg_gen.extract_knowledge_graph(combined_text)
            
            logger.info({
                'msg': 'Ingested research to knowledge engine',
                'session_id': session.session_id,
                'entities': len(kg.entities),
                'relations': len(kg.relations)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Knowledge ingestion failed: {e}")
            return False
    
    async def _search_google(
        self,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if not self.google_api_key or not self.google_cse_id:
            return []
        
        try:
            url = 'https://www.googleapis.com/customsearch/v1'
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': min(max_results, 10)
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append(SearchResult(
                    title=item['title'],
                    url=item['link'],
                    snippet=item.get('snippet', ''),
                    source='google',
                    relevance_score=0.8
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    async def _search_github(
        self,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Search GitHub issues"""
        if not self.client:
            return []
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        try:
            url = 'https://api.github.com/search/issues'
            params = {
                'q': query,
                'sort': 'updated',
                'order': 'desc'
            }
            
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', [])[:max_results]:
                results.append(SearchResult(
                    title=f"[GitHub] {item['title']}",
                    url=item['html_url'],
                    snippet=item.get('body', '')[:200] + '...',
                    source='github',
                    relevance_score=self._calculate_relevance(item.get('body', ''), query)
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    async def _search_stackoverflow(
        self,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Search StackOverflow"""
        if not self.client:
            return []
        
        try:
            url = 'https://api.stackexchange.com/2.3/search'
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'intitle': query,
                'site': 'stackoverflow',
                'pagesize': max_results
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append(SearchResult(
                    title=f"[StackOverflow] {item['title']}",
                    url=item['link'],
                    snippet=f"Score: {item.get('score', 0)}, Answers: {item.get('answer_count', 0)}",
                    source='stackoverflow',
                    relevance_score=min(item.get('score', 0) / 10, 1.0)
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"StackOverflow search failed: {e}")
            return []
    
    def _parse_with_bs4(self, url: str, html_content: str) -> PageContent:
        """Parse HTML using BeautifulSoup"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else 'No Title'
        
        # Extract text (remove scripts and styles)
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        
        # Extract code blocks
        code_blocks = []
        for code in soup.find_all(['code', 'pre']):
            code_text = code.get_text(strip=True)
            if len(code_text) > 20:  # Only substantial code blocks
                code_blocks.append(code_text)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                links.append(href)
            elif href.startswith('/'):
                links.append(urljoin(url, href))
        
        # Extract metadata
        metadata = {
            'description': soup.find('meta', attrs={'name': 'description'}),
            'keywords': soup.find('meta', attrs={'name': 'keywords'})
        }
        
        return PageContent(
            url=url,
            title=title,
            text_content=text[:5000],  # Limit text
            code_blocks=code_blocks[:10],
            links=list(set(links))[:20],  # Deduplicate and limit
            metadata={k: str(v) if v else None for k, v in metadata.items()}
        )
    
    def _parse_basic(self, url: str, html_content: str) -> PageContent:
        """Basic HTML parsing without BeautifulSoup"""
        # Simple regex-based extraction
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
        title = title_match.group(1) if title_match else 'No Title'
        
        # Remove scripts and tags
        text = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = ' '.join(text.split())  # Normalize whitespace
        
        return PageContent(
            url=url,
            title=title,
            text_content=text[:5000],
            code_blocks=[],
            links=[],
            metadata={}
        )
    
    def _extract_error_knowledge(
        self,
        page: PageContent,
        error_message: str
    ) -> Optional[Dict[str, Any]]:
        """Extract knowledge relevant to an error"""
        knowledge = {
            'error_patterns': [],
            'solutions': [],
            'related_issues': []
        }
        
        # Look for solutions in code blocks
        for code in page.code_blocks:
            if 'fix' in code.lower() or 'solution' in code.lower():
                knowledge['solutions'].append(code[:500])
        
        # Look for error mentions
        if error_message.lower() in page.text_content.lower():
            # Extract surrounding context
            idx = page.text_content.lower().find(error_message.lower())
            start = max(0, idx - 200)
            end = min(len(page.text_content), idx + 200)
            context = page.text_content[start:end]
            
            knowledge['error_patterns'].append(context)
        
        return knowledge if any(knowledge.values()) else None
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        query_terms = query.lower().split()
        
        matches = sum(1 for term in query_terms if term in text_lower)
        return min(matches / len(query_terms), 1.0) if query_terms else 0.0
    
    def _log_research(self, session: ResearchSession):
        """Log research session"""
        self.research_history.append({
            'session_id': session.session_id,
            'query': session.query,
            'results_count': len(session.results),
            'pages_visited': len(session.pages_visited),
            'knowledge_extracted': len(session.knowledge_extracted),
            'timestamp': session.start_time
        })
        
        # Keep last 1000
        self.research_history = self.research_history[-1000:]
    
    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get research history"""
        return self.research_history.copy()
    
    async def close(self):
        """Close the browser agent"""
        if self.client:
            await self.client.aclose()
