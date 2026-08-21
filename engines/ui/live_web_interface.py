"""
Live Web Interface - Web Browsing & Research Agent

Provides headless browser capabilities for live web research,
documentation lookup, and knowledge ingestion. Enables the system
to leave the simulation and gather fresh information.

Key Features:
- Headless browser automation (Playwright/Puppeteer)
- GitHub Issues and documentation crawling
- Knowledge extraction and ingestion into OneKE
- Automated research workflows
- MultiOn integration for AI-powered browsing
"""
from __future__ import annotations


import os
import re
import json
import time
import asyncio
import hashlib
import logging
from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, urldefrag
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class BrowserEngine(Enum):
    """Supported browser engines"""
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    MULTION = "multion"


class ResearchDepth(Enum):
    """Depth of research to perform"""
    SHALLOW = "shallow"      # Single page
    STANDARD = "standard"    # Page + linked pages
    DEEP = "deep"           # Recursive crawling
    COMPREHENSIVE = "comprehensive"  # Full site analysis


@dataclass
class WebPage:
    """Represents a crawled web page"""
    url: str
    title: str
    content: str
    text_content: str
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    crawl_depth: int = 0
    
    @property
    def content_hash(self) -> str:
        """Generate hash of content for deduplication"""
        return hashlib.md5(self.content.encode()).hexdigest()[:16]
    
    def to_knowledge_entry(self) -> Dict[str, Any]:
        """Convert to knowledge graph entry format"""
        return {
            "source": self.url,
            "title": self.title,
            "content": self.text_content[:5000],  # Truncate for storage
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ResearchQuery:
    """Research query specification"""
    query: str
    target_sources: List[str] = field(default_factory=list)
    max_results: int = 10
    depth: ResearchDepth = ResearchDepth.STANDARD
    time_range: Optional[str] = None  # "day", "week", "month", "year"
    required_fields: List[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    """Result of web research"""
    query: str
    pages: List[WebPage] = field(default_factory=list)
    summary: str = ""
    key_findings: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_crawled: int = 0
    total_filtered: int = 0
    execution_time_seconds: float = 0.0
    
    def to_knowledge_artifact(self) -> Dict[str, Any]:
        """Convert to knowledge artifact for ingestion"""
        return {
            "type": "web_research",
            "query": self.query,
            "summary": self.summary,
            "sources": [p.url for p in self.pages],
            "findings": self.key_findings,
            "timestamp": self.timestamp.isoformat(),
            "content": "\n\n".join(p.text_content[:2000] for p in self.pages)
        }


@dataclass
class BrowserConfig:
    """Configuration for browser"""
    engine: BrowserEngine = BrowserEngine.PLAYWRIGHT
    headless: bool = True
    user_agent: str = "OpenEvolve Research Bot 1.0"
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout_ms: int = 30000
    wait_for_load: bool = True
    javascript_enabled: bool = True
    cookies: Dict[str, str] = field(default_factory=dict)


class WebBrowser:
    """Headless web browser using Playwright or Selenium"""
    
    def __init__(self, config: BrowserConfig = None):
        self.config = config or BrowserConfig()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._session_cookies: List[Dict] = []
    
    async def initialize(self):
        """Initialize browser"""
        if self.config.engine == BrowserEngine.PLAYWRIGHT:
            await self._init_playwright()
        elif self.config.engine == BrowserEngine.SELENIUM:
            self._init_selenium()
    
    async def _init_playwright(self):
        """Initialize Playwright browser"""
        try:
            from playwright.async_api import async_playwright
            
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless
            )
            self._context = await self._browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                },
                user_agent=self.config.user_agent
            )
            
            # Add cookies if provided
            if self.config.cookies:
                await self._context.add_cookies([
                    {"name": k, "value": v, "domain": ".github.com", "path": "/"}
                    for k, v in self.config.cookies.items()
                ])
            
            self._page = await self._context.new_page()
            
        except ImportError:
            raise RuntimeError("Playwright not installed. Run: pip install playwright")
    
    def _init_selenium(self):
        """Initialize Selenium browser"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            
            chrome_options = Options()
            if self.config.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"--window-size={self.config.viewport_width},{self.config.viewport_height}")
            chrome_options.add_argument(f"--user-agent={self.config.user_agent}")
            
            self._browser = webdriver.Chrome(options=chrome_options)
            
        except ImportError:
            raise RuntimeError("Selenium not installed. Run: pip install selenium")
    
    async def navigate(self, url: str, wait_for_selector: str = None) -> WebPage:
        """Navigate to URL and return page content"""
        if self.config.engine == BrowserEngine.PLAYWRIGHT:
            return await self._navigate_playwright(url, wait_for_selector)
        else:
            return self._navigate_selenium(url)
    
    async def _navigate_playwright(
        self,
        url: str,
        wait_for_selector: str = None
    ) -> WebPage:
        """Navigate using Playwright"""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        
        response = await self._page.goto(
            url,
            wait_until="networkidle" if self.config.wait_for_load else "load",
            timeout=self.config.timeout_ms
        )
        
        if wait_for_selector:
            await self._page.wait_for_selector(wait_for_selector, timeout=10000)
        
        # Extract page data
        title = await self._page.title()
        content = await self._page.content()
        
        # Extract text content
        text_content = await self._page.evaluate("""
            () => document.body.innerText
                .replace(/\\s+/g, ' ')
                .trim()
                .substring(0, 50000)
        """)
        
        # Extract links
        links = await self._page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                .map(a => a.href)
                .filter(href => href.startsWith('http'))
        """)
        
        # Extract metadata
        metadata = await self._page.evaluate("""
            () => ({
                description: document.querySelector('meta[name="description"]')?.content || '',
                author: document.querySelector('meta[name="author"]')?.content || '',
                keywords: document.querySelector('meta[name="keywords"]')?.content || '',
                og_title: document.querySelector('meta[property="og:title"]')?.content || ''
            })
        """)
        
        return WebPage(
            url=url,
            title=title,
            content=content,
            text_content=text_content,
            links=list(set(links))[:100],  # Limit links
            metadata=metadata
        )
    
    def _navigate_selenium(self, url: str) -> WebPage:
        """Navigate using Selenium"""
        self._browser.get(url)
        
        from selenium.webdriver.common.by import By
        
        title = self._browser.title
        content = self._browser.page_source
        
        # Extract text
        body = self._browser.find_element(By.TAG_NAME, "body")
        text_content = body.text[:50000]
        
        # Extract links
        links = [
            elem.get_attribute("href")
            for elem in self._browser.find_elements(By.TAG_NAME, "a")
            if elem.get_attribute("href")
        ]
        
        return WebPage(
            url=url,
            title=title,
            content=content,
            text_content=text_content,
            links=list(set(links))[:100]
        )
    
    async def search_github_issues(
        self,
        repo: str,
        query: str,
        max_results: int = 10
    ) -> List[WebPage]:
        """Search GitHub issues for a repository"""
        search_url = f"https://github.com/{repo}/issues?q={query.replace(' ', '+')}"
        
        page = await self.navigate(search_url)
        
        # Extract issue links
        issue_pattern = rf"https://github\.com/{re.escape(repo)}/issues/\\d+"
        issue_links = list(set(
            re.findall(issue_pattern, page.content)
        ))[:max_results]
        
        pages = [page]
        
        # Fetch individual issues
        for link in issue_links[:max_results]:
            try:
                issue_page = await self.navigate(link)
                pages.append(issue_page)
                await asyncio.sleep(0.5)  # Be polite
            except Exception as e:
                logger.warning(f"Failed to fetch issue {link}: {e}")
        
        return pages
    
    async def extract_code_blocks(self, page: WebPage) -> List[Dict[str, Any]]:
        """Extract code blocks from a page"""
        if self.config.engine != BrowserEngine.PLAYWRIGHT:
            return []
        
        code_blocks = await self._page.evaluate("""
            () => Array.from(document.querySelectorAll('pre code, .highlight pre, .code-block'))
                .map(block => ({
                    language: block.className.match(/language-(\\w+)/)?.[1] || 
                              block.className.match(/\\b(\\w+)$/)?.[1] || 'text',
                    code: block.textContent.trim()
                }))
                .filter(block => block.code.length > 10)
        """)
        
        return code_blocks
    
    async def take_screenshot(self, selector: str = None) -> bytes:
        """Take screenshot of current page"""
        if self.config.engine == BrowserEngine.PLAYWRIGHT:
            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    return await element.screenshot()
            return await self._page.screenshot(full_page=True)
        else:
            return self._browser.get_screenshot_as_png()
    
    async def close(self):
        """Close browser"""
        if self.config.engine == BrowserEngine.PLAYWRIGHT:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        else:
            if self._browser:
                self._browser.quit()


class MultiOnBrowser:
    """Integration with MultiOn for AI-powered browsing"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MULTION_API_KEY")
        self._session_id: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        """Check if MultiOn is available"""
        return self.api_key is not None
    
    async def browse(self, task: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Use MultiOn to perform a browsing task
        
        Args:
            task: Natural language description of the task
            url: Starting URL (optional)
            
        Returns:
            Result of the browsing task
        """
        if not self.is_available:
            raise RuntimeError("MultiOn API key not configured")
        
        try:
            import multion
            
            client = multion.Client(api_key=self.api_key)
            
            if self._session_id:
                # Continue existing session
                response = client.step(
                    session_id=self._session_id,
                    cmd=task
                )
            else:
                # Start new session
                response = client.browse(
                    cmd=task,
                    url=url,
                    local=True
                )
                self._session_id = response.get("session_id")
            
            return {
                "status": response.get("status"),
                "message": response.get("message"),
                "url": response.get("url"),
                "screenshot": response.get("screenshot"),
                "session_id": self._session_id
            }
            
        except ImportError:
            raise RuntimeError("MultiOn not installed. Run: pip install multion")
        except Exception as e:
            logger.error(f"MultiOn browsing failed: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close MultiOn session"""
        if self._session_id:
            try:
                import multion
                client = multion.Client(api_key=self.api_key)
                client.close_session(self._session_id)
            except Exception as e:
                logger.warning(f"Failed to close MultiOn session: {e}")


class KnowledgeIngestor:
    """Ingests web content into the Knowledge Engine"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path or "./knowledge_base"
        self._ingested_hashes: set = set()
    
    async def ingest_webpage(self, page: WebPage) -> bool:
        """Ingest a web page into the knowledge base"""
        content_hash = page.content_hash
        
        if content_hash in self._ingested_hashes:
            logger.info(f"Skipping duplicate content: {page.url}")
            return False
        
        # Store in knowledge base
        knowledge_entry = page.to_knowledge_entry()
        
        # Save to file (in production, this would go to Graphiti/DeepKE)
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        
        filename = f"{hashlib.md5(page.url.encode()).hexdigest()[:16]}.json"
        filepath = os.path.join(self.knowledge_base_path, filename)
        
        with open(filepath, "w") as f:
            json.dump(knowledge_entry, f, indent=2)
        
        self._ingested_hashes.add(content_hash)
        logger.info(f"Ingested: {page.url}")
        
        return True
    
    async def ingest_research_result(self, result: ResearchResult) -> int:
        """Ingest all pages from a research result"""
        ingested_count = 0
        
        for page in result.pages:
            if await self.ingest_webpage(page):
                ingested_count += 1
        
        # Save research summary
        if result.summary:
            summary_file = os.path.join(
                self.knowledge_base_path,
                f"research_{int(time.time())}.json"
            )
            with open(summary_file, "w") as f:
                json.dump(result.to_knowledge_artifact(), f, indent=2)
        
        return ingested_count


class ResearchAgent:
    """
    Main Research Agent - Live Web Interface
    
    Performs web research, crawls documentation,
    and ingests fresh knowledge into the system.
    """
    
    def __init__(
        self,
        browser_config: BrowserConfig = None,
        enable_multion: bool = False
    ):
        self.browser_config = browser_config or BrowserConfig()
        self.browser = WebBrowser(self.browser_config)
        self.multion = MultiOnBrowser() if enable_multion else None
        self.ingestor = KnowledgeIngestor()
        self._rate_limiter = asyncio.Semaphore(3)  # Max 3 concurrent requests
    
    async def initialize(self):
        """Initialize the research agent"""
        await self.browser.initialize()
    
    async def research(
        self,
        query: ResearchQuery
    ) -> ResearchResult:
        """
        Perform comprehensive web research
        
        Args:
            query: Research query specification
            
        Returns:
            ResearchResult with findings
        """
        start_time = time.time()
        
        # Determine search strategy based on target sources
        if "github.com" in str(query.target_sources):
            pages = await self._research_github(query)
        elif "docs" in query.query.lower() or "documentation" in query.query.lower():
            pages = await self._research_documentation(query)
        else:
            pages = await self._research_general(query)
        
        # Filter and deduplicate
        unique_pages = self._deduplicate_pages(pages)
        
        # Extract key findings
        key_findings = self._extract_findings(unique_pages, query)
        
        # Generate summary
        summary = self._generate_summary(unique_pages, query)
        
        execution_time = time.time() - start_time
        
        result = ResearchResult(
            query=query.query,
            pages=unique_pages[:query.max_results],
            summary=summary,
            key_findings=key_findings,
            total_crawled=len(pages),
            total_filtered=len(pages) - len(unique_pages),
            execution_time_seconds=execution_time
        )
        
        # Auto-ingest into knowledge base
        await self.ingestor.ingest_research_result(result)
        
        return result
    
    async def _research_github(self, query: ResearchQuery) -> List[WebPage]:
        """Research on GitHub"""
        pages = []
        
        # Parse repo from query or target sources
        repo = None
        for source in query.target_sources:
            if "github.com" in source:
                match = re.search(r"github\.com/([^/]+/[^/]+)", source)
                if match:
                    repo = match.group(1)
                    break
        
        if repo:
            # Search issues
            issue_pages = await self.browser.search_github_issues(
                repo,
                query.query,
                query.max_results
            )
            pages.extend(issue_pages)
        
        return pages
    
    async def _research_documentation(self, query: ResearchQuery) -> List[WebPage]:
        """Research documentation sites"""
        pages = []
        
        # Crawl documentation pages
        for source in query.target_sources:
            async with self._rate_limiter:
                try:
                    page = await self.browser.navigate(source)
                    pages.append(page)
                    
                    # Crawl linked pages if depth allows
                    if query.depth in [ResearchDepth.DEEP, ResearchDepth.COMPREHENSIVE]:
                        for link in page.links[:5]:  # Limit to 5 linked pages
                            if any(doc_site in link for doc_site in ["docs.", "documentation"]):
                                try:
                                    linked_page = await self.browser.navigate(link)
                                    linked_page.crawl_depth = 1
                                    pages.append(linked_page)
                                    await asyncio.sleep(1)
                                except Exception as e:
                                    logger.warning(f"Failed to crawl {link}: {e}")
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Failed to navigate {source}: {e}")
        
        return pages
    
    async def _research_general(self, query: ResearchQuery) -> List[WebPage]:
        """General web research"""
        pages = []
        
        for source in query.target_sources:
            async with self._rate_limiter:
                try:
                    page = await self.browser.navigate(source)
                    pages.append(page)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f"Failed to navigate {source}: {e}")
        
        return pages
    
    def _deduplicate_pages(self, pages: List[WebPage]) -> List[WebPage]:
        """Remove duplicate pages based on content hash"""
        seen_hashes = set()
        unique = []
        
        for page in pages:
            if page.content_hash not in seen_hashes:
                seen_hashes.add(page.content_hash)
                unique.append(page)
        
        return unique
    
    def _extract_findings(
        self,
        pages: List[WebPage],
        query: ResearchQuery
    ) -> List[Dict[str, Any]]:
        """Extract key findings from pages"""
        findings = []
        
        for page in pages:
            # Simple keyword-based extraction
            text_lower = page.text_content.lower()
            query_terms = query.query.lower().split()
            
            relevance_score = sum(
                1 for term in query_terms if term in text_lower
            ) / len(query_terms)
            
            if relevance_score > 0.3:
                findings.append({
                    "source": page.url,
                    "title": page.title,
                    "relevance": relevance_score,
                    "snippet": page.text_content[:500] + "..." if len(page.text_content) > 500 else page.text_content
                })
        
        # Sort by relevance
        findings.sort(key=lambda x: x["relevance"], reverse=True)
        return findings[:10]
    
    def _generate_summary(self, pages: List[WebPage], query: ResearchQuery) -> str:
        """Generate research summary"""
        if not pages:
            return "No results found."
        
        summary = f"""
Research Query: {query.query}
Sources Analyzed: {len(pages)}
Key Sources:
"""
        for page in pages[:5]:
            summary += f"- {page.title}: {page.url}\n"
        
        return summary
    
    async def fetch_error_solution(
        self,
        error_message: str,
        context: str = ""
    ) -> Optional[ResearchResult]:
        """
        Fetch solution for an error from the web
        
        This is used when Blue Team hits an error they haven't seen.
        Instead of hallucinating a fix, they research the actual solution.
        
        Args:
            error_message: The error message to search for
            context: Additional context about the error
            
        Returns:
            ResearchResult with potential solutions
        """
        # Clean error message for search
        clean_error = re.sub(r'\\s+', ' ', error_message)[:200]
        
        # Determine likely sources
        sources = []
        if "z3" in error_message.lower():
            sources.append("https://github.com/Z3Prover/z3/issues")
        if "python" in error_message.lower():
            sources.append("https://stackoverflow.com/search")
        
        query = ResearchQuery(
            query=f"{clean_error} {context}",
            target_sources=sources or ["https://stackoverflow.com/search"],
            max_results=5,
            depth=ResearchDepth.SHALLOW
        )
        
        return await self.research(query)
    
    async def monitor_documentation_updates(
        self,
        doc_urls: List[str],
        interval_hours: int = 24
    ) -> AsyncGenerator[List[WebPage], None]:
        """
        Monitor documentation for updates
        
        Yields pages that have changed since last check.
        """
        last_hashes: Dict[str, str] = {}
        
        while True:
            changed_pages = []
            
            for url in doc_urls:
                try:
                    page = await self.browser.navigate(url)
                    current_hash = page.content_hash
                    
                    if url in last_hashes and last_hashes[url] != current_hash:
                        changed_pages.append(page)
                        logger.info(f"Documentation updated: {url}")
                    
                    last_hashes[url] = current_hash
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to check {url}: {e}")
            
            if changed_pages:
                yield changed_pages
            
            await asyncio.sleep(interval_hours * 3600)
    
    async def close(self):
        """Cleanup resources"""
        await self.browser.close()


# Convenience functions for quick usage
async def quick_research(
    query: str,
    sources: List[str] = None
) -> ResearchResult:
    """Quick research function"""
    agent = ResearchAgent()
    await agent.initialize()
    
    try:
        research_query = ResearchQuery(
            query=query,
            target_sources=sources or [],
            max_results=5
        )
        return await agent.research(research_query)
    finally:
        await agent.close()


# Example usage
if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("LIVE WEB INTERFACE DEMO - Research Agent")
        print("=" * 60)
        
        # Initialize agent
        agent = ResearchAgent()
        await agent.initialize()
        
        print("\n[OK] Research Agent initialized")
        print(f"  Browser: {agent.browser_config.engine.value}")
        print(f"  Headless: {agent.browser_config.headless}")
        
        print("\n" + "=" * 60)
        print("Example Use Cases:")
        print("  1. Blue Team hits Z3 error")
        print("     -> Research GitHub Issues for solutions")
        print("  2. New library documentation needed")
        print("     -> Crawl docs and ingest into OneKE")
        print("  3. Monitor documentation for updates")
        print("  4. MultiOn AI-powered browsing")
        
        await agent.close()
        print("\n[OK] Demo complete")
    
    asyncio.run(demo())
