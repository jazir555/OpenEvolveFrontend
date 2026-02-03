"""
Regulatory Ingestor Module
Scrapes regulatory sources, monitors RSS feeds, and tracks regulatory changes.

Author: AI Architecture Team
Date: 2026-01-30
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import re

# Try importing web scraping libraries
try:
    import aiohttp
    import feedparser
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    aiohttp = None
    feedparser = None
    BeautifulSoup = None


class SourceType(Enum):
    """Types of regulatory sources"""
    RSS_FEED = "rss_feed"
    WEB_PAGE = "web_page"
    EMAIL_ALERT = "email_alert"
    API_ENDPOINT = "api_endpoint"
    DOCUMENT_REPO = "document_repo"


@dataclass
class RegulatoryChange:
    """Represents a regulatory change"""
    source: str
    title: str
    description: str
    url: str
    published_date: datetime
    change_type: str  # 'new_rule', 'amendment', 'repeal', 'guidance'
    affected_areas: List[str] = field(default_factory=list)
    content_hash: str = ""
    raw_content: str = ""

    def __post_init__(self):
        """Generate content hash if not provided"""
        if not self.content_hash and self.raw_content:
            self.content_hash = hashlib.sha256(
                self.raw_content.encode()
            ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source': self.source,
            'title': self.title,
            'description': self.description,
            'url': self.url,
            'published_date': self.published_date.isoformat(),
            'change_type': self.change_type,
            'affected_areas': self.affected_areas,
            'content_hash': self.content_hash,
            'raw_content': self.raw_content[:1000]  # Truncate for storage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegulatoryChange':
        """Create from dictionary"""
        return cls(
            source=data['source'],
            title=data['title'],
            description=data['description'],
            url=data['url'],
            published_date=datetime.fromisoformat(data['published_date']),
            change_type=data['change_type'],
            affected_areas=data.get('affected_areas', []),
            content_hash=data.get('content_hash', ''),
            raw_content=data.get('raw_content', '')
        )


@dataclass
class RegulationDocument:
    """Represents a full regulatory document"""
    document_id: str
    title: str
    source: str
    url: str
    version: str
    effective_date: Optional[datetime] = None
    content: str = ""
    related_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegulatoryIngestor:
    """
    Scrapes regulatory sources and tracks changes

    Monitors:
    - SEC releases and updates
    - FINRA rule changes
    - ESMA regulations
    - Custom RSS feeds
    - Email alerts

    Example:
        >>> ingestor = RegulatoryIngestor(sources=["https://sec.gov/rss"])
        >>> changes = await ingestor.scan_sources()
        >>> print(f"Found {len(changes)} changes")
    """

    def __init__(
        self,
        sources: Optional[List[str]] = None,
        cache_dir: str = "./cache/regulatory",
        check_interval_hours: int = 24,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize regulatory ingestor

        Args:
            sources: List of regulatory source URLs
            cache_dir: Directory for caching regulatory documents
            check_interval_hours: Hours between checks (for caching)
            logger: Logger instance
        """
        self.sources = sources or self._get_default_sources()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.check_interval = timedelta(hours=check_interval_hours)

        self.logger = logger or self._setup_logging()

        # Track seen changes to avoid duplicates
        self.seen_hashes: Set[str] = set()
        self._load_seen_hashes()

        # Store changes
        self.changes: List[RegulatoryChange] = []

        # Check if web scraping is available
        if not WEB_SCRAPING_AVAILABLE:
            self.logger.warning(
                "Web scraping libraries not available. "
                "Install aiohttp, feedparser, and beautifulsoup4 for full functionality."
            )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("RegulatoryIngestor")
        logger.setLevel(logging.INFO)
        return logger

    def _get_default_sources(self) -> List[str]:
        """Get default regulatory sources"""
        return [
            # SEC
            "https://www.sec.gov/news/pressreleases.rss",
            "https://www.sec.gov/rules/final/htm",

            # FINRA
            "https://www.finra.org/rules-guidance/rulebooks",
            "https://www.finra.org/newsroom",

            # ESMA
            "https://www.esma.europa.eu/press-releases",
        ]

    def _load_seen_hashes(self):
        """Load previously seen content hashes"""
        hash_file = self.cache_dir / "seen_hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    self.seen_hashes = set(json.load(f))
            except Exception as e:
                self.logger.error(f"Failed to load seen hashes: {e}")

    def _save_seen_hashes(self):
        """Save seen content hashes"""
        hash_file = self.cache_dir / "seen_hashes.json"
        try:
            with open(hash_file, 'w') as f:
                json.dump(list(self.seen_hashes), f)
        except Exception as e:
            self.logger.error(f"Failed to save seen hashes: {e}")

    async def scan_sources(self) -> List[Dict[str, Any]]:
        """
        Scan all configured sources for changes

        Returns:
            List of new regulatory changes (as dictionaries)
        """
        self.logger.info(f"Scanning {len(self.sources)} regulatory sources")

        new_changes = []

        for source in self.sources:
            try:
                source_type = self._detect_source_type(source)

                if source_type == SourceType.RSS_FEED:
                    changes = await self._scan_rss_feed(source)
                elif source_type == SourceType.WEB_PAGE:
                    changes = await self._scan_web_page(source)
                else:
                    self.logger.warning(f"Unsupported source type: {source_type}")
                    changes = []

                # Filter new changes
                for change in changes:
                    if change.content_hash not in self.seen_hashes:
                        new_changes.append(change)
                        self.seen_hashes.add(change.content_hash)

                self.logger.info(f"Found {len(changes)} changes from {source}")

            except Exception as e:
                self.logger.error(f"Error scanning {source}: {e}", exc_info=True)

        # Store changes
        self.changes.extend(new_changes)
        self._save_seen_hashes()
        self._cache_changes(new_changes)

        # Convert to dictionaries
        return [change.to_dict() for change in new_changes]

    def _detect_source_type(self, source: str) -> SourceType:
        """Detect the type of regulatory source"""
        if source.endswith('.rss') or 'rss' in source.lower():
            return SourceType.RSS_FEED
        elif source.startswith('http'):
            return SourceType.WEB_PAGE
        else:
            return SourceType.WEB_PAGE  # Default

    async def _scan_rss_feed(self, url: str) -> List[RegulatoryChange]:
        """Scan RSS feed for changes"""
        if not WEB_SCRAPING_AVAILABLE:
            self.logger.warning("RSS scanning not available")
            return []

        changes = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        self.logger.error(f"HTTP {response.status} for {url}")
                        return []

                    feed_content = await response.text()

            # Parse RSS feed
            feed = feedparser.parse(feed_content)

            for entry in feed.entries[:50]:  # Limit to 50 most recent
                # Parse change type from title/category
                change_type = self._parse_change_type(entry.get('title', ''))

                # Extract affected areas
                affected_areas = self._extract_affected_areas(
                    entry.get('title', '') + ' ' + entry.get('description', '')
                )

                change = RegulatoryChange(
                    source=url,
                    title=entry.get('title', ''),
                    description=entry.get('description', ''),
                    url=entry.get('link', ''),
                    published_date=self._parse_date(entry.get('published')),
                    change_type=change_type,
                    affected_areas=affected_areas,
                    raw_content=entry.get('description', '')
                )
                changes.append(change)

        except Exception as e:
            self.logger.error(f"Error scanning RSS feed {url}: {e}")

        return changes

    async def _scan_web_page(self, url: str) -> List[RegulatoryChange]:
        """Scan web page for regulatory changes"""
        if not WEB_SCRAPING_AVAILABLE:
            self.logger.warning("Web page scanning not available")
            return []

        changes = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        self.logger.error(f"HTTP {response.status} for {url}")
                        return []

                    html = await response.text()

            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')

            # Look for recent releases/updates
            # This is a simplified example - real implementation would be more sophisticated
            articles = soup.find_all(['article', 'div'], class_=['release', 'update', 'news-item'])

            for article in articles[:20]:  # Limit to 20
                title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                link = article.find('a')
                url = link.get('href', '') if link else url

                # Ensure absolute URL
                if url and not url.startswith('http'):
                    base_url = '/'.join(url.split('/')[:3])
                    url = base_url + url

                change_type = self._parse_change_type(title)
                affected_areas = self._extract_affected_areas(title)

                change = RegulatoryChange(
                    source=url,
                    title=title,
                    description=article.get_text(strip=True)[:500],
                    url=url,
                    published_date=datetime.utcnow(),
                    change_type=change_type,
                    affected_areas=affected_areas,
                    raw_content=article.get_text(strip=True)
                )
                changes.append(change)

        except Exception as e:
            self.logger.error(f"Error scanning web page {url}: {e}")

        return changes

    def _parse_change_type(self, text: str) -> str:
        """Parse change type from text"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['repeal', 'rescinded', 'withdrawn']):
            return 'repeal'
        elif any(word in text_lower for word in ['amend', 'amendment', 'modification', 'revision']):
            return 'amendment'
        elif any(word in text_lower for word in ['guidance', 'interpretation', 'faq']):
            return 'guidance'
        else:
            return 'new_rule'

    def _extract_affected_areas(self, text: str) -> List[str]:
        """Extract affected regulatory areas from text"""
        areas = []

        # Common regulatory keywords
        keywords = {
            'trading': ['trading', 'market', 'execution', 'order'],
            'reporting': ['reporting', 'disclosure', 'filing', 'form'],
            'risk': ['risk', 'margin', 'capital', 'liquidity'],
            'investor': ['investor', 'customer', 'retail', 'client'],
            'crypto': ['crypto', 'digital asset', 'virtual currency'],
            'esg': ['esg', 'environmental', 'social', 'governance'],
        }

        text_lower = text.lower()

        for area, terms in keywords.items():
            if any(term in text_lower for term in terms):
                areas.append(area)

        return areas

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse date from various formats"""
        if not date_str:
            return datetime.utcnow()

        # Try common formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',  # RFC 2822
            '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
            '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 UTC
            '%Y-%m-%d',                  # Simple date
            '%d %b %Y',                  # Day Month Year
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue

        return datetime.utcnow()

    def _cache_changes(self, changes: List[RegulatoryChange]):
        """Cache changes to disk"""
        cache_file = self.cache_dir / f"changes_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump([c.to_dict() for c in changes], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to cache changes: {e}")

    async def has_changes(self) -> bool:
        """Check if there are unprocessed changes"""
        return len(self.changes) > 0

    async def get_changes(self) -> List[Dict[str, Any]]:
        """Get all pending changes"""
        return [change.to_dict() for change in self.changes]

    async def ingest_changes(self, changes: List[Dict[str, Any]]):
        """Manually ingest changes (e.g., from API or email)"""
        for change_dict in changes:
            change = RegulatoryChange.from_dict(change_dict)
            if change.content_hash not in self.seen_hashes:
                self.changes.append(change)
                self.seen_hashes.add(change.content_hash)

        self._save_seen_hashes()
        self._cache_changes(self.changes)

    async def parse_regulatory_document(
        self,
        document_url: str,
        document_id: str
    ) -> Optional[RegulationDocument]:
        """
        Parse and store a full regulatory document

        Args:
            document_url: URL to the document
            document_id: Unique identifier for the document

        Returns:
            Parsed RegulationDocument or None
        """
        if not WEB_SCRAPING_AVAILABLE:
            self.logger.warning("Document parsing not available")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(document_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        return None

                    content = await response.text()

            # Parse content
            soup = BeautifulSoup(content, 'html.parser')

            # Extract title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else document_id

            # Extract main content
            # This is simplified - real implementation would be more sophisticated
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            content_text = main_content.get_text(strip=True) if main_content else content

            # Extract rules (simplified)
            rules = re.findall(r'(Rule \d+|§ \d+|Section \d+)', content_text)

            document = RegulationDocument(
                document_id=document_id,
                title=title_text,
                source=document_url,
                url=document_url,
                version="1.0",
                content=content_text,
                related_rules=rules
            )

            # Cache document
            doc_file = self.cache_dir / f"doc_{document_id}.json"
            with open(doc_file, 'w') as f:
                json.dump({
                    'document_id': document.document_id,
                    'title': document.title,
                    'source': document.source,
                    'url': document.url,
                    'version': document.version,
                    'content': document.content[:10000],  # Truncate
                    'related_rules': document.related_rules
                }, f)

            return document

        except Exception as e:
            self.logger.error(f"Error parsing document {document_url}: {e}")
            return None

    async def get_document_history(self, document_id: str) -> List[RegulationDocument]:
        """
        Get version history for a document

        Args:
            document_id: Document identifier

        Returns:
            List of document versions
        """
        # This would query the cache for all versions of a document
        # Simplified implementation
        return []

    async def clear_processed_changes(self):
        """Clear processed changes from memory"""
        self.changes.clear()
