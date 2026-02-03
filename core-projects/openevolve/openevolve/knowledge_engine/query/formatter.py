"""
Result Formatter

Formats search results for different output types and contexts.

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

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from ..hybrid.search import SearchResult

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Output format types"""
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    TABLE = "table"
    BULLET = "bullet"


@dataclass
class FormattedResult:
    """A formatted search result"""
    content: str
    format: OutputFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_string(self) -> str:
        """Get result as string"""
        return self.content


class ResultFormatter:
    """
    Formats search results for various output types
    
    Supports:
    - JSON (machine-readable)
    - Markdown (documentation)
    - Plain text (console)
    - HTML (web display)
    - Table (structured data)
    """
    
    def __init__(self):
        self.templates = self._build_templates()
    
    def _build_templates(self) -> Dict[str, str]:
        """Build format templates"""
        return {
            "header": "# Search Results\n\n",
            "result_markdown": "## {title}\n\n{content}\n\n**Score:** {score:.2f}\n\n---\n\n",
            "result_text": "{title} (Score: {score:.2f})\n{content}\n\n",
            "result_html": """
            <div class="result">
                <h3>{title}</h3>
                <p>{content}</p>
                <span class="score">Score: {score:.2f}</span>
            </div>
            """,
        }
    
    def format(
        self,
        results: List[SearchResult],
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        query: Optional[str] = None,
        include_metadata: bool = False
    ) -> FormattedResult:
        """
        Format results for output
        
        Args:
            results: Search results to format
            output_format: Desired output format
            query: Original query (for context)
            include_metadata: Whether to include metadata
        
        Returns:
            Formatted result
        """
        if output_format == OutputFormat.JSON:
            return self._format_json(results, query, include_metadata)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(results, query)
        elif output_format == OutputFormat.TEXT:
            return self._format_text(results, query)
        elif output_format == OutputFormat.HTML:
            return self._format_html(results, query)
        elif output_format == OutputFormat.TABLE:
            return self._format_table(results)
        elif output_format == OutputFormat.BULLET:
            return self._format_bullet(results)
        else:
            return self._format_text(results, query)
    
    def _format_json(
        self,
        results: List[SearchResult],
        query: Optional[str],
        include_metadata: bool
    ) -> FormattedResult:
        """Format as JSON"""
        data = {
            "query": query,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat(),
            "results": []
        }
        
        for r in results:
            result_data = {
                "id": r.id,
                "score": r.score,
                "content": r.content,
            }
            
            if include_metadata:
                result_data["metadata"] = r.metadata
                result_data["source"] = r.source
                result_data["node_type"] = r.node_type
            
            data["results"].append(result_data)
        
        content = json.dumps(data, indent=2)
        
        return FormattedResult(
            content=content,
            format=OutputFormat.JSON,
            metadata={"count": len(results)}
        )
    
    def _format_markdown(
        self,
        results: List[SearchResult],
        query: Optional[str]
    ) -> FormattedResult:
        """Format as Markdown"""
        lines = []
        
        # Header
        if query:
            lines.append(f"# Results for: \"{query}\"\n")
        else:
            lines.append("# Search Results\n")
        
        lines.append(f"Found {len(results)} results\n")
        
        # Results
        for i, r in enumerate(results, 1):
            title = r.metadata.get('name', r.metadata.get('title', f"Result {i}"))
            content = r.content[:500] + "..." if len(r.content) > 500 else r.content
            
            lines.append(f"## {i}. {title}")
            lines.append(f"**Score:** {r.score:.3f} | **Source:** {r.source}")
            if r.node_type:
                lines.append(f"**Type:** {r.node_type}")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        content = "\n".join(lines)
        
        return FormattedResult(
            content=content,
            format=OutputFormat.MARKDOWN,
            metadata={"count": len(results)}
        )
    
    def _format_text(
        self,
        results: List[SearchResult],
        query: Optional[str]
    ) -> FormattedResult:
        """Format as plain text"""
        lines = []
        
        if query:
            lines.append(f"Results for: {query}")
            lines.append("-" * 50)
        
        lines.append(f"Found {len(results)} results\n")
        
        for i, r in enumerate(results, 1):
            title = r.metadata.get('name', f"Result {i}")
            lines.append(f"{i}. {title} (Score: {r.score:.3f})")
            
            content = r.content[:200] + "..." if len(r.content) > 200 else r.content
            lines.append(f"   {content}")
            lines.append("")
        
        content = "\n".join(lines)
        
        return FormattedResult(
            content=content,
            format=OutputFormat.TEXT,
            metadata={"count": len(results)}
        )
    
    def _format_html(
        self,
        results: List[SearchResult],
        query: Optional[str]
    ) -> FormattedResult:
        """Format as HTML"""
        lines = []
        
        lines.append("<div class=\"search-results\">")
        
        if query:
            lines.append(f"<h1>Results for: {query}</h1>")
        
        lines.append(f"<p>Found {len(results)} results</p>")
        lines.append("<div class=\"results-list\">")
        
        for r in results:
            title = r.metadata.get('name', r.id)
            content = r.content[:300] + "..." if len(r.content) > 300 else r.content
            
            lines.append("<div class=\"result\">")
            lines.append(f"<h3>{title}</h3>")
            lines.append(f"<p>{content}</p>")
            lines.append(f"<span class=\"score\">Score: {r.score:.3f}</span>")
            if r.source:
                lines.append(f"<span class=\"source\">Source: {r.source}</span>")
            lines.append("</div>")
        
        lines.append("</div>")
        lines.append("</div>")
        
        content = "\n".join(lines)
        
        return FormattedResult(
            content=content,
            format=OutputFormat.HTML,
            metadata={"count": len(results)}
        )
    
    def _format_table(self, results: List[SearchResult]) -> FormattedResult:
        """Format as a table"""
        if not results:
            return FormattedResult(content="No results", format=OutputFormat.TABLE)
        
        # Header
        headers = ["Rank", "ID", "Score", "Source", "Type", "Content Preview"]
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        col_widths[5] = 40  # Content preview width
        
        # Build table
        lines = []
        
        # Header row
        header_row = " | ".join(
            h.ljust(col_widths[i])
            for i, h in enumerate(headers)
        )
        lines.append(header_row)
        lines.append("-" * len(header_row))
        
        # Data rows
        for i, r in enumerate(results, 1):
            content_preview = r.content[:col_widths[5]].replace("|", "/")
            row = [
                str(i).ljust(col_widths[0]),
                r.id[:20].ljust(col_widths[1]),
                f"{r.score:.3f}".ljust(col_widths[2]),
                r.source.ljust(col_widths[3]) if r.source else " " * col_widths[3],
                (r.node_type or "").ljust(col_widths[4]),
                content_preview.ljust(col_widths[5])
            ]
            lines.append(" | ".join(row))
        
        content = "\n".join(lines)
        
        return FormattedResult(
            content=content,
            format=OutputFormat.TABLE,
            metadata={"count": len(results)}
        )
    
    def _format_bullet(self, results: List[SearchResult]) -> FormattedResult:
        """Format as bullet points"""
        lines = []
        
        for r in results:
            title = r.metadata.get('name', r.id)
            lines.append(f"• {title} ({r.score:.2f})")
        
        content = "\n".join(lines)
        
        return FormattedResult(
            content=content,
            format=OutputFormat.BULLET,
            metadata={"count": len(results)}
        )
    
    def format_summary(
        self,
        results: List[SearchResult],
        max_length: int = 500
    ) -> str:
        """Generate a summary of results"""
        if not results:
            return "No results found."
        
        # Group by source
        by_source = {}
        for r in results:
            source = r.source or "unknown"
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(r)
        
        lines = [f"Found {len(results)} results:\n"]
        
        for source, source_results in by_source.items():
            lines.append(f"• {len(source_results)} from {source}")
        
        lines.append("")
        
        # Add top result summary
        if results:
            top = results[0]
            lines.append(f"Top result: {top.metadata.get('name', top.id)}")
            preview = top.content[:200] + "..." if len(top.content) > 200 else top.content
            lines.append(preview)
        
        summary = "\n".join(lines)
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
