"""
PostgreSQL Backend Adapter for Unified Knowledge Graph Manager.

Provides document storage using PostgreSQL with JSONB support.
License: PostgreSQL License (permissive, similar to MIT/BSD)
Follows CLAUDE.md principles: Runtime Truth, Configuration Explicitness, UTC.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
import uuid

from .base import (
    KnowledgeGraphBackend,
    BackendType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)

logger = logging.getLogger(__name__)


class PostgreSQLBackend(KnowledgeGraphBackend):
    """
    PostgreSQL backend adapter for document storage with JSONB support.
    
    Uses PostgreSQL's JSONB type for flexible document storage while maintaining
    ACID compliance and full SQL query capabilities.
    
    License: PostgreSQL License (permissive, similar to MIT/BSD)
    
    Environment Variables Required:
        POSTGRESQL_URI: PostgreSQL connection URI (e.g., postgresql://user:pass@localhost/dbname)
        POSTGRESQL_TABLE: Table name (default: knowledge_entries)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = BackendType.POSTGRESQL
        self.pool = None
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration"""
        if 'uri' not in self.config:
            raise ValueError("PostgreSQL backend requires 'uri' in config")

        self.uri = self.config['uri']
        self.table_name = self.config.get('table', 'knowledge_entries')
        self.schema = self.config.get('schema', 'public')

        logger.info(f"PostgreSQL backend configured for: {self.schema}.{self.table_name}")

    async def connect(self) -> bool:
        """Establish connection to PostgreSQL - Runtime Truth"""
        try:
            import asyncpg

            self.pool = await asyncpg.create_pool(
                self.uri,
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            # Verify connection - Runtime Truth
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
                
                # Create table if not exists
                await self._ensure_table(conn)
                
                # Create indexes
                await self._ensure_indexes(conn)

            self.is_healthy = True
            logger.info(f"Successfully connected to PostgreSQL: {self.schema}.{self.table_name}")

            return True

        except ImportError:
            logger.error("asyncpg package not installed. Install with: pip install asyncpg")
            raise ImportError("asyncpg package required for PostgreSQLBackend")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise ConnectionError(f"PostgreSQL connection failed: {e}")

    async def _ensure_table(self, conn):
        """Create table if it doesn't exist"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema}.{self.table_name} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{{}}',
            embedding FLOAT[],
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        await conn.execute(create_table_sql)
        logger.info(f"Ensured table exists: {self.schema}.{self.table_name}")

    async def _ensure_indexes(self, conn):
        """Create indexes for better performance"""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source ON {self.schema}.{self.table_name}(source);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp ON {self.schema}.{self.table_name}(timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata ON {self.schema}.{self.table_name} USING GIN (metadata);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_content_search ON {self.schema}.{self.table_name} USING GIN (to_tsvector('english', content));",
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
        
        logger.info("Created PostgreSQL indexes")

    async def disconnect(self) -> None:
        """Close PostgreSQL connection"""
        if self.pool:
            await self.pool.close()
            self.is_healthy = False
            logger.info("Disconnected from PostgreSQL")

    async def health_check(self) -> bool:
        """Check PostgreSQL health"""
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL health check failed: {e}")
            self.is_healthy = False
            return False

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Add knowledge to PostgreSQL"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        start_time = datetime.now(timezone.utc)

        try:
            entry_id = str(uuid.uuid4())
            
            async with self.pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self.schema}.{self.table_name} 
                    (id, source, content, metadata, embedding, timestamp, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    entry_id,
                    entry.source,
                    entry.content,
                    json.dumps(entry.metadata or {}),
                    entry.embedding,
                    entry.timestamp,
                    datetime.now(timezone.utc)
                )

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Added knowledge to PostgreSQL in {elapsed_ms:.2f}ms: {entry_id}")

            return entry_id

        except Exception as e:
            logger.error(f"Failed to add knowledge to PostgreSQL: {e}")
            raise ConnectionError(f"PostgreSQL add_knowledge failed: {e}")

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> SearchResults:
        """Search knowledge in PostgreSQL using full-text search"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        start_time = datetime.now(timezone.utc)

        try:
            conditions = []
            params = []
            param_idx = 1

            # Full-text search
            if query:
                conditions.append(f"to_tsvector('english', content) @@ plainto_tsquery('english', ${param_idx})")
                params.append(query)
                param_idx += 1

            # Apply filters
            if filters:
                if "source" in filters:
                    conditions.append(f"source = ${param_idx}")
                    params.append(filters["source"])
                    param_idx += 1
                    
                if "date_after" in filters:
                    conditions.append(f"timestamp >= ${param_idx}")
                    params.append(filters["date_after"])
                    param_idx += 1
                    
                if "tags" in filters:
                    conditions.append(f"metadata @> ${param_idx}")
                    params.append(json.dumps({"tags": filters["tags"]}))
                    param_idx += 1

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            async with self.pool.acquire() as conn:
                # Get total count
                count_sql = f"SELECT COUNT(*) FROM {self.schema}.{self.table_name} {where_clause}"
                total_count = await conn.fetchval(count_sql, *params)

                # Execute search with ranking
                if query:
                    search_sql = f"""
                        SELECT id, source, content, metadata, timestamp, created_at,
                               ts_rank(to_tsvector('english', content), plainto_tsquery('english', ${param_idx})) as rank
                        FROM {self.schema}.{self.table_name}
                        {where_clause}
                        ORDER BY rank DESC, timestamp DESC
                        LIMIT ${param_idx + 1} OFFSET ${param_idx + 2}
                    """
                    params.extend([query, limit, offset])
                else:
                    search_sql = f"""
                        SELECT id, source, content, metadata, timestamp, created_at
                        FROM {self.schema}.{self.table_name}
                        {where_clause}
                        ORDER BY timestamp DESC
                        LIMIT ${param_idx} OFFSET ${param_idx + 1}
                    """
                    params.extend([limit, offset])

                rows = await conn.fetch(search_sql, *params)

            results = []
            for row in rows:
                results.append({
                    "id": str(row["id"]),
                    "source": row["source"],
                    "content": row["content"],
                    "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                })

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return SearchResults(
                query=query,
                results=results,
                total_count=total_count,
                backend_used="postgresql",
                search_time_ms=elapsed_ms,
                metadata={"filters": filters}
            )

        except Exception as e:
            logger.error(f"PostgreSQL search failed: {e}")
            raise ConnectionError(f"PostgreSQL search failed: {e}")

    async def analyze(
        self,
        analysis_type: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze PostgreSQL knowledge collection"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        start_time = datetime.now(timezone.utc)

        try:
            async with self.pool.acquire() as conn:
                if analysis_type == "source_distribution":
                    rows = await conn.fetch(
                        f"""
                        SELECT source, COUNT(*) as count 
                        FROM {self.schema}.{self.table_name}
                        GROUP BY source 
                        ORDER BY count DESC
                        """
                    )
                    results = {"by_source": [{"source": row["source"], "count": row["count"]} for row in rows]}

                elif analysis_type == "tag_distribution":
                    # Extract tags from JSONB metadata
                    rows = await conn.fetch(
                        f"""
                        SELECT jsonb_array_elements_text(metadata->'tags') as tag, COUNT(*) as count
                        FROM {self.schema}.{self.table_name}
                        WHERE metadata->'tags' IS NOT NULL
                        GROUP BY tag
                        ORDER BY count DESC
                        LIMIT 20
                        """
                    )
                    results = {"by_tag": [{"tag": row["tag"], "count": row["count"]} for row in rows]}

                elif analysis_type == "temporal_analysis":
                    rows = await conn.fetch(
                        f"""
                        SELECT 
                            DATE(created_at) as date,
                            COUNT(*) as count
                        FROM {self.schema}.{self.table_name}
                        GROUP BY DATE(created_at)
                        ORDER BY date DESC
                        LIMIT 30
                        """
                    )
                    results = {"timeline": [{"date": str(row["date"]), "count": row["count"]} for row in rows]}

                elif analysis_type == "content_statistics":
                    row = await conn.fetchrow(
                        f"""
                        SELECT 
                            AVG(LENGTH(content)) as avg_length,
                            MAX(LENGTH(content)) as max_length,
                            MIN(LENGTH(content)) as min_length,
                            COUNT(*) as total_docs
                        FROM {self.schema}.{self.table_name}
                        """
                    )
                    results = {
                        "average_content_length": int(row["avg_length"] or 0),
                        "max_content_length": row["max_length"] or 0,
                        "min_content_length": row["min_length"] or 0,
                        "total_documents": row["total_docs"]
                    }

                else:
                    raise ValueError(f"Unsupported analysis type: {analysis_type}")

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return AnalysisResult(
                analysis_type=analysis_type,
                target=target or "collection",
                results=results,
                backend_used="postgresql",
                analysis_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"PostgreSQL analysis failed: {e}")
            raise ConnectionError(f"PostgreSQL analysis failed: {e}")

    async def get_statistics(self) -> GraphStatistics:
        """Get PostgreSQL collection statistics"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        try:
            async with self.pool.acquire() as conn:
                # Get total count
                total_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self.schema}.{self.table_name}"
                )

                # Get stats by source
                rows = await conn.fetch(
                    f"""
                    SELECT source, COUNT(*) as count 
                    FROM {self.schema}.{self.table_name}
                    GROUP BY source
                    ORDER BY count DESC
                    """
                )
                source_stats = {row["source"]: row["count"] for row in rows}

            return GraphStatistics(
                node_count=total_count,
                edge_count=0,  # PostgreSQL doesn't have native graph edges
                backend="postgresql",
                metadata={
                    "schema": self.schema,
                    "table": self.table_name,
                    "by_source": source_stats
                },
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to get PostgreSQL statistics: {e}")
            raise ConnectionError(f"PostgreSQL statistics failed: {e}")

    async def visualize(
        self,
        output_format: str = 'html',
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate visualization from PostgreSQL"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        try:
            async with self.pool.acquire() as conn:
                if output_format == 'json':
                    rows = await conn.fetch(
                        f"""
                        SELECT id, source, content, metadata, timestamp, created_at
                        FROM {self.schema}.{self.table_name}
                        LIMIT 100
                        """
                    )
                    
                    results = []
                    for row in rows:
                        results.append({
                            "id": str(row["id"]),
                            "source": row["source"],
                            "content": row["content"],
                            "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                            "created_at": row["created_at"].isoformat() if row["created_at"] else None
                        })

                    return json.dumps({
                        "documents": results,
                        "schema": self.schema,
                        "table": self.table_name
                    }, indent=2)

                elif output_format == 'html':
                    stats = await self.get_statistics()
                    
                    rows = await conn.fetch(
                        f"""
                        SELECT id, source, content, timestamp
                        FROM {self.schema}.{self.table_name}
                        ORDER BY timestamp DESC
                        LIMIT 10
                        """
                    )
                    
                    recent_docs = []
                    for row in rows:
                        recent_docs.append({
                            "id": str(row["id"]),
                            "source": row["source"],
                            "content": row["content"][:100] + "...",
                            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                        })

                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>PostgreSQL Knowledge Collection</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .stats {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                            .stat-item {{ margin: 10px 0; }}
                            .docs {{ margin-top: 20px; }}
                            .doc {{ background: white; padding: 10px; margin: 10px 0; border: 1px solid #ddd; }}
                        </style>
                    </head>
                    <body>
                        <h1>PostgreSQL Knowledge Collection</h1>
                        <div class="stats">
                            <div class="stat-item"><strong>Schema:</strong> {self.schema}</div>
                            <div class="stat-item"><strong>Table:</strong> {self.table_name}</div>
                            <div class="stat-item"><strong>Total Documents:</strong> {stats.node_count}</div>
                            <div class="stat-item"><strong>Sources:</strong> {len(stats.metadata.get('by_source', {}))}</div>
                        </div>
                        <h2>Recent Documents</h2>
                        <div class="docs">
                    """

                    for doc in recent_docs:
                        html += f"""
                            <div class="doc">
                                <strong>ID:</strong> {doc['id']}<br>
                                <strong>Source:</strong> {doc['source']}<br>
                                <strong>Content:</strong> {doc['content']}<br>
                                <strong>Timestamp:</strong> {doc['timestamp']}
                            </div>
                        """

                    html += """
                        </div>
                    </body>
                    </html>
                    """
                    return html

                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"PostgreSQL visualization failed: {e}")
            raise ConnectionError(f"PostgreSQL visualization failed: {e}")

    async def delete_knowledge(self, entry_id: str) -> bool:
        """Delete knowledge from PostgreSQL"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    f"DELETE FROM {self.schema}.{self.table_name} WHERE id = $1",
                    entry_id
                )
                # Result is like "DELETE 1"
                return "DELETE 1" in result

        except Exception as e:
            logger.error(f"PostgreSQL delete failed: {e}")
            raise ConnectionError(f"PostgreSQL delete failed: {e}")

    async def update_knowledge(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update knowledge in PostgreSQL"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        try:
            # Don't allow updating id
            if "id" in updates:
                del updates["id"]

            # Build dynamic update
            set_clauses = []
            params = []
            param_idx = 1
            
            for key, value in updates.items():
                if key == "metadata":
                    set_clauses.append(f"{key} = ${param_idx}::jsonb")
                else:
                    set_clauses.append(f"{key} = ${param_idx}")
                params.append(value)
                param_idx += 1
            
            # Always update updated_at
            set_clauses.append(f"updated_at = ${param_idx}")
            params.append(datetime.now(timezone.utc))
            param_idx += 1
            
            # Add id for WHERE clause
            params.append(entry_id)

            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    UPDATE {self.schema}.{self.table_name}
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_idx}
                    """,
                    *params
                )
                # Result is like "UPDATE 1"
                return "UPDATE 1" in result

        except Exception as e:
            logger.error(f"PostgreSQL update failed: {e}")
            raise ConnectionError(f"PostgreSQL update failed: {e}")

    async def clear_all(self) -> int:
        """Clear all knowledge from PostgreSQL - Destructive operation"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        try:
            async with self.pool.acquire() as conn:
                # Get count before deletion
                count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self.schema}.{self.table_name}"
                )

                # Delete all documents
                await conn.execute(f"DELETE FROM {self.schema}.{self.table_name}")

                logger.warning(f"Cleared {count} documents from PostgreSQL")
                return count

        except Exception as e:
            logger.error(f"PostgreSQL clear failed: {e}")
            raise ConnectionError(f"PostgreSQL clear failed: {e}")

    async def batch_add_knowledge(
        self,
        entries: List[KnowledgeEntry]
    ) -> List[str]:
        """Batch add knowledge to PostgreSQL efficiently"""
        if not self.is_healthy:
            raise ConnectionError("PostgreSQL backend not healthy")

        start_time = datetime.now(timezone.utc)

        try:
            ids = []
            records = []
            
            for entry in entries:
                entry_id = str(uuid.uuid4())
                ids.append(entry_id)
                
                records.append((
                    entry_id,
                    entry.source,
                    entry.content,
                    json.dumps(entry.metadata or {}),
                    entry.embedding,
                    entry.timestamp,
                    datetime.now(timezone.utc)
                ))

            async with self.pool.acquire() as conn:
                await conn.copy_records_to_table(
                    self.table_name,
                    schema_name=self.schema,
                    records=records,
                    columns=['id', 'source', 'content', 'metadata', 'embedding', 'timestamp', 'created_at']
                )

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Batch added {len(ids)} entries to PostgreSQL in {elapsed_ms:.2f}ms")

            return ids

        except Exception as e:
            logger.error(f"PostgreSQL batch add failed: {e}")
            raise ConnectionError(f"PostgreSQL batch add failed: {e}")
