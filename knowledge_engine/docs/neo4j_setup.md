# Neo4j Setup Guide (DEPRECATED - ORPHANED)

**⚠️ IMPORTANT NOTICE ⚠️**

This guide is **DEPRECATED** and **ARCHIVED** for reference only.

**Neo4j has been completely removed from the active codebase.**

- All Neo4j references removed from `enhanced_storage.py`
- All Neo4j references removed from `knowledge_storage.py`
- All Neo4j references removed from `real_database_integration.py`
- Zero GPL Neo4j database code remains in active code path
- The `neo4j_backend.py` file exists as an orphaned adapter (not imported or used)

## Use Memgraph Instead

**Migration completed:** Neo4j → Memgraph

| Feature | Neo4j (Orphaned) | Memgraph (Active) |
|---------|------------------|-------------------|
| **License** | GPL (copyleft) | Apache 2.0 (permissive) |
| **Status** | Orphaned, zero references | Active, fully supported |
| **Cypher Support** | Native | ✅ Fully compatible |
| **Bolt Protocol** | Native | ✅ Compatible |
| **Python Driver** | `neo4j` | ✅ Same `neo4j` driver |

## Migration Resources

- **Migration Guide:** `NEO4J_TO_MEMGRAPH_MIGRATION.md`
- **Memgraph Setup:** See Memgraph documentation
- **License Compliance:** `LICENSE_COMPLIANCE_REPORT.md`

## Technical Note

The `neo4j` Python driver package is still used by the codebase, but **only for Memgraph connectivity**:
- Memgraph is Apache 2.0 licensed (permissive)
- Memgraph uses the Bolt protocol (same as Neo4j)
- The `neo4j` Python driver is Apache 2.0 licensed
- No GPL Neo4j database features are used

---

*This document is preserved for historical reference only.*
*Last updated: 2026-01-30*
