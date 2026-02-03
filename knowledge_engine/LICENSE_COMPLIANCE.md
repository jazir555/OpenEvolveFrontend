# OpenEvolve Knowledge Engine - License Compliance

**Date:** 2026-02-03  
**Status:** ✅ COMPLIANT - All Permissive Licenses

---

## License Policy

The OpenEvolve Knowledge Engine **ONLY** uses dependencies with permissive open-source licenses:

### ✅ ACCEPTED LICENSES
- **MIT** - Massachusetts Institute of Technology License
- **Apache 2.0** - Apache License 2.0
- **BSD** (2-Clause, 3-Clause) - Berkeley Software Distribution
- **PostgreSQL** - PostgreSQL License
- **MPL 2.0** - Mozilla Public License 2.0

### ❌ REJECTED LICENSES
- **GPL** (v1, v2, v3) - GNU General Public License
- **AGPL** - GNU Affero General Public License
- **LGPL** - GNU Lesser General Public License
- **SSPL** - Server Side Public License (MongoDB, Elasticsearch)
- **Commercial/Proprietary** - Any closed-source licenses

---

## Backend Licenses

| Backend | License | Status |
|---------|---------|--------|
| PostgreSQL | PostgreSQL License | ✅ Approved |
| Memgraph | Apache 2.0 | ✅ Approved |
| Qdrant | Apache 2.0 | ✅ Approved |
| Redis | BSD 3-Clause | ✅ Approved |
| KarateClub | MIT | ✅ Approved |
| In-Memory | MIT | ✅ Approved |
| **Neo4j** | **GPL v3** | ❌ **BLOCKED** |
| **MongoDB** | **SSPL** | ❌ **BLOCKED** |
| **Elasticsearch** | **SSPL** | ❌ **BLOCKED** |

### Alternatives for Blocked Backends

Instead of **Neo4j (GPL)**, use:
- **Memgraph** (Apache 2.0) - Drop-in Cypher-compatible replacement
- **PostgreSQL** with graph extensions (PostgreSQL License)

Instead of **MongoDB (SSPL)**, use:
- **PostgreSQL** (PostgreSQL License) - JSON/JSONB support
- **Cassandra** (Apache 2.0) - For distributed document storage

Instead of **Elasticsearch (SSPL)**, use:
- **OpenSearch** (Apache 2.0) - Fork of Elasticsearch 7.10
- **Typesense** (GPL v3 for certain features, check version) - Use with caution
- **Meilisearch** (MIT) - Alternative search engine

---

## Dependency Licenses

### Core Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| numpy | BSD 3-Clause | Numerical computing |
| sentence-transformers | Apache 2.0 | Embedding generation |
| psutil | BSD 3-Clause | System monitoring |
| asyncpg | Apache 2.0 | PostgreSQL async driver |

### Cloud Storage Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| boto3 | Apache 2.0 | AWS S3 integration |
| google-cloud-storage | Apache 2.0 | GCS integration |
| azure-storage-blob | MIT | Azure Blob integration |

### Optional Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| gqlalchemy | Apache 2.0 | Memgraph driver |
| qdrant-client | Apache 2.0 | Qdrant vector DB |
| redis | MIT | Redis cache |
| networkx | BSD 3-Clause | Graph analysis |
| scikit-learn | BSD 3-Clause | ML utilities |
| z3-solver | MIT | Theorem proving |
| torch | BSD 3-Clause | Neural networks |

---

## License Verification

All dependencies are verified using:
1. Official package documentation
2. PyPI license classifiers
3. Source repository LICENSE files
4. FOSSA or similar license scanning tools (recommended)

---

## Compliance Checklist

- [x] No GPL dependencies in codebase
- [x] No AGPL dependencies in codebase
- [x] No SSPL dependencies in codebase
- [x] All backends use permissive licenses
- [x] All cloud SDKs use permissive licenses
- [x] Documentation clearly states license requirements
- [x] Migration path provided for non-permissive alternatives

---

## Adding New Dependencies

When adding new dependencies:

1. **Check the license** using:
   ```bash
   pip show <package>
   # Check "License" field
   ```

2. **Verify on PyPI**:
   - Visit https://pypi.org/project/<package>/
   - Check "License" classifier

3. **Review source repository**:
   - Check the LICENSE file in the GitHub/GitLab repository

4. **Get approval** if uncertain about license compatibility

5. **Document** in this file

---

## Prohibited Dependencies

The following are explicitly prohibited:

```python
# Graph Databases
neo4j              # GPL v3
arangodb           # Complex multi-license

# Document Databases  
mongodb            # SSPL
couchbase          # Proprietary features

# Search Engines
elasticsearch      # SSPL (after 7.10)

# Other
timescaledb        # Timescale License (proprietary features)
```

---

## Questions?

If you're unsure about a license:
1. Check https://opensource.org/licenses for OSI-approved licenses
2. Consult with the project maintainers
3. When in doubt, choose a clearly permissive alternative

---

**Enforcement:** Any PR adding non-permissive dependencies will be rejected.
