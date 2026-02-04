# Knowledge Engine Configuration Guide

This guide documents all environment variables used by the OpenEvolve Knowledge Engine, organized by category.

## Table of Contents

- [Required Configuration](#required-configuration)
- [Optional Configuration](#optional-configuration)
- [Validation](#validation)
- [Configuration Template](#configuration-template)
- [Troubleshooting](#troubleshooting)

---

## Required Configuration

The following environment variables **must** be set for the Knowledge Engine to function:

### Core Knowledge Graph

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `GRAPHITI_URI` | URI for Graphiti/Neo4j knowledge graph | `bolt://localhost:7687` | Yes |
| `GRAPHITI_USER` | Username for knowledge graph database | `neo4j` | Yes |
| `GRAPHITI_PASSWORD` | Password for knowledge graph database | `your_secure_password` | Yes |
| `NEO4J_URI` | Alternative Neo4j URI | `bolt://localhost:7687` | No |
| `NEO4J_USER` | Alternative Neo4j user | `neo4j` | No |
| `NEO4J_PASSWORD` | Alternative Neo4j password | `your_secure_password` | No |

**Note:** `NEO4J_*` variables are alternatives to `GRAPHITI_*` variables. You only need to set one pair.

### LLM Providers

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM operations | `sk-proj-...` | Yes |

**Note:** While `ANTHROPIC_API_KEY` and other provider keys are optional, you need at least one LLM provider configured.

---

## Optional Configuration

### LLM Providers (Continued)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | - | `sk-ant-...` |
| `LLM_API_KEY` | Generic LLM API key (if not provider-specific) | - | `your_api_key` |
| `LLM_API_BASE` | Base URL for LLM API | `https://api.openai.com/v1` | `https://api.openai.com/v1` |
| `LLM_DEFAULT_MODEL` | Default LLM model to use | `gpt-4o` | `gpt-4o` |
| `LLM_TEMPERATURE` | Default temperature for LLM generation | `0.1` | `0.1` |
| `LLM_MAX_TOKENS` | Default max tokens for LLM generation | `2000` | `2000` |
| `LLM_TIMEOUT` | LLM request timeout in seconds | `120` | `120` |
| `LLM_MAX_RETRIES` | Maximum retries for LLM requests | `3` | `3` |

### KGGen Integration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `KGGEN_ENTITY_MODEL` | Model for entity extraction | `gpt-4o` | `gpt-4o` |
| `KGGEN_RELATION_MODEL` | Model for relation extraction | `gpt-4o` | `gpt-4o` |
| `KGGEN_TIMEOUT_MS` | Timeout for KGGen operations (ms) | `30000` | `30000` |
| `KGGEN_CHUNK_SIZE` | Chunk size for text processing | `5000` | `5000` |

### OneKE Integration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ONEKE_MODEL_NAME` | OneKE model to use | `oneke/OneKE-13B` | `oneke/OneKE-13B` |
| `ONEKE_DEVICE` | Device for OneKE inference (cuda/cpu) | `cuda` | `cuda` |
| `ONEKE_TIMEOUT_MS` | Timeout for OneKE operations (ms) | `60000` | `60000` |
| `ONEKE_TASK_TIMEOUT` | Task timeout for OneKE (seconds) | `300` | `300` |
| `ONEKE_MAX_RETRIES` | Maximum retries for OneKE tasks | `3` | `3` |

### Qdrant Vector Store

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `QDRANT_HOST` | Qdrant host address | `localhost` | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` | `6333` |

### PostgreSQL Database

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `POSTGRESQL_URI` | PostgreSQL connection URI | - | `postgresql://user:pass@localhost:5432/openevolve` |
| `DB_HOST` | Database host | `localhost` | `localhost` |
| `DB_PORT` | Database port | `5432` | `5432` |
| `DB_USERNAME` | Database username | `openevolve` | `openevolve` |
| `DB_PASSWORD` | Database password (REQUIRED in production) | - | `your_secure_password` |
| `DB_NAME` | Database name | `openevolve_kg` | `openevolve_kg` |

### Redis Cache

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `REDIS_HOST` | Redis host address | `localhost` | `localhost` |
| `REDIS_PORT` | Redis port | `6379` | `6379` |

### Elasticsearch

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ELASTICSEARCH_HOSTS` | Elasticsearch hosts (comma-separated) | `http://localhost:9200` | `http://localhost:9200` |
| `ELASTICSEARCH_API_KEY` | Elasticsearch API key | - | `your_elasticsearch_api_key` |
| `ELASTICSEARCH_INDEX_PREFIX` | Prefix for Elasticsearch indices | `openevolve` | `openevolve` |

### AWS S3 Storage

| Variable | Description | Example |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key ID (REQUIRED if using S3) | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key (REQUIRED if using S3) | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_REGION` | AWS region | `us-east-1` |
| `AWS_ENDPOINT_URL` | Custom endpoint URL (for MinIO compatibility) | `http://localhost:9000` |

### Google Cloud Storage

| Variable | Description | Example |
|----------|-------------|---------|
| `GCS_PROJECT_ID` | GCP project ID (REQUIRED if using GCS) | `my-gcp-project` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account credentials JSON | `/path/to/credentials.json` |
| `GCS_CREDENTIALS_JSON` | Raw GCP credentials JSON string | `{"type": "service_account", ...}` |

### Azure Blob Storage

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_STORAGE_ACCOUNT` | Azure storage account name (REQUIRED if using Azure) | `mystorageaccount` |
| `AZURE_STORAGE_KEY` | Azure storage key | `your_storage_key` |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure storage connection string | `DefaultEndpointsProtocol=https;AccountName=...` |
| `AZURE_STORAGE_SAS_TOKEN` | Azure SAS token | `your_sas_token` |

### SFTP Storage

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SFTP_HOST` | SFTP host (REQUIRED if using SFTP) | - | `sftp.example.com` |
| `SFTP_PORT` | SFTP port | `22` | `22` |
| `SFTP_USERNAME` | SFTP username | - | `sftp_user` |
| `SFTP_PASSWORD` | SFTP password | - | `sftp_password` |
| `SFTP_PRIVATE_KEY_PATH` | Path to SFTP private key | - | `/path/to/private_key` |
| `SFTP_KEY_PASSPHRASE` | Passphrase for SFTP private key | - | `key_passphrase` |

### Math Knowledge Integration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MATH_KNOWLEDGE_DB_URL` | Math knowledge database URL | `sqlite:///math_knowledge.db` | `sqlite:///math_knowledge.db` |
| `MATH_KNOWLEDGE_Z3_TIMEOUT_MS` | Z3 solver timeout (ms) | `30000` | `30000` |
| `MATH_KNOWLEDGE_Z3_MEMORY_MB` | Z3 solver memory limit (MB) | `4096` | `4096` |

### Server Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SERVER_HOST` | Server host address | `0.0.0.0` | `0.0.0.0` |
| `SERVER_PORT` | Server port | `8000` | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` | `INFO` |

---

## Validation

The Knowledge Engine includes automatic configuration validation that runs on module import.

### Validation Modes

Set the `KE_VALIDATE_CONFIG` environment variable to control validation behavior:

| Mode | Description |
|------|-------------|
| `warn` (default) | Validate and warn on errors, but allow import to continue |
| `strict` | Fail on both errors and warnings |
| `off` | Disable validation (not recommended) |

### Running Validation Manually

```python
from knowledge_engine.config_validation import validate_config

# Basic validation
result = validate_config()
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")

# Strict validation (fails on warnings too)
result = validate_config(strict=True)
```

### Generating Configuration Template

To generate a template `.env` file:

```python
from knowledge_engine.config_validation import get_config_template

template = get_config_template()
with open('.env.template', 'w') as f:
    f.write(template)
```

Or via command line:

```bash
python -c "from knowledge_engine.config_validation import get_config_template; print(get_config_template())" > .env.template
```

---

## Configuration Template

```bash
# Knowledge Engine Configuration Template
#
# Copy this file to .env and fill in the required values
#
# IMPORTANT: Never commit .env to version control!

# Core Knowledge Graph
# REQUIRED: URI for Graphiti/Neo4j knowledge graph
GRAPHITI_URI=bolt://localhost:7687

# REQUIRED: Username for knowledge graph database
GRAPHITI_USER=neo4j

# REQUIRED: Password for knowledge graph database
GRAPHITI_PASSWORD=your_secure_password

# Optional: Alternative Neo4j URI (if different from GRAPHITI_URI)
# NEO4J_URI=bolt://localhost:7687

# Optional: Alternative Neo4j user
# NEO4J_USER=neo4j

# Optional: Alternative Neo4j password
# NEO4J_PASSWORD=your_secure_password

# LLM Providers
# REQUIRED: OpenAI API key for LLM operations
OPENAI_API_KEY=sk-proj-...

# Optional: Anthropic API key for Claude models
# ANTHROPIC_API_KEY=unset

# Optional: Generic LLM API key (if not provider-specific)
# LLM_API_KEY=unset

# Optional: Base URL for LLM API
# LLM_API_BASE=https://api.openai.com/v1

# Optional: Default LLM model to use
# LLM_DEFAULT_MODEL=gpt-4o

# Optional: Default temperature for LLM generation
# LLM_TEMPERATURE=0.1

# Optional: Default max tokens for LLM generation
# LLM_MAX_TOKENS=2000

# Optional: LLM request timeout in seconds
# LLM_TIMEOUT=120

# Optional: Maximum retries for LLM requests
# LLM_MAX_RETRIES=3

# KGGen Integration
# Optional: Model for entity extraction
# KGGEN_ENTITY_MODEL=gpt-4o

# Optional: Model for relation extraction
# KGGEN_RELATION_MODEL=gpt-4o

# Optional: Timeout for KGGen operations (milliseconds)
# KGGEN_TIMEOUT_MS=30000

# Optional: Chunk size for text processing
# KGGEN_CHUNK_SIZE=5000

# OneKE Integration
# Optional: OneKE model to use
# ONEKE_MODEL_NAME=oneke/OneKE-13B

# Optional: Device for OneKE inference (cuda/cpu)
# ONEKE_DEVICE=cuda

# Optional: Timeout for OneKE operations (milliseconds)
# ONEKE_TIMEOUT_MS=60000

# Optional: Task timeout for OneKE (seconds)
# ONEKE_TASK_TIMEOUT=300

# Optional: Maximum retries for OneKE tasks
# ONEKE_MAX_RETRIES=3

# Qdrant Vector Store
# Optional: Qdrant host address
# QDRANT_HOST=localhost

# Optional: Qdrant port
# QDRANT_PORT=6333

# PostgreSQL Database
# Optional: PostgreSQL connection URI
# POSTGRESQL_URI=unset

# Optional: Database host
# DB_HOST=localhost

# Optional: Database port
# DB_PORT=5432

# Optional: Database username
# DB_USERNAME=openevolve

# Optional: Database password (REQUIRED in production)
# DB_PASSWORD=unset

# Optional: Database name
# DB_NAME=openevolve_kg

# Redis Cache
# Optional: Redis host address
# REDIS_HOST=localhost

# Optional: Redis port
# REDIS_PORT=6379

# Elasticsearch
# Optional: Elasticsearch hosts (comma-separated)
# ELASTICSEARCH_HOSTS=http://localhost:9200

# Optional: Elasticsearch API key
# ELASTICSEARCH_API_KEY=unset

# Optional: Prefix for Elasticsearch indices
# ELASTICSEARCH_INDEX_PREFIX=openevolve

# AWS S3 Storage
# Optional: AWS access key ID (REQUIRED if using S3 storage)
# AWS_ACCESS_KEY_ID=unset

# Optional: AWS secret access key (REQUIRED if using S3 storage)
# AWS_SECRET_ACCESS_KEY=unset

# Optional: AWS region
# AWS_REGION=us-east-1

# Optional: Custom endpoint URL (for MinIO compatibility)
# AWS_ENDPOINT_URL=unset

# Google Cloud Storage
# Optional: GCP project ID (REQUIRED if using GCS)
# GCS_PROJECT_ID=unset

# Optional: Path to GCP service account credentials JSON
# GOOGLE_APPLICATION_CREDENTIALS=unset

# Optional: Raw GCP credentials JSON string
# GCS_CREDENTIALS_JSON=unset

# Azure Blob Storage
# Optional: Azure storage account name (REQUIRED if using Azure)
# AZURE_STORAGE_ACCOUNT=unset

# Optional: Azure storage key
# AZURE_STORAGE_KEY=unset

# Optional: Azure storage connection string
# AZURE_STORAGE_CONNECTION_STRING=unset

# Optional: Azure SAS token
# AZURE_STORAGE_SAS_TOKEN=unset

# SFTP Storage
# Optional: SFTP host (REQUIRED if using SFTP)
# SFTP_HOST=unset

# Optional: SFTP port
# SFTP_PORT=22

# Optional: SFTP username
# SFTP_USERNAME=unset

# Optional: SFTP password
# SFTP_PASSWORD=unset

# Optional: Path to SFTP private key
# SFTP_PRIVATE_KEY_PATH=unset

# Optional: Passphrase for SFTP private key
# SFTP_KEY_PASSPHRASE=unset

# Math Knowledge
# Optional: Math knowledge database URL
# MATH_KNOWLEDGE_DB_URL=sqlite:///math_knowledge.db

# Optional: Z3 solver timeout (milliseconds)
# MATH_KNOWLEDGE_Z3_TIMEOUT_MS=30000

# Optional: Z3 solver memory limit (MB)
# MATH_KNOWLEDGE_Z3_MEMORY_MB=4096

# Server
# Optional: Server host address
# SERVER_HOST=0.0.0.0

# Optional: Server port
# SERVER_PORT=8000

# Optional: Logging level
# LOG_LEVEL=INFO
```

---

## Troubleshooting

### Common Issues

#### 1. "Configuration validation failed"

**Problem:** Required environment variables are missing.

**Solution:**
1. Run validation to see which variables are missing:
   ```bash
   python -c "from knowledge_engine.config_validation import validate_config; validate_config()"
   ```
2. Set the missing variables in your environment or `.env` file
3. Restart your application

#### 2. "Environment variable 'XXX' is set to empty string"

**Problem:** An environment variable is set but has no value.

**Solution:**
- Check for empty assignments in your `.env` file:
  ```bash
  # BAD
  OPENAI_API_KEY=

  # GOOD
  OPENAI_API_KEY=sk-proj-...
  ```
- Or unset the variable if it's not needed:
  ```bash
  unset OPTIONAL_VAR
  ```

#### 3. Cloud storage credentials not working

**Problem:** AWS/GCS/Azure credentials fail at runtime.

**Solution:**
1. Ensure you're setting all required credentials for your provider
2. Check that credentials have proper permissions
3. For AWS, verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are both set
4. For GCS, verify `GCS_PROJECT_ID` is set along with credentials
5. For Azure, verify `AZURE_STORAGE_ACCOUNT` is set

#### 4. Validation works but components still fail

**Problem:** Configuration passes validation but components fail at runtime.

**Solution:**
- Use strict validation: `KE_VALIDATE_CONFIG=strict`
- Check for typos in variable names
- Verify values are correct (e.g., no extra whitespace, proper URIs)
- Check component-specific logs for detailed error messages

### Getting Help

For additional help:
1. Check the logs: Set `LOG_LEVEL=DEBUG` for verbose output
2. Run validation manually: See "Running Validation Manually" above
3. Review this documentation for your specific configuration needs
4. Check component-specific documentation in the `/docs` directory

---

## Best Practices

1. **Use a `.env` file**: Keep all environment variables in a `.env` file (never commit it)
2. **Use templates**: Generate `.env.template` files and commit those instead
3. **Validate early**: Run validation in CI/CD before deployment
4. **Use strict mode in production**: Set `KE_VALIDATE_CONFIG=strict` for production deployments
5. **Rotate secrets**: Regularly rotate API keys and passwords
6. **Use strong passwords**: Never use default or weak passwords in production
7. **Document custom variables**: If you add custom variables, document them in your team's wiki

---

## Security Notes

1. **Never commit `.env` files**: Add `.env` to your `.gitignore`
2. **Use secret management**: For production, use a proper secret management system (e.g., HashiCorp Vault, AWS Secrets Manager)
3. **Limit permissions**: Grant minimum required permissions to API keys
4. **Audit access**: Regularly audit who has access to your secrets
5. **Rotate credentials**: Implement regular credential rotation policies

---

## Additional Resources

- [OpenEvolve Documentation](../README.md)
- [Component-Specific Configuration](./docs/)
- [Deployment Guide](./docs/operations/deployment.md)
- [Security Best Practices](./docs/operations/security.md)
