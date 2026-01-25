# CodeIndexer - Complete Documentation

## Overview

The CodeIndexer is a production-grade repository analysis tool that uses LLM-powered semantic analysis to understand code structure, find relationships, and enable intelligent code search.

## Features

- **Automated Repository Indexing**: Recursively analyze code repositories
- **Semantic Code Search**: Natural language queries over code
- **Similarity Detection**: Find files similar to a reference file
- **Relationship Mapping**: Discover dependencies and relationships between files
- **Incremental Updates**: Update indexes without full re-indexing
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
- **LLM Provider Support**: Anthropic (Claude), OpenAI (GPT), Google (Gemini)

## Installation

### Requirements

```bash
# Core requirements
pip install pyyaml aiohttp httpx

# LLM providers (install at least one)
pip install anthropic  # For Claude
pip install openai     # For GPT
pip install google-generativeai  # For Gemini
```

### Configuration

Create a configuration file `indexer_config.yaml`:

```yaml
# Paths Configuration
paths:
  code_base_path: "code_base"
  output_dir: "indexes"

# File Analysis Settings
file_analysis:
  supported_extensions:
    - ".py"
    - ".js"
    - ".ts"
    - ".java"
    - ".cpp"
    - ".go"
    - ".rs"

  skip_directories:
    - "__pycache__"
    - "node_modules"
    - "venv"
    - ".git"

  max_file_size: 1048576  # 1MB

# LLM Configuration
llm:
  model_provider: "anthropic"  # or "openai", "google"
  max_tokens: 4000
  temperature: 0.3
  system_prompt: "You are a code analysis expert."

  # Rate limiting
  request_delay: 0.1
  max_retries: 3

# Performance Settings
performance:
  enable_concurrent_analysis: true
  max_concurrent_files: 5

# Debug Settings
debug:
  mock_llm_responses: false
  verbose_output: true
  save_raw_responses: false
```

## Quick Start

### 1. Set up API Keys

```bash
# Option 1: Environment variables
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# Option 2: Config file (mcp_agent.secrets.yaml)
anthropic_api_key: "your-key-here"
openai_api_key: "your-key-here"
```

### 2. Basic Usage

```python
import asyncio
from knowledge_engine.indexer import CodeIndexer

async def main():
    # Initialize indexer
    indexer = CodeIndexer(
        code_base_path="path/to/your/repositories",
        target_structure="Your project structure description",
        output_dir="indexes",
        indexer_config_path="indexer_config.yaml"
    )

    # Index all repositories
    output_files = await indexer.build_all_indexes()
    print(f"Created {len(output_files)} indexes")

    # Search code
    results = await indexer.search_code("database connection handler")
    for result in results:
        print(f"Found: {result['file_path']} (relevance: {result['relevance_score']})")

asyncio.run(main())
```

## Detailed Usage

### Repository Indexing

```python
# Index a single repository
repo_index = await indexer.process_repository(Path("my_repo"))

# Access results
print(f"Repository: {repo_index.repo_name}")
print(f"Total files: {repo_index.total_files}")
print(f"Files analyzed: {len(repo_index.file_summaries)}")

# Inspect file summaries
for summary in repo_index.file_summaries:
    print(f"\nFile: {summary.file_path}")
    print(f"Type: {summary.file_type}")
    print(f"Functions: {', '.join(summary.main_functions)}")
    print(f"Summary: {summary.summary}")
```

### Code Search

```python
# Natural language search
results = await indexer.search_code(
    query="authentication middleware",
    filters={
        "file_type": "python",
        "min_confidence": 0.7
    }
)

# Process results
for match in results:
    print(f"\nFile: {match['file_path']}")
    print(f"Relevance: {match['relevance_score']}")
    print(f"Reason: {match['reason']}")
    print(f"Key aspects: {', '.join(match['relevant_snippets'])}")
```

### Find Similar Files

```python
# Find files similar to a reference file
similar_files = await indexer.find_similar_files(
    file_path="src/auth/login.py",
    threshold=0.6,
    top_k=10
)

for similar in similar_files:
    print(f"\nSimilar: {similar['file_path']}")
    print(f"Similarity: {similar['similarity_score']}")
    print(f"Reason: {similar['reason']}")
    print(f"Shared: {', '.join(similar['shared_aspects'])}")
```

### Get File Relationships

```python
# Get detailed relationships for a file
relationships = await indexer.get_file_relationships(
    file_path="src/utils/helpers.py"
)

print(f"File: {relationships['file_summary']['file_path']}")
print(f"Total relationships: {relationships['metadata']['total_relationships']}")

for rel in relationships['relationships']:
    print(f"\n→ {rel['target_file_path']}")
    print(f"  Type: {rel['relationship_type']}")
    print(f"  Confidence: {rel['confidence_score']}")
    print(f"  Usage: {rel['usage_suggestions']}")
```

### Incremental Updates

```python
# Update index for a modified file
success = await indexer.update_index(
    file_path="src/auth/login.py",
    repo_name="my_repo"
)

if success:
    print("Index updated successfully")

# Remove deleted file from index
success = await indexer.remove_from_index(
    file_path="src/deprecated.py",
    repo_name="my_repo"
)
```

### Statistics and Reporting

```python
# Get index statistics
stats = indexer.get_index_statistics()

print(f"Total repositories: {stats['total_repositories']}")
for repo in stats['repositories']:
    print(f"\n{repo['repo_name']}:")
    print(f"  Files: {repo['total_files']}")
    print(f"  Lines of code: {repo['total_lines_of_code']}")
    print(f"  Relationships: {repo['total_relationships']}")

    # File type breakdown
    for file_type, count in repo['file_types'].items():
        print(f"  {file_type}: {count}")
```

## Advanced Features

### Custom Target Structure

The CodeIndexer can analyze code against a target structure to find relevant implementations:

```python
target_structure = """
Project: E-commerce Backend

Structure:
- src/
  - auth/          # Authentication & authorization
  - database/      # Database models & queries
  - api/           # REST API endpoints
  - utils/         # Utility functions
  - config/        # Configuration management
"""

indexer = CodeIndexer(
    code_base_path="reference_implementations",
    target_structure=target_structure,
    enable_pre_filtering=True  # Use LLM to pre-filter relevant files
)
```

### File Pre-Filtering

Enable LLM-based file filtering to focus on relevant code:

```python
indexer = CodeIndexer(
    code_base_path="large_codebase",
    target_structure=your_target,
    enable_pre_filtering=True  # Only analyze relevant files
)

# The LLM will filter files before full analysis
await indexer.build_all_indexes()
```

### Caching for Performance

Enable content caching to avoid re-analyzing unchanged files:

```python
# In indexer_config.yaml
performance:
  enable_content_caching: true
  max_cache_size: 100

# Caching is automatic based on file mtime and size
await indexer.build_all_indexes()
```

### Mock Mode for Testing

Use mock LLM responses for testing without API calls:

```python
# In indexer_config.yaml
debug:
  mock_llm_responses: true

# Or programmatically
indexer = CodeIndexer(
    code_base_path="test_repo",
    mock_llm_responses=True
)
```

## API Reference

### CodeIndexer Class

#### Constructor

```python
CodeIndexer(
    code_base_path: str = None,
    target_structure: str = None,
    output_dir: str = None,
    config_path: str = "mcp_agent.secrets.yaml",
    indexer_config_path: str = None,
    enable_pre_filtering: bool = True
)
```

#### Methods

##### `async build_all_indexes() -> Dict[str, str]`
Build indexes for all repositories in code_base.

**Returns**: Dictionary mapping repo names to index file paths

##### `async process_repository(repo_path: Path) -> RepoIndex`
Process a single repository and create complete index.

**Returns**: RepoIndex object with file summaries and relationships

##### `async search_code(query: str, filters: Dict = None, index_path: str = None) -> List[Dict]`
Search code using natural language query.

**Parameters**:
- `query`: Natural language search query
- `filters`: Optional filters like {"file_type": "python", "min_confidence": 0.7}
- `index_path`: Optional path to specific index file

**Returns**: List of matching code snippets with relevance scores

##### `async find_similar_files(file_path: str, threshold: float = 0.6, top_k: int = 10) -> List[Dict]`
Find files similar to the given file.

**Returns**: List of similar files with similarity scores

##### `async get_file_relationships(file_path: str, index_path: str = None) -> Dict`
Get detailed relationships for a specific file.

**Returns**: Dictionary with file summary, relationships, and metadata

##### `async update_index(file_path: str, repo_name: str = None) -> bool`
Update index for a specific file (incremental update).

**Returns**: True if update successful

##### `async remove_from_index(file_path: str, repo_name: str = None) -> bool`
Remove a file from the index.

**Returns**: True if removal successful

##### `get_index_statistics(repo_name: str = None) -> Dict`
Get statistics about indexed repositories.

**Returns**: Dictionary with statistics

### Data Models

#### FileSummary
```python
@dataclass
class FileSummary:
    file_path: str              # Relative path to file
    file_type: str              # File type description
    main_functions: List[str]   # Main functions/classes
    key_concepts: List[str]     # Important concepts
    dependencies: List[str]     # External dependencies
    summary: str                # File description
    lines_of_code: int          # LOC count
    last_modified: str          # ISO timestamp
```

#### FileRelationship
```python
@dataclass
class FileRelationship:
    repo_file_path: str              # Source file
    target_file_path: str            # Target file
    relationship_type: str           # direct_match, partial_match, reference, utility
    confidence_score: float          # 0.0 to 1.0
    helpful_aspects: List[str]       # Helpful aspects
    potential_contributions: List[str]  # Potential uses
    usage_suggestions: str           # How to use
```

#### RepoIndex
```python
@dataclass
class RepoIndex:
    repo_name: str                     # Repository name
    total_files: int                   # Total files
    file_summaries: List[FileSummary]  # All file summaries
    relationships: List[FileRelationship]  # All relationships
    analysis_metadata: Dict[str, Any]  # Metadata
```

## Best Practices

### 1. Environment Setup

Always use environment variables for API keys:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### 2. Configuration Management

Keep configuration in YAML files, not hardcoded:

```yaml
# indexer_config.yaml
llm:
  model_provider: "anthropic"
  max_tokens: 4000

performance:
  enable_concurrent_analysis: true
  max_concurrent_files: 5
```

### 3. Error Handling

Always handle async operations properly:

```python
try:
    results = await indexer.search_code("database connection")
    if not results:
        print("No results found")
    else:
        for result in results:
            print(result['file_path'])
except Exception as e:
    print(f"Search failed: {e}")
```

### 4. Performance Optimization

For large codebases:

```python
# Enable concurrent processing
indexer.enable_concurrent_analysis = True
indexer.max_concurrent_files = 10

# Enable caching
indexer.enable_content_caching = True

# Use pre-filtering
indexer.enable_pre_filtering = True
```

### 5. Incremental Updates

Don't re-index entire repositories for small changes:

```python
# Just update modified files
await indexer.update_index("src/modified_file.py")
```

## Troubleshooting

### No API Key Found

```
ValueError: No valid LLM API key found
```

**Solution**: Set environment variables or config file:
```bash
export ANTHROPIC_API_KEY="your-key"
```

### Timeout Errors

```
TimeoutError: LLM call timed out
```

**Solution**: Increase timeout in config:
```yaml
llm:
  max_retries: 5
  retry_delay: 2.0
```

### Out of Memory

**Solution**: Reduce concurrent files or enable streaming:
```yaml
performance:
  enable_concurrent_analysis: false  # Disable concurrency
  max_concurrent_files: 3
```

### Mock Mode Not Working

**Solution**: Ensure mock is enabled in config:
```yaml
debug:
  mock_llm_responses: true
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest knowledge_engine/tests/test_indexer.py -v

# Run specific test
pytest knowledge_engine/tests/test_indexer.py::TestCodeIndexer::test_search_code -v

# Run with coverage
pytest knowledge_engine/tests/test_indexer.py --cov=knowledge_engine.indexer
```

## Examples

See the `examples/` directory for complete examples:

- `basic_indexing.py`: Simple repository indexing
- `code_search.py`: Natural language code search
- `similarity_analysis.py`: Find similar files
- `incremental_updates.py`: Update indexes incrementally
- `custom_target.py`: Custom target structure analysis

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [repository_url]/issues
- Documentation: [repository_url]/wiki
- Email: support@example.com
