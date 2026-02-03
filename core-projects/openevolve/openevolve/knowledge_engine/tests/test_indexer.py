"""
Comprehensive Test Suite for CodeIndexer

Tests:
- Repository indexing
- Code search functionality
- Similarity matching
- Incremental updates
- Edge cases and error handling
"""

import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pytest

# Import the CodeIndexer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.indexer import CodeIndexer, FileSummary, FileRelationship, RepoIndex


class TestCodeIndexer:
    """Test suite for CodeIndexer functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp = Path(tempfile.mkdtemp())
        yield temp
        # Cleanup
        if temp.exists():
            shutil.rmtree(temp)

    @pytest.fixture
    def sample_repo(self, temp_dir):
        """Create a sample repository for testing"""
        repo_path = temp_dir / "test_repo"

        # Create directory structure
        (repo_path / "src").mkdir(parents=True)
        (repo_path / "tests").mkdir(parents=True)

        # Create sample Python files
        (repo_path / "src" / "main.py").write_text("""
def main():
    '''Main entry point'''
    print("Hello World")

class DataProcessor:
    def process(self, data):
        return data.strip()
""", encoding="utf-8")

        (repo_path / "src" / "utils.py").write_text("""
def helper_function(x):
    '''Helper utility'''
    return x * 2

class Helper:
    pass
""", encoding="utf-8")

        (repo_path / "tests" / "test_main.py").write_text("""
def test_main():
    assert True
""", encoding="utf-8")

        # Create config file
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text("""
llm:
  model_provider: "mock"
  mock_llm_responses: true
  max_tokens: 1000

performance:
  enable_concurrent_analysis: false

debug:
  mock_llm_responses: true
  verbose_output: false
""", encoding="utf-8")

        return repo_path, config_path

    @pytest.fixture
    def sample_index(self, temp_dir):
        """Create a sample index file for testing"""
        index_file = temp_dir / "test_index.json"

        index_data = {
            "repo_name": "test_repo",
            "total_files": 2,
            "file_summaries": [
                {
                    "file_path": "src/main.py",
                    "file_type": "Python module",
                    "main_functions": ["main", "DataProcessor"],
                    "key_concepts": ["entry point", "data processing"],
                    "dependencies": [],
                    "summary": "Main entry point with data processing class",
                    "lines_of_code": 10,
                    "last_modified": "2025-01-01T00:00:00"
                },
                {
                    "file_path": "src/utils.py",
                    "file_type": "Python module",
                    "main_functions": ["helper_function", "Helper"],
                    "key_concepts": ["utility", "helper"],
                    "dependencies": [],
                    "summary": "Utility functions and helper classes",
                    "lines_of_code": 8,
                    "last_modified": "2025-01-01T00:00:00"
                }
            ],
            "relationships": [
                {
                    "repo_file_path": "src/utils.py",
                    "target_file_path": "src/main.py",
                    "relationship_type": "utility",
                    "confidence_score": 0.8,
                    "helpful_aspects": ["helper functions"],
                    "potential_contributions": ["utility methods"],
                    "usage_suggestions": "Use helper functions in main"
                }
            ],
            "analysis_metadata": {
                "analysis_date": datetime.now().isoformat(),
                "total_relationships_found": 1,
                "analyzer_version": "1.4.0"
            }
        }

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        return index_file

    @pytest.mark.asyncio
    async def test_indexer_initialization(self, sample_repo):
        """Test CodeIndexer initialization"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path.parent),
            target_structure="Test project structure",
            output_dir=str(repo_path.parent / "indexes"),
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        assert indexer.code_base_path == repo_path.parent
        assert indexer.output_dir == repo_path.parent / "indexes"
        assert indexer.mock_llm_responses == True
        assert indexer.supported_extensions == {
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
            ".cs", ".php", ".rb", ".go", ".rs", ".scala", ".kt",
            ".swift", ".m", ".mm", ".r", ".matlab", ".sql", ".sh",
            ".bat", ".ps1", ".yaml", ".yml", ".json", ".xml", ".toml"
        }

    @pytest.mark.asyncio
    async def test_get_all_repo_files(self, sample_repo):
        """Test repository file discovery"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        files = indexer.get_all_repo_files(repo_path)

        assert len(files) == 3  # main.py, utils.py, test_main.py
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" in file_names

    def test_generate_file_tree(self, sample_repo):
        """Test file tree generation"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        tree = indexer.generate_file_tree(repo_path)

        assert "test_repo" in tree
        assert "src" in tree
        assert "tests" in tree
        assert "main.py" in tree
        assert "utils.py" in tree

    @pytest.mark.asyncio
    async def test_analyze_file_content(self, sample_repo):
        """Test individual file analysis"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        test_file = repo_path / "src" / "main.py"
        summary = await indexer.analyze_file_content(test_file)

        assert isinstance(summary, FileSummary)
        # Normalize path for comparison (Windows vs Unix)
        assert summary.file_path.replace("\\", "/") == "src/main.py"
        assert summary.file_type != "error"
        assert len(summary.main_functions) > 0
        assert summary.lines_of_code > 0
        assert summary.last_modified != ""

    @pytest.mark.asyncio
    async def test_find_relationships(self, sample_repo):
        """Test relationship discovery"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test project with main and utils",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        file_summary = FileSummary(
            file_path="src/utils.py",
            file_type="Python module",
            main_functions=["helper_function"],
            key_concepts=["utility"],
            dependencies=[],
            summary="Utility functions",
            lines_of_code=5,
            last_modified=datetime.now().isoformat()
        )

        relationships = await indexer.find_relationships(file_summary)

        assert isinstance(relationships, list)
        # With mock responses, should get at least one relationship
        if relationships:
            assert isinstance(relationships[0], FileRelationship)
            assert relationships[0].repo_file_path == "src/utils.py"
            assert relationships[0].confidence_score >= 0.0
            assert relationships[0].confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_process_repository(self, sample_repo):
        """Test complete repository processing"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path.parent),
            target_structure="Test project structure",
            output_dir=str(repo_path.parent / "indexes"),
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        repo_index = await indexer.process_repository(repo_path)

        assert isinstance(repo_index, RepoIndex)
        assert repo_index.repo_name == "test_repo"
        assert repo_index.total_files >= 3
        assert len(repo_index.file_summaries) >= 3
        assert "analysis_date" in repo_index.analysis_metadata

    @pytest.mark.asyncio
    async def test_build_all_indexes(self, sample_repo):
        """Test building indexes for all repositories"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path.parent),
            target_structure="Test project",
            output_dir=str(repo_path.parent / "indexes"),
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        output_files = await indexer.build_all_indexes()

        assert isinstance(output_files, dict)
        assert "test_repo" in output_files
        assert Path(output_files["test_repo"]).exists()

    def test_load_index(self, sample_index):
        """Test loading index from file"""
        indexer = CodeIndexer()
        index_data = indexer.load_index(str(sample_index))

        assert index_data is not None
        assert index_data["repo_name"] == "test_repo"
        assert len(index_data["file_summaries"]) == 2
        assert len(index_data["relationships"]) == 1

    def test_query_index_by_keyword(self, sample_index):
        """Test keyword search in index"""
        indexer = CodeIndexer()  # Dummy instance for method call
        index_data = indexer.load_index(str(sample_index))

        # Search for "main"
        results = indexer.query_index_by_keyword(index_data, "main")

        assert len(results) > 0
        assert any("main" in r["file_path"].lower() or "main" in r["summary"].lower()
                   for r in results)

    @pytest.mark.asyncio
    async def test_search_code(self, sample_repo, sample_index):
        """Test natural language code search"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path.parent),
            output_dir=str(sample_index.parent),
            indexer_config_path=str(config_path)
        )

        # Move sample index to output directory
        output_dir = Path(repo_path.parent) / "indexes"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_index, output_dir / "test_repo_index.json")

        results = await indexer.search_code("main function")

        assert isinstance(results, list)
        # With mock responses, may or may not get results depending on mock

    @pytest.mark.asyncio
    async def test_find_similar_files(self, sample_repo, sample_index):
        """Test finding similar files"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path.parent),
            output_dir=str(sample_index.parent),
            indexer_config_path=str(config_path)
        )

        # Setup index
        output_dir = Path(repo_path.parent) / "indexes"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_index, output_dir / "test_repo_index.json")

        # Create a reference file
        ref_file = repo_path / "reference.py"
        ref_file.write_text("""
def helper(x):
    return x * 2

class Processor:
    pass
""", encoding="utf-8")

        similar = await indexer.find_similar_files(str(ref_file))

        assert isinstance(similar, list)
        # Results may vary with mock responses

    @pytest.mark.asyncio
    async def test_get_file_relationships(self, sample_index):
        """Test getting file relationships"""
        indexer = CodeIndexer()
        relationships = await indexer.get_file_relationships(
            "src/utils.py",
            str(sample_index)
        )

        assert isinstance(relationships, dict)
        if relationships:  # If found
            assert "file_summary" in relationships
            assert "relationships" in relationships
            assert "metadata" in relationships

    @pytest.mark.asyncio
    async def test_update_index(self, sample_repo, sample_index, temp_dir):
        """Test incremental index update"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            output_dir=str(temp_dir / "indexes"),
            indexer_config_path=str(config_path)
        )

        # Copy sample index
        output_dir = temp_dir / "indexes"
        output_dir.mkdir(parents=True, exist_ok=True)
        index_file = output_dir / "test_repo_index.json"
        shutil.copy(sample_index, index_file)

        # Modify the file
        (repo_path / "src" / "main.py").write_text("""
def main():
    '''Updated main function'''
    print("Updated Hello World")
    return 0
""", encoding="utf-8")

        # Update index
        success = await indexer.update_index(str(repo_path / "src" / "main.py"), "test_repo")

        assert success == True

    @pytest.mark.asyncio
    async def test_remove_from_index(self, sample_index, temp_dir):
        """Test removing file from index"""
        indexer = CodeIndexer(
            output_dir=str(temp_dir / "indexes")
        )

        # Copy sample index
        output_dir = temp_dir / "indexes"
        output_dir.mkdir(parents=True, exist_ok=True)
        index_file = output_dir / "test_repo_index.json"
        shutil.copy(sample_index, index_file)

        # Remove file
        success = await indexer.remove_from_index("src/utils.py", "test_repo")

        assert success == True

        # Verify removal
        index_data = indexer.load_index(str(index_file))
        assert not any(s["file_path"] == "src/utils.py" for s in index_data["file_summaries"])

    def test_get_index_statistics(self, sample_index, temp_dir):
        """Test getting index statistics"""
        indexer = CodeIndexer(
            output_dir=str(temp_dir / "indexes")
        )

        # Copy sample index
        output_dir = temp_dir / "indexes"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_index, output_dir / "test_repo_index.json")

        stats = indexer.get_index_statistics("test_repo")

        assert isinstance(stats, dict)
        assert stats["total_repositories"] == 1
        assert len(stats["repositories"]) == 1

        repo_stats = stats["repositories"][0]
        assert repo_stats["repo_name"] == "test_repo"
        assert repo_stats["total_files"] >= 2
        assert "total_lines_of_code" in repo_stats

    @pytest.mark.asyncio
    async def test_error_handling_invalid_file(self, sample_repo):
        """Test error handling for invalid file"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        # Try to analyze non-existent file
        fake_file = repo_path / "nonexistent.py"
        summary = await indexer.analyze_file_content(fake_file)

        # Should return an error summary, not raise exception
        assert summary.file_type == "error"

    @pytest.mark.asyncio
    async def test_large_file_handling(self, sample_repo):
        """Test handling of files exceeding size limit"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        # Set a small size limit for testing
        indexer.max_file_size = 100

        # Create a file exceeding limit
        large_file = repo_path / "large.py"
        large_file.write_text("x" * 200, encoding="utf-8")

        summary = await indexer.analyze_file_content(large_file)

        # Should be skipped
        assert "skipped" in summary.file_type.lower()
        assert summary.lines_of_code == 0

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, sample_repo):
        """Test concurrent file processing"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        # Enable concurrent processing
        indexer.enable_concurrent_analysis = True
        indexer.max_concurrent_files = 2

        files = indexer.get_all_repo_files(repo_path)
        summaries, relationships = await indexer._process_files_concurrently(files)

        assert len(summaries) == len(files)
        assert isinstance(relationships, list)

    def test_filter_files_by_paths(self, sample_repo):
        """Test file filtering by paths"""
        repo_path, config_path = sample_repo

        indexer = CodeIndexer(
            code_base_path=str(repo_path),
            target_structure="Test",
            indexer_config_path=str(config_path),
            enable_pre_filtering=False
        )

        all_files = indexer.get_all_repo_files(repo_path)

        # Filter for only src files
        selected = ["src/main.py", "src/utils.py"]
        filtered = indexer.filter_files_by_paths(all_files, selected, repo_path)

        assert len(filtered) == 2
        assert all(f.parent.name == "src" for f in filtered)


class TestLLMUtils:
    """Test suite for LLM utilities"""

    @pytest.mark.asyncio
    async def test_extract_json_from_response(self):
        """Test JSON extraction from LLM responses"""
        from knowledge_engine.llm_utils import extract_json_from_response

        # Test direct JSON
        response1 = '{"key": "value"}'
        result1 = extract_json_from_response(response1)
        assert result1 == {"key": "value"}

        # Test JSON in code block
        response2 = '```json\n{"key": "value"}\n```'
        result2 = extract_json_from_response(response2)
        assert result2 == {"key": "value"}

        # Test JSON with surrounding text
        response3 = 'Here is the result: {"key": "value"} and more text'
        result3 = extract_json_from_response(response3)
        assert result3 == {"key": "value"}

        # Test invalid JSON
        response4 = 'This is not JSON'
        result4 = extract_json_from_response(response4)
        assert result4 == {}

    def test_create_llm_prompt(self):
        """Test LLM prompt creation"""
        from knowledge_engine.llm_utils import create_llm_prompt

        template = "Analyze this {language} code: {code}"
        result = create_llm_prompt(template, language="Python", code="print('hello')")

        assert result == "Analyze this Python code: print('hello')"

    @pytest.mark.asyncio
    async def test_call_llm_fallback(self):
        """Test LLM call fallback mechanism"""
        from knowledge_engine.llm_utils import call_llm

        # Call without API key (should use fallback)
        result = await call_llm(
            prompt="Test",
            api_key=None,
            timeout=5.0
        )

        assert isinstance(result, str)
        assert len(result) > 0


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
