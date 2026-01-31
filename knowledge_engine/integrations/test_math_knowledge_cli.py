"""
Test Suite for math_knowledge_cli.py

Tests all CLI commands and functionality.
"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMathKnowledgeCLI:
    """Test the math knowledge CLI."""
    
    def test_cli_creation(self):
        """Test that CLI can be created."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        assert cli is not None
    
    def test_cli_parser(self):
        """Test CLI argument parser."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        # Test parser creation
        assert cli.parser is not None
        
        # Test that commands are registered
        args = cli.parser.parse_args(['solve', '--problem', 'x + y = 10'])
        assert args.command == 'solve'
        assert args.problem == 'x + y = 10'
    
    def test_solve_command_parsing(self):
        """Test solve command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args([
            'solve',
            '--problem', 'x > 0',
            '--solver', 'z3',
            '--timeout', '60'
        ])
        
        assert args.command == 'solve'
        assert args.problem == 'x > 0'
        assert args.solver == 'z3'
        assert args.timeout == 60
    
    def test_prove_command_parsing(self):
        """Test prove command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args([
            'prove',
            '--theorem', 'forall n, n + 0 = n',
            '--timeout', '300'
        ])
        
        assert args.command == 'prove'
        assert args.theorem == 'forall n, n + 0 = n'
        assert args.timeout == 300
    
    def test_search_command_parsing(self):
        """Test search command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args([
            'search',
            '--query', 'linear system',
            '--top-k', '10'
        ])
        
        assert args.command == 'search'
        assert args.query == 'linear system'
        assert args.top_k == 10
    
    def test_config_command_parsing(self):
        """Test config command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args([
            'config',
            '--show'
        ])
        
        assert args.command == 'config'
        assert args.show is True
    
    def test_benchmark_command_parsing(self):
        """Test benchmark command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args([
            'benchmark',
            '--suite', 'comprehensive',
            '--iterations', '50'
        ])
        
        assert args.command == 'benchmark'
        assert args.suite == 'comprehensive'
        assert args.iterations == 50
    
    def test_server_command_parsing(self):
        """Test server command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args([
            'server',
            '--start',
            '--port', '8765'
        ])
        
        assert args.command == 'server'
        assert args.start is True
        assert args.port == 8765
    
    def test_knowledge_command_parsing(self):
        """Test knowledge command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args([
            'knowledge',
            '--stats'
        ])
        
        assert args.command == 'knowledge'
        assert args.stats is True
    
    def test_health_command_parsing(self):
        """Test health command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args(['health'])
        
        assert args.command == 'health'
    
    def test_version_command_parsing(self):
        """Test version command argument parsing."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        args = cli.parser.parse_args(['version'])
        
        assert args.command == 'version'


class TestCLIHandlerMethods:
    """Test CLI handler methods exist."""
    
    def test_handler_methods_exist(self):
        """Test that all command handlers exist."""
        from math_knowledge_cli import MathKnowledgeCLI
        cli = MathKnowledgeCLI()
        
        commands = [
            '_cmd_solve',
            '_cmd_prove',
            '_cmd_search',
            '_cmd_config',
            '_cmd_benchmark',
            '_cmd_server',
            '_cmd_knowledge',
            '_cmd_health',
            '_cmd_version',
        ]
        
        for cmd in commands:
            assert hasattr(cli, cmd), f"Missing handler: {cmd}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
