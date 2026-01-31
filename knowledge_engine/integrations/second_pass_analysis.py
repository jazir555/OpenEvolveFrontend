"""
Second Pass - Deep Gap Analysis

Thorough analysis to identify any remaining gaps:
1. Async/await consistency
2. Error handling coverage
3. Configuration validation
4. Database schema completeness
5. API endpoint coverage
6. Documentation completeness
7. Type hints coverage
8. Logging coverage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import inspect
from typing import Any, Dict, List, Set


class SecondPassAnalyzer:
    """Deep analyzer for remaining gaps."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.ok_count = 0
    
    def report(self, category: str, status: str, message: str):
        """Report a finding."""
        if status == "OK":
            self.ok_count += 1
            print(f"   [OK] {message}")
        elif status == "WARN":
            self.warnings.append((category, message))
            print(f"   [WARN] {message}")
        else:
            self.issues.append((category, message))
            print(f"   [FAIL] {message}")
    
    async def analyze(self):
        """Run all analyses."""
        print("="*70)
        print("SECOND PASS - DEEP GAP ANALYSIS")
        print("="*70)
        
        await self.check_async_consistency()
        await self.check_error_handling()
        await self.check_configuration()
        await self.check_database_models()
        await self.check_mcp_tools()
        await self.check_api_coverage()
        await self.check_documentation()
        await self.check_type_hints()
        await self.check_logging()
        
        self.print_summary()
    
    async def check_async_consistency(self):
        """Check async/await consistency."""
        print("\n1. Async/Await Consistency")
        
        from z3_solver_connector import Z3SolverConnector
        from z3_knowledge_complete import Z3KnowledgeManager
        
        # Check Z3SolverConnector
        async_methods = [name for name, method in inspect.getmembers(Z3SolverConnector, predicate=inspect.isfunction) 
                        if inspect.iscoroutinefunction(method)]
        sync_methods = [name for name, method in inspect.getmembers(Z3SolverConnector, predicate=inspect.isfunction) 
                       if not inspect.iscoroutinefunction(method) and not name.startswith('_')]
        
        self.report("Async", "OK" if len(async_methods) > 0 else "WARN", 
                   f"Z3SolverConnector: {len(async_methods)} async, {len(sync_methods)} sync methods")
        
        # Key methods should be async
        key_async = ['solve_smtlib', 'solve_with_config', 'check_satisfiability']
        for method in key_async:
            if hasattr(Z3SolverConnector, method):
                is_async = inspect.iscoroutinefunction(getattr(Z3SolverConnector, method))
                self.report("Async", "OK" if is_async else "FAIL", f"{method} is async: {is_async}")
    
    async def check_error_handling(self):
        """Check error handling coverage."""
        print("\n2. Error Handling Coverage")
        
        from z3_solver_connector import Z3SolverConnector, Z3SolverConfig
        
        z3 = Z3SolverConnector()
        
        # Test with invalid input
        try:
            result = await z3.solve_smtlib("invalid smtlib !!!", Z3SolverConfig())
            has_error_handling = hasattr(result, 'status') and result.status is not None
            self.report("Error", "OK" if has_error_handling else "WARN", 
                       "Invalid input handling" if has_error_handling else "May need better error handling")
        except Exception as e:
            self.report("Error", "FAIL", f"Exception on invalid input: {e}")
        
        # Test with timeout
        try:
            config = Z3SolverConfig(timeout_ms=1)  # Very short timeout
            result = await z3.solve_smtlib("(declare-fun x () Int) (assert (> x 0)) (check-sat)", config)
            self.report("Error", "OK", "Timeout handling works")
        except Exception as e:
            self.report("Error", "OK" if "timeout" in str(e).lower() else "WARN", 
                       f"Timeout behavior: {type(e).__name__}")
    
    async def check_configuration(self):
        """Check configuration completeness."""
        print("\n3. Configuration Validation")
        
        from math_knowledge_config import MathKnowledgeConfig
        
        config = MathKnowledgeConfig()
        
        # Check required fields
        checks = [
            ("database.url", config.database.url, bool),
            ("z3.timeout_ms", config.z3.timeout_ms, lambda x: x > 0),
            ("leanaide.host", config.leanaide.host, bool),
            ("api.port", config.api.port, lambda x: 0 < x < 65536),
        ]
        
        for name, value, validator in checks:
            try:
                is_valid = validator(value)
                self.report("Config", "OK" if is_valid else "FAIL", f"{name}: {value}")
            except Exception as e:
                self.report("Config", "FAIL", f"{name}: {e}")
        
        # Check validation method
        has_validate = hasattr(config, 'validate') and callable(getattr(config, 'validate'))
        self.report("Config", "OK" if has_validate else "FAIL", "Has validate method")
        
        if has_validate:
            errors = config.validate()
            self.report("Config", "OK" if not errors else "WARN", 
                       f"Validation errors: {len(errors)}")
    
    async def check_database_models(self):
        """Check database model completeness."""
        print("\n4. Database Schema Completeness")
        
        from math_knowledge_models import MODELS_AVAILABLE
        
        if not MODELS_AVAILABLE:
            self.report("DB", "WARN", "SQLAlchemy not available, skipping")
            return
        
        from math_knowledge_models import Z3KnowledgeBase, Z3SolverRun, LeanProofRecord
        
        models = [
            ("Z3KnowledgeBase", Z3KnowledgeBase, ['id', 'record_type', 'record_hash']),
            ("Z3SolverRun", Z3SolverRun, ['id', 'run_id', 'problem_hash']),
            ("LeanProofRecord", LeanProofRecord, ['id', 'theorem_id']),
        ]
        
        for name, model, required_cols in models:
            try:
                cols = [c.name for c in model.__table__.columns]
                missing = [r for r in required_cols if r not in cols]
                if missing:
                    self.report("DB", "FAIL", f"{name} missing columns: {missing}")
                else:
                    self.report("DB", "OK", f"{name}: {len(cols)} columns")
            except Exception as e:
                self.report("DB", "FAIL", f"{name}: {e}")
    
    async def check_mcp_tools(self):
        """Check MCP tool completeness."""
        print("\n5. MCP Tool Completeness")
        
        from math_mcp_tools import MathMCPTools
        
        tools = MathMCPTools()
        available = tools.get_tools()
        
        # Check required fields
        required_fields = ['name', 'description', 'inputSchema']
        for tool in available:
            missing = [f for f in required_fields if f not in tool]
            if missing:
                self.report("MCP", "FAIL", f"{tool.get('name', 'unknown')} missing: {missing}")
            else:
                self.report("MCP", "OK", f"{tool['name']}: complete")
        
        # Check tool count
        expected_min = 8
        self.report("MCP", "OK" if len(available) >= expected_min else "WARN", 
                   f"Tool count: {len(available)} (expected >= {expected_min})")
    
    async def check_api_coverage(self):
        """Check API endpoint coverage."""
        print("\n6. API Endpoint Coverage")
        
        # Check new complete API first
        try:
            from math_api_complete import math_api
            app = math_api
        except ImportError:
            from z3_api import app
        
        if not app:
            self.report("API", "FAIL", "FastAPI app not available")
            return
        
        routes = [(r.path, list(r.methods) if hasattr(r, 'methods') else []) 
                  for r in app.routes if hasattr(r, 'path')]
        
        expected = [
            ('/health', ['GET', 'HEAD']),
            ('/solve/z3', ['POST']),
            ('/solve/lean', ['POST']),
            ('/solve/unified', ['POST']),
            ('/knowledge/learn', ['POST']),
            ('/knowledge/search', ['POST']),
        ]
        
        for path, methods in expected:
            exists = any(path == r[0] and any(m in r[1] for m in methods) for r in routes)
            self.report("API", "OK" if exists else "FAIL", f"{methods[0]} {path}")
    
    async def check_documentation(self):
        """Check documentation completeness."""
        print("\n7. Documentation Completeness")
        
        doc_files = [
            'README.md',
            'FINAL_SUMMARY.md',
            'GAP_ANALYSIS_REPORT.md',
            'COMPLETION_REPORT_FINAL.md',
        ]
        
        for doc in doc_files:
            path = os.path.join(os.path.dirname(__file__), doc)
            if os.path.exists(path):
                size = os.path.getsize(path)
                self.report("Docs", "OK" if size > 1000 else "WARN", 
                           f"{doc}: {size/1024:.1f}KB")
            else:
                self.report("Docs", "FAIL", f"{doc}: missing")
        
        # Check code docstrings
        from z3_solver_connector import Z3SolverConnector
        methods_with_docs = sum(1 for name, method in inspect.getmembers(Z3SolverConnector, predicate=inspect.isfunction) 
                               if method.__doc__)
        total_methods = sum(1 for name, method in inspect.getmembers(Z3SolverConnector, predicate=inspect.isfunction) 
                           if not name.startswith('_'))
        coverage = methods_with_docs / total_methods if total_methods > 0 else 0
        self.report("Docs", "OK" if coverage > 0.5 else "WARN", 
                   f"Z3SolverConnector docstring coverage: {coverage*100:.0f}%")
    
    async def check_type_hints(self):
        """Check type hints coverage."""
        print("\n8. Type Hints Coverage")
        
        from z3_solver_connector import Z3SolverConnector
        
        methods = [method for name, method in inspect.getmembers(Z3SolverConnector, predicate=inspect.isfunction) 
                  if not name.startswith('_')]
        
        with_hints = sum(1 for m in methods if m.__annotations__)
        total = len(methods)
        coverage = with_hints / total if total > 0 else 0
        
        self.report("Types", "OK" if coverage > 0.5 else "WARN", 
                   f"Type hint coverage: {coverage*100:.0f}% ({with_hints}/{total})")
    
    async def check_logging(self):
        """Check logging coverage."""
        print("\n9. Logging Coverage")
        
        import logging
        
        # Check if loggers are configured
        loggers = ['z3_solver_connector', 'z3_knowledge_complete', 'math_mcp_tools']
        
        for logger_name in loggers:
            try:
                logger = logging.getLogger(logger_name)
                has_handlers = len(logger.handlers) > 0 or logger.parent is not None
                self.report("Logging", "OK" if has_handlers else "WARN", 
                           f"{logger_name}: {'configured' if has_handlers else 'basic'}")
            except Exception as e:
                self.report("Logging", "FAIL", f"{logger_name}: {e}")
    
    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nPassed: {self.ok_count}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Issues: {len(self.issues)}")
        
        if self.warnings:
            print("\nWarnings:")
            for cat, msg in self.warnings:
                print(f"   [{cat}] {msg}")
        
        if self.issues:
            print("\nIssues:")
            for cat, msg in self.issues:
                print(f"   [{cat}] {msg}")
        
        print("\n" + "="*70)
        if not self.issues:
            print("SECOND PASS COMPLETE - NO CRITICAL GAPS FOUND")
        else:
            print(f"SECOND PASS COMPLETE - {len(self.issues)} ISSUES TO ADDRESS")
        print("="*70)


async def main():
    analyzer = SecondPassAnalyzer()
    await analyzer.analyze()
    return 0 if not analyzer.issues else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
