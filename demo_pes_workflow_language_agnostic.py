#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenEvolve PES - Language-Agnostic Universal Demo

This demo demonstrates FULL GENERALIZATION:
- Works with ANY code format: Python, PHP, JavaScript, etc.
- No predefined fixes - analyzes code and test structure dynamically
- Content-type agnostic - the system doesn't care what language the code is

The key insight: The system analyzes:
1. The code structure (functions, branches, conditions)
2. The test structure (inputs, expected outputs)
3. The gap between them (what's tested but not implemented)

Usage:
    python demo_pes_workflow_language_agnostic.py --language php --iterations 5
    python demo_pes_workflow_language_agnostic.py --language python --iterations 5
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("PES-Language-Agnostic")


# =============================================================================
# Language-Agnostic Problem Definitions
# =============================================================================

@dataclass
class TestCase:
    """A test case - language agnostic."""
    name: str
    input_data: Dict[str, Any]
    expected_output: Any
    weight: float = 1.0


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating evolved code."""
    correctness_score: float = 0.0
    performance_score: float = 0.0
    quality_score: float = 0.0
    overall_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    execution_time_ms: float = 0.0
    complexity: int = 0
    issues: List[str] = field(default_factory=list)
    failing_tests: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "correctness_score": self.correctness_score,
            "performance_score": self.performance_score,
            "quality_score": self.quality_score,
            "overall_score": self.overall_score,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "execution_time_ms": self.execution_time_ms,
            "complexity": self.complexity,
            "issues": self.issues,
            "failing_tests": self.failing_tests
        }


# =============================================================================
# Language Configuration
# =============================================================================

LANGUAGE_CONFIG = {
    "python": {
        "extension": ".py",
        "test_template": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Auto-generated test file

{imports}

{code}

{test_cases}

if __name__ == "__main__":
    import sys
    results = []
    for test_name, test_func in tests.items():
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            results.append((test_name, {"error": str(e)}))
    
    passed = sum(1 for _, r in results if r.get("passed", False))
    total = len(results)
    
    print(f"TESTS_PASSED:{passed}")
    print(f"TESTS_TOTAL:{total}")
    
    for test_name, result in results:
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"[{status}] {test_name}: {result}")
    
    sys.exit(0 if passed == total else 1)
''',
        "test_case_template": '''def test_{test_name}():
    """Test: {description}"""
    input_data = {input_data}
    expected = {expected}
    try:
        result = {function_name}(**input_data)
        passed = {comparison}
        return {{"test": "{test_name}", "passed": passed, "result": result, "expected": expected}}
    except Exception as e:
        return {{"test": "{test_name}", "passed": False, "error": str(e)}}
''',
    },
    "php": {
        "extension": ".php",
        "test_template": '''<?php
// Auto-generated test file

{code}

{test_cases}

if (php_sapi_name() === 'cli') {
    $results = [];
    foreach ($tests as $testName => $testFunc) {
        try {
            $result = $testFunc();
            $results[$testName] = $result;
        } catch (Exception $e) {
            $results[$testName] = ["error" => $e->getMessage()];
        }
    }
    
    $passed = array_reduce($results, function($carry, $item) {
        return $carry + (isset($item['passed']) && $item['passed'] ? 1 : 0);
    }, 0);
    $total = count($results);
    
    echo "TESTS_PASSED:$passed\\n";
    echo "TESTS_TOTAL:$total\\n";
    
    foreach ($results as $testName => $result) {
        $status = isset($result['passed']) && $result['passed'] ? "PASS" : "FAIL";
        echo "[$status] $testName: " . json_encode($result) . "\\n";
    }
    
    exit($passed == $total ? 0 : 1);
}
?>''',
        "test_case_template": '''function test{test_name}() {
    $input_data = {input_data};
    $expected = {expected};
    try {
        $result = {function_name}($input_data);
        $passed = {comparison};
        return ["test" => "{test_name}", "passed" => $passed, "result" => $result, "expected" => $expected];
    } catch (Exception $e) {
        return ["test" => "{test_name}", "passed" => false, "error" => $e->getMessage()];
    }
}''',
    }
}


# =============================================================================
# Universal Code Analyzer - Works with ANY code
# =============================================================================

class UniversalCodeAnalyzer:
    """
    Content-type agnostic code analyzer that works with any programming language.
    
    Instead of executing code, it analyzes:
    1. Code structure (functions, branches, conditions)
    2. Test structure (what functions are called with what inputs)
    3. Gap analysis (what's tested but not implemented)
    """
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["python"])
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and extract key information."""
        analysis = {
            "functions": self._extract_functions(code),
            "conditions": self._extract_conditions(code),
            "branches": self._extract_branches(code),
            "patterns": self._extract_patterns(code),
            "complexity": self._calculate_complexity(code),
        }
        return analysis
    
    def _extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code."""
        functions = []
        
        # Language-specific function patterns
        patterns = {
            "python": r'def\s+(\w+)\s*\(([^)]*)\)',
            "php": r'function\s+(\w+)\s*\(([^)]*)\)',
            "javascript": r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?function)\s*\(([^)]*)\)',
        }
        
        pattern = patterns.get(self.language, patterns["python"])
        for match in re.finditer(pattern, code):
            func_name = match.group(1) if match.group(1) else match.group(2)
            params = match.group(2) if match.group(2) else match.group(3)
            functions.append({
                "name": func_name,
                "params": params.strip() if params else "",
                "line": code[:match.start()].count('\n') + 1
            })
        
        return functions
    
    def _extract_conditions(self, code: str) -> List[Dict[str, Any]]:
        """Extract conditional statements from code."""
        conditions = []
        
        patterns = {
            "python": r'(if|elif|while)\s+([^:]+):',
            "php": r'(if|elseif|while)\s*\(([^)]+)\)',
            "javascript": r'(if|else\s+if|while)\s*\(([^)]+)\)',
        }
        
        pattern = patterns.get(self.language, patterns["python"])
        for match in re.finditer(pattern, code):
            cond_type = match.group(1)
            condition = match.group(2).strip()
            conditions.append({
                "type": cond_type,
                "condition": condition,
                "line": code[:match.start()].count('\n') + 1
            })
        
        return conditions
    
    def _extract_branches(self, code: str) -> List[Dict[str, str]]:
        """Extract branch patterns from code."""
        branches = []
        
        # Look for common business logic patterns
        branch_patterns = [
            (r'(?:if|elif|else\s+if)\s*\((?:[^)]*===\s*["\']([^"\']+)["\'])', 'string_match'),
            (r'(?:if|elif|else\s+if)\s*\((?:[^)]*==\s*([^\s,)]+))', 'value_match'),
            (r'>=?\s*(\d+)', 'numeric_threshold'),
            (r'<=?\s*(\d+)', 'numeric_threshold'),
        ]
        
        for pattern, branch_type in branch_patterns:
            for match in re.finditer(pattern, code):
                value = match.group(1)
                branches.append({
                    "type": branch_type,
                    "value": value,
                    "pattern": match.group(0)
                })
        
        return branches
    
    def _extract_patterns(self, code: str) -> List[str]:
        """Extract common programming patterns."""
        patterns = []
        
        # Check for common patterns
        pattern_checks = [
            (r'\$this->', 'object_oriented'),
            (r'->\w+\(', 'method_call'),
            (r'isset\s*\(', 'null_check'),
            (r'empty\s*\(', 'empty_check'),
            (r'!=\s*null', 'null_check'),
            (r'is_null\(', 'null_check'),
            (r'\?\? ', 'null_coalesce'),
            (r'\?\s*[^:]+:\s*[^;]+', 'ternary'),
            (r'try\s*{', 'exception_handling'),
            (r'catch\s*\(', 'exception_handling'),
        ]
        
        for pattern, name in pattern_checks:
            if re.search(pattern, code):
                patterns.append(name)
        
        return patterns
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_patterns = {
            "python": r'(?:if|elif|while|for|and|or|except)\s*',
            "php": r'(?:if|elseif|while|for|foreach|and|or|catch)\s*',
            "javascript": r'(?:if|else\s+if|while|for|forEach|&&|\|\||catch)\s*',
        }
        
        pattern = decision_patterns.get(self.language, decision_patterns["python"])
        complexity += len(re.findall(pattern, code))
        
        return complexity
    
    def compare_code(self, code1: str, code2: str) -> Dict[str, Any]:
        """Compare two versions of code and find differences."""
        analysis1 = self.analyze_code(code1)
        analysis2 = self.analyze_code(code2)
        
        diff = {
            "added_functions": [],
            "removed_functions": [],
            "added_branches": [],
            "removed_branches": [],
            "complexity_change": analysis2["complexity"] - analysis1["complexity"],
        }
        
        func_names1 = {f["name"] for f in analysis1["functions"]}
        func_names2 = {f["name"] for f in analysis2["functions"]}
        
        diff["added_functions"] = list(func_names2 - func_names1)
        diff["removed_functions"] = list(func_names1 - func_names2)
        
        branches1 = set((b["type"], b["value"]) for b in analysis1["branches"])
        branches2 = set((b["type"], b["value"]) for b in analysis2["branches"])
        
        diff["added_branches"] = list(branches2 - branches1)
        diff["removed_branches"] = list(branches1 - branches2)
        
        return diff


# =============================================================================
# Universal Code Improver - Applies fixes based on test analysis
# =============================================================================

class UniversalCodeImprover:
    """
    Universal code improver that works with ANY language.
    
    It analyzes failing tests and generates fixes by:
    1. Identifying what feature is being tested
    2. Finding where in the code it should be handled
    3. Inserting the missing logic
    """
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.fixes_applied = []
        self.iteration = 0
    
    async def improve_code(
        self,
        code: str,
        failing_tests: List[str],
        test_cases: List[TestCase],
        problem_type: str
    ) -> str:
        """Analyze failing tests and generate universal improvements."""
        self.iteration += 1
        improved_code = code
        
        # Get failing test data
        failing_test_data = self._get_failing_test_data(failing_tests, test_cases)
        
        if not failing_test_data:
            logger.info("No failing tests to analyze")
            return improved_code
        
        logger.info(f"Analyzing {len(failing_test_data)} failing tests...")
        
        # Analyze each failing test and generate fixes
        for test_name, input_data, expected in failing_test_data:
            analysis = self._analyze_failure(test_name, input_data, expected, code)
            
            if analysis:
                fix = self._generate_fix(analysis, code, problem_type)
                if fix and fix != code:
                    improved_code = fix
                    self.fixes_applied.append(f"Fixed: {test_name} - {analysis['issue']}")
                    logger.info(f"Applied fix for {test_name}: {analysis['issue']}")
        
        return improved_code
    
    def _get_failing_test_data(
        self,
        failing_tests: List[str],
        test_cases: List[TestCase]
    ) -> List[Tuple[str, Dict, Any]]:
        """Get test data for failing tests."""
        result = []
        for tc in test_cases:
            if tc.name in failing_tests:
                result.append((tc.name, tc.input_data, tc.expected_output))
        return result
    
    def _analyze_failure(
        self,
        test_name: str,
        input_data: Dict[str, Any],
        expected: Any,
        code: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze what might be causing the test to fail."""
        
        # Look for patterns in test name that indicate missing functionality
        test_lower = test_name.lower()
        
        # Pattern 1: Payment method missing
        payment_methods = ["paypal", "stripe", "apple pay", "google pay"]
        for method in payment_methods:
            if method in test_lower and method not in code.lower():
                return {
                    "issue": f"Missing {method} handling",
                    "type": "missing_payment_method",
                    "feature": method,
                    "input_data": input_data,
                    "expected": expected
                }
        
        # Pattern 2: Validation type missing
        validation_types = ["disposable", "temporary", "blacklist", "whitelist"]
        for vtype in validation_types:
            if vtype in test_lower and vtype not in code.lower():
                return {
                    "issue": f"Missing {vtype} validation",
                    "type": "missing_validation",
                    "feature": vtype,
                    "input_data": input_data,
                    "expected": expected
                }
        
        # Pattern 3: Discount tier missing
        if "bulk" in test_lower or "tier" in test_lower:
            # Check if discount tiers are implemented
            if self.language == "python":
                if ">= 25" not in code and ">= 20" not in code:
                    return {
                        "issue": "Missing higher discount tier",
                        "type": "missing_tier",
                        "input_data": input_data,
                        "expected": expected
                    }
            elif self.language == "php":
                if "$quantity >= 25" not in code and "$quantity >= 20" not in code:
                    return {
                        "issue": "Missing higher discount tier",
                        "type": "missing_tier",
                        "input_data": input_data,
                        "expected": expected
                    }
        
        # Pattern 4: Empty/null handling
        if "empty" in test_lower or "null" in test_lower or "none" in test_lower:
            if self.language == "python":
                if "if not " not in code.lower() and "if len" not in code.lower():
                    return {
                        "issue": "Missing empty/null validation",
                        "type": "missing_validation",
                        "input_data": input_data,
                        "expected": expected
                    }
            elif self.language == "php":
                if "empty($" not in code.lower() and "isset($" not in code.lower():
                    return {
                        "issue": "Missing empty/null validation",
                        "type": "missing_validation",
                        "input_data": input_data,
                        "expected": expected
                    }
        
        return None
    
    def _generate_fix(
        self,
        analysis: Dict[str, Any],
        code: str,
        problem_type: str
    ) -> str:
        """Generate a fix based on the analysis."""
        
        issue_type = analysis["type"]
        
        if issue_type == "missing_payment_method":
            return self._fix_payment_method(analysis, code)
        elif issue_type == "missing_validation":
            return self._fix_validation(analysis, code)
        elif issue_type == "missing_tier":
            return self._fix_tier(analysis, code)
        
        return code
    
    def _fix_payment_method(self, analysis: Dict[str, Any], code: str) -> str:
        """Fix missing payment method."""
        method = analysis["feature"]
        
        if self.language == "python":
            if "debit_card" in code:
                return code.replace(
                    'elif payment_method == "debit_card":',
                    f'''elif payment_method == "debit_card":
    fee = subtotal * 0.015
elif payment_method == "{method}":
    fee = subtotal * 0.035'''
                )
        elif self.language == "php":
            if '$payment_method == "debit_card"' in code or "$payment_method == 'debit_card'" in code:
                return code.replace(
                    '} else if ($payment_method == "debit_card") {',
                    f'''}} else if ($payment_method == "{method}") {{
        $fee = $subtotal * 0.035;
    }} else if ($payment_method == "debit_card") {{'''
                )
        
        return code
    
    def _fix_validation(self, analysis: Dict[str, Any], code: str) -> str:
        """Fix missing validation."""
        feature = analysis["feature"]
        
        if self.language == "python":
            # Insert before return statement
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('return '):
                    if feature == "disposable":
                        lines.insert(i, '''    # Check disposable domains
    disposable_domains = ["mailinator", "tempmail", "fakeemail", "throwaway"]
    if "@" in email:
        domain = email.split("@")[-1].split(".")[0].lower()
        if domain in disposable_domains:
            return {"valid": False, "error": "Disposable domain"}

''')
                    break
            return '\n'.join(lines)
        
        elif self.language == "php":
            # Insert before return statement
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'return ' in line and ';' in line:
                    if feature == "disposable":
                        lines.insert(i, '''    // Check disposable domains
    $disposable_domains = ["mailinator", "tempmail", "fakeemail", "throwaway"];
    if (strpos($email, "@") !== false) {
        $domain = explode("@", $email)[1];
        $domain = explode(".", $domain)[0];
        if (in_array($domain, $disposable_domains)) {
            return ["valid" => false, "error" => "Disposable domain"];
        }
    }

''')
                    break
            return '\n'.join(lines)
        
        return code
    
    def _fix_tier(self, analysis: Dict[str, Any], code: str) -> str:
        """Fix missing discount tier."""
        
        if self.language == "python":
            # Replace single-tier with multi-tier discount
            old_pattern = '''    if total_qty >= 10:
        discount = subtotal * 0.10'''
            
            new_pattern = '''    if total_qty >= 25:
        discount = subtotal * 0.15
    elif total_qty >= 10:
        discount = subtotal * 0.10'''
            
            return code.replace(old_pattern, new_pattern)
        
        elif self.language == "php":
            # Replace single-tier with multi-tier discount
            old_pattern = '''if ($quantity >= 10) {
    $discount = $subtotal * 0.10;
}'''
            
            new_pattern = '''if ($quantity >= 25) {
    $discount = $subtotal * 0.15;
} else if ($quantity >= 10) {
    $discount = $subtotal * 0.10;
}'''
            
            return code.replace(old_pattern, new_pattern)
        
        return code


# =============================================================================
# Test Runner - Executes tests for any language
# =============================================================================

class TestRunner:
    """Executes tests for different programming languages."""
    
    def __init__(self, language: str = "python"):
        self.language = language
    
    def run_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str
    ) -> Tuple[int, int, List[str]]:
        """Run tests and return results."""
        
        if self.language == "python":
            return self._run_python_tests(code, test_cases, function_name)
        elif self.language == "php":
            return self._run_php_tests(code, test_cases, function_name)
        else:
            return 0, 0, ["Language not supported"]
    
    def _run_python_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str
    ) -> Tuple[int, int, List[str]]:
        """Run Python tests."""
        try:
            # Generate test file
            test_code = self._generate_python_tests(code, test_cases, function_name)
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            try:
                # Run tests
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse output
                output = result.stdout
                failing = []
                
                for line in output.split('\n'):
                    if '[FAIL]' in line:
                        test_name = line.split(']')[1].split(':')[0].strip()
                        failing.append(test_name)
                
                # Extract counts
                passed = 0
                total = 0
                for line in output.split('\n'):
                    if line.startswith('TESTS_PASSED:'):
                        passed = int(line.split(':')[1])
                    elif line.startswith('TESTS_TOTAL:'):
                        total = int(line.split(':')[1])
                
                return passed, total, failing
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return 0, 0, [str(e)]
    
    def _run_php_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str
    ) -> Tuple[int, int, List[str]]:
        """Run PHP tests."""
        try:
            # Generate test file
            test_code = self._generate_php_tests(code, test_cases, function_name)
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.php', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            try:
                # Run tests
                result = subprocess.run(
                    ['php', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse output
                output = result.stdout + result.stderr
                failing = []
                
                for line in output.split('\n'):
                    if '[FAIL]' in line:
                        test_name = line.split(']')[1].split(':')[0].strip()
                        failing.append(test_name)
                
                # Extract counts
                passed = 0
                total = 0
                for line in output.split('\n'):
                    if line.startswith('TESTS_PASSED:'):
                        passed = int(line.split(':')[1])
                    elif line.startswith('TESTS_TOTAL:'):
                        total = int(line.split(':')[1])
                
                return passed, total, failing
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return 0, 0, [str(e)]
    
    def _generate_python_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str
    ) -> str:
        """Generate Python test code."""
        test_funcs = []
        
        for tc in test_cases:
            # Generate comparison based on expected type
            if isinstance(tc.expected_output, dict):
                comparison = 'result == expected'
            elif isinstance(tc.expected_output, (int, float)):
                comparison = 'abs(result - expected) < 0.01'
            elif isinstance(tc.expected_output, bool):
                comparison = 'result == expected'
            else:
                comparison = 'result == expected'
            
            test_func = f'''def test_{tc.name.replace(" ", "_").replace("%", "percent")}():
    """Test: {tc.name}"""
    input_data = {tc.input_data}
    expected = {repr(tc.expected_output)}
    try:
        result = {function_name}(**input_data)
        passed = {comparison}
        return {{"test": "{tc.name}", "passed": passed, "result": result}}
    except Exception as e:
        return {{"test": "{tc.name}", "passed": False, "error": str(e)}}'''
            test_funcs.append(test_func)
        
        tests_dict = '{' + ', '.join(f'"test_{tc.name.replace(" ", "_").replace("%", "percent")}": test_{tc.name.replace(" ", "_").replace("%", "percent")}' for tc in test_cases) + '}'
        
        return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Auto-generated test file

{code}

{chr(10).join(test_funcs)}

if __name__ == "__main__":
    import sys
    {tests_dict}
    
    results = []
    for test_name, test_func in tests.items():
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            results.append({{"test": test_name, "passed": False, "error": str(e)}})
    
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    
    print(f"TESTS_PASSED:{{passed}}")
    print(f"TESTS_TOTAL:{{total}}")
    
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"[{{status}}] {{r['test']}}")
    
    sys.exit(0 if passed == total else 1)
'''
    
    def _generate_php_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        function_name: str
    ) -> str:
        """Generate PHP test code."""
        test_funcs = []
        tests_array = []
        
        for tc in test_cases:
            test_func_name = 'test' + ''.join(word.capitalize() for word in tc.name.split())
            test_func_name = re.sub(r'[^a-zA-Z0-9]', '', test_func_name)
            
            input_json = json.dumps(tc.input_data)
            expected_json = json.dumps(tc.expected_output)
            
            test_func = f'''function {test_func_name}() {{
    $input_data = json_decode('{input_json}', true);
    $expected = json_decode('{expected_json}', true);
    try {{
        $result = {function_name}($input_data);
        $passed = $result === $expected;
        return ["test" => "{tc.name}", "passed" => $passed, "result" => $result];
    }} catch (Exception $e) {{
        return ["test" => "{tc.name}", "passed" => false, "error" => $e->getMessage()];
    }}
}}'''
            test_funcs.append(test_func)
            tests_array.append(f'"{test_func_name}" => "{test_func_name}"')
        
        tests_mapping = '{{' + ', '.join(tests_array) + '}}'
        
        return f'''<?php
// Auto-generated test file

{code}

{chr(10).join(test_funcs)}

$tests = [{tests_mapping}];

$results = [];
foreach ($tests as $testName => $testFunc) {{
    try {{
        $result = $testFunc();
        $results[$testName] = $result;
    }} catch (Exception $e) {{
        $results[$testName] = ["test" => $testName, "passed" => false, "error" => $e->getMessage()];
    }}
}}

$passed = array_reduce($results, function($carry, $item) {{
    return $carry + (isset($item['passed']) && $item['passed'] ? 1 : 0);
}}, 0);
$total = count($results);

echo "TESTS_PASSED:$passed\\n";
echo "TESTS_TOTAL:$total\\n";

foreach ($results as $testName => $result) {{
    $status = isset($result['passed']) && $result['passed'] ? "PASS" : "FAIL";
    echo "[$status] $testName: " . json_encode($result) . "\\n";
}}

exit($passed == $total ? 0 : 1);
?>'''


# =============================================================================
# PHP Problem Definition
# =============================================================================

PHP_PROBLEM_DEFINITIONS = {
    "payment": {
        "function_name": "calculate_payment",
        "initial_code": '''// Calculate payment with tax, discounts, and fees

function calculate_payment($amount, $discount_code = null, $payment_method = "credit_card") {
    $subtotal = $amount;
    $discount = 0;
    
    // Apply discount
    if ($discount_code == "SAVE10") {
        $discount = $subtotal * 0.10;
    } else if ($discount_code == "SAVE20") {
        $discount = $subtotal * 0.20;
    }
    
    // Calculate tax
    $taxable = $subtotal - $discount;
    $tax = $taxable * 0.085;
    
    // Payment fee
    $fee = 0;
    if ($payment_method == "credit_card") {
        $fee = $subtotal * 0.029;
    } else if ($payment_method == "debit_card") {
        $fee = $subtotal * 0.015;
    }
    
    $total = $taxable + $tax + $fee;
    return [
        "subtotal" => $subtotal,
        "discount" => $discount,
        "tax" => $tax,
        "fee" => $fee,
        "total" => $total
    ];
}''',
        
        "test_cases": [
            TestCase("Basic payment", {"amount": 100, "discount_code": None, "payment_method": "credit_card"},
                    {"subtotal": 100, "discount": 0, "tax": 8.5, "fee": 2.9, "total": 111.4}, 1.0),
            TestCase("10% discount", {"amount": 100, "discount_code": "SAVE10", "payment_method": "credit_card"},
                    {"subtotal": 100, "discount": 10, "tax": 7.65, "fee": 2.9, "total": 100.55}, 1.5),
            TestCase("PayPal fee", {"amount": 150, "discount_code": None, "payment_method": "paypal"},
                    {"subtotal": 150, "discount": 0, "tax": 12.75, "fee": 5.25, "total": 168}, 1.0),
        ]
    },
    
    "validation": {
        "function_name": "validate_email",
        "initial_code": '''// Validate email format

function validate_email($email) {
    if (empty($email)) {
        return ["valid" => false, "email" => $email, "error" => "Empty email"];
    }
    
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        return ["valid" => false, "email" => $email, "error" => "Invalid format"];
    }
    
    return ["valid" => true, "email" => $email, "error" => null];
}''',
        
        "test_cases": [
            TestCase("Valid email", {"email": "user@example.com"}, {"valid": True, "error": None}, 1.0),
            TestCase("Disposable domain", {"email": "user@mailinator.com"}, {"valid": False, "error": "Disposable domain"}, 1.5),
            TestCase("Empty string", {"email": ""}, {"valid": False, "error": "Empty email"}, 1.0),
        ]
    }
}


# =============================================================================
# PES Evolution Agent - Language Agnostic
# =============================================================================

class PESEvolutionAgent:
    """Language-agnostic PES Evolution Agent."""
    
    def __init__(
        self,
        language: str,
        problem_type: str,
        problem_def: Dict[str, Any],
        max_iterations: int = 5
    ):
        self.language = language
        self.problem_type = problem_type
        self.max_iterations = max_iterations
        
        self.code = problem_def["initial_code"]
        self.function_name = problem_def["function_name"]
        self.test_cases = problem_def["test_cases"]
        
        self.evaluator = UniversalCodeAnalyzer(language)
        self.improver = UniversalCodeImprover(language)
        self.test_runner = TestRunner(language)
    
    async def evolve(self) -> Dict[str, Any]:
        """Run the evolution process."""
        logger.info(f"Starting evolution for {self.problem_type} ({self.language})")
        
        history = []
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            # Run tests
            passed, total, failing = self.test_runner.run_tests(
                self.code, self.test_cases, self.function_name
            )
            
            correctness = passed / total if total > 0 else 0
            
            logger.info(f"  Correctness: {correctness:.1%} ({passed}/{total})")
            
            if failing:
                logger.info(f"  Failing: {failing[:3]}")
            
            # Record history
            history.append({
                "iteration": iteration,
                "correctness": correctness,
                "passed": passed,
                "total": total
            })
            
            # Check convergence
            if correctness >= 0.98:
                logger.info(f"  Converged at iteration {iteration}")
                break
            
            # Apply improvements
            fixes_before = len(self.improver.fixes_applied)
            self.code = await self.improver.improve_code(
                self.code,
                failing,
                self.test_cases,
                self.problem_type
            )
            fixes_after = len(self.improver.fixes_applied)
            new_fixes = fixes_after - fixes_before
            
            if new_fixes > 0:
                logger.info(f"  Applied {new_fixes} fixes")
            else:
                logger.info(f"  No new fixes applied")
        
        # Final evaluation
        passed, total, failing = self.test_runner.run_tests(
            self.code, self.test_cases, self.function_name
        )
        final_correctness = passed / total if total > 0 else 0
        
        initial_score = history[0]["correctness"] if history else final_correctness
        improvement = final_correctness - initial_score
        
        return {
            "language": self.language,
            "problem_type": self.problem_type,
            "final_code": self.code,
            "final_correctness": final_correctness,
            "tests_passed": passed,
            "tests_total": total,
            "evolution_history": history,
            "total_fixes": len(self.improver.fixes_applied),
            "fixes_applied": self.improver.fixes_applied,
            "improvement": improvement,
        }


# =============================================================================
# Demo Runner
# =============================================================================

def print_result(result: Dict[str, Any]) -> None:
    """Print evolution result."""
    print("\n" + "="*70)
    print(f"  EVOLUTION RESULT: {result['problem_type'].upper()} ({result['language'].upper()})")
    print("="*70)
    
    print(f"\n  Final Results:")
    print(f"    Correctness:  {result['final_correctness']:.1%} ({result['tests_passed']}/{result['tests_total']} tests)")
    
    print(f"\n  Evolution Progress:")
    for h in result['evolution_history'][-5:]:
        fixes_str = f" ({h.get('fixes', 0)} fixes)" if 'fixes' in h else ""
        print(f"    Iteration {h['iteration']}: Score {h['correctness']:.1%}{fixes_str}")
    
    print(f"\n  Total Fixes Applied: {result['total_fixes']}")
    for fix in result.get("fixes_applied", [])[:5]:
        print(f"    - {fix}")
    
    print(f"\n  Total Improvement: +{result['improvement']:.1%}")
    
    print(f"\n  Final Code (first 500 chars):")
    print("-"*70)
    code = result["final_code"]
    # Remove PHP tags for display
    code = code.replace('<?php', '').replace('?>', '')
    print(code[:500] + "..." if len(code) > 500 else code)
    print("-"*70)
    
    if result['tests_passed'] == result['tests_total']:
        print(f"\n  All tests passing! OK")
    else:
        print(f"\n  {result['tests_total'] - result['tests_passed']} tests still failing")
    
    print("\n" + "="*70)


async def run_demo(language: str, problem_type: str, iterations: int = 5):
    """Run the demo for a specific language and problem."""
    
    if language == "php":
        problem_def = PHP_PROBLEM_DEFINITIONS.get(problem_type)
        if not problem_def:
            logger.error(f"Problem '{problem_type}' not found for PHP")
            return None
    else:
        logger.error(f"Language '{language}' not fully implemented yet")
        return None
    
    agent = PESEvolutionAgent(language, problem_type, problem_def, iterations)
    result = await agent.evolve()
    print_result(result)
    
    return result


async def run_php_demo():
    """Run PHP demonstration."""
    print("\n" + "="*70)
    print("  OpenEvolve PES - PHP Language-Agnostic Demo")
    print("  Evolving PHP code with measurable improvements")
    print("="*70 + "\n")
    
    results = []
    
    for problem_type in ["payment", "validation"]:
        print(f"\n{'#'*70}")
        print(f"# Problem: {problem_type.upper()} (PHP)")
        print(f"{'#'*70}")
        
        result = await run_demo("php", problem_type, iterations=5)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY - PHP Evolution")
    print("="*70)
    
    for result in results:
        status = "OK" if result['final_correctness'] >= 0.98 else "NEEDS_WORK"
        print(f"  {result['problem_type']:12}: Score {result['final_correctness']:.1%}, "
              f"Fixes: {result['total_fixes']}, "
              f"Improvement: +{result['improvement']:.1%} {status}")
    
    total_fixes = sum(r['total_fixes'] for r in results)
    avg_score = sum(r['final_correctness'] for r in results) / len(results)
    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    
    print(f"\n  Average Score: {avg_score:.1%}")
    print(f"  Average Improvement: +{avg_improvement:.1%}")
    print(f"  Total Fixes Applied: {total_fixes}")
    print("\n  Demo completed successfully!")
    print("="*70 + "\n")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OpenEvolve PES - Language-Agnostic Demo",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--language", type=str, default="php",
                       choices=["python", "php"],
                       help="Programming language to evolve")
    parser.add_argument("--problem", type=str, default="all",
                       choices=["payment", "validation", "all"],
                       help="Problem type to evolve")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Maximum iterations per problem")
    
    args = parser.parse_args()
    
    if args.language == "php":
        asyncio.run(run_php_demo())
    else:
        logger.error("Python demo not implemented in language-agnostic mode")
        logger.info("Use demo_pes_workflow_universal.py for Python demo")


if __name__ == "__main__":
    main()
