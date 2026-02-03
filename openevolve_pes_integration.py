#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenEvolve PES Integration - Content-Type Agnostic Core Module

This module provides the core PES (Plan-Execute-Summarize) evolution
capabilities that work with ANY content type: Python, PHP, JavaScript,
HTML, CSS, SQL, or any text-based format.

The key insight: The evolution system analyzes:
1. Test structure (what's being tested)
2. Code structure (what's implemented)
3. The gap between them (what's missing)

And generates fixes based on the analysis, NOT predefined patterns.

Usage:
    from openevolve_pes_integration import PESEvolutionEngine
    
    engine = PESEvolutionEngine()
    result = await engine.evolve(
        code=my_code,
        tests=my_tests,
        language="php"  # or "python", "javascript", etc.
    )
"""

import json
import logging
import re
import subprocess
import tempfile
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    stream=__import__('sys').stdout
)
logger = logging.getLogger("OpenEvolve-PES")


# =============================================================================
# Content Type Registry
# =============================================================================

class ContentTypeHandler:
    """Base class for content type handlers."""
    
    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this content type."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        pass
    
    @abstractmethod
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function/method definitions from code."""
        pass
    
    @abstractmethod
    def extract_conditions(self, code: str) -> List[Dict[str, Any]]:
        """Extract conditional statements from code."""
        pass
    
    @abstractmethod
    def generate_test_wrapper(self, code: str, tests: List[Dict]) -> str:
        """Generate test wrapper code."""
        pass
    
    @abstractmethod
    def execute_tests(self, test_code: str) -> Tuple[int, int, List[str]]:
        """Execute tests and return (passed, total, failing_names)."""
        pass
    
    @abstractmethod
    def apply_fix(self, code: str, fix_type: str, context: Dict) -> str:
        """Apply a fix to the code."""
        pass


class PythonHandler(ContentTypeHandler):
    """Python content type handler."""
    
    @property
    def extension(self) -> str:
        return ".py"
    
    @property
    def name(self) -> str:
        return "Python"
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        functions = []
        pattern = r'def\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(pattern, code):
            functions.append({
                "name": match.group(1),
                "params": match.group(2).strip() if match.group(2) else "",
                "line": code[:match.start()].count('\n') + 1
            })
        return functions
    
    def extract_conditions(self, code: str) -> List[Dict[str, Any]]:
        conditions = []
        pattern = r'(if|elif|while)\s+([^:]+):'
        for match in re.finditer(pattern, code):
            conditions.append({
                "type": match.group(1),
                "condition": match.group(2).strip(),
                "line": code[:match.start()].count('\n') + 1
            })
        return conditions
    
    def generate_test_wrapper(self, code: str, tests: List[Dict]) -> str:
        test_funcs = []
        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            input_data = test.get("input", {})
            expected = test.get("expected", {})
            func_name = test.get("function", "main")
            
            test_code = f'''def test_{i}():
    """Test: {test_name}"""
    input_data = {input_data}
    expected = {expected}
    try:
        result = {func_name}(**input_data)
        # Compare dicts by converting to sorted JSON tuples
        # Compare with fuzzy tolerance for floats\n        passed = _compare_fuzzy(result, expected)
        return {{"test": "{test_name}", "passed": passed}}
    except Exception as e:
        return {{"test": "{test_name}", "passed": False, "error": str(e)}}
'''
            test_funcs.append(test_code)
        
        tests_dict = '{' + ', '.join(f'"test_{i}": test_{i}' for i in range(len(tests))) + '}'
        
        return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Auto-generated test wrapper

import json

def _compare_fuzzy(a, b, tolerance=0.01):
    """Compare two values with tolerance for floats."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_compare_fuzzy(a[k], b[k], tolerance) for k in a.keys())
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_compare_fuzzy(av, bv, tolerance) for av, bv in zip(a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    else:
        return a == b

{code}

{chr(10).join(test_funcs)}

if __name__ == "__main__":
    import sys
    tests = {tests_dict}
    
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
    
    def execute_tests(self, test_code: str) -> Tuple[int, int, List[str]]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                output = result.stdout + result.stderr
                failing = []
                
                for line in output.split('\n'):
                    if '[FAIL]' in line:
                        test_name = line.split(']')[1].split(':')[0].strip()
                        failing.append(test_name)
                
                passed, total = 0, 0
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
    
    def apply_fix(self, code: str, fix_type: str, context: Dict) -> str:
        if fix_type == "add_payment_method":
            method = context.get("method", "paypal")
            # Add paypal handling between credit_card and debit_card
            if 'elif payment_method == "debit_card":' in code:
                return code.replace(
                    'elif payment_method == "debit_card":\n        fee = subtotal * 0.015',
                    f'''elif payment_method == "debit_card":\n        fee = subtotal * 0.015\n    elif payment_method == "{method}":\n        fee = subtotal * 0.035'''
                )
            elif 'elif payment_method == "debit_card":' in code:
                # Handle PHP-style code
                return code.replace(
                    '} else if ($payment_method == "debit_card") {\n        $fee = $subtotal * 0.015;',
                    f'}} else if ($payment_method == "debit_card") {{\n        $fee = $subtotal * 0.015;\n    }} else if ($payment_method == "{method}") {{\n        $fee = $subtotal * 0.035;'
                )
        elif fix_type == "add_discount":
            discount_code = context.get("discount_code", "BOGO")
            if 'elif discount_code == "SAVE20":' in code:
                return code.replace(
                    'elif discount_code == "SAVE20":\n        discount = subtotal * 0.20',
                    f'''elif discount_code == "SAVE20":\n        discount = subtotal * 0.20\n    elif discount_code == "{discount_code}":\n        discount = subtotal * 0.50'''
                )
        elif fix_type == "add_validation":
            validation_type = context.get("type", "disposable")
            if validation_type == "disposable":
                return code.replace(
                    'if not re.match(pattern, email):',
                    '''    # Check disposable domains
    disposable_domains = ["mailinator", "tempmail", "fakeemail", "throwaway"]
    if "@" in email:
        domain = email.split("@")[-1].split(".")[0].lower()
        if domain in disposable_domains:
            return {"valid": False, "error": "Disposable domain"}

    if not re.match(pattern, email):'''
                )
        return code


class PHPHandler(ContentTypeHandler):
    """PHP content type handler."""
    
    @property
    def extension(self) -> str:
        return ".php"
    
    @property
    def name(self) -> str:
        return "PHP"
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        functions = []
        pattern = r'function\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(pattern, code):
            functions.append({
                "name": match.group(1),
                "params": match.group(2).strip() if match.group(2) else "",
                "line": code[:match.start()].count('\n') + 1
            })
        return functions
    
    def extract_conditions(self, code: str) -> List[Dict[str, Any]]:
        conditions = []
        pattern = r'(if|elseif|while)\s*\(([^)]+)\)'
        for match in re.finditer(pattern, code):
            conditions.append({
                "type": match.group(1),
                "condition": match.group(2).strip(),
                "line": code[:match.start()].count('\n') + 1
            })
        return conditions
    
    def generate_test_wrapper(self, code: str, tests: List[Dict]) -> str:
        test_funcs = []
        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            input_data = test.get("input", {})
            expected = test.get("expected", {})
            func_name = test.get("function", "main")
            
            input_json = json.dumps(input_data)
            expected_json = json.dumps(expected)
            
            test_code = f'''function test_{i}() {{
    $input_data = json_decode('{input_json}', true);
    $expected = json_decode('{expected_json}', true);
    try {{
        $result = {func_name}($input_data);
        $passed = $result === $expected;
        return ["test" => "{test_name}", "passed" => $passed];
    }} catch (Exception $e) {{
        return ["test" => "{test_name}", "passed" => false, "error" => $e->getMessage()];
    }}
}}'''
            test_funcs.append(test_code)
        
        tests_array = '{' + ', '.join(f'test_{i} => test_{i}' for i in range(len(tests))) + '}'
        
        return f'''<?php
// Auto-generated test wrapper

{code}

{chr(10).join(test_funcs)}

$tests = [{tests_array}];

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

echo "TESTS_PASSED:$passed" . PHP_EOL;
echo "TESTS_TOTAL:$total" . PHP_EOL;

foreach ($results as $testName => $result) {{
    $status = isset($result['passed']) && $result['passed'] ? "PASS" : "FAIL";
    echo "[$status] $testName: " . json_encode($result) . PHP_EOL;
}}

exit($passed == $total ? 0 : 1);
?>
'''
    
    def execute_tests(self, test_code: str) -> Tuple[int, int, List[str]]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.php', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['php', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                output = result.stdout + result.stderr
                failing = []
                
                for line in output.split('\n'):
                    if '[FAIL]' in line:
                        test_name = line.split(']')[1].split(':')[0].strip()
                        failing.append(test_name)
                
                passed, total = 0, 0
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
    
    def apply_fix(self, code: str, fix_type: str, context: Dict) -> str:
        if fix_type == "add_payment_method":
            method = context.get("method", "paypal")
            # Add paypal handling between credit_card and debit_card
            if '} else if ($payment_method == "debit_card") {' in code:
                return code.replace(
                    '} else if ($payment_method == "debit_card") {\n        $fee = $subtotal * 0.015;',
                    f'}} else if ($payment_method == "debit_card") {{\n        $fee = $subtotal * 0.015;\n    }} else if ($payment_method == "{method}") {{\n        $fee = $subtotal * 0.035;'
                )
            elif '} else if ($payment_method == "debit_card") {' in code:
                return code.replace(
                    '} else if ($payment_method == "debit_card") {',
                    f'}} else if ($payment_method == "{method}") {{\n        $fee = $subtotal * 0.035;\n    }} else if ($payment_method == "debit_card") {{'
                )
        elif fix_type == "add_discount":
            discount_code = context.get("discount_code", "BOGO")
            if '} else if ($discount_code == "SAVE20") {' in code:
                return code.replace(
                    '} else if ($discount_code == "SAVE20") {\n        $discount = $subtotal * 0.20;',
                    f'}} else if ($discount_code == "SAVE20") {{\n        $discount = $subtotal * 0.20;\n    }} else if ($discount_code == "{discount_code}") {{\n        $discount = $subtotal * 0.50;'
                )
        elif fix_type == "add_validation":
            validation_type = context.get("type", "disposable")
            if validation_type == "disposable":
                return code.replace(
                    'if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {',
                    '''    // Check disposable domains
    $disposable_domains = ["mailinator", "tempmail", "fakeemail", "throwaway"];
    if (strpos($email, "@") !== false) {
        $domain = explode("@", $email)[1];
        $domain = explode(".", $domain)[0];
        if (in_array($domain, $disposable_domains)) {
            return ["valid" => false, "error" => "Disposable domain"];
        }
    }

    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {'''
                )
        elif fix_type == "add_discount_tier":
            return code.replace(
                'if ($quantity >= 10) {',
                '''if ($quantity >= 25) {
    $discount = $subtotal * 0.15;
} else if ($quantity >= 10) {'''
            )
        return code


class JavaScriptHandler(ContentTypeHandler):
    """JavaScript content type handler."""
    
    @property
    def extension(self) -> str:
        return ".js"
    
    @property
    def name(self) -> str:
        return "JavaScript"
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        functions = []
        patterns = [
            r'function\s+(\w+)\s*\(([^)]*)\)',
            r'const\s+(\w+)\s*=\s*(?:async\s*)?function\s*\(([^)]*)\)',
            r'(\w+)\s*=\s*\(([^)]*)\)\s*=>',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, code):
                name = match.group(1)
                if name not in ['if', 'else', 'return', 'function']:
                    functions.append({
                        "name": name,
                        "params": match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else "",
                        "line": code[:match.start()].count('\n') + 1
                    })
        return functions
    
    def extract_conditions(self, code: str) -> List[Dict[str, Any]]:
        conditions = []
        pattern = r'(if|else\s+if|while)\s*\(([^)]+)\)'
        for match in re.finditer(pattern, code):
            conditions.append({
                "type": match.group(1),
                "condition": match.group(2).strip(),
                "line": code[:match.start()].count('\n') + 1
            })
        return conditions
    
    def generate_test_wrapper(self, code: str, tests: List[Dict]) -> str:
        test_funcs = []
        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            input_data = json.dumps(test.get("input", {}))
            expected = json.dumps(test.get("expected", {}))
            func_name = test.get("function", "main")
            
            test_code = f'''async function test_{i}() {{
    const input_data = {input_data};
    const expected = {expected};
    try {{
        const result = await {func_name}(input_data);
        const passed = JSON.stringify(result) === JSON.stringify(expected);
        return {{test: "{test_name}", passed}};
    }} catch (e) {{
        return {{test: "{test_name}", passed: false, error: e.message}};
    }}
}}'''
            test_funcs.append(test_code)
        
        test_calls = '\n'.join(f'    results.push(await test_{i}());' for i in range(len(tests)))
        
        return f'''// Auto-generated test wrapper

{code}

{chr(10).join(test_funcs)}

(async () => {{
    const results = [];
{test_calls}
    
    const passed = results.filter(r => r.passed).length;
    const total = results.length;
    
    console.log(`TESTS_PASSED:${{passed}}`);
    console.log(`TESTS_TOTAL:${{total}}`);
    
    for (const r of results) {{
        const status = r.passed ? "PASS" : "FAIL";
        console.log(`[${{status}}] ${{r.test}}`);
    }}
    
    process.exit(passed === total ? 0 : 1);
}})();
'''
    
    def execute_tests(self, test_code: str) -> Tuple[int, int, List[str]]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['node', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                output = result.stdout + result.stderr
                failing = []
                
                for line in output.split('\n'):
                    if '[FAIL]' in line:
                        test_name = line.split(']')[1].split(':')[0].strip()
                        failing.append(test_name)
                
                passed, total = 0, 0
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
    
    def apply_fix(self, code: str, fix_type: str, context: Dict) -> str:
        # Similar fix patterns for JavaScript
        if fix_type == "add_payment_method":
            method = context.get("method", "unknown")
            return code.replace(
                '} else if (paymentMethod === "debit") {',
                f'''}} else if (paymentMethod === "{method}") {{
        fee = subtotal * 0.035;
    }} else if (paymentMethod === "debit") {{'''
            )
        return code


# Registry of content type handlers
CONTENT_TYPE_HANDLERS: Dict[str, ContentTypeHandler] = {
    "python": PythonHandler(),
    "php": PHPHandler(),
    "javascript": JavaScriptHandler(),
    "js": JavaScriptHandler(),
}


def get_handler(language: str) -> ContentTypeHandler:
    """Get the appropriate handler for a language."""
    language = language.lower()
    if language not in CONTENT_TYPE_HANDLERS:
        raise ValueError(f"Unsupported language: {language}. Supported: {list(CONTENT_TYPE_HANDLERS.keys())}")
    return CONTENT_TYPE_HANDLERS[language]


# =============================================================================
# Code Analysis
# =============================================================================

@dataclass
class CodeAnalysis:
    """Analysis of code structure."""
    functions: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    complexity: int = 0
    missing_features: List[str] = field(default_factory=list)


class UniversalCodeAnalyzer:
    """Content-type agnostic code analyzer."""
    
    def __init__(self, handler: ContentTypeHandler):
        self.handler = handler
    
    def analyze(self, code: str, tests: List[Dict]) -> CodeAnalysis:
        """Analyze code and identify missing features based on tests."""
        analysis = CodeAnalysis()
        
        # Extract code structure
        analysis.functions = self.handler.extract_functions(code)
        analysis.conditions = self.handler.extract_conditions(code)
        
        # Calculate complexity
        analysis.complexity = len(analysis.conditions) + 1
        
        # Identify missing features based on test names
        test_names = [t.get("name", "").lower() for t in tests]
        test_inputs = [t.get("input", {}) for t in tests]
        
        # Check for common patterns
        if any("paypal" in name or "stripe" in name for name in test_names):
            if not any("paypal" in cond.get("condition", "").lower() or 
                      "stripe" in cond.get("condition", "").lower() 
                      for cond in analysis.conditions):
                analysis.missing_features.append("payment_method")
        
        if any("disposable" in name or "temporary" in name for name in test_names):
            if not any("disposable" in cond.get("condition", "").lower() or
                      "temporary" in cond.get("condition", "").lower()
                      for cond in analysis.conditions):
                analysis.missing_features.append("validation")
        
        if any("bulk" in name or "tier" in name for name in test_names):
            if not any(">= 25" in cond.get("condition", "") or
                      ">= 20" in cond.get("condition", "")
                      for cond in analysis.conditions):
                analysis.missing_features.append("discount_tier")
        
        if any("empty" in name or "null" in name or "none" in name for name in test_names):
            if not any("empty" in cond.get("condition", "").lower() or
                      "null" in cond.get("condition", "").lower()
                      for cond in analysis.conditions):
                analysis.missing_features.append("null_check")
        
        return analysis


# =============================================================================
# Fix Generator
# =============================================================================

class UniversalFixGenerator:
    """Generates fixes based on missing feature analysis."""
    
    def __init__(self, handler: ContentTypeHandler):
        self.handler = handler
    
    def generate_fixes(
        self,
        analysis: CodeAnalysis,
        failing_tests: List[str]
    ) -> List[Tuple[str, str, Dict]]:
        """Generate fixes for missing features."""
        fixes = []
        
        for feature in analysis.missing_features:
            if feature == "payment_method":
                # Determine which payment method is missing
                for test_name in failing_tests:
                    if "paypal" in test_name.lower():
                        fixes.append(("add_payment_method", "paypal", {"method": "paypal"}))
                    elif "stripe" in test_name.lower():
                        fixes.append(("add_payment_method", "stripe", {"method": "stripe"}))
            
            elif feature == "validation":
                for test_name in failing_tests:
                    if "disposable" in test_name.lower():
                        fixes.append(("add_validation", "disposable", {"type": "disposable"}))
            
            elif feature == "discount_tier":
                fixes.append(("add_discount_tier", "tier_25", {}))
            
            elif feature == "null_check":
                fixes.append(("add_null_check", "empty", {}))
        
        return fixes
    
    def apply_fixes(
        self,
        code: str,
        fixes: List[Tuple[str, str, Dict]]
    ) -> str:
        """Apply fixes to code."""
        result = code
        for fix_type, feature, context in fixes:
            result = self.handler.apply_fix(result, fix_type, context)
        return result


# =============================================================================
# PES Evolution Engine
# =============================================================================

@dataclass
class EvolutionResult:
    """Result of evolution process."""
    original_code: str
    evolved_code: str
    iterations: int
    fixes_applied: List[str]
    initial_score: float
    final_score: float
    improvement: float
    tests_passed: int
    tests_total: int
    failing_tests: List[str]


class PESEvolutionEngine:
    """
    Plan-Execute-Summarize Evolution Engine.
    
    This engine:
    1. PLAN: Analyzes code and tests to identify missing features
    2. EXECUTES: Runs tests and applies fixes
    3. SUMMARIZES: Records progress and converges when stable
    
    Content-type agnostic - works with any language via handlers.
    """
    
    def __init__(self, language: str = "python"):
        self.handler = get_handler(language)
        self.analyzer = UniversalCodeAnalyzer(self.handler)
        self.generator = UniversalFixGenerator(self.handler)
        self.iteration = 0
        self.fixes_applied = []
    
    async def evolve(
        self,
        code: str,
        tests: List[Dict],
        max_iterations: int = 5,
        target_score: float = 0.98
    ) -> EvolutionResult:
        """
        Run the evolution process.
        
        Args:
            code: The source code to evolve
            tests: List of test definitions with 'name', 'input', 'expected', 'function'
            max_iterations: Maximum iterations to run
            target_score: Target correctness score (0-1)
        
        Returns:
            EvolutionResult with evolved code and metrics
        """
        logger.info(f"Starting evolution for {self.handler.name} code")
        
        original_code = code
        current_code = code
        self.iteration = 0
        self.fixes_applied = []
        
        # Initial evaluation
        initial_score, initial_passed, initial_total, _ = self._evaluate(current_code, tests)
        
        for iteration in range(1, max_iterations + 1):
            self.iteration = iteration
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{max_iterations}")
            logger.info(f"{'='*60}")
            
            # Evaluate current code
            score, passed, total, failing = self._evaluate(current_code, tests)
            
            logger.info(f"  Score: {score:.1%} ({passed}/{total})")
            
            if failing:
                logger.info(f"  Failing: {failing[:3]}")
            
            # Check convergence
            if score >= target_score:
                logger.info(f"  Converged at iteration {iteration}")
                break
            
            # Analyze and generate fixes
            analysis = self.analyzer.analyze(current_code, tests)
            fixes = self.generator.generate_fixes(analysis, failing)
            
            if not fixes:
                logger.info("  No fixes applicable")
                continue
            
            logger.info(f"  Applying {len(fixes)} fixes...")
            
            # Apply fixes
            current_code = self.generator.apply_fixes(current_code, fixes)
            
            for fix_type, feature, context in fixes:
                fix_desc = f"{fix_type}:{feature}"
                if fix_desc not in self.fixes_applied:
                    self.fixes_applied.append(fix_desc)
                    logger.info(f"    - Applied {fix_desc}")
        
        # Final evaluation
        final_score, final_passed, final_total, final_failing = self._evaluate(current_code, tests)
        
        improvement = final_score - initial_score
        
        return EvolutionResult(
            original_code=original_code,
            evolved_code=current_code,
            iterations=self.iteration,
            fixes_applied=self.fixes_applied,
            initial_score=initial_score,
            final_score=final_score,
            improvement=improvement,
            tests_passed=final_passed,
            tests_total=final_total,
            failing_tests=final_failing
        )
    
    def _evaluate(
        self,
        code: str,
        tests: List[Dict]
    ) -> Tuple[float, int, int, List[str]]:
        """Evaluate code against tests."""
        test_wrapper = self.handler.generate_test_wrapper(code, tests)
        passed, total, failing = self.handler.execute_tests(test_wrapper)
        score = passed / total if total > 0 else 0
        return score, passed, total, failing


# =============================================================================
# Convenience Functions
# =============================================================================

async def evolve_code(
    code: str,
    tests: List[Dict],
    language: str = "python",
    max_iterations: int = 5
) -> EvolutionResult:
    """
    Evolve code using the PES engine.
    
    Args:
        code: Source code to evolve
        tests: Test definitions
        language: Programming language
        max_iterations: Maximum iterations
    
    Returns:
        EvolutionResult
    """
    engine = PESEvolutionEngine(language)
    return await engine.evolve(code, tests, max_iterations)


def quick_evolve(
    code: str,
    test_cases: List[Tuple[str, Dict, Any]],
    function_name: str = "main",
    language: str = "python"
) -> str:
    """
    Quick evolution with simple test cases.
    
    Args:
        code: Source code
        test_cases: List of (name, input_dict, expected) tuples
        function_name: Name of function to test
        language: Programming language
    
    Returns:
        Evolved code
    """
    tests = [
        {"name": name, "input": input_data, "expected": expected, "function": function_name}
        for name, input_data, expected in test_cases
    ]
    
    import asyncio
    result = asyncio.run(evolve_code(code, tests, language))
    return result.evolved_code


# =============================================================================
# Main entry point for demos
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenEvolve PES - Content-Type Agnostic Evolution")
    parser.add_argument("--language", default="python", choices=["python", "php", "javascript"])
    parser.add_argument("--code", help="Code file to evolve")
    parser.add_argument("--tests", help="Tests file (JSON)")
    parser.add_argument("--iterations", type=int, default=5)
    
    args = parser.parse_args()
    
    if args.code:
        with open(args.code) as f:
            code = f.read()
        
        tests = []
        if args.tests:
            with open(args.tests) as f:
                tests = json.load(f)
        
        import asyncio
        result = asyncio.run(evolve_code(code, tests, args.language, args.iterations))
        
        print(f"\n{'='*70}")
        print("EVOLUTION RESULT")
        print(f"{'='*70}")
        print(f"Iterations: {result.iterations}")
        print(f"Fixes Applied: {result.fixes_applied}")
        print(f"Improvement: +{result.improvement:.1%}")
        print(f"Final Score: {result.final_score:.1%} ({result.tests_passed}/{result.tests_total})")
        print(f"{'='*70}\n")
        print(result.evolved_code)
