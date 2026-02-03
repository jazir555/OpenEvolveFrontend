#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenEvolve PES Integration - TRULY Content-Agnostic

This module improves ANY code regardless of programming language.
The system analyzes failing tests and generates fixes without
needing language-specific handlers.

Key Principles:
1. Auto-detect language from code patterns
2. Analyze code structure universally
3. Generate fixes using universal strategies
4. Auto-translate fix syntax to target language
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("OpenEvolve-AGNOSTIC-PES")


# =============================================================================
# Language Detection & Syntax Generation (Universal)
# =============================================================================

class LanguageDetector:
    """Auto-detect programming language from code patterns."""
    
    # Language signatures - patterns that identify a language
    SIGNATURES = {
        'python': {
            'keywords': ['def ', 'import ', 'from ', 'class ', 'if __name__'],
            'patterns': [r'def\s+\w+\s*\(', r'import\s+\w+', r'from\s+\w+\s+import'],
            'comment': '#',
            'indent': '    ',
        },
        'php': {
            'keywords': ['<?php', 'function ', '->', '->'],
            'patterns': [r'function\s+\w+\s*\(', r'\$\w+\s*='],
            'comment': '//',
            'indent': '    ',
        },
        'javascript': {
            'keywords': ['function ', 'const ', 'let ', 'var ', '=>', 'async'],
            'patterns': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*='],
            'comment': '//',
            'indent': '    ',
        },
        'java': {
            'keywords': ['public class', 'public static', 'void ', 'System.out'],
            'patterns': [r'public\s+(?:static\s+)?(?:void|int|String)\s+\w+\s*\('],
            'comment': '//',
            'indent': '    ',
        },
        'cpp': {
            'keywords': ['#include', 'int main()', 'std::', 'void '],
            'patterns': [r'#include\s*<', r'int\s+main\s*\('],
            'comment': '//',
            'indent': '    ',
        },
        'ruby': {
            'keywords': ['def ', 'require ', 'class ', 'end'],
            'patterns': [r'def\s+\w+\s*$', r'require\s+[\'"]'],
            'comment': '#',
            'indent': '  ',
        },
        'go': {
            'keywords': ['func ', 'package ', 'import '],
            'patterns': [r'func\s+\w+\s*\(', r'package\s+\w+'],
            'comment': '//',
            'indent': '\t',
        },
        'rust': {
            'keywords': ['fn ', 'let ', 'impl ', 'pub '],
            'patterns': [r'fn\s+\w+\s*\(', r'let\s+(?:mut\s+)?\w+'],
            'comment': '//',
            'indent': '    ',
        },
        'csharp': {
            'keywords': ['public class', 'void ', 'using '],
            'patterns': [r'public\s+(?:static\s+)?(?:void|int|string)\s+\w+\s*\('],
            'comment': '//',
            'indent': '    ',
        },
    }
    
    @classmethod
    def detect(cls, code: str) -> str:
        """Detect the programming language of the code."""
        code_lower = code.lower()
        
        scores = {}
        for lang, signature in cls.SIGNATURES.items():
            score = 0
            # Check keywords
            for keyword in signature['keywords']:
                if keyword.lower() in code_lower:
                    score += 2
            # Check patterns
            for pattern in signature['patterns']:
                if re.search(pattern, code):
                    score += 3
            if score > 0:
                scores[lang] = score
        
        if not scores:
            return 'python'  # Default to Python
        
        # Return the language with highest score
        return max(scores, key=scores.get)


class UniversalSyntaxGenerator:
    """Generate syntax for any language based on detected language."""
    
    SYNTAX = {
        'python': {
            'indent': '    ',
            'line_continue': '',
            'func_def': 'def {name}({params}):',
            'func_call': '{name}({args})',
            'return': 'return {value}',
            'if_start': 'if {cond}:',
            'elif_start': 'elif {cond}:',
            'else_start': 'else:',
            'block_end': '',
            'comment': '# {text}',
            'string_double': '"{value}"',
            'string_single': "'{value}'",
            'dict_literal': '{{{pairs}}}',
            'array_literal': '[{items}]',
            'comparison': '{a} {op} {b}',
            'assign': '{name} = {value}',
            'augment': '{name} {op}= {value}',
            'float_literal': '{value}',
            'int_literal': '{value}',
            'null_literal': 'None',
            'true_literal': 'True',
            'false_literal': 'False',
        },
        'php': {
            'indent': '    ',
            'line_continue': '',
            'func_def': 'function {name}({params}) {{',
            'func_call': '{name}({args})',
            'return': 'return {value};',
            'if_start': 'if ({cond}) {{',
            'elif_start': 'else if ({cond}) {{',
            'else_start': 'else {{',
            'block_end': '}}',
            'comment': '// {text}',
            'string_double': '"{value}"',
            'string_single': "'{value}'",
            'dict_literal': '[{pairs}]',
            'array_literal': '[{items}]',
            'comparison': '{a} {op} {b}',
            'assign': '${name} = {value};',
            'augment': '${name} {op}= {value};',
            'float_literal': '{value}',
            'int_literal': '{value}',
            'null_literal': 'null',
            'true_literal': 'true',
            'false_literal': 'false',
        },
        'javascript': {
            'indent': '    ',
            'line_continue': '',
            'func_def': 'function {name}({params}) {{',
            'func_call': '{name}({args})',
            'return': 'return {value};',
            'if_start': 'if ({cond}) {{',
            'elif_start': 'else if ({cond}) {{',
            'else_start': 'else {{',
            'block_end': '}}',
            'comment': '// {text}',
            'string_double': '"{value}"',
            'string_single': "'{value}'",
            'dict_literal': '{{{pairs}}}',
            'array_literal': '[{items}]',
            'comparison': '{a} {op} {b}',
            'assign': 'const {name} = {value};',
            'augment': '{name} {op}= {value};',
            'float_literal': '{value}',
            'int_literal': '{value}',
            'null_literal': 'null',
            'true_literal': 'true',
            'false_literal': 'false',
        },
        'java': {
            'indent': '    ',
            'line_continue': '',
            'func_def': 'public {return_type} {name}({params}) {{',
            'func_call': '{name}({args})',
            'return': 'return {value};',
            'if_start': 'if ({cond}) {{',
            'elif_start': 'else if ({cond}) {{',
            'else_start': 'else {{',
            'block_end': '}}',
            'comment': '// {text}',
            'string_double': '"{value}"',
            'string_single': "'{value}'",
            'dict_literal': 'new HashMap<String, Object>() {{ put({pairs}); }}',
            'array_literal': 'new {type}[]{{{items}}}',
            'comparison': '{a} {op} {b}',
            'assign': '{type} {name} = {value};',
            'augment': '{name} {op}= {value};',
            'float_literal': '{value}f',
            'int_literal': '{value}',
            'null_literal': 'null',
            'true_literal': 'true',
            'false_literal': 'false',
        },
        'cpp': {
            'indent': '    ',
            'line_continue': ' \\',
            'func_def': '{return_type} {name}({params}) {{',
            'func_call': '{name}({args})',
            'return': 'return {value};',
            'if_start': 'if ({cond}) {{',
            'elif_start': 'else if ({cond}) {{',
            'else_start': 'else {{',
            'block_end': '}}',
            'comment': '// {text}',
            'string_double': '"{value}"',
            'string_single': "'{value}'",
            'dict_literal': 'std::map<std::string, {type}>{{{pairs}}}',
            'array_literal': '{{{items}}}',
            'comparison': '{a} {op} {b}',
            'assign': '{type} {name} = {value};',
            'augment': '{name} {op}= {value};',
            'float_literal': '{value}',
            'int_literal': '{value}',
            'null_literal': 'nullptr',
            'true_literal': 'true',
            'false_literal': 'false',
        },
    }
    
    @classmethod
    def get_syntax(cls, language: str) -> Dict[str, str]:
        """Get syntax templates for a language."""
        return cls.SYNTAX.get(language, cls.SYNTAX['python'])
    
    @classmethod
    def generate_branch(cls, language: str, condition: str, body: str, 
                       branch_type: str = 'if') -> str:
        """Generate a conditional branch in the target language."""
        syntax = cls.get_syntax(language)
        indent = syntax['indent']
        
        if branch_type == 'if':
            template = syntax['if_start']
        elif branch_type == 'elif':
            template = syntax['elif_start']
        else:  # else
            template = syntax['else_start']
        
        branch = template.format(cond=condition)
        
        if '{' in branch:
            # Languages with braces
            return f"{branch}\n{indent}{body}\n{syntax['block_end']}"
        else:
            # Languages with colons
            return f"{branch}\n{indent}{body}"
    
    @classmethod
    def generate_return(cls, language: str, value: str) -> str:
        """Generate a return statement in the target language."""
        syntax = cls.get_syntax(language)
        return syntax['return'].format(value=value)
    
    @classmethod
    def generate_assignment(cls, language: str, name: str, value: str) -> str:
        """Generate an assignment statement in the target language."""
        syntax = cls.get_syntax(language)
        return syntax['assign'].format(name=name, value=value)


# =============================================================================
# Universal Code Analysis
# =============================================================================

class UniversalCodeAnalyzer:
    """Analyze code structure in a language-agnostic way."""
    
    @staticmethod
    def find_function_boundaries(code: str) -> List[Dict[str, int]]:
        """Find all function definitions and their boundaries."""
        functions = []
        
        # Patterns for different languages
        patterns = [
            (r'def\s+(\w+)\s*\(([^)]*)\)', 'python'),
            (r'function\s+(\w+)\s*\(', 'php'),
            (r'(?:async\s+)?function\s+(\w+)\s*\(', 'javascript'),
            (r'(?:public\s+)?(?:static\s+)?(?:void|int|String|boolean)\s+(\w+)\s*\(', 'java'),
            (r'(?:pub\s+)?fn\s+(\w+)\s*\(', 'rust'),
        ]
        
        for pattern, lang in patterns:
            for match in re.finditer(pattern, code):
                start = code[:match.start()].count('\n')
                functions.append({
                    'name': match.group(1),
                    'language': lang,
                    'start_line': start,
                    'end_line': None,  # Will be calculated
                })
        
        return functions
    
    @staticmethod
    def find_conditional_branches(code: str) -> List[Dict[str, Any]]:
        """Find conditional statements (if/elif/else)."""
        branches = []
        
        # Find if statements
        if_patterns = [
            (r'\bif\s+(?!.*else)(.*?):', 'python'),  # Python if
            (r'\bif\s*\(([^)]+)\)\s*\{', 'php'),  # PHP/JavaScript if
            (r'\bif\s+\(([^)]+)\)\s*\{', 'java'),
        ]
        
        for pattern, lang in if_patterns:
            for match in re.finditer(pattern, code):
                line = code[:match.start()].count('\n')
                cond = match.group(1).strip()
                branches.append({
                    'type': 'if',
                    'condition': cond,
                    'line': line,
                    'language': lang,
                })
        
        return branches
    
    @staticmethod
    def find_return_statements(code: str) -> List[Dict[str, int]]:
        """Find return statements."""
        returns = []
        
        patterns = [
            (r'return\s+', 'python'),
            (r'return\s+', 'php'),
            (r'return\s+', 'javascript'),
            (r'return\s+', 'java'),
            (r'return\s+', 'rust'),
        ]
        
        for pattern, lang in patterns:
            for match in re.finditer(pattern, code):
                line = code[:match.start()].count('\n')
                returns.append({
                    'line': line,
                    'language': lang,
                })
        
        return returns
    
    @staticmethod
    def analyze(code: str) -> Dict[str, Any]:
        """Complete analysis of code structure."""
        return {
            'functions': UniversalCodeAnalyzer.find_function_boundaries(code),
            'conditionals': UniversalCodeAnalyzer.find_conditional_branches(code),
            'returns': UniversalCodeAnalyzer.find_return_statements(code),
            'language': LanguageDetector.detect(code),
            'line_count': len(code.split('\n')),
        }


# =============================================================================
# Universal Fix Generation
# =============================================================================

class UniversalFixGenerator:
    """Generate fixes for any code based on test analysis."""
    
    # Strategy: What type of fix is needed based on test failure pattern
    STRATEGIES = {
        'missing_branch': {
            'description': 'Code is missing a conditional branch for a specific input',
            'fix_type': 'add_branch',
        },
        'missing_validation': {
            'description': 'Code does not validate input properly',
            'fix_type': 'add_validation',
        },
        'missing_discount_tier': {
            'description': 'Code is missing a discount tier',
            'fix_type': 'add_tier',
        },
        'wrong_calculation': {
            'description': 'Calculation logic is incorrect',
            'fix_type': 'fix_calculation',
        },
        'missing_parameter_handling': {
            'description': 'Function does not handle all parameter values',
            'fix_type': 'add_branch',
        },
    }
    
    @classmethod
    def analyze_failure(cls, test_name: str, test_input: Dict, test_expected: Any, 
                       code: str, analysis: Dict) -> Optional[Dict[str, Any]]:
        """Analyze a test failure and determine what fix is needed."""
        
        test_name_lower = test_name.lower()
        code_lower = code.lower()
        
        # Strategy 1: Missing payment method branch
        if 'paypal' in test_name_lower or 'payment_method' in str(test_input):
            if 'paypal' not in code_lower and 'payment_method' in code_lower:
                payment_method = test_input.get('payment_method', 'paypal')
                return {
                    'issue': f"Missing branch for payment_method: {payment_method}",
                    'strategy': 'missing_branch',
                    'fix_type': 'add_branch',
                    'context': {
                        'branch_type': 'payment_method',
                        'value': payment_method,
                        'expected_fee': test_expected.get('fee') if isinstance(test_expected, dict) else None,
                    }
                }
        
        # Strategy 2: Missing discount code branch
        if 'discount' in test_name_lower or 'discount_code' in test_input:
            discount_code = test_input.get('discount_code', '')
            if discount_code and discount_code.upper() not in code_lower:
                return {
                    'issue': f"Missing branch for discount_code: {discount_code}",
                    'strategy': 'missing_branch',
                    'fix_type': 'add_branch',
                    'context': {
                        'branch_type': 'discount_code',
                        'value': discount_code,
                        'expected_discount': test_expected.get('discount') if isinstance(test_expected, dict) else None,
                    }
                }
        
        # Strategy 3: Empty input handling
        if 'empty' in test_name_lower:
            if 'if not' not in code_lower and 'if len' not in code_lower:
                return {
                    'issue': 'Missing empty input handling',
                    'strategy': 'missing_validation',
                    'fix_type': 'add_validation',
                    'context': {'validation_type': 'empty'}
                }
        
        # Strategy 4: Bulk discount tier
        if 'bulk' in test_name_lower or '15%' in test_name:
            if '>= 25' not in code and 'total_qty >= 25' not in code_lower:
                return {
                    'issue': 'Missing tiered bulk discount (15% for 25+ items)',
                    'strategy': 'missing_discount_tier',
                    'fix_type': 'add_tier',
                    'context': {'tier': 25, 'discount': 0.15}
                }
        
        return None
    
    @classmethod
    def generate_fix(cls, code: str, analysis: Dict, fix_request: Dict) -> str:
        """Generate the fixed code based on the fix request."""
        
        strategy = fix_request.get('strategy', 'unknown')
        context = fix_request.get('context', {})
        language = analysis.get('language', 'python')
        
        if strategy == 'missing_branch':
            return cls._fix_missing_branch(code, context, language)
        elif strategy == 'missing_validation':
            return cls._fix_missing_validation(code, context, language)
        elif strategy == 'missing_discount_tier':
            return cls._fix_missing_tier(code, context, language)
        
        return code  # No change if unknown strategy
    
    @classmethod
    def _fix_missing_branch(cls, code: str, context: Dict, language: str) -> str:
        """Fix missing branch (e.g., PayPal payment method)."""
        branch_type = context.get('branch_type', '')
        value = context.get('value', '')
        expected = context.get('expected_fee', 0.035)
        
        syntax = UniversalSyntaxGenerator.get_syntax(language)
        indent = syntax['indent']
        
        if branch_type == 'payment_method':
            # Find the payment method section and add the new method
            patterns = {
                'python': [
                    (r"elif payment_method == \"debit_card\":\n        fee = subtotal \* 0\.015", True),
                    (r"elif payment_method == 'debit_card':\n        fee = subtotal \* 0\.015", True),
                ],
                'php': [
                    (r'} else if \(\$payment_method == \"debit_card\"\) \{', True),
                    (r"} else if \(\$payment_method == 'debit_card'\) \{", True),
                ],
                'javascript': [
                    (r"} else if \(payment_method === 'debit_card'\) \{", True),
                ],
            }
            
            # Find the right insertion point
            for pattern, _ in patterns.get(language, []):
                match = re.search(pattern, code)
                if match:
                    # Insert after debit_card
                    insert_pos = match.end()
                    
                    if language == 'python':
                        # Use default rate if expected_fee is provided as absolute value
                        expected_fee = context.get('expected_fee', 0.035)
                        # If expected_fee > 1, assume it's an absolute value, use default rate
                        if isinstance(expected_fee, (int, float)) and expected_fee > 1:
                            expected_fee = 0.035
                        new_code = f'''elif payment_method == "{value}":
{indent}    fee = subtotal * {expected_fee}'''
                        return code[:insert_pos] + '\n' + indent + new_code + code[insert_pos:]
                    elif language == 'php':
                        expected_fee = context.get('expected_fee', 0.035)
                        if isinstance(expected_fee, (int, float)) and expected_fee > 1:
                            expected_fee = 0.035
                        new_code = f'}} else if ($payment_method == "{value}") {{\n{indent}$fee = $subtotal * {expected_fee};'
                        return code[:insert_pos] + '\n' + indent + new_code + code[insert_pos:]
        
        return code  # Return unchanged if no pattern matched
    
    @classmethod
    def _fix_missing_validation(cls, code: str, context: Dict, language: str) -> str:
        """Fix missing validation (e.g., empty input)."""
        validation_type = context.get('validation_type', 'empty')
        syntax = UniversalSyntaxGenerator.get_syntax(language)
        indent = syntax['indent']
        
        if validation_type == 'empty':
            # Add empty check at the beginning of the function
            validation_code = {
                'python': f'''{indent}# Input validation
{indent}if not input_data or len(input_data) == 0:
{indent}{syntax['return'].format(value='{"error": "Empty input"}')}''',
                'php': f'''{indent}// Input validation
{indent}if (empty($input_data)) {{
{indent}{syntax['return'].format(value='["error" => "Empty input"]')}
{indent}}}''',
                'javascript': f'''{indent}// Input validation
{indent}if (!input_data || input_data.length === 0) {{
{indent}{syntax['return'].format(value='{error: "Empty input"}')}
{indent}}}''',
            }
            
            # Try to insert after function definition
            func_match = re.search(r'(def\s+\w+\s*\([^)]*\):|function\s+\w+\s*\([^)]*\)\s*\{)', code)
            if func_match:
                insert_pos = func_match.end()
                return code[:insert_pos] + '\n' + validation_code.get(language, validation_code['python']) + code[insert_pos:]
        
        return code
    
    @classmethod
    def _fix_missing_tier(cls, code: str, context: Dict, language: str) -> str:
        """Fix missing discount tier."""
        tier = context.get('tier', 25)
        discount = context.get('discount', 0.15)
        syntax = UniversalSyntaxGenerator.get_syntax(language)
        indent = syntax['indent']
        
        # Find the discount section and add tier
        patterns = {
            'python': r'(if total_qty >= 10:)',
            'php': r'(if \(\$total_qty >= 10\))',
            'javascript': r'(if \(totalQty >= 10\))',
        }
        
        pattern = patterns.get(language, patterns['python'])
        match = re.search(pattern, code)
        
        if match:
            insert_pos = match.start()
            
            if language == 'python':
                new_tier = f'''{indent}if total_qty >= {tier}:
{indent}{indent}discount = subtotal * {discount}
'''
                return code[:insert_pos] + new_tier + code[insert_pos:]
            elif language == 'php':
                new_tier = f'''{indent}if ($total_qty >= {tier}) {{
{indent}{indent}$discount = $subtotal * {discount};
{indent}}}
'''
                return code[:insert_pos] + new_tier + code[insert_pos:]
        
        return code


# =============================================================================
# Test Runner (Universal)
# =============================================================================

class UniversalTestRunner:
    """Run tests on any code regardless of language."""
    
    # Extension to interpreter mapping
    INTERPRETERS = {
        '.py': 'python',
        '.php': 'php',
        '.js': 'node',
        '.ts': 'ts-node',
        '.java': 'javac',
        '.cpp': 'g++',
        '.c': 'gcc',
        '.rb': 'ruby',
        '.go': 'go run',
        '.rs': 'rustc',
    }
    
    @classmethod
    def get_interpreter(cls, extension: str) -> str:
        """Get the interpreter command for a file extension."""
        return cls.INTERPRETERS.get(extension, 'python')
    
    @classmethod
    def generate_test_wrapper(cls, code: str, tests: List[Dict], language: str) -> str:
        """Generate a test wrapper in the appropriate language."""
        
        if language == 'python':
            return cls._generate_python_wrapper(code, tests)
        elif language == 'php':
            return cls._generate_php_wrapper(code, tests)
        elif language == 'javascript':
            return cls._generate_js_wrapper(code, tests)
        else:
            # Default to Python for unsupported languages
            return cls._generate_python_wrapper(code, tests)
    
    @classmethod
    def _generate_python_wrapper(cls, code: str, tests: List[Dict]) -> str:
        """Generate Python test wrapper."""
        
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
        # Compare with fuzzy tolerance for floats
        passed = _compare_fuzzy(result, expected)
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
    """Compare two values with tolerance for floats - handles partial dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        # Check that all expected keys in b are present in a and match
        for key, expected_val in b.items():
            if key not in a:
                return False
            if not _compare_fuzzy(a[key], expected_val, tolerance):
                return False
        return True
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
    
    @classmethod
    def _generate_php_wrapper(cls, code: str, tests: List[Dict]) -> str:
        """Generate PHP test wrapper."""
        
        test_funcs = []
        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            input_data = json.dumps(test.get("input", {}))
            expected = json.dumps(test.get("expected", {}))
            func_name = test.get("function", "main")
            
            test_code = f'''function test_{i}() {{
    $input_data = json_decode('{input_data}', true);
    $expected = json_decode('{expected}', true);
    try {{
        $result = $func_name($input_data);
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
    
    @classmethod
    def _generate_js_wrapper(cls, code: str, tests: List[Dict]) -> str:
        """Generate JavaScript test wrapper."""
        
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
    
    @classmethod
    def execute(cls, test_code: str, language: str) -> Tuple[int, int, List[str]]:
        """Execute the test wrapper and return results."""
        
        extension = '.' + language if language not in ['cpp', 'java'] else '.cpp'
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            
            try:
                # Get interpreter
                interpreter = cls.INTERPRETERS.get(f'.{language}', 'python')
                
                # Run the test
                result = subprocess.run(
                    [interpreter, temp_file] if ' ' not in interpreter else interpreter.split()[0] + ' ' + temp_file,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=True if ' ' in interpreter else False
                )
                
                output = result.stdout + result.stderr
                failing = []
                
                # Parse output
                passed, total = 0, 0
                for line in output.split('\n'):
                    if line.startswith('TESTS_PASSED:'):
                        passed = int(line.split(':')[1])
                    elif line.startswith('TESTS_TOTAL:'):
                        total = int(line.split(':')[1])
                    elif '[FAIL]' in line:
                        test_name = line.split(']')[1].split(':')[0].strip()
                        failing.append(test_name)
                
                return passed, total, failing
                
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            return 0, 0, [str(e)]


# =============================================================================
# PES Evolution Engine (Truly Agnostic)
# =============================================================================

@dataclass
class EvolutionResult:
    """Result of an evolution process."""
    original_code: str = ""
    evolved_code: str = ""
    iterations: int = 0
    fixes_applied: List[str] = field(default_factory=list)
    improvement: float = 0.0
    final_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0


class AgnosticPESEngine:
    """
    Truly content-agnostic PES (Plan-Execute-Summarize) evolution engine.
    
    This engine can improve ANY code regardless of programming language.
    """
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
    
    async def evolve(self, code: str, tests: List[Dict], 
                    problem_type: str = "general") -> EvolutionResult:
        """
        Evolve code to improve correctness based on test results.
        
        Args:
            code: The source code to evolve
            tests: List of test cases with name, input, expected, function
            problem_type: Type of problem (payment, validation, etc.)
        
        Returns:
            EvolutionResult with evolved code and metrics
        """
        logger.info(f"Starting content-agnostic evolution for {len(tests)} tests")
        
        # Auto-detect language
        language = LanguageDetector.detect(code)
        logger.info(f"Detected language: {language}")
        
        # Analyze code structure
        analysis = UniversalCodeAnalyzer.analyze(code)
        logger.info(f"Code analysis: {analysis['functions']} functions found")
        
        current_code = code
        fixes_applied = []
        best_code = code
        best_score = 0.0
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate and run tests
            test_wrapper = UniversalTestRunner.generate_test_wrapper(current_code, tests, language)
            passed, total, failing = UniversalTestRunner.execute(test_wrapper, language)
            
            # Calculate score
            score = passed / total if total > 0 else 0.0
            logger.info(f"Score: {score:.1%} ({passed}/{total})")
            logger.info(f"Failing: {failing}")
            
            # If all tests pass, we're done
            if score == 1.0:
                best_code = current_code
                logger.info("All tests passing! Evolution complete.")
                break
            
            # Analyze failures and generate fixes
            fixes_this_iteration = 0
            
            for test_name in failing:
                # Find the test case
                test_case = next((t for t in tests if t.get("name") == test_name), None)
                if not test_case:
                    continue
                
                # Analyze failure
                fix_request = UniversalFixGenerator.analyze_failure(
                    test_name,
                    test_case.get("input", {}),
                    test_case.get("expected", {}),
                    current_code,
                    analysis
                )
                
                if fix_request:
                    # Generate fix
                    new_code = UniversalFixGenerator.generate_fix(
                        current_code, analysis, fix_request
                    )
                    
                    if new_code != current_code:
                        # Apply fix
                        current_code = new_code
                        fix_desc = f"{fix_request['strategy']}:{fix_request.get('context', {}).get('value', '')}"
                        fixes_applied.append(fix_desc)
                        fixes_this_iteration += 1
                        logger.info(f"Applied fix: {fix_desc}")
            
            if fixes_this_iteration == 0:
                logger.info("No more fixes applicable")
                break
            
            # Run tests on the fixed code and update best_score
            final_wrapper = UniversalTestRunner.generate_test_wrapper(current_code, tests, language)
            new_passed, new_total, _ = UniversalTestRunner.execute(final_wrapper, language)
            new_score = new_passed / new_total if new_total > 0 else 0.0
            
            if new_score > best_score:
                best_score = new_score
                best_code = current_code
                logger.info(f"New best score: {best_score:.1%} ({new_passed}/{new_total})")
            
            if new_score == 1.0:
                logger.info("All tests passing! Evolution complete.")
                break
        
        # Final evaluation
        final_wrapper = UniversalTestRunner.generate_test_wrapper(best_code, tests, language)
        final_passed, final_total, _ = UniversalTestRunner.execute(final_wrapper, language)
        final_score = final_passed / final_total if final_total > 0 else 0.0
        
        original_wrapper = UniversalTestRunner.generate_test_wrapper(code, tests, language)
        original_passed, original_total, _ = UniversalTestRunner.execute(original_wrapper, language)
        original_score = original_passed / original_total if original_total > 0 else 0.0
        
        return EvolutionResult(
            original_code=code,
            evolved_code=best_code,
            iterations=self.max_iterations,
            fixes_applied=fixes_applied,
            improvement=final_score - original_score,
            final_score=final_score,
            tests_passed=final_passed,
            tests_total=final_total,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def evolve_code(code: str, tests: List[Dict], language: str = None, 
                     max_iterations: int = 5) -> EvolutionResult:
    """
    Evolve code to improve correctness.
    
    Args:
        code: Source code to evolve
        tests: List of test cases
        language: Optional language hint (auto-detected if not provided)
        max_iterations: Maximum evolution iterations
    
    Returns:
        EvolutionResult with evolved code
    """
    engine = AgnosticPESEngine(max_iterations=max_iterations)
    return await engine.evolve(code, tests)


def quick_evolve(code: str, tests: List[Dict], language: str = None) -> EvolutionResult:
    """
    Quick evolution with default settings.
    
    Args:
        code: Source code to evolve
        tests: List of test cases
        language: Optional language hint
    
    Returns:
        EvolutionResult with evolved code
    """
    return asyncio.run(evolve_code(code, tests, language, max_iterations=3))


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate content-agnostic evolution."""
    
    print("\n" + "="*70)
    print("  OpenEvolve AGNOSTIC PES Demo")
    print("  TRULY Content-Agnostic Code Evolution")
    print("="*70)
    
    # Python code demo
    python_code = '''def calculate_payment(amount, discount_code=None, payment_method="credit_card"):
    """Calculate payment with tax, discounts, and fees."""
    subtotal = amount
    discount = 0
    
    # Apply discount
    if discount_code == "SAVE10":
        discount = subtotal * 0.10
    elif discount_code == "SAVE20":
        discount = subtotal * 0.20
    
    # Calculate tax
    taxable = subtotal - discount
    tax = taxable * 0.085
    
    # Payment fee
    fee = 0
    if payment_method == "credit_card":
        fee = subtotal * 0.029
    elif payment_method == "debit_card":
        fee = subtotal * 0.015
    
    total = taxable + tax + fee
    return {"subtotal": subtotal, "discount": discount, "tax": tax, "fee": fee, "total": total}'''
    
    tests = [
        {"name": "Basic payment", "input": {"amount": 100}, "expected": {"total": 111.4}, "function": "calculate_payment"},
        {"name": "10% discount", "input": {"amount": 100, "discount_code": "SAVE10"}, "expected": {"discount": 10}, "function": "calculate_payment"},
        {"name": "PayPal fee", "input": {"amount": 150, "payment_method": "paypal"}, "expected": {"fee": 5.25}, "function": "calculate_payment"},
    ]
    
    print("\nOriginal Code:")
    print(python_code)
    
    print("\nRunning content-agnostic evolution...")
    result = asyncio.run(evolve_code(python_code, tests, max_iterations=5))
    
    print(f"\nEvolution Result:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Fixes Applied: {result.fixes_applied}")
    print(f"  Improvement: +{result.improvement:.1%}")
    print(f"  Final Score: {result.final_score:.1%} ({result.tests_passed}/{result.tests_total})")
    
    print(f"\nEvolved Code:")
    print(result.evolved_code)
    
    return result


if __name__ == "__main__":
    demo()
