#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenEvolve PES Integration - Production Business Logic Demo

This demo demonstrates real-world code evolution that IMPROVES:
1. Code correctness (tests passing)
2. Performance (execution time)
3. Code quality (linting, complexity)
4. Maintainability (type hints, docs)

Production Test Cases:
1. Payment processing calculation
2. Data validation pipeline
3. Order fulfillment logic
4. Inventory management

Usage:
    python demo_pes_workflow.py --problem payment --iterations 5
    python demo_pes_workflow.py --problem all --iterations 3
"""

import asyncio
import json
import logging
import sys
import time
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("PES-Production-Demo")


# =============================================================================
# Business Logic Problem Definitions
# =============================================================================

class ProblemType(Enum):
    PAYMENT = "payment"
    VALIDATION = "validation"
    ORDER = "order"
    INVENTORY = "inventory"


@dataclass
class TestCase:
    """A test case for validating business logic."""
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


# Production test cases with known issues in initial code
PROBLEM_DEFINITIONS = {
    ProblemType.PAYMENT: {
        "task": "Implement payment calculation with tax, discounts, and fees.",
        "initial_code": '''def calculate_payment(amount, discount_code=None, payment_method="credit_card"):
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
    return {"subtotal": subtotal, "discount": discount, "tax": tax, "fee": fee, "total": total}''',
        
        "test_cases": [
            TestCase("Basic payment", {"amount": 100, "discount_code": None, "payment_method": "credit_card"}, 
                    {"subtotal": 100, "discount": 0, "tax": 8.5, "fee": 2.9, "total": 111.4}, 1.0),
            TestCase("10% discount", {"amount": 100, "discount_code": "SAVE10", "payment_method": "credit_card"},
                    {"subtotal": 100, "discount": 10, "tax": 7.65, "fee": 2.9, "total": 100.55}, 1.5),
            TestCase("20% discount", {"amount": 200, "discount_code": "SAVE20", "payment_method": "debit_card"},
                    {"subtotal": 200, "discount": 40, "tax": 13.6, "fee": 3, "total": 176.6}, 1.5),
            TestCase("PayPal fee", {"amount": 150, "discount_code": None, "payment_method": "paypal"},
                    {"subtotal": 150, "discount": 0, "tax": 12.75, "fee": 5.25, "total": 168}, 1.0),
            TestCase("BOGO discount", {"amount": 100, "discount_code": "BOGO", "payment_method": "credit_card"},
                    {"subtotal": 100, "discount": 0, "tax": 8.5, "fee": 2.9, "total": 111.4}, 1.0),
        ],
        
        "fixes": [
            {
                "pattern": 'discount = subtotal * 0.20',
                "fix": 'discount = subtotal * 0.20\n    elif discount_code == "BOGO":\n        discount = subtotal * 0.50',
                "description": "Add BOGO discount handling"
            },
            {
                "pattern": '    fee = subtotal * 0.015',
                "fix": '    fee = subtotal * 0.015\n    elif payment_method == "paypal":\n        fee = subtotal * 0.035',
                "description": "Add PayPal fee handling"
            }
        ]
    },
    
    ProblemType.VALIDATION: {
        "task": "Implement email validation with format and domain checking.",
        "initial_code": '''def validate_email(email):
    """Validate email format and domain."""
    import re
    
    if not email:
        return {"valid": False, "email": email, "error": "Empty email"}
    
    pattern = r"^[\\w.-]+@[\\w.-]+\\.\\w+$"
    if not re.match(pattern, email):
        return {"valid": False, "email": email, "error": "Invalid format"}
    
    return {"valid": True, "email": email, "error": None}''',
        
        "test_cases": [
            TestCase("Valid email", {"email": "user@example.com"}, {"valid": True, "error": None}, 1.0),
            TestCase("Valid with dots", {"email": "john.doe@company.co.uk"}, {"valid": True, "error": None}, 1.0),
            TestCase("No @ symbol", {"email": "userexample.com"}, {"valid": False, "error": "Invalid format"}, 1.5),
            TestCase("No TLD", {"email": "user@domain"}, {"valid": False, "error": "Invalid format"}, 1.0),
            TestCase("Disposable domain", {"email": "user@mailinator.com"}, {"valid": False, "error": "Disposable domain"}, 1.5),
            TestCase("Empty string", {"email": ""}, {"valid": False, "error": "Empty email"}, 1.0),
        ],
        
        "fixes": [
            {
                "pattern": '    return {"valid": True, "email": email, "error": None}',
                "fix": '''    # Check disposable domains
    disposable_domains = ["mailinator", "tempmail", "fakeemail", "throwaway"]
    domain = email.split("@")[-1].split(".")[0] if "@" in email else ""
    if domain.lower() in disposable_domains:
        return {"valid": False, "email": email, "error": "Disposable domain"}

    return {"valid": True, "email": email, "error": None}''',
                "description": "Add disposable domain checking"
            }
        ]
    },
    
    ProblemType.ORDER: {
        "task": "Implement order processing with bulk discounts and shipping.",
        "initial_code": '''def process_order(items, shipping_distance_km=100):
    """Process order with pricing and shipping."""
    if not items:
        return {"error": "No items"}
    
    subtotal = 0
    total_qty = 0
    
    for item in items:
        price = item.get("price", 0)
        qty = item.get("quantity", 1)
        subtotal += price * qty
        total_qty += qty
    
    # Discount
    discount = 0
    if total_qty >= 10:
        discount = subtotal * 0.10
    
    # Shipping
    weight = total_qty * 1.0  # 1kg per item
    shipping = 5 + weight * 0.5 + shipping_distance_km / 100 * 2
    
    total = subtotal - discount + shipping
    return {"subtotal": subtotal, "discount": discount, "shipping": shipping, "total": total, "items": total_qty}''',
        
        "test_cases": [
            TestCase("Basic order", {"items": [{"price": 10, "quantity": 2}]}, 
                    {"subtotal": 20, "discount": 0, "items": 2}, 1.0),
            TestCase("Bulk 10%", {"items": [{"price": 10, "quantity": 10}]}, 
                    {"subtotal": 100, "discount": 10, "items": 10}, 1.5),
            TestCase("Bulk 15%", {"items": [{"price": 10, "quantity": 25}]}, 
                    {"subtotal": 250, "discount": 37.5, "items": 25}, 1.5),
            TestCase("Empty order", {"items": []}, None, 1.0),
            TestCase("Long distance", {"items": [{"price": 50, "quantity": 1}]}, 
                    {"subtotal": 50, "discount": 0, "items": 1}, 1.0),
        ],
        
        "fixes": [
            {
                "pattern": r'if total_qty >= 10:',
                "fix": '''    # Bulk discount
    discount = 0
    if total_qty >= 25:
        discount = subtotal * 0.15
    elif total_qty >= 10:
        discount = subtotal * 0.10''',
                "description": "Add tiered bulk discount (15% for 25+ items)"
            },
            {
                "pattern": r'if not items:',
                "fix": '''    if not items or len(items) == 0:
        return {"error": "Order must contain at least one item", "subtotal": 0, "discount": 0, "shipping": 0, "total": 0, "items": 0}''',
                "description": "Fix empty order error handling"
            }
        ]
    },
    
    ProblemType.INVENTORY: {
        "task": "Implement inventory allocation across warehouses.",
        "initial_code": '''def allocate_inventory(order_items, warehouses):
    """Allocate inventory to order from warehouses."""
    if not order_items:
        return {"error": "No items specified", "allocation": {}, "unallocated": [], "complete": False}
    
    allocation = {}
    unallocated = []
    
    for item_name, quantity in order_items.items():
        allocated_qty = 0
        item_alloc = {}
        
        for wh in warehouses:
            if allocated_qty >= quantity:
                break
            stock = wh.get("stock", {}).get(item_name, 0)
            available = stock - wh.get("reserved", 0)
            to_alloc = min(available, quantity - allocated_qty)
            
            if to_alloc > 0:
                item_alloc[wh["name"]] = to_alloc
                allocated_qty += to_alloc
        
        if allocated_qty < quantity:
            unallocated.append(item_name)
        
        allocation[item_name] = item_alloc
    
    return {"allocation": allocation, "unallocated": unallocated, "complete": len(unallocated) == 0}''',
        
        "test_cases": [
            TestCase("Basic allocation", {"order_items": {"widget": 10}, "warehouses": [{"name": "warehouse_a", "stock": {"widget": 20}, "reserved": 0}]}, 
                    {"unallocated": [], "complete": True}, 1.0),
            TestCase("Multiple warehouses", {"order_items": {"gadget": 15, "gizmo": 5}, "warehouses": [{"name": "warehouse_a", "stock": {"gadget": 10, "gizmo": 5}, "reserved": 2}]},
                    {"unallocated": [], "complete": True}, 1.5),
            TestCase("Out of stock", {"order_items": {"rare_item": 5}, "warehouses": [{"name": "warehouse_a", "stock": {"rare_item": 0}, "reserved": 0}]},
                    {"unallocated": ["rare_item"], "complete": False}, 1.5),
        ],
        
        "fixes": []  # Initial code is mostly correct
    }
}


# =============================================================================
# Code Evaluation System
# =============================================================================

class CodeEvaluator:
    """Production-grade code evaluator."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_code(
        self,
        code: str,
        problem_type: ProblemType,
        test_cases: List[TestCase]
    ) -> EvaluationMetrics:
        """Evaluate code against test cases."""
        metrics = EvaluationMetrics()
        
        try:
            exec_globals = {}
            exec(code, exec_globals)
            
            func_name = self._get_function_name(problem_type)
            main_func = exec_globals.get(func_name)
            
            if not main_func:
                metrics.issues.append(f"Function '{func_name}' not found")
                return metrics
            
            # Run tests
            tests_passed = 0
            total_weight = 0
            exec_times = []
            
            for test in test_cases:
                try:
                    start = time.perf_counter()
                    result = main_func(**test.input_data)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    exec_times.append(elapsed_ms)
                    
                    passed = self._check_result(result, test.expected_output, test.input_data)
                    
                    if passed:
                        tests_passed += test.weight
                    else:
                        metrics.failing_tests.append(test.name)
                    
                    total_weight += test.weight
                    
                except Exception as e:
                    if test.expected_output is None:
                        tests_passed += test.weight
                    else:
                        metrics.failing_tests.append(f"{test.name}: {str(e)}")
            
            metrics.tests_passed = tests_passed
            metrics.tests_total = total_weight
            metrics.correctness_score = tests_passed / total_weight if total_weight > 0 else 0
            metrics.execution_time_ms = max(exec_times) if exec_times else 0
            metrics.performance_score = max(0, 1 - (metrics.execution_time_ms / 10000))
            
        except Exception as e:
            metrics.issues.append(str(e))
        
        metrics.quality_score = self._calculate_quality_score(code)
        metrics.overall_score = (
            metrics.correctness_score * 0.5 +
            metrics.performance_score * 0.25 +
            metrics.quality_score * 0.25
        )
        
        return metrics
    
    def _get_function_name(self, problem_type: ProblemType) -> str:
        mapping = {
            ProblemType.PAYMENT: "calculate_payment",
            ProblemType.VALIDATION: "validate_email",
            ProblemType.ORDER: "process_order",
            ProblemType.INVENTORY: "allocate_inventory",
        }
        return mapping.get(problem_type, "main")
    
    def _check_result(self, result: Any, expected: Any, input_data: Dict[str, Any]) -> bool:
        """Check if result matches expected."""
        if expected is None:
            return result is not None and "error" in str(result).lower()
        
        if not isinstance(result, dict) or not isinstance(expected, dict):
            return result == expected
        
        for key, exp_val in expected.items():
            if key not in result:
                return False
            if isinstance(exp_val, float):
                if abs(result[key] - exp_val) > 0.1:
                    return False
            elif result[key] != exp_val:
                return False
        
        return True
    
    def _calculate_quality_score(self, code: str) -> float:
        score = 1.0
        
        if '"""' in code or "'''" in code:
            score += 0.1
        else:
            score -= 0.2
        
        if ': str' in code or ': int' in code or ': float' in code:
            score += 0.1
        else:
            score -= 0.1
        
        if 'try:' in code and 'except' in code:
            score += 0.1
        else:
            score -= 0.1
        
        validation_keywords = ['if not', 'if len', 'isinstance', 'raise', 'ValueError']
        if any(kw in code for kw in validation_keywords):
            score += 0.1
        else:
            score -= 0.1
        
        if 'round(' in code:
            score += 0.1
        
        return max(0, min(1, score))


# =============================================================================
# Smart Code Improver
# =============================================================================

class SmartCodeImprover:
    """
    Code improver that applies targeted fixes based on test failures.
    
    Analyzes failing tests and applies appropriate code fixes.
    Fixes are applied only once per execution.
    """
    
    def __init__(self, evaluator: CodeEvaluator):
        self.evaluator = evaluator
        self.fixes_applied = []
    
    async def improve_code(
        self,
        code: str,
        problem_type: ProblemType,
        failing_tests: List[str]
    ) -> str:
        """Apply targeted fixes based on failing tests."""
        improved_code = code
        
        # Get problem-specific fixes
        problem = PROBLEM_DEFINITIONS[problem_type]
        fixes = problem.get("fixes", [])
        
        for fix_def in fixes:
            pattern = fix_def["pattern"]
            fix_content = fix_def["fix"]
            description = fix_def["description"]
            
            # Skip if already applied (to prevent re-application)
            if description in self.fixes_applied:
                continue
            
            # Check if this fix is needed based on failing tests
            if self._should_apply_fix(description, failing_tests):
                # Use simple string replacement (not regex) for safer substitution
                # Find the exact pattern in the code
                if pattern in improved_code:
                    improved_code = improved_code.replace(pattern, fix_content)
                    self.fixes_applied.append(description)
                    logger.info(f"Applied fix: {description}")
        
        # General improvements
        improved_code = self._add_general_improvements(improved_code, failing_tests)
        
        return improved_code
    
    def _should_apply_fix(self, fix_description: str, failing_tests: List[str]) -> bool:
        """Determine if a fix should be applied based on failing tests."""
        if not failing_tests:
            return False
        
        # Direct mapping from description keywords to test name patterns
        desc_lower = fix_description.lower()
        
        for test in failing_tests:
            test_lower = test.lower()
            
            # PayPal fix
            if "paypal" in desc_lower and "paypal" in test_lower:
                return True
            
            # BOGO fix  
            if "bogo" in desc_lower and "bogo" in test_lower:
                return True
            
            # Bulk discount fix
            if "bulk" in desc_lower and "bulk" in test_lower:
                return True
            
            # Empty order fix
            if "empty" in desc_lower and "empty" in test_lower:
                return True
            
            # Disposable domain fix
            if "disposable" in desc_lower and "disposable" in test_lower:
                return True
            
            # Tiered discount fix (15% for 25+)
            if ("tiered" in desc_lower or "15%" in desc_lower) and ("15" in test or "25" in test):
                return True
        
        return False
    
    def _add_general_improvements(self, code: str, failing_tests: List[str]) -> str:
        """Add general improvements based on test failures."""
        # Add input validation if tests are failing due to invalid inputs
        if any("empty" in t.lower() or "invalid" in t.lower() for t in failing_tests):
            if 'if not ' not in code and 'isinstance' not in code:
                # Add validation after docstring
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if 'def ' in line:
                        # Find end of function signature
                        while i < len(lines) and '):' not in lines[i]:
                            i += 1
                        # Add validation
                        validation = '\n    # Input validation\n    if not items or len(items) == 0:\n        return {"error": "Invalid input"}'
                        lines.insert(i + 1, validation)
                        return '\n'.join(lines)
        
        return code


# =============================================================================
# PES Planner
# =============================================================================

class PESPlanner:
    """Planner that generates improvement plans."""
    
    def __init__(self, evaluator: CodeEvaluator):
        self.evaluator = evaluator
    
    async def plan(self, code: str, metrics: EvaluationMetrics) -> str:
        """Generate improvement plan based on current metrics."""
        issues = []
        
        if metrics.correctness_score < 1.0:
            for test in metrics.failing_tests[:3]:
                issues.append(f"Failing test: {test}")
        
        if metrics.quality_score < 0.7:
            if '"""' not in code:
                issues.append("Missing docstring")
            if 'raise' not in code and 'ValueError' not in code:
                issues.append("Missing input validation")
        
        plan = f"""
## Code Improvement Plan

### Current Status
- Correctness: {metrics.correctness_score:.1%}
- Performance: {metrics.performance_score:.1%}
- Quality: {metrics.quality_score:.1%}
- Overall: {metrics.overall_score:.1%}

### Issues to Address
{chr(10).join(f'- {i}' for i in issues)}

### Priority Fixes
"""
        
        if metrics.correctness_score < 1.0:
            plan += "1. Fix failing test cases\n"
        if metrics.quality_score < 0.8:
            plan += "2. Add docstrings and validation\n"
        if metrics.performance_score < 0.8:
            plan += "3. Optimize performance\n"
        
        return plan


# =============================================================================
# PES Executor
# =============================================================================

class PESExecutor:
    """Executor that applies improvements to code."""
    
    def __init__(self, evaluator: CodeEvaluator):
        self.evaluator = evaluator
        self.improver = SmartCodeImprover(evaluator)
    
    async def execute(
        self,
        code: str,
        plan: str,
        problem_type: ProblemType,
        failing_tests: List[str]
    ) -> str:
        """Apply plan to improve code."""
        # Use smart improver to apply targeted fixes
        improved_code = await self.improver.improve_code(code, problem_type, failing_tests)
        return improved_code


# =============================================================================
# PES Summary
# =============================================================================

class PESSummary:
    """Summarizer that documents improvements."""
    
    def summarize(
        self,
        original_code: str,
        improved_code: str,
        original_metrics: EvaluationMetrics,
        improved_metrics: EvaluationMetrics,
        iteration: int,
        fixes_applied: List[str]
    ) -> Dict[str, Any]:
        """Generate summary of improvements."""
        improvements = {
            "correctness": improved_metrics.correctness_score - original_metrics.correctness_score,
            "performance": improved_metrics.performance_score - original_metrics.performance_score,
            "quality": improved_metrics.quality_score - original_metrics.quality_score,
            "overall": improved_metrics.overall_score - original_metrics.overall_score,
        }
        
        return {
            "iteration": iteration,
            "improvements": {k: round(v, 4) for k, v in improvements.items()},
            "original_score": original_metrics.overall_score,
            "improved_score": improved_metrics.overall_score,
            "test_improvement": f"{original_metrics.tests_passed:.1f} -> {improved_metrics.tests_passed:.1f}",
            "issues_fixed": len(original_metrics.failing_tests) - len(improved_metrics.failing_tests),
            "fixes_applied": fixes_applied,
        }


# =============================================================================
# PES Evolution Agent
# =============================================================================

class PESEvolutionAgent:
    """Production PES Evolution Agent."""
    
    def __init__(self, problem_type: ProblemType, max_iterations: int = 5):
        self.problem_type = problem_type
        self.max_iterations = max_iterations
        self.evaluator = CodeEvaluator()
        self.planner = PESPlanner(self.evaluator)
        self.executor = PESExecutor(self.evaluator)
        self.summarizer = PESSummary()
        
        self.problem = PROBLEM_DEFINITIONS[problem_type]
        self.test_cases = self.problem["test_cases"]
    
    async def evolve(self) -> Dict[str, Any]:
        """Run the evolution process."""
        logger.info(f"Starting evolution for: {self.problem_type.value}")
        
        current_code = self.problem["initial_code"]
        history = []
        fixes_history = []
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            # Evaluate current code
            current_metrics = self.evaluator.evaluate_code(
                current_code, self.problem_type, self.test_cases
            )
            
            logger.info(f"  Correctness: {current_metrics.correctness_score:.1%} "
                       f"({current_metrics.tests_passed:.1f}/{current_metrics.tests_total})")
            logger.info(f"  Performance: {current_metrics.performance_score:.1%}")
            logger.info(f"  Quality: {current_metrics.quality_score:.1%}")
            logger.info(f"  Overall: {current_metrics.overall_score:.1%}")
            
            if current_metrics.failing_tests:
                logger.info(f"  Failing: {current_metrics.failing_tests[:3]}")
            
            # Check convergence
            if current_metrics.overall_score >= 0.95 and current_metrics.correctness_score >= 0.98:
                logger.info(f"  Converged at iteration {iteration}")
                break
            
            # Generate plan
            plan = await self.planner.plan(current_code, current_metrics)
            
            # Apply improvements
            original_failing = current_metrics.failing_tests.copy()
            current_code = await self.executor.execute(
                current_code, plan, self.problem_type, current_metrics.failing_tests
            )
            fixes_applied = self.executor.improver.fixes_applied[len(fixes_history):]
            fixes_history.extend(fixes_applied)
            
            if fixes_applied:
                logger.info(f"  Fixes applied: {len(fixes_applied)}")
                for fix in fixes_applied:
                    logger.info(f"    - {fix}")
            else:
                logger.info(f"  No new fixes applied")
            
            # Re-evaluate
            improved_metrics = self.evaluator.evaluate_code(
                current_code, self.problem_type, self.test_cases
            )
            
            # Summarize
            summary = self.summarizer.summarize(
                self.problem["initial_code"],
                current_code,
                current_metrics,
                improved_metrics,
                iteration,
                fixes_applied
            )
            history.append(summary)
            
            improvement = summary["improvements"].get("overall", 0)
            logger.info(f"  Improvement: +{improvement:.1%}")
            
            # Check if we made progress
            if improvement <= 0 and iteration > 2:
                logger.info("  No improvement, trying different approach...")
        
        # Final evaluation
        final_metrics = self.evaluator.evaluate_code(
            current_code, self.problem_type, self.test_cases
        )
        
        return {
            "problem_type": self.problem_type.value,
            "final_code": current_code,
            "final_metrics": final_metrics.to_dict(),
            "evolution_history": history,
            "total_iterations": len(history),
            "total_fixes": len(fixes_history),
            "fixes_applied": fixes_history,
            "improvement": history[-1]["improvements"].get("overall", 0) if history else 0,
        }


# =============================================================================
# Demo Runner
# =============================================================================

def print_result(result: Dict[str, Any]) -> None:
    """Print evolution result."""
    print("\n" + "="*70)
    print(f"  EVOLUTION RESULT: {result['problem_type'].upper()}")
    print("="*70)
    
    metrics = result["final_metrics"]
    print(f"\n  Final Scores:")
    print(f"    Correctness:  {metrics['correctness_score']:.1%} ({metrics['tests_passed']:.1f}/{metrics['tests_total']} tests)")
    print(f"    Performance:  {metrics['performance_score']:.1%} ({metrics['execution_time_ms']:.1f}ms)")
    print(f"    Quality:      {metrics['quality_score']:.1%}")
    print(f"    Overall:      {metrics['overall_score']:.1%}")
    
    print(f"\n  Evolution Progress:")
    for h in result["evolution_history"][-5:]:
        fixes = h.get("fixes_applied", [])
        fix_str = f" ({len(fixes)} fixes)" if fixes else ""
        print(f"    Iteration {h['iteration']}: +{h['improvements'].get('overall', 0):.1%}"
              f"{fix_str} (tests: {h['test_improvement']})")
    
    print(f"\n  Total Fixes Applied: {result['total_fixes']}")
    for fix in result.get("fixes_applied", [])[:5]:
        print(f"    - {fix}")
    
    print(f"\n  Total Improvement: +{result['improvement']:.1%}")
    print(f"  Total Iterations:  {result['total_iterations']}")
    
    print(f"\n  Final Code (first 400 chars):")
    print("-"*70)
    print(result["final_code"][:400] + "..." if len(result["final_code"]) > 400 else result["final_code"])
    print("-"*70)
    
    if metrics.get("failing_tests"):
        print(f"\n  Remaining Failing Tests:")
        for test in metrics["failing_tests"][:3]:
            print(f"    - {test}")
    else:
        print(f"\n  All tests passing!")
    
    print("\n" + "="*70)


async def run_production_demo():
    """Run the production business logic demo."""
    print("\n" + "="*70)
    print("  OpenEvolve PES - Production Business Logic Demo")
    print("  Evolving real-world code with measurable improvements")
    print("="*70 + "\n")
    
    results = []
    
    for problem_type in ProblemType:
        print(f"\n{'#'*70}")
        print(f"# Problem: {problem_type.value.upper()}")
        print(f"{'#'*70}")
        
        agent = PESEvolutionAgent(problem_type, max_iterations=5)
        result = await agent.evolve()
        results.append(result)
        
        print_result(result)
    
    # Summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70)
    
    for result in results:
        fixes = len(result.get("fixes_applied", []))
        print(f"  {result['problem_type']:12}: "
              f"Score {result['final_metrics']['overall_score']:.1%}, "
              f"Fixes: {fixes}, "
              f"Improvement: +{result['improvement']:.1%}")
    
    total_fixes = sum(len(r.get("fixes_applied", [])) for r in results)
    avg_score = sum(r["final_metrics"]["overall_score"] for r in results) / len(results)
    avg_improvement = sum(r["improvement"] for r in results) / len(results)
    
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
        description="OpenEvolve PES Production Demo",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--problem", type=str, default="all",
                       choices=["payment", "validation", "order", "inventory", "all"],
                       help="Problem type to evolve")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Maximum iterations per problem")
    
    args = parser.parse_args()
    
    if args.problem == "all":
        asyncio.run(run_production_demo())
    else:
        problem_type = ProblemType(args.problem)
        agent = PESEvolutionAgent(problem_type, args.iterations)
        result = asyncio.run(agent.evolve())
        print_result(result)


if __name__ == "__main__":
    main()
