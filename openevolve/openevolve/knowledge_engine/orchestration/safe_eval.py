"""
Safe Expression Evaluator

Replaces dangerous eval() with a safe expression parser for pipeline conditions.
Supports basic comparison operators, logical operators, and context variable access.
"""

import ast
import operator
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SafeExpressionEvaluator:
    """
    Safe expression evaluator for pipeline conditions.
    
    Supports:
    - Comparison operators: ==, !=, <, >, <=, >=
    - Logical operators: and, or, not
    - Mathematical operators: +, -, *, /, //, %
    - Context variable access: context['key'], context.get('key')
    - Literals: strings, numbers, lists, booleans, None
    - Built-in functions: len(), str(), int(), float(), bool()
    """
    
    # Allowed operators
    OPERATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
    }
    
    # Allowed built-in functions
    BUILTINS = {
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'any': any,
        'all': all,
        'get': lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default,
    }
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator with context.
        
        Args:
            context: Dictionary of variables available to expressions
        """
        self.context = context or {}
    
    def eval(self, expression: str) -> Any:
        """
        Safely evaluate an expression.
        
        Args:
            expression: Python expression string
            
        Returns:
            Result of evaluation
            
        Raises:
            ValueError: If expression contains unsafe operations
            SyntaxError: If expression is invalid
        """
        if not expression or not expression.strip():
            return True  # Empty condition defaults to True
        
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except SyntaxError as e:
            logger.error(f"Invalid expression syntax: {expression}")
            raise ValueError(f"Invalid expression: {e}")
        except Exception as e:
            logger.error(f"Expression evaluation failed: {expression}")
            raise ValueError(f"Evaluation error: {e}")
    
    def _eval_node(self, node: ast.AST) -> Any:
        """Evaluate an AST node"""
        
        # Literals
        if isinstance(node, ast.Constant):
            return node.value
        
        if isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        
        if isinstance(node, ast.NameConstant):  # Python < 3.8
            return node.value
        
        # Lists and tuples
        if isinstance(node, ast.List):
            return [self._eval_node(e) for e in node.elts]
        
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(e) for e in node.elts)
        
        if isinstance(node, ast.Set):
            return {self._eval_node(e) for e in node.elts}
        
        if isinstance(node, ast.Dict):
            return {
                self._eval_node(k): self._eval_node(v)
                for k, v in zip(node.keys, node.values)
            }
        
        # Variables
        if isinstance(node, ast.Name):
            if node.id in self.context:
                return self.context[node.id]
            if node.id in self.BUILTINS:
                return self.BUILTINS[node.id]
            raise ValueError(f"Undefined variable: {node.id}")
        
        # Attribute access (e.g., context.get)
        if isinstance(node, ast.Attribute):
            obj = self._eval_node(node.value)
            return getattr(obj, node.attr)
        
        # Subscript (e.g., context['key'])
        if isinstance(node, ast.Subscript):
            obj = self._eval_node(node.value)
            slice_val = self._eval_node(node.slice)
            return obj[slice_val]
        
        # Binary operations (comparisons, math)
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                if type(op) not in self.OPERATORS:
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                if not self.OPERATORS[type(op)](left, right):
                    return False
                left = right
            return True
        
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if type(node.op) not in self.OPERATORS:
                raise ValueError(f"Unsupported binary operation: {type(node.op).__name__}")
            return self.OPERATORS[type(node.op)](left, right)
        
        # Unary operations (not, -, +)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if type(node.op) not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
            return self.OPERATORS[type(node.op)](operand)
        
        # Boolean operations (and, or)
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(v) for v in node.values]
            if type(node.op) not in self.OPERATORS:
                raise ValueError(f"Unsupported boolean operation: {type(node.op).__name__}")
            
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        
        # Function calls (limited to builtins)
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            
            # Check if function is allowed
            if func not in self.BUILTINS.values():
                raise ValueError(f"Function not allowed: {func}")
            
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {
                kw.arg: self._eval_node(kw.value)
                for kw in node.keywords
            }
            
            return func(*args, **kwargs)
        
        # Conditional expressions
        if isinstance(node, ast.IfExp):
            test = self._eval_node(node.test)
            if test:
                return self._eval_node(node.body)
            else:
                return self._eval_node(node.orelse)
        
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Safely evaluate an expression with context.
    
    Args:
        expression: Python expression string
        context: Dictionary of variables available to expressions
        
    Returns:
        Result of evaluation
        
    Example:
        >>> safe_eval("len(context.get('items', [])) > 5", {'context': {'items': [1,2,3,4,5,6]}})
        True
        
        >>> safe_eval("data_type == 'financial' and len(text) > 100", 
        ...           {'data_type': 'financial', 'text': 'A' * 150})
        True
    """
    evaluator = SafeExpressionEvaluator(context)
    return evaluator.eval(expression)


# Predefined condition evaluators for common use cases
class ConditionEvaluator:
    """Predefined condition evaluators for pipeline stages"""
    
    @staticmethod
    def data_type_is(context: Dict[str, Any], expected_type: str) -> bool:
        """Check if data type matches"""
        return context.get('data_type') == expected_type
    
    @staticmethod
    def has_key(context: Dict[str, Any], key: str) -> bool:
        """Check if context has key with non-empty value"""
        value = context.get(key)
        if value is None:
            return False
        if isinstance(value, (list, dict, str)):
            return len(value) > 0
        return True
    
    @staticmethod
    def min_length(context: Dict[str, Any], key: str, min_len: int) -> bool:
        """Check if value has minimum length"""
        value = context.get(key, [])
        return len(value) >= min_len
    
    @staticmethod
    def domain_is(context: Dict[str, Any], expected_domain: str) -> bool:
        """Check if domain matches"""
        return context.get('domain') == expected_domain
