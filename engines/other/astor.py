"""
astor - AST manipulation library compatibility stub.

Astor is a library for generating Python source code from AST (Abstract Syntax Trees).
This stub provides minimal implementations to allow imports to succeed.

Note: This is NOT a functional replacement for astor. Install astor package
for full functionality: pip install astor
"""

import ast
import warnings
from typing import Any, Optional

warnings.warn(
    "Using astor stub module. This is not a functional replacement for astor. "
    "Install astor package for full functionality.",
    RuntimeWarning,
    stacklevel=2
)

__version__ = "0.8.1-stub"


def to_source(node: ast.AST, indent_with: str = "    ", add_line_information: bool = False) -> str:
    """
    Convert an AST node to Python source code.
    
    This stub provides basic functionality using ast.unparse (Python 3.9+)
    or a fallback for older versions.
    
    Args:
        node: AST node to convert
        indent_with: String to use for indentation
        add_line_information: Whether to add line comments
        
    Returns:
        Python source code as string
    """
    try:
        # Python 3.9+ has ast.unparse
        return ast.unparse(node)
    except AttributeError:
        # Fallback for older Python versions
        return f"# astor.to_source stub - install astor for full functionality\n{ast.dump(node)}"


def parse_file(filename: str) -> ast.AST:
    """
    Parse a Python file and return its AST.
    
    Args:
        filename: Path to the Python file
        
    Returns:
        AST module node
    """
    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()
    return ast.parse(source, filename=filename)


class CodeGen:
    """Stub implementation of astor CodeGen."""
    
    def __init__(self, indent_with: str = "    "):
        self.indent_with = indent_with
    
    def visit(self, node: ast.AST) -> str:
        """Visit an AST node and return source code."""
        return to_source(node, self.indent_with)


def dump(node: ast.AST, annotate_fields: bool = True, include_attributes: bool = False) -> str:
    """
    Return a formatted dump of the tree in node.
    
    This is a compatibility wrapper around ast.dump.
    
    Args:
        node: AST node to dump
        annotate_fields: Add field names
        include_attributes: Include attributes
        
    Returns:
        Formatted string representation of AST
    """
    return ast.dump(node, annotate_fields=annotate_fields, include_attributes=include_attributes)


# Compatibility aliases
gen_unparse = to_source
code_gen = CodeGen

__all__ = [
    'to_source', 'parse_file', 'CodeGen', 'dump',
    'gen_unparse', 'code_gen', '__version__'
]
