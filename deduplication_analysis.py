"""
Code Deduplication Analysis and Automated Fixing

This module analyzes UI component code for duplication and provides
automated refactoring recommendations.
"""

import ast
import os
from typing import List, Dict, Tuple, Set
from difflib import SequenceMatcher
from collections import defaultdict


class CodeBlock:
    """Represents a code block for duplication analysis."""
    
    def __init__(self, file_path: str, start_line: int, end_line: int, code: str):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.code = code
        self.hash = hash(code.strip())
    
    def __repr__(self):
        return f"CodeBlock({self.file_path}:{self.start_line}-{self.end_line})"


class DeduplicationAnalyzer:
    """Analyzes code for duplication and generates refactoring recommendations."""
    
    def __init__(self, min_lines: int = 5, similarity_threshold: float = 0.8):
        """
        Initialize the analyzer.
        
        Args:
            min_lines: Minimum number of lines to consider for duplication
            similarity_threshold: Minimum similarity score (0-1) to consider as duplicate
        """
        self.min_lines = min_lines
        self.similarity_threshold = similarity_threshold
        self.code_blocks: List[CodeBlock] = []
        self.duplicates: List[Tuple[CodeBlock, CodeBlock, float]] = []
    
    def analyze_file(self, file_path: str) -> None:
        """
        Analyze a Python file for code blocks.
        
        Args:
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract function definitions
            tree = ast.parse(''.join(lines), filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.end_lineno
                    
                    if end_line - start_line >= self.min_lines:
                        code = ''.join(lines[start_line-1:end_line])
                        block = CodeBlock(file_path, start_line, end_line, code)
                        self.code_blocks.append(block)
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def find_duplicates(self) -> None:
        """Find duplicate code blocks."""
        self.duplicates = []
        
        # Compare all pairs of code blocks
        for i, block1 in enumerate(self.code_blocks):
            for block2 in self.code_blocks[i+1:]:
                # Skip if same file and overlapping lines
                if (block1.file_path == block2.file_path and
                    not (block1.end_line < block2.start_line or block2.end_line < block1.start_line)):
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(block1.code, block2.code)
                
                if similarity >= self.similarity_threshold:
                    self.duplicates.append((block1, block2, similarity))
    
    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate similarity between two code blocks.
        
        Args:
            code1: First code block
            code2: Second code block
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize code (remove whitespace variations)
        norm1 = ' '.join(code1.split())
        norm2 = ' '.join(code2.split())
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def generate_report(self) -> str:
        """
        Generate a deduplication report.
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("CODE DEDUPLICATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append(f"Total code blocks analyzed: {len(self.code_blocks)}")
        report.append(f"Duplicate pairs found: {len(self.duplicates)}")
        report.append(f"Minimum lines threshold: {self.min_lines}")
        report.append(f"Similarity threshold: {self.similarity_threshold}")
        report.append("")
        
        # Group duplicates by similarity
        high_similarity = [d for d in self.duplicates if d[2] >= 0.95]
        medium_similarity = [d for d in self.duplicates if 0.85 <= d[2] < 0.95]
        low_similarity = [d for d in self.duplicates if d[2] < 0.85]
        
        report.append(f"High similarity (>=95%): {len(high_similarity)}")
        report.append(f"Medium similarity (85-95%): {len(medium_similarity)}")
        report.append(f"Low similarity (<85%): {len(low_similarity)}")
        report.append("")
        
        # Detailed duplicates
        report.append("=" * 80)
        report.append("DUPLICATE CODE BLOCKS")
        report.append("=" * 80)
        report.append("")
        
        for i, (block1, block2, similarity) in enumerate(sorted(self.duplicates, key=lambda x: x[2], reverse=True), 1):
            report.append(f"Duplicate #{i} (Similarity: {similarity:.2%})")
            report.append(f"  Location 1: {block1.file_path}:{block1.start_line}-{block1.end_line}")
            report.append(f"  Location 2: {block2.file_path}:{block2.start_line}-{block2.end_line}")
            report.append(f"  Lines: {block1.end_line - block1.start_line + 1}")
            report.append("")
        
        return "\n".join(report)
    
    def generate_recommendations(self) -> List[Dict]:
        """
        Generate refactoring recommendations.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Group similar duplicates
        duplicate_groups = self._group_duplicates()
        
        for group_id, group in enumerate(duplicate_groups, 1):
            if len(group) < 2:
                continue
            
            # Analyze the group to suggest extraction
            first_block = group[0][0]
            
            recommendation = {
                "id": group_id,
                "type": "extract_function",
                "priority": "high" if group[0][1] >= 0.95 else "medium",
                "occurrences": len(group),
                "locations": [
                    f"{block.file_path}:{block.start_line}-{block.end_line}"
                    for block, _ in group
                ],
                "suggested_function_name": self._suggest_function_name(first_block),
                "suggested_module": "ui_utils.py",
                "estimated_lines_saved": sum(
                    block.end_line - block.start_line
                    for block, _ in group
                ) - (first_block.end_line - first_block.start_line)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _group_duplicates(self) -> List[List[Tuple[CodeBlock, float]]]:
        """Group similar duplicates together."""
        groups = []
        used_blocks = set()
        
        for block1, block2, similarity in sorted(self.duplicates, key=lambda x: x[2], reverse=True):
            # Find existing group or create new one
            found_group = False
            
            for group in groups:
                # Check if either block is already in this group
                if any(block1.hash == b.hash or block2.hash == b.hash for b, _ in group):
                    # Add both blocks if not already present
                    if block1.hash not in [b.hash for b, _ in group]:
                        group.append((block1, similarity))
                    if block2.hash not in [b.hash for b, _ in group]:
                        group.append((block2, similarity))
                    found_group = True
                    break
            
            if not found_group:
                groups.append([(block1, similarity), (block2, similarity)])
        
        return groups
    
    def _suggest_function_name(self, block: CodeBlock) -> str:
        """Suggest a function name based on the code block."""
        # Try to extract function name from the code
        try:
            tree = ast.parse(block.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Use the original function name as base
                    return f"shared_{node.name}"
        except:
            pass
        
        return "shared_utility_function"


def analyze_ui_components() -> Tuple[str, List[Dict]]:
    """
    Analyze UI components for code duplication.
    
    Returns:
        Tuple of (report, recommendations)
    """
    analyzer = DeduplicationAnalyzer(min_lines=5, similarity_threshold=0.8)
    
    # Analyze UI component files
    ui_files = [
        "ui_components.py",
        "ui_utils.py",
        "ui_models.py",
        "ui_config.py"
    ]
    
    for file_path in ui_files:
        if os.path.exists(file_path):
            print(f"Analyzing {file_path}...")
            analyzer.analyze_file(file_path)
    
    # Find duplicates
    print("Finding duplicates...")
    analyzer.find_duplicates()
    
    # Generate report and recommendations
    report = analyzer.generate_report()
    recommendations = analyzer.generate_recommendations()
    
    return report, recommendations


def save_report(report: str, recommendations: List[Dict], output_file: str = "deduplication_report.txt") -> None:
    """
    Save deduplication report to file.
    
    Args:
        report: Report string
        recommendations: List of recommendations
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write("\n\n")
        f.write("=" * 80)
        f.write("\nREFACTORING RECOMMENDATIONS")
        f.write("\n" + "=" * 80)
        f.write("\n\n")
        
        for rec in recommendations:
            f.write(f"Recommendation #{rec['id']} (Priority: {rec['priority'].upper()})\n")
            f.write(f"  Type: {rec['type']}\n")
            f.write(f"  Occurrences: {rec['occurrences']}\n")
            f.write(f"  Suggested function: {rec['suggested_function_name']}\n")
            f.write(f"  Suggested module: {rec['suggested_module']}\n")
            f.write(f"  Estimated lines saved: {rec['estimated_lines_saved']}\n")
            f.write(f"  Locations:\n")
            for loc in rec['locations']:
                f.write(f"    - {loc}\n")
            f.write("\n")
    
    print(f"Report saved to {output_file}")


if __name__ == "__main__":
    print("Starting code deduplication analysis...")
    print()
    
    report, recommendations = analyze_ui_components()
    
    print(report)
    print()
    
    if recommendations:
        print(f"Generated {len(recommendations)} refactoring recommendations")
        save_report(report, recommendations)
    else:
        print("No significant code duplication found!")
