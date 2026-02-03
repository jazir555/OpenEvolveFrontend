"""
Answer Extraction Module for ACE (Agentic Context Engine)

This module provides robust utilities for extracting final answers and structured data
from LLM responses, with multiple fallback strategies to handle various response formats.

Functions:
    - extract_boxed_content: Extract content from LaTeX \\boxed{} notation
    - extract_final_answer: Extract final answer with 5 fallback strategies
    - extract_json_from_text: Extract JSON with multiple fallback strategies
    - find_json_objects: Find all JSON objects using balanced brace counting
"""

import re
import json
from typing import Optional, List, Dict, Any


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from LaTeX \\boxed{} notation using balanced brace counting.

    This function handles nested braces within \\boxed{} commands, which can occur
    in complex mathematical expressions.

    Args:
        text: String potentially containing \\boxed{} notation

    Returns:
        Extracted content inside the box, or None if no valid box found

    Examples:
        >>> extract_boxed_content(r"The answer is \\boxed{42}")
        '42'
        >>> extract_boxed_content(r"Result: \\boxed{\\frac{1}{2}}")
        r'\\frac{1}{2}'
        >>> extract_boxed_content("No box here")
        None
    """
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None

    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start

    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None


def extract_final_answer(response: str) -> str:
    """
    Extract final answer from model response using 5 fallback strategies.

    Strategies (tried in order):
        1. Direct JSON parsing for {"final_answer": "..."}
        2. Regex for final_answer field (double quotes, then single quotes)
        3. Finish[] format (common in math reasoning tasks)
        4. "The final answer is:" pattern with boxed content
        5. Extract from boxed content with "The final answer is:" prefix

    Args:
        response: Model response text

    Returns:
        Extracted final answer string, or "No final answer found" if all strategies fail

    Examples:
        >>> extract_final_answer('{"final_answer": "42"}')
        '42'
        >>> extract_final_answer('Finish[42]')
        '42'
        >>> extract_final_answer('The final answer is: \\\\boxed{42}')
        '42'
    """
    # Strategy 1: Direct JSON parsing
    try:
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", ""))
        if answer:
            return answer
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        pass

    # Strategy 2: Regex for final_answer field
    # Try double quotes first
    matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
    if matches:
        answer = matches[-1]
        return answer

    # Try single quotes
    matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
    if matches:
        answer = matches[-1]
        return answer

    # Handle JSON format without quotes (for simple expressions)
    matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
    if matches:
        answer = matches[-1].strip()
        # Clean up trailing characters
        answer = re.sub(r'[,}]*$', '', answer)
        return answer

    # Strategy 3: Finish[] format (common in math reasoning)
    matches = re.findall(r'Finish\[(.*?)\]', response)
    if matches:
        answer = matches[-1]
        return answer

    # Strategy 4: "The final answer is:" pattern with boxed
    final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
    match = re.search(final_answer_pattern, response)
    if match:
        # Extract boxed content starting from this match
        remaining_text = response[match.start():]
        boxed_content = extract_boxed_content(remaining_text)
        if boxed_content:
            return boxed_content

    # Strategy 5: More general pattern for "final answer is X"
    matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
    if matches:
        answer = matches[-1].strip()
        # Clean up common formatting
        answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
        answer = answer.replace('$', '').strip()
        if answer:
            return answer

    return "No final answer found"


def find_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Find all JSON objects in text using balanced brace counting.

    This function handles deeply nested structures and quoted strings correctly.
    It scans through text looking for JSON objects and parses them.

    Args:
        text: String potentially containing JSON objects

    Returns:
        List of parsed JSON objects (dictionaries)

    Examples:
        >>> find_json_objects('{"a": 1} and {"b": 2}')
        [{'a': 1}, {'b': 2}]
        >>> find_json_objects('nested: {"outer": {"inner": "value"}}')
        [{'outer': {'inner': 'value'}}]
    """
    json_objects = []
    i = 0

    while i < len(text):
        # Look for opening brace
        if text[i] == '{':
            start = i
            brace_count = 0
            in_string = False
            escape_next = False
            j = i

            # Count braces while handling strings
            while j < len(text):
                char = text[j]

                if escape_next:
                    escape_next = False
                    j += 1
                    continue

                if char == '\\':
                    escape_next = True
                    j += 1
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    j += 1
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete object
                            try:
                                obj = json.loads(text[start:j + 1])
                                json_objects.append(obj)
                                i = j + 1
                                break
                            except json.JSONDecodeError:
                                # Not valid JSON, continue searching
                                pass
                j += 1
            else:
                i += 1
        else:
            i += 1

    return json_objects


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text with multiple fallback strategies.

    Strategies (tried in order):
        1. Parse entire text as JSON
        2. Extract from ```json``` code blocks
        3. Find JSON objects using balanced brace counting

    Args:
        text: String potentially containing JSON

    Returns:
        Parsed JSON object (dictionary), or None if no valid JSON found

    Examples:
        >>> extract_json_from_text('{"key": "value"}')
        {'key': 'value'}
        >>> extract_json_from_text('Here is the result: ```json\\n{"key": "value"}\\n```')
        {'key': 'value'}
        >>> extract_json_from_text('No JSON here')
        None
    """
    # Strategy 1: Parse entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json``` code blocks
    json_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Also try ```code``` blocks (sometimes used for JSON)
    code_block_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Strategy 3: Find JSON objects using balanced brace counting
    json_objects = find_json_objects(text)
    if json_objects:
        # Return the first valid JSON object found
        return json_objects[0]

    return None
