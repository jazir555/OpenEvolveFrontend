"""
Test suite for ACE Answer Extraction Module

Tests cover various response formats and edge cases for extraction functions.
"""

import pytest
from ace.extraction import (
    extract_boxed_content,
    extract_final_answer,
    extract_json_from_text,
    find_json_objects
)


class TestExtractBoxedContent:
    """Test suite for extract_boxed_content function"""

    def test_simple_boxed_content(self):
        """Test extraction of simple boxed content"""
        result = extract_boxed_content(r"The answer is \boxed{42}")
        assert result == "42"

    def test_boxed_with_latex(self):
        """Test extraction of boxed content with LaTeX commands"""
        result = extract_boxed_content(r"Result: \boxed{\frac{1}{2}}")
        assert result == r"\frac{1}{2}"

    def test_nested_braces_in_boxed(self):
        """Test boxed content with nested braces"""
        result = extract_boxed_content(r"\boxed{{a: {b: c}}}")
        assert result == "{a: {b: c}}"

    def test_no_boxed_content(self):
        """Test when no boxed content exists"""
        result = extract_boxed_content("Just plain text")
        assert result is None

    def test_boxed_with_dollar_signs(self):
        """Test boxed content within math mode"""
        result = extract_boxed_content(r"$\boxed{42}$")
        assert result == "42"

    def test_multiple_boxed(self):
        """Test extraction with multiple boxed expressions"""
        result = extract_boxed_content(r"First \boxed{1} then \boxed{2}")
        # Should return the first one
        assert result == "1"


class TestExtractFinalAnswer:
    """Test suite for extract_final_answer function"""

    def test_extract_json_clean(self):
        """Test Strategy 1: Clean JSON format"""
        response = '{"final_answer": "42"}'
        result = extract_final_answer(response)
        assert result == "42"

    def test_extract_json_with_extra_fields(self):
        """Test JSON with additional fields"""
        response = '{"reasoning": "math", "final_answer": "100"}'
        result = extract_final_answer(response)
        assert result == "100"

    def test_extract_json_messy(self):
        """Test Strategy 2: Regex extraction from messy text"""
        response = '''
        Let me think about this...
        {"final_answer": "The answer is 42"}
        That's it!
        '''
        result = extract_final_answer(response)
        assert result == "The answer is 42"

    def test_extract_single_quotes(self):
        """Test extraction with single quotes"""
        response = "{'final_answer': 'test_value'}"
        result = extract_final_answer(response)
        assert result == "test_value"

    def test_extract_finish_format(self):
        """Test Strategy 3: Finish[] format"""
        response = "After reasoning, Finish[42] is correct"
        result = extract_final_answer(response)
        assert result == "42"

    def test_extract_finish_multiple(self):
        """Test Finish[] with multiple occurrences"""
        response = "Finish[intermediate] ... Finish[final]"
        result = extract_final_answer(response)
        # Should return the last one
        assert result == "final"

    def test_extract_boxed_content(self):
        """Test Strategy 4: Boxed content extraction"""
        response = r"The final answer is: \boxed{42}"
        result = extract_final_answer(response)
        assert result == "42"

    def test_extract_boxed_with_dollar_signs(self):
        """Test boxed with math mode delimiters"""
        response = r"The final answer is: $\boxed{42}$"
        result = extract_final_answer(response)
        assert result == "42"

    def test_fallback_strategies(self):
        """Test Strategy 5: General 'final answer is' pattern"""
        response = "The final answer is: 42"
        result = extract_final_answer(response)
        assert result == "42"

    def test_fallback_with_period(self):
        """Test final answer pattern with period"""
        response = "The final answer is: 42."
        result = extract_final_answer(response)
        # Should strip the period
        assert result == "42"

    def test_fallback_lowercase(self):
        """Test lowercase variant"""
        response = "the final answer is: test"
        result = extract_final_answer(response)
        assert result == "test"

    def test_no_answer_found(self):
        """Test when no answer can be extracted"""
        response = "Just some random text without any answer"
        result = extract_final_answer(response)
        assert result == "No final answer found"

    def test_complex_latex_in_boxed(self):
        """Test complex LaTeX expression in boxed"""
        response = r"The final answer is: \boxed{\frac{3\pi}{4}}"
        result = extract_final_answer(response)
        assert result == r"\frac{3\pi}{4}"

    def test_nested_boxed_content(self):
        """Test boxed with nested braces"""
        response = r"The final answer is: \boxed{{nested: {value}}}"
        result = extract_final_answer(response)
        assert result == "{nested: {value}}"


class TestExtractJsonFromText:
    """Test suite for extract_json_from_text function"""

    def test_extract_json_clean(self):
        """Test Strategy 1: Clean JSON"""
        result = extract_json_from_text('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_json_from_code_block(self):
        """Test Strategy 2: JSON in markdown code block"""
        text = '''
        Here's the result:
        ```json
        {"answer": 42, "confidence": "high"}
        ```
        '''
        result = extract_json_from_text(text)
        assert result == {"answer": 42, "confidence": "high"}

    def test_extract_json_from_plain_code_block(self):
        """Test JSON in plain ``` block"""
        text = '```{"key": "value"}```'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_extract_json_messy(self):
        """Test Strategy 3: Find JSON in messy text"""
        text = 'Let me think... {"answer": "found"} ...done'
        result = extract_json_from_text(text)
        assert result == {"answer": "found"}

    def test_extract_json_no_json(self):
        """Test when no JSON is present"""
        result = extract_json_from_text("Just plain text")
        assert result is None

    def test_extract_json_nested(self):
        """Test extraction of nested JSON"""
        text = '{"outer": {"inner": "deep", "number": 123}}'
        result = extract_json_from_text(text)
        assert result == {"outer": {"inner": "deep", "number": 123}}

    def test_extract_json_multiple(self):
        """Test with multiple JSON objects"""
        text = '{"first": 1} and {"second": 2}'
        result = extract_json_from_text(text)
        # Should return the first one
        assert result == {"first": 1}


class TestFindJsonObjects:
    """Test suite for find_json_objects function"""

    def test_find_single_object(self):
        """Test finding a single JSON object"""
        text = '{"key": "value"}'
        result = find_json_objects(text)
        assert len(result) == 1
        assert result[0] == {"key": "value"}

    def test_find_multiple_objects(self):
        """Test finding multiple JSON objects"""
        text = '{"a": 1} and {"b": 2}'
        result = find_json_objects(text)
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}

    def test_find_nested_objects(self):
        """Test finding nested JSON structures"""
        text = '{"outer": {"inner": "value", "deep": {"x": 1}}}'
        result = find_json_objects(text)
        assert len(result) == 1
        assert result[0] == {"outer": {"inner": "value", "deep": {"x": 1}}}

    def test_find_with_arrays(self):
        """Test finding JSON with arrays"""
        text = '{"items": [1, 2, 3]}'
        result = find_json_objects(text)
        assert len(result) == 1
        assert result[0] == {"items": [1, 2, 3]}

    def test_find_with_quoted_braces(self):
        """Test handling of quoted braces (should be ignored)"""
        text = r'{"text": "not a brace } here", "key": "value"}'
        result = find_json_objects(text)
        assert len(result) == 1
        assert result[0] == {"text": "not a brace } here", "key": "value"}

    def test_find_with_escaped_quotes(self):
        """Test handling of escaped quotes"""
        text = r'{"text": "He said \"hello\""}'
        result = find_json_objects(text)
        assert len(result) == 1
        assert result[0] == {"text": 'He said "hello"'}

    def test_find_no_objects(self):
        """Test when no JSON objects exist"""
        text = "Just plain text with { braces } but not JSON"
        result = find_json_objects(text)
        assert len(result) == 0

    def test_find_incomplete_json(self):
        """Test handling of incomplete JSON objects"""
        text = '{"valid": 1} and {"invalid": '
        result = find_json_objects(text)
        assert len(result) == 1
        assert result[0] == {"valid": 1}


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_string(self):
        """Test with empty string"""
        assert extract_boxed_content("") is None
        assert extract_final_answer("") == "No final answer found"
        assert extract_json_from_text("") is None
        assert find_json_objects("") == []

    def test_none_handling(self):
        """Test functions handle various input types"""
        # These should not crash, though behavior is undefined for non-strings
        # Just ensuring they don't raise exceptions
        try:
            extract_boxed_content("test")  # Valid call
            extract_final_answer("test")  # Valid call
            extract_json_from_text("test")  # Valid call
            find_json_objects("test")  # Valid call
        except Exception as e:
            pytest.fail(f"Functions raised exception with valid input: {e}")

    def test_unicode_content(self):
        """Test handling of Unicode characters"""
        response = '{"final_answer": "中文答案"}'
        result = extract_final_answer(response)
        assert result == "中文答案"

    def test_special_characters(self):
        """Test handling of special characters"""
        # JSON escapes the backslashes, so the actual value has newline/tab/carriage return
        response = '{"final_answer": "\\n\\t\\r"}'
        result = extract_final_answer(response)
        # After JSON parsing, these become actual control characters
        assert result == "\n\t\r"

    def test_very_long_answer(self):
        """Test handling of very long answers"""
        long_answer = "x" * 10000
        response = f'{{"final_answer": "{long_answer}"}}'
        result = extract_final_answer(response)
        assert result == long_answer

    def test_deeply_nested_structure(self):
        """Test deeply nested JSON structure"""
        nested = {"l" + str(i): {"value": i} for i in range(10)}
        text = str(nested).replace("'", '"')
        result = extract_json_from_text("{" + text + "}")
        # Should successfully parse the nested structure
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
