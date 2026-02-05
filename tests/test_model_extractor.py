import unittest
import sys
import os

# Add the parent directory to the sys.path to allow imports from lean4_system
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lean4_system.model_extractor import MathematicalModelExtractor
from lean4_system.lean4_data_models import ProofObligation

class TestMathematicalModelExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = MathematicalModelExtractor()

    def test_extract_single_function(self):
        """
        Tests that the extractor can identify a single function in a Python script.
        """
        solution_content = """
def my_sort(arr):
    return sorted(arr)
"""
        properties = ["correctness"]
        obligations = self.extractor.extract(solution_content, properties)
        
        self.assertEqual(len(obligations), 1)
        self.assertIsInstance(obligations[0], ProofObligation)
        self.assertEqual(obligations[0].name, "correctness_of_my_sort_0")
        self.assertIn("my_sort", obligations[0].statement)

    def test_extract_multiple_functions(self):
        """
        Tests that the extractor can identify multiple functions in a Python script.
        """
        solution_content = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
        properties = ["correctness"]
        obligations = self.extractor.extract(solution_content, properties)
        
        self.assertEqual(len(obligations), 2)
        func_names = {ob.name for ob in obligations}
        self.assertIn("correctness_of_add_0", func_names)
        self.assertIn("correctness_of_subtract_1", func_names)

    def test_no_functions_found(self):
        """
        Tests that a single obligation is created for content with no functions.
        """
        solution_content = "x = 1\ny = 2\nz = x + y"
        properties = ["correctness"]
        obligations = self.extractor.extract(solution_content, properties)
        
        self.assertEqual(len(obligations), 1)
        self.assertEqual(obligations[0].name, "overall_content_correctness")
        self.assertIn("content_is_correct", obligations[0].statement)

    def test_multiple_properties(self):
        """
        Tests that obligations are created for each specified property.
        """
        solution_content = "def calculate(x):\n    return x * 2"
        properties = ["correctness", "termination"]
        obligations = self.extractor.extract(solution_content, properties)

        self.assertEqual(len(obligations), 2)
        prop_names = {ob.name for ob in obligations}
        self.assertIn("correctness_of_calculate_0", prop_names)
        self.assertIn("termination_of_calculate_1", prop_names)
        
    def test_empty_content(self):
        """
        Tests that no obligations are created for empty content, except the default one.
        """
        solution_content = ""
        properties = ["correctness"]
        obligations = self.extractor.extract(solution_content, properties)
        self.assertEqual(len(obligations), 1)
        self.assertEqual(obligations[0].name, "overall_content_correctness")

if __name__ == '__main__':
    unittest.main()
