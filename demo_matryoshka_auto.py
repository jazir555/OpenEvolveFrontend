#!/usr/bin/env python3
"""Auto-run version of demo_matryoshka_unified_memory.py"""

import sys
from io import StringIO

# Simulate user input at key points
inputs = [
    "",  # Start demo
    "",  # Continue to solution
    "",  # Continue to detailed indexing
    "",  # Continue to comparison
]

# Redirect stdin to provide automatic inputs
class MockInput:
    def __init__(self, inputs):
        self.inputs = inputs
        self.index = 0
    
    def __call__(self, prompt=None):
        if self.index < len(self.inputs):
            value = self.inputs[self.index]
            self.index += 1
            return value
        return ""

# Patch input
original_input = input
import demo_matryoshka_unified_memory

# Replace input function
demo_matryoshka_unified_memory.input = MockInput(inputs)

# Run demo
demo_matryoshka_unified_memory.run_demo()
