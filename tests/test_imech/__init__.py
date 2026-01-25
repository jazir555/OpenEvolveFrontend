"""
Test suite for I_mech (Mechanistic Isomorphism Validator)

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from phase2.imech import *
