#!/usr/bin/env python3
import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')

try:
    import openevolve_agnostic_pes as pes
    print('Import successful', file=sys.stderr)
    
    code = 'def test(): pass'
    analysis = pes.UniversalCodeAnalyzer.analyze(code)
    print('Analysis: ' + str(analysis), file=sys.stderr)
    
except Exception as e:
    print('Error: ' + str(e), file=sys.stderr)
    import traceback
    traceback.print_exc()
