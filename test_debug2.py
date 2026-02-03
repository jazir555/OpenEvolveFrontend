import sys
sys.stdout = open('test_output.txt', 'w', encoding='utf-8')

import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')
try:
    from leanaide_pes_handler import LeanPESHandler, LeanCodeAnalyzer
    handler = LeanPESHandler()
    code = 'theorem add_comm (n m : Nat) : n + m = m + n := by sorry'
    print('Input:', code)
    
    # Test analyzer
    analysis = LeanCodeAnalyzer.analyze_structure(code)
    print('Analysis:', analysis)
    print('Has sorry:', analysis['has_sorry'])
    print('Theorems:', analysis['theorems'])
    print('Theorem analysis:', analysis.get('theorem_analysis', 'NOT FOUND'))
    
    # Test selector
    if analysis['theorem_analysis']:
        thm = analysis['theorem_analysis'][0]
        proof = handler.selector.generate_proof(thm)
        print('Generated proof:', proof)
    
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

print('\nDone', file=sys.__stdout__)
