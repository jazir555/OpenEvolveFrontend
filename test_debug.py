import sys
sys.stdout = open('test_output.txt', 'w', encoding='utf-8')

import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')
try:
    from leanaide_pes_handler import LeanPESHandler
    handler = LeanPESHandler()
    code = 'theorem add_comm (n m : Nat) : n + m = m + n := by sorry'
    print('Input:', code)
    print('Has by sorry:', 'by sorry' in code)
    
    # Direct test
    if 'by sorry' in code:
        new_code = code.replace('by sorry', 'by\n    test_tactic')
        print('After replace:', new_code)
    
    # Now use the handler
    result = handler.complete_single_proof(code)
    print('Handler result:', result)
    
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

print('\nDone', file=sys.__stdout__)
