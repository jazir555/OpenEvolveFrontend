import sys
sys.stdout = open('test_output.txt', 'w', encoding='utf-8')

import sys
sys.path.insert(0, 'c:/Users/mmeadow/Documents/OpenEvolve/Frontend')
try:
    from leanaide_pes_handler import LeanPESHandler
    handler = LeanPESHandler()
    code = 'theorem add_comm (n m : Nat) : n + m = m + n := by sorry'
    print('Input:', code)
    result = handler.complete_single_proof(code)
    print('Output:', result)
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

print('\nDone', file=sys.__stdout__)
