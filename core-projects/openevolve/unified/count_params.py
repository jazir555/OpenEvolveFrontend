#!/usr/bin/env python
"""Count parameters in unified configuration"""

import sys
sys.path.insert(0, '.')

from config import *

# Count parameters in each config class
classes = [
    ('UnifiedEvolutionConfig', UnifiedEvolutionConfig),
    ('LLMConfig', LLMConfig),
    ('DatabaseConfig', DatabaseConfig),
    ('EvaluatorConfig', EvaluatorConfig),
    ('PESConfig', PESConfig),
    ('QDConfig', QDConfig),
    ('MOConfig', MOConfig),
    ('AdversarialConfig', AdversarialConfig),
]

total = 0
print('PARAMETER COUNT BY CLASS:')
print('=' * 60)

for name, cls in classes:
    fields = len(cls.__fields__)
    total += fields
    print(f'{name:30s}: {fields:3d} parameters')

print('=' * 60)
print(f'{"TOTAL":30s}: {total:3d} parameters')
print()
print('SUB-CONFIGS IN UnifiedEvolutionConfig:')
print('  - llm: LLMConfig')
print('  - database: DatabaseConfig')
print('  - evaluator: EvaluatorConfig')
print('  - pes: PESConfig')
print('  - qd: QDConfig')
print('  - mo: MOConfig')
print('  - adversarial: AdversarialConfig')
print()
print('EXCLUDING SUB-CONFIGS FROM MAIN:')
main_only = len(UnifiedEvolutionConfig.__fields__) - 7  # Exclude 7 sub-configs
print(f'  UnifiedEvolutionConfig (direct): {main_only} parameters')
print(f'  Sub-configs total: {total - len(UnifiedEvolutionConfig.__fields__)} parameters')
print(f'  UNIQUE parameters: {main_only + (total - len(UnifiedEvolutionConfig.__fields__))}')
