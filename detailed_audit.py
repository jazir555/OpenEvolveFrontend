#!/usr/bin/env python3
"""Detailed Real vs Mocked Analysis with CAV-NLP Integration"""

import re
import os
import logging

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

logger = logging.getLogger(__name__)

class AuditConfig:
    """Configuration for detailed audit with CAV-NLP support."""
    
    def __init__(self, use_cav_nlp: bool = True):
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for detailed audit")


def formalize_audit_finding_with_cav_nlp(
    finding: Dict[str, Any],
    config: Optional[AuditConfig] = None
) -> Dict[str, Any]:
    """
    Formalize an audit finding using CAV-NLP.
    
    Args:
        finding: Audit finding to formalize
        config: Audit configuration with CAV-NLP settings
        
    Returns:
        Formalized audit result
    """
    if not config or not config.use_cav_nlp:
        return {
            'success': False,
            'error': 'CAV-NLP not available',
            'finding': finding
        }
    
    try:
        # Create description from finding
        description = finding.get('description', '')
        if not description and 'real_examples' in finding:
            description = str(finding['real_examples'])
        
        # Use enhanced solver to formalize
        formalization = config.enhanced_solver.formalize_natural_language(
            description,
            context={
                'file': finding.get('file', 'unknown'),
                'real_patterns': finding.get('real_patterns', 0),
                'mock_indicators': finding.get('mock_indicators', 0),
                'is_real': finding.get('is_mostly_real', False)
            }
        )
        
        result = {
            'success': formalization.get('success', False),
            'finding': finding,
            'constraints': formalization.get('constraints', []),
            'properties': formalization.get('properties', {}),
            'confidence': formalization.get('confidence', 0.0),
            'formalized': formalization.get('formalized_problem', '')
        }
        
        logger.debug(f"Formalized audit finding with confidence: {result['confidence']:.2f}")
        return result
        
    except Exception as e:
        logger.error(f"CAV-NLP formalization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'finding': finding
        }


def analyze_file_real_vs_mock(file_path):
    """Analyze if a file has real implementations or is mocked"""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return {'error': str(e)}
    
    lines = content.split('\n')
    total_lines = len(lines)
    
    # Count patterns
    real_patterns = [
        ('z3.Solver()', 'z3.Solver()'),
        ('z3.Optimize()', 'z3.Optimize()'),
        ('spla.solve', 'scipy.sparse.linalg'),
        ('np.linalg', 'numpy linear algebra'),
        ('sqlite3.connect', 'SQLite database'),
        ('requests.get', 'HTTP requests'),
        ('openai.', 'OpenAI API'),
        ('jwt.encode', 'JWT encoding'),
        ('hashlib.', 'Hash operations'),
        ('threading.', 'Threading'),
        ('asyncio.', 'AsyncIO'),
        ('subprocess.run', 'Subprocess'),
    ]
    
    mock_indicators = [
        ('pass', 'Empty function (pass)'),
        ('NotImplementedError', 'Not implemented'),
        ('TODO:', 'TODO marker'),
        ('FIXME:', 'FIXME marker'),
        ('mock', 'Mock reference'),
        ('simulated', 'Simulated data'),
        ('placeholder', 'Placeholder'),
    ]
    
    real_count = 0
    mock_count = 0
    real_found = []
    mock_found = []
    
    for line in lines:
        for pattern, desc in real_patterns:
            if pattern in line:
                real_count += 1
                if desc not in [r[0] for r in real_found]:
                    real_found.append((desc, line.strip()[:60]))
        
        for pattern, desc in mock_indicators:
            if pattern in line.lower():
                mock_count += 1
                if len(mock_found) < 10:
                    mock_found.append((desc, line.strip()[:60]))
    
    is_mostly_real = real_count > mock_count * 2 if mock_count > 0 else real_count > 10
    
    return {
        'total_lines': total_lines,
        'real_patterns': real_count,
        'mock_indicators': mock_count,
        'real_examples': real_found[:5],
        'mock_examples': mock_found[:5],
        'is_mostly_real': is_mostly_real
    }


def check_external_integrations():
    """Check if external integrations are real or mocked"""
    integrations = {
        'DeepKE': ['deepke', 'from deepke', 'import deepke'],
        'OneKE': ['oneke', 'from oneke', 'import oneke'],
        'OpenAI': ['openai.ChatCompletion', 'openai.chat.completions', 'client.chat.completions'],
        'Z3': ['z3.Solver()', 'z3.Optimize()', 'from z3 import'],
        'PhysicsNeMo': ['physicsnemo', 'import physicsnemo'],
        'Neo4j': ['neo4j', 'GraphDatabase'],
    }
    
    results = {}
    
    # Walk through Python files
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except:
                    continue
                
                for integration, patterns in integrations.items():
                    for pattern in patterns:
                        if pattern in content:
                            if integration not in results:
                                results[integration] = {'files': [], 'pattern': pattern}
                            if filepath not in results[integration]['files']:
                                results[integration]['files'].append(filepath)
    
    return results


def main():
    print('='*70)
    print('DETAILED REAL VS MOCKED ANALYSIS')
    print('='*70)
    
    files = [
        'security_framework.py',
        'physics_validator_real.py', 
        'z3prover_advanced.py',
        'ml_pattern_clustering.py',
        'gauntlet_types.py',
    ]
    
    for file in files:
        print('\n[' + file + ']')
        result = analyze_file_real_vs_mock(file)
        if result is None:
            print('  File not found')
            continue
        if 'error' in result:
            print('  Error: ' + str(result['error']))
            continue
        
        status = 'REAL' if result['is_mostly_real'] else 'LIKELY_MOCKED'
        print('  Status: ' + status)
        print('  Lines: ' + str(result['total_lines']))
        print('  Real patterns: ' + str(result['real_patterns']))
        print('  Mock indicators: ' + str(result['mock_indicators']))
        
        if result['real_examples']:
            print('  Real implementation examples:')
            for desc, line in result['real_examples']:
                print('    - ' + desc + ': ' + line[:50] + '...')
        
        if result['mock_examples'] and not result['is_mostly_real']:
            print('  Mock indicators found:')
            for desc, line in result['mock_examples'][:3]:
                print('    - ' + desc + ': ' + line[:50] + '...')
    
    # External integrations
    print('\n' + '='*70)
    print('EXTERNAL INTEGRATION CHECK')
    print('='*70)
    
    integrations = check_external_integrations()
    for name, data in integrations.items():
        print('\n[' + name + ']')
        print('  Pattern found: ' + data['pattern'])
        print('  Files: ' + str(len(data['files'])))
        for f in data['files'][:5]:
            print('    - ' + f)


if __name__ == '__main__':
    main()
