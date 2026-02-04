"""
Simple test for RESE Phase IV implementation
"""
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "glue" / "schemas"))
sys.path.insert(0, str(Path(__file__).parent / "glue" / "adapters" / "rese-phase4" / "src"))

import os
os.environ['PHASE4_ASSEMBLY_TIMEOUT_MS'] = '25000'
os.environ['PHASE4_VALIDATION_LEVEL'] = 'basic'

def test_schemas():
    print("Testing schemas...")
    from rese_phase4_schemas import (
        ArchitectureAssembly,
        ParadigmShift,
        SynthesizedKnowledge,
        Phase4Config,
        AssemblyStatus,
        ParadigmShiftType,
        ValidationLevel,
        IntegrationStrategy,
    )
    print("[OK] All schemas imported")

    # Test ParadigmShift
    shift = ParadigmShift(
        description='Test paradigm shift',
        shift_type=ParadigmShiftType.STRUCTURAL,
        confidence=0.85,
    )
    print(f"[OK] ParadigmShift created: {shift.shift_id}")

    # Test SynthesizedKnowledge
    knowledge = SynthesizedKnowledge(
        description='Test knowledge',
        paradigm_shifts=[shift],
    )
    print(f"[OK] SynthesizedKnowledge created: {knowledge.knowledge_id}")

    # Test ArchitectureAssembly
    assembly = ArchitectureAssembly(
        synthesized_knowledge=knowledge,
        paradigm_shifts=[shift],
        status=AssemblyStatus.VALIDATED,
    )
    print(f"[OK] ArchitectureAssembly created: {assembly.assembly_id}")

    # Test serialization
    assembly_dict = assembly.to_dict()
    print(f"[OK] Serialization works")

    # Test deserialization
    assembly2 = ArchitectureAssembly.from_dict(assembly_dict)
    assert assembly2.assembly_id == assembly.assembly_id
    print(f"[OK] Deserialization works")

    return True

def test_executor():
    print("\nTesting executor...")
    from phase4_executor import ArchitectureAssemblyExecutor, StructuredLogger
    print("[OK] Executor imported")

    # Create executor
    executor = ArchitectureAssemblyExecutor()
    print(f"[OK] Executor created")

    # Execute simple assembly
    assembly = executor.execute(
        phase1_patterns=[
            {
                'pattern_id': 'p1-001',
                'type': 'structural',
                'description': 'Pattern from Phase I',
                'confidence': 0.8,
            }
        ],
        phase2_patterns=[
            {
                'pattern_id': 'p2-001',
                'type': 'structural',
                'description': 'Pattern from Phase II',
                'confidence': 0.75,
            }
        ],
    )
    print(f"[OK] Assembly executed: {assembly.assembly_id}")
    print(f"  - Status: {assembly.status.value}")
    print(f"  - Confidence: {assembly.confidence:.2f}")
    print(f"  - Paradigm shifts: {len(assembly.paradigm_shifts)}")

    return True

def test_adapter():
    print("\nTesting adapter...")
    from adapter import Phase4Adapter
    print("[OK] Adapter imported")

    # Create adapter
    adapter = Phase4Adapter()
    print(f"[OK] Adapter created")

    # Test health check
    health = adapter.health_check()
    print(f"[OK] Health check: {health['status']}")

    # Test assembly
    request = {
        'request_id': 'test-req-001',
        'phase1_patterns': [
            {
                'pattern_id': 'p1-001',
                'type': 'structural',
                'description': 'Pattern from Phase I',
                'confidence': 0.8,
            }
        ],
        'phase2_patterns': [
            {
                'pattern_id': 'p2-001',
                'type': 'structural',
                'description': 'Pattern from Phase II',
                'confidence': 0.75,
            }
        ],
        'phase3_patterns': [
            {
                'pattern_id': 'p3-001',
                'type': 'functional',
                'description': 'Pattern from Phase III',
                'confidence': 0.85,
            }
        ],
    }

    response = adapter.assemble_architecture(request)
    print(f"[OK] Assembly request processed")
    print(f"  - Response ID: {response['response_id']}")
    print(f"  - Status: {response['status']}")

    assembly_data = response['assembly']
    print(f"  - Assembly ID: {assembly_data['assembly_id']}")
    print(f"  - Validation passed: {response['metadata']['validation_passed']}")

    return True

if __name__ == '__main__':
    print("=" * 60)
    print("RESE Phase IV: Simple Test")
    print("=" * 60)

    try:
        test_schemas()
        test_executor()
        test_adapter()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[FAILED] TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
