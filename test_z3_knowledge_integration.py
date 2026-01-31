"""
Test script for Z3 Knowledge Integration
"""

import asyncio
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_z3_knowledge_extraction():
    """Test Z3 knowledge extraction."""
    print("=" * 60)
    print("Z3 Knowledge Integration Test")
    print("=" * 60)
    
    # Test 1: Z3 Knowledge Extraction Module
    print("\n[Test 1] Z3 Knowledge Extraction Module")
    try:
        from z3_knowledge_extraction import (
            Z3KnowledgeExtractor,
            ProofPattern,
            ConstraintPattern,
            SolutionStrategy,
            get_z3_knowledge_extractor
        )
        
        extractor = get_z3_knowledge_extractor()
        
        # Learn a strategy
        strategy = extractor.learn_strategy(
            problem_features={
                "type": "linear",
                "var_count": 5,
                "constraint_count": 10
            },
            tactics_used=["simplify", "solve-eqs", "smt"],
            config_used={"timeout": 30, "threads": 4},
            success=True,
            solving_time=2.5
        )
        
        print(f"  [OK] Learned strategy: {strategy.name}")
        print(f"  [OK] Success rate: {strategy.success_rate():.1%}")
        
        # Analyze constraints
        constraints = [
            "(> x 0)",
            "(< x 10)",
            "(= y (+ x 5))",
            "(> (* x y) 0)"
        ]
        
        patterns = extractor.analyze_constraints(constraints, 1.5, True)
        print(f"  [OK] Found {len(patterns)} constraint patterns)")
        
        # Get summary
        summary = extractor.get_knowledge_summary()
        print(f"  [OK] Strategies: {summary['strategies']['count']}")
        print(f"  [OK] Constraint patterns: {summary['constraint_patterns']['count']}")
        
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False
    
    # Test 2: Database Models
    print("\n[Test 2] Database Models")
    try:
        from knowledge_engine.integrations.z3_database_models import (
            create_z3_tables,
            Z3KnowledgeEntry,
            Z3ProofPattern,
            Z3Strategy,
            Z3ConstraintPattern
        )
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create in-memory database
        engine = create_z3_tables("sqlite:///:memory:")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create knowledge entry
        entry = Z3KnowledgeEntry(
            entry_type="strategy",
            content_hash="test_hash_123",
            content="Test strategy content",
            metadata_json={"test": True},
            problem_domain="linear",
            confidence=0.85
        )
        session.add(entry)
        session.commit()
        
        # Create strategy
        strategy = Z3Strategy(
            knowledge_entry_id=entry.id,
            strategy_name="Test Strategy",
            problem_pattern="linear_vars_5",
            recommended_tactics=["simplify", "smt"],
            solver_configuration={"timeout": 30},
            expected_avg_time=1.5,
            success_count=10,
            failure_count=2
        )
        session.add(strategy)
        session.commit()
        
        # Verify
        result = session.query(Z3Strategy).first()
        assert result is not None
        assert result.success_rate == 10/12
        
        print(f"  [OK] Created {session.query(Z3KnowledgeEntry).count()} knowledge entries")
        print(f"  [OK] Created {session.query(Z3Strategy).count()} strategies")
        print(f"  [OK] Success rate calculation: {result.success_rate:.1%}")
        
        session.close()
        
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Integration Layer (without storage)
    print("\n[Test 3] Integration Layer")
    try:
        from knowledge_engine.integrations.z3_knowledge_integration import (
            Z3KnowledgeIntegration,
            Z3KnowledgeEntry
        )
        
        # Create integration without storage
        integration = Z3KnowledgeIntegration(storage_engine=None)
        
        # Create mock result
        class MockResult:
            success = True
            model = type('Model', (), {'assignments': {'x': 5, 'y': 10}})()
            constraints = ["(> x 0)", "(< x 10)", "(= y (+ x 5))"]
            solving_time = 1.5
            tactics_used = ["simplify", "solve-eqs", "smt"]
            config = {"timeout": 30}
        
        # Extract knowledge
        extracted = await integration.extract_from_solver_result(
            result=MockResult(),
            problem_statement="Find x and y satisfying constraints",
            problem_type="linear"
        )
        
        print(f"  [OK] Extracted {len(extracted['insights'])} insights")
        print(f"  [OK] Extracted {len(extracted['patterns'])} patterns")
        print(f"  [OK] Extracted {len(extracted['strategies'])} strategies")
        
        # Get summary
        summary = await integration.get_knowledge_summary()
        print(f"  [OK] Storage available: {summary['storage_available']}")
        print(f"  [OK] Total extractions: {summary['extraction_stats']['total_extractions']}")
        
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Auto-Extraction
    print("\n[Test 4] Auto-Extraction")
    try:
        from knowledge_engine.integrations.z3_auto_extraction import (
            Z3AutoExtractionManager,
            auto_extract_knowledge
        )
        
        manager = Z3AutoExtractionManager()
        # Don't initialize - test without storage
        
        # Test decorator
        @auto_extract_knowledge(problem_type="test")
        async def test_solver():
            class Result:
                success = True
                model = type('Model', (), {'assignments': {'x': 1}})()
            return Result()
        
        result = await test_solver()
        print(f"  [OK] Decorator works (extraction skipped - no storage)")
        
        # Test stats
        stats = manager.get_stats()
        print(f"  [OK] Stats: {stats}")
        
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_z3_knowledge_extraction())
    exit(0 if success else 1)
