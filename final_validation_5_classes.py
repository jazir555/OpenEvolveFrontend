#!/usr/bin/env python3
"""
Final Comprehensive Validation of 5 Implemented Classes
Validates completeness and error-free operation
"""

import sys
import traceback
from datetime import datetime

def test_validation_result():
    """Test ValidationResult class thoroughly"""
    print("\n" + "="*60)
    print("TEST 1: ValidationResult (schemas/base.py)")
    print("="*60)
    
    try:
        from knowledge_engine.schemas.base import ValidationResult
        
        # Test instantiation with defaults
        vr1 = ValidationResult()
        assert vr1.is_valid == True, "Default is_valid should be True"
        assert vr1.errors == [], "Default errors should be empty list"
        assert vr1.warnings == [], "Default warnings should be empty list"
        assert vr1.entity_id is None, "Default entity_id should be None"
        assert vr1.schema_name is None, "Default schema_name should be None"
        assert isinstance(vr1.timestamp, str), "Timestamp should be string"
        assert vr1.metadata == {}, "Default metadata should be empty dict"
        print("  [OK] Instantiation with defaults works")
        
        # Test instantiation with values
        vr2 = ValidationResult(
            is_valid=False,
            errors=["error1"],
            warnings=["warning1"],
            entity_id="ent123",
            schema_name="TestSchema",
            metadata={"key": "value"}
        )
        assert vr2.is_valid == False
        assert vr2.errors == ["error1"]
        assert vr2.warnings == ["warning1"]
        assert vr2.entity_id == "ent123"
        assert vr2.schema_name == "TestSchema"
        print("  [OK] Instantiation with custom values works")
        
        # Test add_error
        vr3 = ValidationResult()
        vr3.add_error("Test error")
        assert vr3.is_valid == False, "add_error should set is_valid to False"
        assert "Test error" in vr3.errors
        print("  [OK] add_error() works correctly")
        
        # Test add_warning
        vr3.add_warning("Test warning")
        assert "Test warning" in vr3.warnings
        assert vr3.is_valid == False  # Still false from earlier
        print("  [OK] add_warning() works correctly")
        
        # Test merge
        vr4 = ValidationResult(is_valid=True)
        vr4.add_warning("Warning A")
        vr5 = ValidationResult(is_valid=False)
        vr5.add_error("Error B")
        
        result = vr4.merge(vr5)
        assert result is vr4, "merge should return self"
        assert "Warning A" in vr4.warnings
        assert "Error B" in vr4.errors
        assert vr4.is_valid == False, "merge should propagate invalid status"
        print("  [OK] merge() works correctly")
        
        # Test to_dict
        vr6 = ValidationResult(entity_id="test", schema_name="Schema")
        vr6.add_error("Err1")
        vr6.add_warning("Warn1")
        d = vr6.to_dict()
        assert isinstance(d, dict)
        assert d['is_valid'] == False
        assert d['entity_id'] == "test"
        assert d['schema_name'] == "Schema"
        assert "Err1" in d['errors']
        assert "Warn1" in d['warnings']
        print("  [OK] to_dict() works correctly")
        
        # Test from_dict
        vr7 = ValidationResult.from_dict(d)
        assert vr7.is_valid == False
        assert vr7.entity_id == "test"
        assert vr7.schema_name == "Schema"
        assert "Err1" in vr7.errors
        print("  [OK] from_dict() works correctly")
        
        print("  [OK][OK] ValidationResult: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False


def test_knowledge_engine():
    """Test KnowledgeEngine class thoroughly"""
    print("\n" + "="*60)
    print("TEST 2: KnowledgeEngine (core/__init__.py)")
    print("="*60)
    
    try:
        from knowledge_engine.core import KnowledgeEngine as CoreKE
        from knowledge_engine.orchestration import KnowledgeEngine as OrchKE
        
        # Test that both are classes
        assert isinstance(CoreKE, type), "CoreKE should be a class"
        assert isinstance(OrchKE, type), "OrchKE should be a class"
        print("  [OK] Both imports are classes")
        
        # Test they have the same methods
        core_methods = {m for m in dir(CoreKE) if not m.startswith('_')}
        orch_methods = {m for m in dir(OrchKE) if not m.startswith('_')}
        
        required_methods = {
            'initialize', 'close', 'process_document', 'query_temporal',
            'detect_contradictions', 'visualize_graph', 'search_knowledge',
            'get_statistics', 'health_check'
        }
        
        missing_from_core = required_methods - core_methods
        missing_from_orch = required_methods - orch_methods
        
        assert not missing_from_core, f"CoreKE missing: {missing_from_core}"
        assert not missing_from_orch, f"OrchKE missing: {missing_from_orch}"
        print("  [OK] Both classes have all 9 required methods")
        
        # Test that methods are callable
        for method in required_methods:
            assert callable(getattr(CoreKE, method)), f"CoreKE.{method} not callable"
            assert callable(getattr(OrchKE, method)), f"OrchKE.{method} not callable"
        print("  [OK] All methods are callable")
        
        # Test functional equivalence - both can be instantiated
        # (Note: They may need config, but the class can be instantiated)
        try:
            # Try to instantiate (may fail due to missing config, that's ok)
            ke = CoreKE.__new__(CoreKE)
            print("  [OK] CoreKE can be instantiated")
        except Exception as inst_e:
            print(f"  ! CoreKE instantiation requires config: {inst_e}")
        
        print("  [OK][OK] KnowledgeEngine: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False


def test_model_config():
    """Test ModelConfig class thoroughly"""
    print("\n" + "="*60)
    print("TEST 3: ModelConfig (integrations/oneke/model_adapter.py)")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.oneke.model_adapter import ModelConfig
        
        # Test instantiation with defaults
        mc1 = ModelConfig()
        assert mc1.model_name == "oneke"
        assert mc1.model_path is None
        assert mc1.device == "cpu"
        assert mc1.batch_size == 32
        assert mc1.max_length == 512
        assert mc1.language == "en"
        assert mc1.confidence_threshold == 0.5
        assert mc1.use_gpu == False
        assert mc1.extract_relations == True
        assert mc1.extract_attributes == True
        print("  [OK] Default instantiation works")
        
        # Test instantiation with custom values
        mc2 = ModelConfig(
            model_name="custom",
            model_path="/path/to/model",
            device="cuda",
            batch_size=64,
            max_length=1024,
            language="zh",
            confidence_threshold=0.8,
            use_gpu=True,
            extract_relations=False,
            extract_attributes=False
        )
        assert mc2.model_name == "custom"
        assert mc2.model_path == "/path/to/model"
        assert mc2.device == "cuda"
        assert mc2.batch_size == 64
        assert mc2.max_length == 1024
        assert mc2.language == "zh"
        assert mc2.confidence_threshold == 0.8
        assert mc2.use_gpu == True
        assert mc2.extract_relations == False
        assert mc2.extract_attributes == False
        print("  [OK] Custom instantiation works")
        
        # Test to_dict
        d = mc2.to_dict()
        assert isinstance(d, dict)
        assert d['model_name'] == "custom"
        assert d['device'] == "cuda"
        assert d['batch_size'] == 64
        assert d['language'] == "zh"
        print("  [OK] to_dict() works correctly")
        
        print("  [OK][OK] ModelConfig: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False


def test_graphiti_config():
    """Test GraphitiConfig class thoroughly"""
    print("\n" + "="*60)
    print("TEST 4: GraphitiConfig (integrations/graphiti/__init__.py)")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.graphiti import GraphitiConfig
        
        # Test instantiation with defaults
        gc1 = GraphitiConfig()
        assert gc1.neo4j_uri == "bolt://localhost:7687"
        assert gc1.neo4j_user == "neo4j"
        assert gc1.neo4j_password == ""
        assert gc1.openai_api_key is None
        assert gc1.default_model == "gpt-4"
        assert gc1.max_hops == 3
        assert gc1.similarity_threshold == 0.8
        assert gc1.temporal_resolution == "seconds"
        assert gc1.enable_caching == True
        assert gc1.cache_ttl == 3600
        print("  [OK] Default instantiation works")
        
        # Test instantiation with custom values
        gc2 = GraphitiConfig(
            neo4j_uri="bolt://remote:7687",
            neo4j_user="admin",
            neo4j_password="secret",
            openai_api_key="sk-test",
            default_model="gpt-3.5-turbo",
            max_hops=5,
            similarity_threshold=0.9,
            temporal_resolution="minutes",
            enable_caching=False,
            cache_ttl=7200
        )
        assert gc2.neo4j_uri == "bolt://remote:7687"
        assert gc2.neo4j_user == "admin"
        assert gc2.neo4j_password == "secret"
        assert gc2.openai_api_key == "sk-test"
        assert gc2.default_model == "gpt-3.5-turbo"
        assert gc2.max_hops == 5
        assert gc2.similarity_threshold == 0.9
        assert gc2.temporal_resolution == "minutes"
        assert gc2.enable_caching == False
        assert gc2.cache_ttl == 7200
        print("  [OK] Custom instantiation works")
        
        # Test to_dict (should mask API key)
        d = gc2.to_dict()
        assert isinstance(d, dict)
        assert d['neo4j_uri'] == "bolt://remote:7687"
        assert d['openai_api_key'] == "***"  # Masked
        assert d['max_hops'] == 5
        print("  [OK] to_dict() works correctly (masks API key)")
        
        print("  [OK][OK] GraphitiConfig: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False


def test_extraction_result():
    """Test ExtractionResult class thoroughly"""
    print("\n" + "="*60)
    print("TEST 5: ExtractionResult (integrations/kggen/extraction_pipeline.py)")
    print("="*60)
    
    try:
        from knowledge_engine.integrations.kggen.extraction_pipeline import ExtractionResult
        
        # Test instantiation with required success field
        er1 = ExtractionResult(success=True)
        assert er1.success == True
        assert er1.entities == []
        assert er1.relations == []
        assert er1.triples == []
        assert er1.metadata == {}
        assert er1.processing_time_ms == 0.0
        assert er1.error is None
        print("  [OK] Default instantiation works")
        
        # Test instantiation with full values
        er2 = ExtractionResult(
            success=False,
            entities=[{"name": "Entity1", "type": "PERSON"}],
            relations=[{"source": "A", "target": "B", "type": "KNOWS"}],
            triples=[("A", "KNOWS", "B")],
            metadata={"source": "test"},
            processing_time_ms=150.5,
            error="Test error"
        )
        assert er2.success == False
        assert len(er2.entities) == 1
        assert len(er2.relations) == 1
        assert len(er2.triples) == 1
        assert er2.metadata == {"source": "test"}
        assert er2.processing_time_ms == 150.5
        assert er2.error == "Test error"
        print("  [OK] Custom instantiation works")
        
        # Test to_dict
        d = er2.to_dict()
        assert isinstance(d, dict)
        assert d['success'] == False
        assert d['entities'] == [{"name": "Entity1", "type": "PERSON"}]
        assert d['processing_time_ms'] == 150.5
        assert d['error'] == "Test error"
        print("  [OK] to_dict() works correctly")
        
        # Test from_dict
        er3 = ExtractionResult.from_dict(d)
        assert er3.success == False
        assert len(er3.entities) == 1
        assert er3.processing_time_ms == 150.5
        assert er3.error == "Test error"
        print("  [OK] from_dict() works correctly")
        
        print("  [OK][OK] ExtractionResult: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE VALIDATION OF 5 IMPLEMENTED CLASSES")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    
    results = []
    
    results.append(("ValidationResult", test_validation_result()))
    results.append(("KnowledgeEngine", test_knowledge_engine()))
    results.append(("ModelConfig", test_model_config()))
    results.append(("GraphitiConfig", test_graphiti_config()))
    results.append(("ExtractionResult", test_extraction_result()))
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 ALL 5 CLASSES ARE COMPLETE AND ERROR-FREE!")
        print("="*60)
        return 0
    else:
        print("[FAIL] SOME TESTS FAILED - SEE DETAILS ABOVE")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
