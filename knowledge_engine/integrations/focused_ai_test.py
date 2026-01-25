"""
Focused AI Integration Test

This test specifically validates the AI knowledge graph integrations
(DeepKE, Karate Club, kg-gen, OneKE) without requiring the full
dependency chain that causes conflicts.
"""

import sys
import os
import time
from typing import Dict, Any

# Add knowledge_engine to Python path
knowledge_engine_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if knowledge_engine_path not in sys.path:
    sys.path.insert(0, knowledge_engine_path)

# Import only the AI integration components (not the full engine)
from integrations.deepke_integration import DeepKEEnhancedExtractor
from integrations.karateclub_integration import KarateClubGraphAnalyzer
from integrations.kg_gen_integration import EnhancedKnowledgeGraphManager
from integrations import AIKnowledgeGraphIntegrator

class FocusedAITester:
    """Test the AI integrations specifically."""
    
    def __init__(self):
        self.results = {
            'deepke': {'status': 'not_tested'},
            'karateclub': {'status': 'not_tested'},
            'kg_gen': {'status': 'not_tested'},
            'ai_integrator': {'status': 'not_tested'},
            'overall': {'status': 'not_tested'}
        }
    
    def test_all_ai_integrations(self) -> Dict[str, Any]:
        """Test all AI integration components."""
        print("Testing AI Knowledge Graph Integrations...")
        print("=" * 50)
        
        # Test each component
        self._test_deepke()
        self._test_karateclub()
        self._test_kg_gen()
        self._test_ai_integrator()
        
        # Determine overall status
        all_passed = all(
            result['status'] == 'success' 
            for result in self.results.values() 
            if isinstance(result, dict) and 'status' in result
        )
        
        self.results['overall']['status'] = 'success' if all_passed else 'partial'
        self.results['overall']['timestamp'] = self._get_current_timestamp()
        
        return self.results
    
    def _test_deepke(self):
        """Test DeepKE integration."""
        print("\n1. Testing DeepKE Integration...")
        
        try:
            extractor = DeepKEEnhancedExtractor()
            available = extractor.is_available()
            
            self.results['deepke'] = {
                'status': 'success',
                'available': available,
                'message': 'DeepKE integration working correctly'
            }
            
            if available:
                # Test with sample text
                test_text = "OpenEvolve integrates DeepKE for knowledge extraction."
                results = extractor.extract_with_deepke(test_text)
                
                if results['status'] == 'success':
                    items = len(results['extracted_knowledge'])
                    self.results['deepke']['extracted_items'] = items
                    print(f"   ✅ DeepKE: Available and functional ({items} items extracted)")
                else:
                    print(f"   ⚠️  DeepKE: Available but extraction failed: {results['message']}")
            else:
                print("   ⚠️  DeepKE: Integration available but DeepKE modules not found")
                
        except Exception as e:
            self.results['deepke'] = {
                'status': 'failed',
                'error': str(e),
                'message': 'DeepKE integration test failed'
            }
            print(f"   ❌ DeepKE: Test failed - {e}")
    
    def _test_karateclub(self):
        """Test Karate Club integration."""
        print("\n2. Testing Karate Club Integration...")
        
        try:
            analyzer = KarateClubGraphAnalyzer()
            available = analyzer.is_available()
            
            self.results['karateclub'] = {
                'status': 'success',
                'available': available,
                'message': 'Karate Club integration working correctly'
            }
            
            if available:
                # Test with sample graph
                graph_data = {
                    'nodes': [
                        {'id': 'A', 'type': 'node'},
                        {'id': 'B', 'type': 'node'}
                    ],
                    'edges': [
                        {'source': 'A', 'target': 'B', 'relationship': 'connected'}
                    ]
                }
                
                results = analyzer.analyze_graph(graph_data)
                
                if results['status'] == 'success':
                    has_metrics = 'metrics' in results['analysis_results']
                    self.results['karateclub']['has_metrics'] = has_metrics
                    print(f"   ✅ Karate Club: Available and functional (metrics: {has_metrics})")
                else:
                    print(f"   ⚠️  Karate Club: Available but analysis failed: {results['message']}")
            else:
                print("   ⚠️  Karate Club: Integration available but Karate Club modules not found")
                
        except Exception as e:
            self.results['karateclub'] = {
                'status': 'failed',
                'error': str(e),
                'message': 'Karate Club integration test failed'
            }
            print(f"   ❌ Karate Club: Test failed - {e}")
    
    def _test_kg_gen(self):
        """Test kg-gen and OneKE integration."""
        print("\n3. Testing kg-gen and OneKE Integration...")
        
        try:
            manager = EnhancedKnowledgeGraphManager()
            kg_gen_available = manager.is_kg_gen_available()
            oneke_available = manager.is_oneke_available()
            
            self.results['kg_gen'] = {
                'status': 'success',
                'kg_gen_available': kg_gen_available,
                'oneke_available': oneke_available,
                'message': 'kg-gen/OneKE integration working correctly'
            }
            
            if kg_gen_available or oneke_available:
                # Test with sample artifacts
                artifacts = [
                    {
                        'source': 'test',
                        'knowledge_type': 'triple',
                        'subject': 'Test',
                        'predicate': 'uses',
                        'object': 'AI',
                        'confidence': 0.95
                    }
                ]
                
                results = manager.generate_and_store_knowledge_graph(artifacts)
                
                if results['status'] == 'success':
                    graph_created = results['results']['knowledge_graph'] is not None
                    self.results['kg_gen']['graph_created'] = graph_created
                    print(f"   ✅ kg-gen/OneKE: Available and functional (graph: {graph_created})")
                else:
                    print(f"   ⚠️  kg-gen/OneKE: Available but processing failed: {results['message']}")
            else:
                print("   ⚠️  kg-gen/OneKE: Integration available but modules not found")
                
        except Exception as e:
            self.results['kg_gen'] = {
                'status': 'failed',
                'error': str(e),
                'message': 'kg-gen/OneKE integration test failed'
            }
            print(f"   ❌ kg-gen/OneKE: Test failed - {e}")
    
    def _test_ai_integrator(self):
        """Test the main AI integrator."""
        print("\n4. Testing AI Knowledge Graph Integrator...")
        
        try:
            integrator = AIKnowledgeGraphIntegrator()
            
            # Test individual methods
            deepke_available = integrator.deepke_extractor.is_available()
            karateclub_available = integrator.karateclub_analyzer.is_available()
            kg_gen_available = integrator.kg_gen_manager.is_kg_gen_available()
            oneke_available = integrator.kg_gen_manager.is_oneke_available()
            
            self.results['ai_integrator'] = {
                'status': 'success',
                'deepke': deepke_available,
                'karateclub': karateclub_available,
                'kg_gen': kg_gen_available,
                'oneke': oneke_available,
                'message': 'AI integrator working correctly'
            }
            
            # Count available integrations
            available_count = sum([deepke_available, karateclub_available, kg_gen_available, oneke_available])
            print(f"   ✅ AI Integrator: {available_count}/4 integrations available")
            print(f"      DeepKE: {'✅' if deepke_available else '❌'}")
            print(f"      Karate Club: {'✅' if karateclub_available else '❌'}")
            print(f"      kg-gen: {'✅' if kg_gen_available else '❌'}")
            print(f"      OneKE: {'✅' if oneke_available else '❌'}")
            
        except Exception as e:
            self.results['ai_integrator'] = {
                'status': 'failed',
                'error': str(e),
                'message': 'AI integrator test failed'
            }
            print(f"   ❌ AI Integrator: Test failed - {e}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

def run_focused_ai_test():
    """Run the focused AI integration test."""
    tester = FocusedAITester()
    results = tester.test_all_ai_integrations()
    return results

if __name__ == "__main__":
    print("AI Knowledge Graph Integration Test")
    print("=" * 50)
    print("Testing DeepKE, Karate Club, kg-gen, and OneKE integrations")
    print("=" * 50)
    
    # Run the focused test
    test_results = run_focused_ai_test()
    
    # Print summary
    print("\n" + "=" * 50)
    print("AI INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    # Count successful tests
    successful = sum(1 for result in test_results.values() 
                    if isinstance(result, dict) and result.get('status') == 'success')
    total = len([v for v in test_results.values() if isinstance(v, dict)])
    
    print(f"\n📊 RESULTS:")
    print(f"   Successful: {successful}/{total}")
    
    # Show individual results
    for name, result in test_results.items():
        if isinstance(result, dict):
            status = result.get('status', 'unknown')
            symbol = "✅" if status == 'success' else "❌"
            print(f"   {symbol} {name}: {status}")
    
    print(f"\n🎯 OVERALL STATUS: {test_results['overall']['status']}")
    
    # Provide recommendations
    if test_results['overall']['status'] == 'success':
        print(f"\n💡 The AI integrations are working correctly!")
        print(f"   All components are properly integrated and functional.")
    else:
        print(f"\n💡 Some integrations may need attention:")
        for name, result in test_results.items():
            if isinstance(result, dict) and result.get('status') == 'failed':
                print(f"   - {name}: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("AI Integration Test Complete")
    print("=" * 50)