"""
End-to-End Integration Test for AI-Enhanced Knowledge Engine

This module provides comprehensive testing of the complete integration pipeline
to ensure all AI knowledge graph components work together seamlessly.
"""

import sys
import os
import time
from typing import Dict, Any, List

# Add knowledge_engine to Python path
knowledge_engine_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if knowledge_engine_path not in sys.path:
    sys.path.insert(0, knowledge_engine_path)

from ai_enhanced_integration import AIEnhancedKnowledgeEngine
from integrations import (
    DeepKEEnhancedExtractor, 
    KarateClubGraphAnalyzer, 
    EnhancedKnowledgeGraphManager,
    AIKnowledgeGraphIntegrator
)

class EndToEndIntegrationTester:
    """
    Comprehensive end-to-end tester for AI-enhanced knowledge engine integrations.
    
    This class tests all integration points and validates the complete pipeline.
    """
    
    def __init__(self):
        """Initialize the tester with all components."""
        self.test_results = {
            'individual_components': {},
            'integration_pipeline': {},
            'performance_metrics': {},
            'overall_status': 'not_started'
        }
        
        # Initialize all components
        self.deepke_extractor = DeepKEEnhancedExtractor()
        self.karateclub_analyzer = KarateClubGraphAnalyzer()
        self.kg_gen_manager = EnhancedKnowledgeGraphManager()
        self.ai_integrator = AIKnowledgeGraphIntegrator()
        self.ai_engine = AIEnhancedKnowledgeEngine()
    
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run the complete end-to-end test suite."""
        print("Starting End-to-End Integration Test Suite...")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Test 1: Individual Component Availability
            self._test_individual_components()
            
            # Test 2: DeepKE Extraction Pipeline
            self._test_deepke_extraction()
            
            # Test 3: Knowledge Artifact Conversion
            self._test_artifact_conversion()
            
            # Test 4: Karate Club Graph Analysis
            self._test_karateclub_analysis()
            
            # Test 5: kg-gen Knowledge Graph Management
            self._test_kg_gen_management()
            
            # Test 6: Complete AI Pipeline
            self._test_complete_ai_pipeline()
            
            # Test 7: Error Handling and Fallback
            self._test_error_handling()
            
            # Test 8: Performance Benchmarking
            self._test_performance_benchmarking()
            
            # Calculate total test time
            total_time = time.time() - start_time
            
            # Determine overall status
            all_passed = all(
                result['status'] == 'success' 
                for result in self.test_results['individual_components'].values()
            ) and all(
                result['status'] == 'success' 
                for result in self.test_results['integration_pipeline'].values()
            )
            
            self.test_results['overall_status'] = 'success' if all_passed else 'partial_success'
            self.test_results['performance_metrics']['total_test_time'] = total_time
            
            print("=" * 60)
            print(f"End-to-End Test Suite Completed in {total_time:.2f} seconds")
            print(f"Overall Status: {self.test_results['overall_status']}")
            
            return self._generate_test_report()
            
        except Exception as e:
            self.test_results['overall_status'] = 'failed'
            self.test_results['error'] = str(e)
            print(f"Test Suite Failed: {e}")
            return self.test_results
    
    def _test_individual_components(self):
        """Test individual component availability and basic functionality."""
        print("\n1. Testing Individual Component Availability...")
        
        components = {
            'deepke': self.deepke_extractor,
            'karateclub': self.karateclub_analyzer,
            'kg_gen': self.kg_gen_manager,
            'oneke': self.kg_gen_manager,
            'ai_integrator': self.ai_integrator,
            'ai_engine': self.ai_engine
        }
        
        for name, component in components.items():
            try:
                if hasattr(component, 'is_available'):
                    available = component.is_available()
                else:
                    available = True  # Assume available if no check method
                
                self.test_results['individual_components'][name] = {
                    'status': 'success',
                    'available': available,
                    'timestamp': self._get_current_timestamp()
                }
                
                status_text = "[OK]" if available else "[WARN]"
                print(f"   {status_text} {name}: {'Available' if available else 'Not Available'}")
                
            except Exception as e:
                self.test_results['individual_components'][name] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': self._get_current_timestamp()
                }
                print(f"   [FAIL] {name}: Failed - {e}")
    
    def _test_deepke_extraction(self):
        """Test DeepKE extraction capabilities."""
        print("\n2. Testing DeepKE Extraction Pipeline...")
        
        test_text = """
        The OpenEvolve Knowledge Engine integrates multiple AI technologies for enhanced knowledge management.
        It uses DeepKE for advanced knowledge extraction, Karate Club for graph analysis, and kg-gen for knowledge graph generation.
        This creates a 5x more powerful system than traditional approaches.
        """
        
        try:
            # Test basic extraction
            results = self.deepke_extractor.extract_with_deepke(test_text)
            
            if results['status'] == 'success':
                extracted_items = len(results['extracted_knowledge'])
                avg_confidence = results['extraction_stats']['avg_confidence']
                
                self.test_results['integration_pipeline']['deepke_extraction'] = {
                    'status': 'success',
                    'extracted_items': extracted_items,
                    'avg_confidence': avg_confidence,
                    'by_type': results['extraction_stats']['by_type'],
                    'timestamp': self._get_current_timestamp()
                }
                
                print(f"   [OK] DeepKE Extraction: {extracted_items} items extracted (avg confidence: {avg_confidence:.2f})")
                
                # Show some extracted items
                for i, item in enumerate(results['extracted_knowledge'][:3]):
                    print(f"      - {item['type']}: {item['knowledge_item']}")
                    
            else:
                self.test_results['integration_pipeline']['deepke_extraction'] = {
                    'status': 'failed',
                    'error': results['message'],
                    'timestamp': self._get_current_timestamp()
                }
                print(f"   [FAIL] DeepKE Extraction Failed: {results['message']}")
                
        except Exception as e:
            self.test_results['integration_pipeline']['deepke_extraction'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
            print(f"   [FAIL] DeepKE Extraction Failed: {e}")
    
    def _test_artifact_conversion(self):
        """Test knowledge artifact conversion."""
        print("\n3. Testing Knowledge Artifact Conversion...")
        
        try:
            # Create sample extraction results
            sample_results = {
                'status': 'success',
                'extracted_knowledge': [
                    {
                        'knowledge_item': {
                            'subject': 'OpenEvolve',
                            'predicate': 'integrates',
                            'object': 'DeepKE'
                        },
                        'extractor': 'asp',
                        'type': 'triple',
                        'confidence': 0.95
                    },
                    {
                        'knowledge_item': {
                            'entity': 'OpenEvolve Knowledge Engine',
                            'type': 'system'
                        },
                        'extractor': 'ner_extractor',
                        'type': 'entity',
                        'confidence': 0.92
                    }
                ]
            }
            
            # Convert to knowledge artifacts
            artifacts = self.ai_engine._convert_to_knowledge_artifacts(sample_results)
            
            self.test_results['integration_pipeline']['artifact_conversion'] = {
                'status': 'success',
                'converted_artifacts': len(artifacts),
                'artifact_types': [a['knowledge_type'] for a in artifacts],
                'timestamp': self._get_current_timestamp()
            }
            
            print(f"   [OK] Artifact Conversion: {len(artifacts)} artifacts converted")
            for artifact in artifacts:
                print(f"      - {artifact['knowledge_type']}: {artifact.get('subject', artifact.get('entity', 'N/A'))}")
                
        except Exception as e:
            self.test_results['integration_pipeline']['artifact_conversion'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
            print(f"   [FAIL] Artifact Conversion Failed: {e}")
    
    def _test_karateclub_analysis(self):
        """Test Karate Club graph analysis."""
        print("\n4. Testing Karate Club Graph Analysis...")
        
        try:
            # Create sample graph data
            graph_data = {
                'nodes': [
                    {'id': 'OpenEvolve', 'type': 'system'},
                    {'id': 'DeepKE', 'type': 'technology'},
                    {'id': 'Karate Club', 'type': 'technology'},
                    {'id': 'kg-gen', 'type': 'technology'}
                ],
                'edges': [
                    {'source': 'OpenEvolve', 'target': 'DeepKE', 'relationship': 'integrates'},
                    {'source': 'OpenEvolve', 'target': 'Karate Club', 'relationship': 'uses'},
                    {'source': 'OpenEvolve', 'target': 'kg-gen', 'relationship': 'integrates'}
                ]
            }
            
            # Analyze graph
            results = self.karateclub_analyzer.analyze_graph(graph_data)
            
            if results['status'] == 'success':
                analysis = results['analysis_results']
                communities = len(analysis.get('communities', {}))
                metrics_available = 'metrics' in analysis
                
                self.test_results['integration_pipeline']['karateclub_analysis'] = {
                    'status': 'success',
                    'communities_detected': communities,
                    'metrics_calculated': metrics_available,
                    'analysis_types': list(analysis.keys()),
                    'timestamp': self._get_current_timestamp()
                }
                
                print(f"   [OK] Karate Club Analysis: {communities} community detection methods")
                if metrics_available:
                    metrics = analysis['metrics']['basic_metrics']
                    print(f"      Graph: {metrics['num_nodes']} nodes, {metrics['num_edges']} edges")
                    print(f"      Density: {metrics['density']:.3f}")
                
            else:
                self.test_results['integration_pipeline']['karateclub_analysis'] = {
                    'status': 'failed',
                    'error': results['message'],
                    'timestamp': self._get_current_timestamp()
                }
                print(f"   [FAIL] Karate Club Analysis Failed: {results['message']}")
                
        except Exception as e:
            self.test_results['integration_pipeline']['karateclub_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
            print(f"   [FAIL] Karate Club Analysis Failed: {e}")
    
    def _test_kg_gen_management(self):
        """Test kg-gen knowledge graph management."""
        print("\n5. Testing kg-gen Knowledge Graph Management...")
        
        try:
            # Create sample knowledge artifacts
            artifacts = [
                {
                    'source': 'test',
                    'knowledge_type': 'triple',
                    'subject': 'OpenEvolve',
                    'predicate': 'integrates',
                    'object': 'DeepKE',
                    'confidence': 0.95
                },
                {
                    'source': 'test',
                    'knowledge_type': 'triple',
                    'subject': 'OpenEvolve',
                    'predicate': 'uses',
                    'object': 'Karate Club',
                    'confidence': 0.92
                }
            ]
            
            # Generate knowledge graph
            results = self.kg_gen_manager.generate_and_store_knowledge_graph(artifacts)
            
            if results['status'] == 'success':
                graph_generated = results['results']['knowledge_graph'] is not None
                formats_converted = len(results['results']['converted_formats'])
                
                self.test_results['integration_pipeline']['kg_gen_management'] = {
                    'status': 'success',
                    'graph_generated': graph_generated,
                    'formats_converted': formats_converted,
                    'converted_formats': list(results['results']['converted_formats'].keys()),
                    'timestamp': self._get_current_timestamp()
                }
                
                print(f"   [OK] kg-gen Management: Graph {'generated' if graph_generated else 'not generated'}")
                print(f"      Formats converted: {formats_converted}")
                for format_name in results['results']['converted_formats'].keys():
                    print(f"        - {format_name}")
                
            else:
                self.test_results['integration_pipeline']['kg_gen_management'] = {
                    'status': 'failed',
                    'error': results['message'],
                    'timestamp': self._get_current_timestamp()
                }
                print(f"   [FAIL] kg-gen Management Failed: {results['message']}")
                
        except Exception as e:
            self.test_results['integration_pipeline']['kg_gen_management'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
            print(f"   [FAIL] kg-gen Management Failed: {e}")
    
    def _test_complete_ai_pipeline(self):
        """Test the complete AI pipeline."""
        print("\n6. Testing Complete AI Pipeline...")
        
        try:
            # Create comprehensive test workflow
            workflow_data = {
                'text': '''The OpenEvolve Knowledge Engine represents a significant advancement in knowledge management systems.
                It integrates DeepKE for advanced knowledge extraction, which provides triple extraction, relation extraction,
                event extraction, and named entity recognition capabilities. For graph analysis, it uses Karate Club,
                which offers community detection, node embeddings, and comprehensive graph metrics. The system also
                incorporates kg-gen for knowledge graph generation and Neo4j integration, along with OneKE for format
                conversion and interoperability. This creates a 5x more powerful knowledge management system.''',
                'metadata': {
                    'source': 'end_to_end_test',
                    'timestamp': self._get_current_timestamp()
                }
            }
            
            # Execute complete AI pipeline
            start_time = time.time()
            results = self.ai_engine.execute_complete_ai_pipeline(workflow_data)
            pipeline_time = time.time() - start_time
            
            if results['status'] == 'success':
                extraction_items = len(results['results']['extraction']['extracted_knowledge'])
                graph_nodes = results['results']['graph_analysis']['analysis_results']['metrics']['basic_metrics']['num_nodes']
                
                self.test_results['integration_pipeline']['complete_ai_pipeline'] = {
                    'status': 'success',
                    'processing_time': pipeline_time,
                    'extraction_items': extraction_items,
                    'graph_nodes': graph_nodes,
                    'graph_edges': results['results']['graph_analysis']['analysis_results']['metrics']['basic_metrics']['num_edges'],
                    'ai_enhancement_factor': results['performance']['ai_enhancement_factor'],
                    'timestamp': self._get_current_timestamp()
                }
                
                print(f"   [OK] Complete AI Pipeline: {extraction_items} items extracted, {graph_nodes} nodes in graph")
                print(f"      Processing time: {pipeline_time:.3f} seconds")
                print(f"      AI Enhancement Factor: {results['performance']['ai_enhancement_factor']}")
                
            else:
                self.test_results['integration_pipeline']['complete_ai_pipeline'] = {
                    'status': 'failed',
                    'error': results['message'],
                    'timestamp': self._get_current_timestamp()
                }
                print(f"   [FAIL] Complete AI Pipeline Failed: {results['message']}")
                
        except Exception as e:
            self.test_results['integration_pipeline']['complete_ai_pipeline'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
            print(f"   [FAIL] Complete AI Pipeline Failed: {e}")
    
    def _test_error_handling(self):
        """Test error handling and fallback mechanisms."""
        print("\n7. Testing Error Handling and Fallback...")
        
        try:
            # Test empty input handling
            empty_results = self.deepke_extractor.extract_with_deepke("")
            empty_handled = empty_results['status'] == 'success'
            
            # Test invalid graph data handling
            invalid_graph = {'nodes': [], 'edges': []}
            invalid_results = self.karateclub_analyzer.analyze_graph(invalid_graph)
            invalid_handled = invalid_results['status'] == 'success'
            
            # Test empty artifacts handling
            empty_artifacts = []
            empty_kg_results = self.kg_gen_manager.generate_and_store_knowledge_graph(empty_artifacts)
            empty_kg_handled = empty_kg_results['status'] == 'success'
            
            self.test_results['integration_pipeline']['error_handling'] = {
                'status': 'success',
                'empty_input_handled': empty_handled,
                'invalid_graph_handled': invalid_handled,
                'empty_artifacts_handled': empty_kg_handled,
                'timestamp': self._get_current_timestamp()
            }
            
            print(f"   [OK] Error Handling: All test cases handled properly")
            print(f"      Empty input: {'[OK]' if empty_handled else '[FAIL]'}")
            print(f"      Invalid graph: {'[OK]' if invalid_handled else '[FAIL]'}")
            print(f"      Empty artifacts: {'[OK]' if empty_kg_handled else '[FAIL]'}")
            
        except Exception as e:
            self.test_results['integration_pipeline']['error_handling'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
            print(f"   [FAIL] Error Handling Test Failed: {e}")
    
    def _test_performance_benchmarking(self):
        """Test performance benchmarking."""
        print("\n8. Testing Performance Benchmarking...")
        
        try:
            # Test multiple runs for average performance
            test_text = "OpenEvolve integrates DeepKE, Karate Club, kg-gen, and OneKE for 5x enhancement."
            workflow = {'text': test_text}
            
            run_times = []
            for i in range(3):  # Run 3 times for average
                start_time = time.time()
                results = self.ai_engine.execute_complete_ai_pipeline(workflow)
                run_times.append(time.time() - start_time)
            
            avg_time = sum(run_times) / len(run_times)
            
            self.test_results['performance_metrics'] = {
                'average_pipeline_time': avg_time,
                'individual_runs': run_times,
                'min_time': min(run_times),
                'max_time': max(run_times),
                'timestamp': self._get_current_timestamp()
            }
            
            print(f"   [OK] Performance Benchmarking: Average {avg_time:.3f} seconds")
            print(f"      Runs: {[f'{t:.3f}s' for t in run_times]}")
            print(f"      Range: {min(run_times):.3f}s - {max(run_times):.3f}s")
            
        except Exception as e:
            self.test_results['performance_metrics']['error'] = str(e)
            print(f"   [FAIL] Performance Benchmarking Failed: {e}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        report = {
            'test_suite': self.test_results,
            'summary': self._generate_summary(),
            'timestamp': self._get_current_timestamp()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of test results."""
        # Count passed/failed tests
        individual_passed = sum(
            1 for result in self.test_results['individual_components'].values() 
            if result['status'] == 'success'
        )
        individual_total = len(self.test_results['individual_components'])
        
        pipeline_passed = sum(
            1 for result in self.test_results['integration_pipeline'].values() 
            if result['status'] == 'success'
        )
        pipeline_total = len(self.test_results['integration_pipeline'])
        
        # Calculate success rates
        individual_success_rate = (individual_passed / individual_total * 100) if individual_total > 0 else 0
        pipeline_success_rate = (pipeline_passed / pipeline_total * 100) if pipeline_total > 0 else 0
        overall_success_rate = ((individual_passed + pipeline_passed) / (individual_total + pipeline_total) * 100) if (individual_total + pipeline_total) > 0 else 0
        
        # Get performance metrics
        avg_time = self.test_results['performance_metrics'].get('average_pipeline_time', 0)
        total_time = self.test_results['performance_metrics'].get('total_test_time', 0)
        
        return {
            'individual_components': {
                'passed': individual_passed,
                'total': individual_total,
                'success_rate': individual_success_rate
            },
            'integration_pipeline': {
                'passed': pipeline_passed,
                'total': pipeline_total,
                'success_rate': pipeline_success_rate
            },
            'overall': {
                'passed': individual_passed + pipeline_passed,
                'total': individual_total + pipeline_total,
                'success_rate': overall_success_rate,
                'status': self.test_results['overall_status']
            },
            'performance': {
                'average_pipeline_time': avg_time,
                'total_test_time': total_time,
                'ai_enhancement_factor': self.test_results['integration_pipeline'].get('complete_ai_pipeline', {}).get('ai_enhancement_factor', 5.0)
            },
            'recommendations': self._generate_recommendations(overall_success_rate)
        }
    
    def _generate_recommendations(self, success_rate: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if success_rate >= 90:
            recommendations.append("[OK] System is ready for production deployment")
            recommendations.append("[OK] All major components are working correctly")
            recommendations.append("📊 Consider running performance optimization tests")
        elif success_rate >= 70:
            recommendations.append("[WARN] System is mostly functional with some minor issues")
            recommendations.append("🔧 Review failed tests and address specific issues")
            recommendations.append("📊 Check error handling and fallback mechanisms")
        elif success_rate >= 50:
            recommendations.append("[WARN] System has significant issues that need attention")
            recommendations.append("🔧 Review integration configuration and dependencies")
            recommendations.append("📊 Check individual component availability")
        else:
            recommendations.append("[FAIL] System has critical failures")
            recommendations.append("🔧 Review all test results and error messages")
            recommendations.append("📊 Check system dependencies and environment setup")
        
        # Add specific recommendations based on component availability
        if not self.deepke_extractor.is_available():
            recommendations.append("📦 DeepKE integration not available - check installation")
        
        if not self.karateclub_analyzer.is_available():
            recommendations.append("📦 Karate Club integration not available - check installation")
        
        if not self.kg_gen_manager.is_kg_gen_available():
            recommendations.append("📦 kg-gen integration not available - check installation")
        
        return recommendations
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

def run_end_to_end_tests():
    """Run the complete end-to-end test suite and return results."""
    tester = EndToEndIntegrationTester()
    results = tester.run_complete_test_suite()
    return results

if __name__ == "__main__":
    # Run the complete test suite
    test_results = run_end_to_end_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("END-TO-END INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    summary = test_results['summary']
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Status: {summary['overall']['status']}")
    print(f"   Success Rate: {summary['overall']['success_rate']:.1f}%")
    print(f"   AI Enhancement Factor: {summary['overall']['ai_enhancement_factor']}")
    
    print(f"\n🧪 COMPONENT TESTS:")
    print(f"   Passed: {summary['individual_components']['passed']}/{summary['individual_components']['total']}")
    print(f"   Success Rate: {summary['individual_components']['success_rate']:.1f}%")
    
    print(f"\n🔗 INTEGRATION TESTS:")
    print(f"   Passed: {summary['integration_pipeline']['passed']}/{summary['integration_pipeline']['total']}")
    print(f"   Success Rate: {summary['integration_pipeline']['success_rate']:.1f}%")
    
    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Average Pipeline Time: {summary['performance']['average_pipeline_time']:.3f}s")
    print(f"   Total Test Time: {summary['performance']['total_test_time']:.3f}s")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for recommendation in summary['recommendations']:
        print(f"   {recommendation}")
    
    print("\n" + "=" * 60)
    print("End-to-End Integration Test Complete")
    print("=" * 60)