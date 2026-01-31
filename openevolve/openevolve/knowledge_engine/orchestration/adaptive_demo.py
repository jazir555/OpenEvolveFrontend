"""
Adaptive Knowledge Engine Demo

Demonstrates the ultimate adaptive knowledge engine that:
1. Automatically classifies ANY input domain
2. Adapts processing strategy dynamically
3. Learns from all users globally
4. Continuously validates through gauntlet
5. Gets smarter and more accurate over time
"""

import json
from datetime import datetime, timezone


def demo_adaptive_classification():
    """Demo automatic domain classification"""
    print("=" * 80)
    print("ADAPTIVE KNOWLEDGE ENGINE - DOMAIN CLASSIFICATION DEMO")
    print("=" * 80)
    
    try:
        from . import DomainClassifier, classify_input
        
        # Test inputs from various domains
        test_inputs = [
            {
                'name': 'Financial Report',
                'text': 'Apple Inc. (AAPL) reported Q4 earnings of $2.18 per share, '
                        'beating analyst estimates. Revenue was $89.5 billion, up 8% year-over-year. '
                        'The company announced a dividend increase and stock buyback program.'
            },
            {
                'name': 'Chemistry Research',
                'text': 'The synthesis of ibuprofen involves the reaction of isobutylbenzene '
                        'with acetic anhydride. The molecular formula C13H18O2 was confirmed '
                        'through mass spectrometry and NMR analysis.'
            },
            {
                'name': 'Medical Record',
                'text': 'Patient presents with hypertension (BP 150/95), prescribed Lisinopril '
                        '10mg daily. History of type 2 diabetes, HbA1c 7.2%. Follow-up in 3 months.'
            },
            {
                'name': 'Legal Document',
                'text': 'WHEREAS, the Party of the First Part (hereinafter "Landlord") agrees '
                        'to lease the premises located at 123 Main Street to the Party of the Second Part '
                        '(hereinafter "Tenant") for a period of 12 months pursuant to Section 2.1.'
            },
            {
                'name': 'Research Paper',
                'text': 'Abstract: This study examines the relationship between climate change '
                        'and agricultural yields. Our methodology combines satellite imagery with '
                        'machine learning models. Results indicate a 15% reduction in crop yields '
                        'by 2050 under current warming scenarios.'
            },
            {
                'name': 'Technology Article',
                'text': 'The new Python 3.12 release introduces significant performance improvements '
                        'through optimized bytecode and reduced memory usage. Key features include '
                        'f-string debugging, type parameter syntax, and improved error messages.'
            }
        ]
        
        classifier = DomainClassifier()
        
        print("\nClassifying various inputs...\n")
        
        for test in test_inputs:
            result = classifier.classify({'text': test['text']})
            
            print(f"Input: {test['name']}")
            print(f"  Primary Domain: {result.primary_domain.value}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Content Type: {result.content_type.value}")
            print(f"  Recommended Components: {', '.join(result.recommended_components[:4])}")
            if result.secondary_domains:
                print(f"  Secondary Domains: {', '.join(d.value for d, _ in result.secondary_domains[:2])}")
            print()
        
        # Show classifier stats
        stats = classifier.get_classifier_stats()
        print(f"Classifier Statistics:")
        print(f"  Total Classifications: {stats['total_classifications']}")
        print(f"  Methods Used: {', '.join(stats['methods_used'])}")
        
    except Exception as e:
        print(f"Demo mode (may need dependencies): {e}")


def demo_global_learning():
    """Demo global learning across users"""
    print("\n" + "=" * 80)
    print("ADAPTIVE KNOWLEDGE ENGINE - GLOBAL LEARNING DEMO")
    print("=" * 80)
    
    try:
        from . import GlobalLearningEngine
        
        # Create global learning engine
        global_learning = GlobalLearningEngine(
            storage_path="demo_global_learning.json",
            enable_sharing=True
        )
        
        print("\nSimulating multi-user learning...\n")
        
        # Simulate contributions from different users
        users = ['alice', 'bob', 'charlie', 'diana']
        
        for i, user in enumerate(users):
            # Simulate execution result
            result = {
                'status': 'success' if i % 3 != 0 else 'partial',
                'domain': 'finance' if i % 2 == 0 else 'chemistry',
                'input_data': {'text': f'Sample text from {user}'},
                'results': {
                    'entities': [
                        {'text': f'Entity1_{user}', 'type': 'ORG'},
                        {'text': f'Entity2_{user}', 'type': 'PERSON'}
                    ]
                },
                'execution': {
                    'duration_ms': 5000 + i * 1000,
                    'quality_score': 0.8 if i % 3 != 0 else 0.5
                }
            }
            
            # Contribute to global learning
            global_learning.contribute_experience(
                user_id=user,
                execution_result=result,
                local_learning={'patterns_found': i + 1}
            )
            
            print(f"User '{user}' contributed experience")
        
        # Show global stats
        stats = global_learning.get_stats()
        print(f"\nGlobal Learning Statistics:")
        print(f"  Total Executions: {stats['total_executions']}")
        print(f"  Unique Users: {stats['unique_users']}")
        print(f"  Learned Patterns: {stats['learned_patterns']}")
        print(f"  Knowledge Entries: {stats['knowledge_entries']}")
        print(f"  Average Pattern Effectiveness: {stats['average_pattern_effectiveness']:.2%}")
        
        # Get recommendations
        print(f"\nGetting recommendations for new input...")
        recommendations = global_learning.get_recommendations(
            {'domain': 'finance', 'data_type': 'report'},
            local_user_id='new_user'
        )
        
        print(f"  Component Configs Available: {len(recommendations['component_configs'])}")
        print(f"  Healing Strategies Available: {len(recommendations['healing_strategies'])}")
        
    except Exception as e:
        print(f"Demo mode: {e}")


def demo_gauntlet_validation():
    """Demo gauntlet validation system"""
    print("\n" + "=" * 80)
    print("ADAPTIVE KNOWLEDGE ENGINE - GAUNTLET VALIDATION DEMO")
    print("=" * 80)
    
    try:
        from . import GauntletIntegration, TestType
        from . import create_integrated_orchestrator
        
        # Create orchestrator and gauntlet
        orchestrator = create_integrated_orchestrator()
        gauntlet = GauntletIntegration(orchestrator)
        
        print("\nCreating validation tests...\n")
        
        # Create test cases
        tests = [
            {
                'name': 'Entity Extraction Accuracy',
                'type': TestType.ACCURACY,
                'input': {
                    'text': 'Apple Inc. is headquartered in Cupertino, California. '
                            'Tim Cook is the CEO.',
                    'data_type': 'company_info'
                },
                'expected': {
                    'entities': [
                        {'text': 'Apple Inc.', 'type': 'ORG'},
                        {'text': 'Cupertino', 'type': 'GPE'},
                        {'text': 'California', 'type': 'GPE'},
                        {'text': 'Tim Cook', 'type': 'PERSON'}
                    ]
                }
            },
            {
                'name': 'Performance Test',
                'type': TestType.PERFORMANCE,
                'input': {
                    'text': 'Short text for performance testing',
                    'data_type': 'test'
                }
            },
            {
                'name': 'Robustness Test',
                'type': TestType.ROBUSTNESS,
                'input': {
                    'text': '',  # Edge case: empty input
                    'data_type': 'edge_case'
                }
            }
        ]
        
        # Create and run tests
        for test_data in tests:
            test = gauntlet.create_test(
                name=test_data['name'],
                test_type=test_data['type'],
                input_data=test_data['input'],
                expected_output=test_data.get('expected'),
                tags=['demo', 'adaptive']
            )
            
            print(f"Running: {test.name}")
            execution = gauntlet.run_test(test.test_id)
            print(f"  Result: {execution.result.value}")
            print(f"  Score: {execution.score:.2%}")
            if execution.issues:
                print(f"  Issues: {execution.issues}")
            print()
        
        # Show gauntlet stats
        stats = gauntlet.get_stats()
        print(f"Gauntlet Statistics:")
        print(f"  Total Tests: {stats['total_tests']}")
        print(f"  Total Executions: {stats['total_executions']}")
        print(f"  Average Score: {stats.get('average_score', 0):.2%}")
        
    except Exception as e:
        print(f"Demo mode: {e}")


def demo_adaptive_orchestrator():
    """Demo the ultimate adaptive orchestrator"""
    print("\n" + "=" * 80)
    print("ADAPTIVE KNOWLEDGE ENGINE - ULTIMATE ADAPTIVE ORCHESTRATOR DEMO")
    print("=" * 80)
    
    print("""
The AdaptiveOrchestrator is the ULTIMATE knowledge engine:

✓ NO DOMAIN PRESETS - Automatically classifies ANY input
✓ DYNAMIC ADAPTATION - Adjusts strategy based on content
✓ GLOBAL LEARNING - Learns from ALL users, improves for EVERYONE
✓ CONTINUOUS VALIDATION - Gauntlet ensures quality
✓ SELF-IMPROVING - Gets more accurate over time

Usage:
    from knowledge_engine import create_adaptive_orchestrator
    
    orchestrator = create_adaptive_orchestrator()
    
    # Works with ANY content - automatically adapts!
    result = orchestrator.process({'text': 'Your content here...'})
    
    # The system learns and improves for next time
    
How it works:
1. Input arrives (any domain, any type)
2. DomainClassifier automatically categorizes it
3. AdaptiveOrchestrator selects optimal components
4. GlobalLearningEngine applies patterns from all users
5. Processing executes with self-healing
6. Results validated through gauntlet
7. Experience contributes to global learning
8. Future executions benefit

Example scenarios:
- Financial report → Finance domain → Causal analysis enabled
- Chemistry paper → Chemistry domain → GlobalChem enabled
- Medical record → Healthcare domain → Drug entity extraction
- Legal contract → Legal domain → Clause extraction
- News article → News domain → Fast processing
- Research paper → Research domain → Comprehensive analysis

All handled AUTOMATICALLY by the adaptive system!
""")
    
    try:
        from . import create_adaptive_orchestrator
        
        print("Creating adaptive orchestrator...")
        orchestrator = create_adaptive_orchestrator(
            user_id='demo_user',
            storage_path='demo_adaptive_learning.json',
            enable_auto_classification=True,
            enable_global_learning=True,
            enable_gauntlet=True
        )
        
        print(f"  ✓ AdaptiveOrchestrator created")
        print(f"  ✓ Auto-classification: {orchestrator.adaptive_config.enable_auto_classification}")
        print(f"  ✓ Global learning: {orchestrator.adaptive_config.enable_global_learning}")
        print(f"  ✓ Gauntlet validation: {orchestrator.adaptive_config.enable_gauntlet}")
        
        # Get adaptive stats
        stats = orchestrator.get_adaptive_stats()
        print(f"\n  Execution Count: {stats['executions']}")
        print(f"  Unique User ID: {stats['user_id_hash']}")
        print(f"  Total Global Executions: {stats['global_learning_stats']['total_executions']}")
        print(f"  Total Global Users: {stats['global_learning_stats']['unique_users']}")
        
    except Exception as e:
        print(f"Demo mode: {e}")


def demo_system_architecture():
    """Show the complete adaptive system architecture"""
    print("\n" + "=" * 80)
    print("ADAPTIVE KNOWLEDGE ENGINE - COMPLETE ARCHITECTURE")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE KNOWLEDGE ENGINE                                │
│              (Universal, Self-Improving, Globally Learning)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Any text content (any domain, any type)                             │
│     ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. DOMAIN CLASSIFICATION                                            │   │
│  │    - Pattern matching (regex keywords)                              │   │
│  │    - LLM-based classification (optional)                            │   │
│  │    - Historical pattern matching                                    │   │
│  │    - Output: Domain + Confidence + Recommended Components           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. DYNAMIC CONFIGURATION                                            │   │
│  │    - Select optimal components based on domain                      │   │
│  │    - Adjust timeouts based on content type                          │   │
│  │    - Enable domain-specific features                                │   │
│  │    - NO hardcoded presets - pure adaptation!                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. GLOBAL LEARNING PATTERNS                                         │   │
│  │    - Apply patterns learned from all users                          │   │
│  │    - Use successful component configurations                        │   │
│  │    - Apply proven healing strategies                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. EXECUTION (with full self-healing)                               │   │
│  │    - Circuit breaker protection                                     │   │
│  │    - Component coordination & gap filling                           │   │
│  │    - 7 healing strategies if failures occur                         │   │
│  │    - Cross-validation between components                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 5. GAUNTLET VALIDATION                                              │   │
│  │    - Accuracy validation (vs expected output)                       │   │
│  │    - Completeness check                                             │   │
│  │    - Consistency validation                                         │   │
│  │    - Performance benchmarking                                       │   │
│  │    - Quality gate check                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 6. GLOBAL LEARNING CONTRIBUTION                                     │   │
│  │    - Share anonymized experience with global pool                   │   │
│  │    - Update pattern effectiveness                                   │   │
│  │    - Contribute to knowledge base                                   │   │
│  │    - ALL users benefit from YOUR execution!                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 7. CONTINUOUS ADAPTATION                                            │   │
│  │    - Track domain-specific performance                              │   │
│  │    - Adjust strategy based on success rates                         │   │
│  │    - Learn optimal configurations                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     ↓                                                                       │
│  OUTPUT: Results + Adaptive Metadata + Feedback Request                     │
│                                                                             │
│  THE SYSTEM IS NOW SMARTER THAN BEFORE!                                     │
│  ✓ Your execution contributed to global learning                            │
│  ✓ Future users will benefit from your patterns                             │
│  ✓ The system learned what works for your domain                            │
│  ✓ Quality validated through gauntlet                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

KEY PRINCIPLE: The more people use it, the better it gets for EVERYONE.
""")


def run_all_demos():
    """Run all adaptive demos"""
    print("\n" + "=" * 80)
    print("ADAPTIVE KNOWLEDGE ENGINE - COMPLETE DEMONSTRATION")
    print("=" * 80)
    print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
    
    demo_adaptive_classification()
    demo_global_learning()
    demo_gauntlet_validation()
    demo_adaptive_orchestrator()
    demo_system_architecture()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"Completed at: {datetime.now(timezone.utc).isoformat()}")
    print("""
The Adaptive Knowledge Engine is a TRUE knowledge engine:

✓ Universal - handles ANY domain automatically
✓ Adaptive - dynamically adjusts to content
✓ Learning - improves from every execution  
✓ Global - benefits all users collectively
✓ Validated - continuous quality assurance
✓ Self-Improving - accuracy increases over time

This is not just an orchestrator - it's a living, learning system
that gets smarter the more it's used!
""")


if __name__ == "__main__":
    run_all_demos()
