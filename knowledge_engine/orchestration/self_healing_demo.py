"""
Self-Healing Learning Orchestrator Demo

Demonstrates the complete self-healing, learning system where:
1. The orchestrator learns from every execution
2. Failures trigger automatic healing strategies
3. Components coordinate to fill each other's gaps
4. Feedback drives continuous improvement
5. The system gets smarter over time
"""

import json
from datetime import datetime, timezone


def demo_self_healing_capabilities():
    """Demonstrate self-healing capabilities"""
    print("=" * 80)
    print("SELF-HEALING LEARNING ORCHESTRATOR DEMO")
    print("=" * 80)
    print()
    
    print("THE VISION:")
    print("-" * 80)
    print("""
The Knowledge Engine is now a self-healing, learning system where:

1. EVERY EXECUTION TEACHES THE SYSTEM
   - Successes reinforce good configurations
   - Failures trigger healing and learning
   - Component performance is continuously tracked
   
2. COMPONENTS COVER EACH OTHER'S GAPS
   - If DeepKE fails on chemistry, GlobalChem steps in
   - If Causal-Learn is unavailable, Karate Club provides structural analysis
   - Gap analysis ensures comprehensive coverage
   
3. FAILURES BECOME LEARNING OPPORTUNITIES
   - Automatic retry with adjusted configuration
   - Component substitution when components fail
   - Fallback pipeline execution
   - Task decomposition for large inputs
   
4. FEEDBACK DRIVES CONTINUOUS IMPROVEMENT
   - User feedback collected automatically
   - A/B testing of configurations
   - Pipeline patterns learned from experience
   - System adapts to your specific use case
""")
    
    print("\n" + "=" * 80)
    print("DEMO 1: Self-Healing in Action")
    print("=" * 80)
    
    try:
        from . import create_self_healing_finance_orchestrator
        
        print("\n1. Creating self-healing finance orchestrator...")
        orchestrator = create_self_healing_finance_orchestrator(
            learning_storage_path="demo_learning.json"
        )
        
        print(f"   [OK] Orchestrator created with {len(orchestrator.components)} components")
        print(f"   [OK] Self-healing enabled: {orchestrator.enable_self_healing}")
        print(f"   [OK] Max healing attempts: {orchestrator.max_healing_attempts}")
        print(f"   [OK] Learning engine initialized with {len(orchestrator.learning_engine.experiences)} experiences")
        
        print("\n2. Simulating execution with potential failures...")
        
        # Demonstrate what happens when a component fails
        print("""
   SCENARIO: NeuralKG component fails during execution
   
   HEALING SEQUENCE:
   
   Attempt 1: Initial execution
   └── Component NeuralKG fails with timeout
   
   Attempt 2: Retry with config (HEALING)
   └── Increase timeout from 30s to 120s
   └── Result: Still failing
   
   Attempt 3: Component substitution (HEALING)
   └── NeuralKG substituted with Karate Club
   └── Karate Club provides graph structure analysis
   └── Result: SUCCESS with substitute
   
   LESSON LEARNED:
   └── Recorded: "NeuralKG has timeout issues on large graphs"
   └── Recorded: "Karate Club is good substitute for NeuralKG"
   └── Future executions will prefer Karate Club for similar inputs
        """)
        
        # Show healing report
        report = orchestrator.get_healing_report()
        print(f"\n3. Healing Report:")
        print(f"   Total healing actions: {report['total_healing_actions']}")
        print(f"   Successful healings: {report['successful_healings']}")
        print(f"   Healing success rate: {report['healing_success_rate']:.1%}")
        
    except Exception as e:
        print(f"   Demo mode (imports not available): {e}")
    
    print("\n" + "=" * 80)
    print("DEMO 2: Learning Engine")
    print("=" * 80)
    
    try:
        from .learning_engine import LearningEngine, LearningExperience
        
        print("\n1. Creating learning engine...")
        learning = LearningEngine()
        
        print("\n2. Simulating multiple learning experiences...")
        
        # Simulate experiences
        experiences = [
            {
                'input_data': {'text': 'Apple Inc. earnings report...', 'data_type': 'financial'},
                'data_type': 'financial',
                'domain': 'finance',
                'components': ['deepke', 'karate_club', 'pami'],
                'success': True,
                'quality': 0.85,
                'time': 5000
            },
            {
                'input_data': {'text': 'Q3 financial results...', 'data_type': 'financial'},
                'data_type': 'financial',
                'domain': 'finance',
                'components': ['deepke', 'karate_club', 'pami', 'neuralkg'],
                'success': True,
                'quality': 0.75,
                'time': 12000
            },
            {
                'input_data': {'text': 'Stock market analysis...', 'data_type': 'financial'},
                'data_type': 'financial',
                'domain': 'finance',
                'components': ['deepke', 'neuralkg'],
                'success': False,
                'quality': 0.2,
                'time': 8000,
                'error': 'timeout'
            }
        ]
        
        for i, exp_data in enumerate(experiences):
            experience = learning.record_experience(
                input_data=exp_data['input_data'],
                data_type=exp_data['data_type'],
                domain=exp_data['domain'],
                pipeline_config={'components': {c: {} for c in exp_data['components']}},
                components_used=exp_data['components'],
                success=exp_data['success'],
                execution_time_ms=exp_data['time'],
                results={'quality': exp_data['quality']} if exp_data['success'] else {},
                errors=[{'type': exp_data.get('error'), 'component': 'neuralkg'}] if not exp_data['success'] else []
            )
            print(f"   [OK] Experience {i+1} recorded: {experience.experience_id}")
            print(f"     Lessons: {experience.lessons_learned}")
        
        print("\n3. Component Performance Profiles:")
        for comp_type, profile in learning.component_profiles.items():
            print(f"   {comp_type}:")
            print(f"     - Success rate: {profile.success_rate:.1%}")
            print(f"     - Avg quality: {profile.average_quality_score:.2f}")
            print(f"     - Total invocations: {profile.total_invocations}")
        
        print("\n4. Recommendations for next execution:")
        recommendations = learning.recommend_components('financial', 'finance', {})
        for rec in recommendations[:3]:
            print(f"   {rec['component']}: expected quality {rec['expected_quality']:.2f}, "
                  f"recommended: {rec['recommended']}")
        
        print("\n5. Learning Summary:")
        summary = learning.get_learning_summary()
        print(f"   Total experiences: {summary['total_experiences']}")
        print(f"   Component profiles: {summary['component_profiles']}")
        print(f"   Learned patterns: {summary['learned_patterns']}")
        print(f"   Global success rate: {summary['global_success_rate']:.1%}")
        
    except Exception as e:
        print(f"   Demo mode: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO 3: Component Coordination & Gap Coverage")
    print("=" * 80)
    
    try:
        from .component_coordination import ComponentCoordinator, analyze_pipeline_gaps
        
        print("\n1. Analyzing pipeline gaps...")
        
        # Analyze a finance pipeline (without chemistry components)
        components = ['deepke', 'karate_club', 'pami', 'neuralkg']
        gap_analysis = analyze_pipeline_gaps(components)
        
        print(f"\n   Pipeline: {components}")
        print(f"\n   Gaps Identified:")
        for gap in gap_analysis['gaps_identified']:
            print(f"     - {gap['component']} has gap: {gap['gap']}")
        
        print(f"\n   Gap Fillers Assigned:")
        for filler in gap_analysis['gap_fillers']:
            print(f"     - {filler['gap']} filled by {filler['filled_by']} "
                  f"(confidence: {filler['confidence']:.2f})")
        
        print("\n2. Coordinating component execution...")
        
        coordinator = ComponentCoordinator()
        plan = coordinator.coordinate_pipeline(
            components=components,
            input_data={'text': 'Financial report...', 'data_type': 'financial'},
            data_type='financial',
            domain='finance'
        )
        
        print(f"\n   Coordination Plan:")
        print(f"     Primary components: {plan['primary_components']}")
        print(f"     Expected confidence: {plan['expected_confidence']:.2f}")
        
        print(f"\n   Data Routing:")
        for stage in plan['data_routing']['stages']:
            print(f"     Stage {stage['stage']}: {stage['components']}")
        
        print(f"\n   Cross-Validation Points:")
        for vp in plan['cross_validation_points']:
            print(f"     - {vp['capability']}: validate between {vp['components']}")
        
    except Exception as e:
        print(f"   Demo mode: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO 4: Feedback Loop & Continuous Improvement")
    print("=" * 80)
    
    try:
        from .feedback_loop import FeedbackCollector, ContinuousImprovementEngine
        from .feedback_loop import FeedbackType, ImprovementArea
        
        print("\n1. Collecting feedback...")
        
        feedback = FeedbackCollector()
        
        # Simulate various feedback
        feedback_entries = [
            {
                'correlation_id': 'exec_001',
                'type': FeedbackType.SUCCESS,
                'rating': 5,
                'components': ['deepke', 'karate_club'],
                'issues': [],
                'suggestions': []
            },
            {
                'correlation_id': 'exec_002',
                'type': FeedbackType.PARTIAL_SUCCESS,
                'rating': 3,
                'components': ['deepke', 'neuralkg'],
                'issues': ['Slow execution'],
                'suggestions': ['Use Karate Club instead']
            },
            {
                'correlation_id': 'exec_003',
                'type': FeedbackType.QUALITY_ISSUE,
                'rating': 2,
                'components': ['deepke', 'pami'],
                'issues': ['Missing entities', 'Incomplete patterns'],
                'suggestions': ['Add GlobalChem for chemical entities']
            }
        ]
        
        for entry_data in feedback_entries:
            entry = feedback.collect_feedback(
                correlation_id=entry_data['correlation_id'],
                input_data={'text': 'sample'},
                components_used=entry_data['components'],
                pipeline_config={},
                feedback_type=entry_data['type'],
                rating=entry_data['rating'],
                issues=entry_data['issues'],
                suggestions=entry_data['suggestions']
            )
            print(f"   [OK] Feedback collected: {entry.feedback_id} - {entry.feedback_type.value}")
        
        print("\n2. Feedback Statistics:")
        stats = feedback.get_feedback_stats()
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   By type: {stats['by_type']}")
        print(f"   Average rating: {stats['average_rating']:.1f}/5")
        
        print("\n3. Analyzing for improvements...")
        
        improvement = ContinuousImprovementEngine(feedback)
        recommendations = improvement.analyze_feedback()
        
        print(f"   Recommendations found: {len(recommendations)}")
        for rec in recommendations:
            print(f"\n   [{rec['priority'].upper()}] {rec['area'].value}")
            print(f"     Issue: {rec['issue']}")
            print(f"     Suggestion: {rec['suggestion']}")
            print(f"     Confidence: {rec['confidence']:.1%}")
        
        print("\n4. Creating improvement experiment...")
        
        experiment = improvement.create_experiment(
            improvement_area=ImprovementArea.COMPONENT_SELECTION,
            hypothesis="Using Karate Club instead of NeuralKG improves performance",
            control_config={'components': ['deepke', 'neuralkg']},
            treatment_config={'components': ['deepke', 'karate_club']}
        )
        
        print(f"   [OK] Experiment created: {experiment.experiment_id}")
        print(f"     Hypothesis: {experiment.hypothesis}")
        print(f"     Status: {experiment.status}")
        
        print("\n5. Improvement Report:")
        report = improvement.get_improvement_report()
        print(f"   Total experiments: {report['total_experiments']}")
        print(f"   Active: {report['active_experiments']}")
        print(f"   Improvements applied: {report['improvements_applied']}")
        
    except Exception as e:
        print(f"   Demo mode: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO 5: Complete Adaptive System")
    print("=" * 80)
    
    print("""
Putting it all together - the complete self-healing, learning, adaptive system:

┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE KNOWLEDGE ORCHESTRATOR                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Financial Report Text                                               │
│     v                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. PRE-EXECUTION ANALYSIS                                           │ │
│  │    - Predict potential failures based on history                    │ │
│  │    - Check component performance for 'financial' data type          │ │
│  │    - NeuralKG flagged as risky (previous timeouts)                  │ │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     v                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. GAP ANALYSIS & COORDINATION                                      │ │
│  │    - Pipeline: DeepKE -> KG-Gen -> Karate Club -> PAMI                │ │
│  │    - Gap: No causal analysis capability                             │ │
│  │    - Filler: Causal-Learn added (optional based on data)            │ │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     v                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. EXECUTION WITH HEALING                                           │ │
│  │    - DeepKE: SUCCESS (extracted 45 entities)                        │ │
│  │    - KG-Gen: SUCCESS (built graph with 45 nodes, 78 edges)          │ │
│  │    - Karate Club: SUCCESS (found 5 communities)                     │ │
│  │    - PAMI: TIMEOUT (pattern mining exceeded 30s)                    │ │
│  │       v TRIGGER HEALING                                             │ │
│  │       - Retry with 120s timeout                                     │ │
│  │       - Still failing                                               │ │
│  │       - Skip PAMI and continue with warning                         │ │
│  │    - Execution completes with partial results                       │ │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     v                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. LEARNING & FEEDBACK                                              │ │
│  │    - Record experience: PAMI timeout on financial data              │ │
│  │    - Update component profile: PAMI reliability v                   │ │
│  │    - Generate lessons: "Consider reducing min_support for PAMI"     │ │
│  │    - Auto-collect feedback: PARTIAL_SUCCESS, rating inferred        │ │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     v                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 5. CONTINUOUS IMPROVEMENT                                           │ │
│  │    - Analysis: PAMI performance degradation detected                │ │
│  │    - Recommendation: Reduce batch size or skip for large inputs     │ │
│  │    - A/B Test: Experiment with adjusted PAMI config created         │ │
│  │    - Next execution will try adjusted configuration                 │ │
│  └─────────────────────────────────────────────────────────────────────┘   │
│     v                                                                       │
│  OUTPUT: Results + Learning Metadata + Feedback Request                     │
│                                                                             │
│  The system is now SMARTER than before!                                     │
│  - Knows PAMI has issues with large financial texts                         │
│  - Will adjust configuration or skip for similar inputs                     │
│  - Can recommend alternatives to users                                      │
│  - Has recorded the pattern for future reference                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("=" * 80)
    print("KEY CAPABILITIES SUMMARY")
    print("=" * 80)
    
    print("""
1. SELF-HEALING (self_healing_orchestrator.py)
   [OK] 7 healing strategies: retry, substitution, fallback, parallel, etc.
   [OK] Automatic failure detection and diagnosis
   [OK] Component substitution matrix for gap coverage
   [OK] Records healing actions and outcomes
   [OK] Learns which strategies work best

2. LEARNING ENGINE (learning_engine.py)
   [OK] Records every execution as learning experience
   [OK] Builds component performance profiles
   [OK] Learns optimal pipeline patterns
   [OK] Predicts failures before they happen
   [OK] Recommends best components for each context

3. COMPONENT COORDINATION (component_coordination.py)
   [OK] Capability registry for all components
   [OK] Automatic gap identification
   [OK] Gap filler assignment
   [OK] Optimal data routing between components
   [OK] Cross-validation of results
   [OK] Result fusion from multiple sources

4. FEEDBACK LOOP (feedback_loop.py)
   [OK] Automatic feedback collection
   [OK] User feedback integration
   [OK] A/B testing framework
   [OK] Continuous improvement engine
   [OK] Improvement recommendations

5. MCP SERVER (mcp_server.py)
   [OK] 26 standardized API methods
   [OK] Expose all capabilities via Model Context Protocol
   [OK] Health monitoring and diagnostics
   [OK] Remote orchestrator management
""")
    
    print("=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("""
# Create a self-healing orchestrator
from knowledge_engine.orchestration import create_self_healing_finance_orchestrator

orchestrator = create_self_healing_finance_orchestrator(
    learning_storage_path="finance_learning.json"
)

# Process - healing happens automatically if needed
result = orchestrator.process({
    'text': 'Financial report text...',
    'data_type': 'financial_report'
})

# Check what was learned
summary = orchestrator.learning_engine.get_learning_summary()
print(f"Experiences: {summary['total_experiences']}")
print(f"Success rate: {summary['global_success_rate']:.1%}")

# Get healing report
healing = orchestrator.get_healing_report()
print(f"Healings performed: {healing['successful_healings']}")

# Check gap coverage analysis
from knowledge_engine.orchestration import analyze_pipeline_gaps
analysis = analyze_pipeline_gaps(['deepke', 'karate_club', 'pami'])
print(f"Gaps found: {len(analysis['gaps_identified'])}")
print(f"Fillers available: {len(analysis['gap_fillers'])}")

# Create adaptive integration with feedback
from knowledge_engine.orchestration import create_adaptive_orchestrator

adaptive = create_adaptive_orchestrator(orchestrator)
result = adaptive.process_with_feedback(
    input_data={'text': '...'},
    collect_user_feedback=True
)

# Submit user feedback
adaptive.submit_user_feedback(
    correlation_id=result['correlation_id'],
    rating=4,
    suggestions=['Add more entity types']
)
""")
    
    print("=" * 80)
    print("SYSTEM ARCHITECTURE")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE ENGINE ORCHESTRATION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SELF-HEALING ORCHESTRATOR                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │    Retry    │  │ Substitute  │  │  Fallback   │  │ Decompose  │ │   │
│  │  │   Strategy  │  │  Strategy   │  │  Strategy   │  │  Strategy  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ^                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LEARNING ENGINE                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ Experiences │  │  Component  │  │  Pipeline   │  │  Failure   │ │   │
│  │  │   Database  │  │  Profiles   │  │  Patterns   │  │ Prediction │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ^                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  COMPONENT COORDINATION                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ Capability  │  │    Gap      │  │    Data     │  │   Cross    │ │   │
│  │  │  Registry   │  │   Filling   │  │   Routing   │  │ Validation │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ^                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FEEDBACK LOOP                                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  Feedback   │  │ Improvement │  │    A/B      │  │  Adaptive  │ │   │
│  │  │  Collector  │  │   Engine    │  │   Tests     │  │  Tuning    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ^                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MCP SERVER                                   │   │
│  │              (26 Methods for External Access)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

The system learns, heals, adapts, and improves continuously!
Every execution makes it smarter. Every failure teaches it something new.
Components work together to cover gaps and ensure robust execution.
""")
    
    print(f"\nDemo completed at: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    demo_self_healing_capabilities()
