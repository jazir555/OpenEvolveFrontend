"""
Enhanced Adversarial Testing - Demonstration

This script demonstrates all the advanced features of the enhanced adversarial testing system.

Features demonstrated:
1. AI-Driven Attack Generation
2. Adaptive Defense Mechanisms
3. Explainability Framework
4. Continuous Learning
5. Ensemble Attack System
6. Advanced Analytics
7. Real-time Adaptation
8. Multi-modal Content Support

Usage:
    python demo_enhanced_adversarial.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import enhanced adversarial system
try:
    from adversarial_advanced import (
        EnhancedAdversarialEngine,
        AdvancedAdversarialConfig,
        create_enhanced_config,
        quick_enhanced_test,
        AdvancedAttackStrategy,
        AdvancedDefenseStrategy,
        ExplainabilityLevel,
        LearningMode
    )
    ENHANCED_ADVERSARIAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Enhanced adversarial system not available: {e}")
    ENHANCED_ADVERSARIAL_AVAILABLE = False


# =============================================================================
# DEMO CONTENT SAMPLES
# =============================================================================

VULNERABLE_CODE = """
def authenticate(username, password):
    '''Authenticate user with username and password'''
    # Vulnerability: SQL Injection
    query = f"SELECT * FROM users WHERE username='{username}'"
    user = database.execute(query)

    # Vulnerability: Plain text password comparison
    if user and user.password == password:
        return True
    return False
"""

SECURE_CODE = """
def authenticate(username, password):
    '''Authenticate user securely'''
    # Parameterized query prevents SQL injection
    query = "SELECT * FROM users WHERE username = ?"
    user = database.execute(query, [username])

    # Hash comparison prevents timing attacks
    if user and verify_password_hash(user.password_hash, password):
        return True
    return False
"""

API_SPECIFICATION = """
POST /api/users/create

Description: Create a new user account

Parameters:
- username: string (required) - Unique username
- password: string (required) - User password
- email: string (optional) - User email address

Authentication: None
Rate Limiting: None

Returns:
- 200: User created successfully
- 400: Invalid input
- 409: Username already exists
"""

DOCUMENTATION = """
# System Architecture

The system uses a client-server architecture with the following components:

1. Load Balancer
2. Application Servers
3. Database Servers

All communication is unencrypted for performance reasons.

Password recovery:
- Click "Forgot Password"
- Answer security question
- Password is displayed in plaintext
"""

COMPLEX_LOGIC = """
class TransactionProcessor:
    def __init__(self):
        self.balance = 0
        self.transactions = []

    def process_transaction(self, amount, type):
        # Race condition vulnerability
        if type == "credit":
            self.balance += amount
        elif type == "debit":
            if self.balance >= amount:
                self.balance -= amount
            else:
                return False

        self.transactions.append({
            "amount": amount,
            "type": type,
            "balance": self.balance
        })
        return True
"""


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_1_basic_enhanced_testing():
    """Demo 1: Basic enhanced testing with defaults"""
    print_section("DEMO 1: Basic Enhanced Testing (Quick Start)")

    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[SKIP] Enhanced adversarial system not available")
        return None

    print("Testing vulnerable authentication function...")
    print("-" * 40)

    result = quick_enhanced_test(
        content=VULNERABLE_CODE,
        content_type="code_python",
        theorem="Authentication function"
    )

    print("\n[OK] Testing completed!")
    print(f"  Duration: {result['duration']:.2f}s")
    print(f"  Iterations: {result['iterations_completed']}")
    print(f"  Final Robustness: {result['final_robustness']:.2%}")
    print(f"\n  Metrics:")
    print(f"    Total Attacks: {result['metrics']['total_attacks']}")
    print(f"    Successful Attacks: {result['metrics']['successful_attacks']}")
    print(f"    Total Defenses: {result['metrics']['total_defenses']}")
    print(f"    Successful Defenses: {result['metrics']['successful_defenses']}")

    # Show some attacks
    if result['attacks']:
        print(f"\n  Sample Attacks:")
        for attack in result['attacks'][:3]:
            status = "[OK]" if attack['success'] else "[FAIL]"
            print(f"    {status} {attack['description'][:60]}...")

    return result


def demo_2_custom_configuration():
    """Demo 2: Custom configuration for specific needs"""
    print_section("DEMO 2: Custom Configuration")

    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[SKIP] Enhanced adversarial system not available")
        return None

    # Create custom configuration
    config = create_enhanced_config(
        # Performance settings
        max_iterations=5,

        # AI-Driven Attacks
        enable_llm_attacks=True,
        llm_attack_model="gpt-4",

        # Adaptive Defense
        enable_adaptive_defense=True,
        defense_adaptation_rate=0.15,

        # Explainability
        explainability_level="detailed",
        explain_to_user=True,

        # Learning
        learning_mode="online",
        learning_rate=0.02,

        # Ensemble
        enable_ensemble=True,
        ensemble_size=5,
        voting_strategy="weighted",

        # Analytics
        enable_advanced_analytics=True,
        generate_reports=True
    )

    print("Configuration created:")
    print(f"  Max Iterations: {config.max_iterations if hasattr(config, 'max_iterations') else 'default'}")
    print(f"  LLM Attacks: {config.enable_llm_attacks}")
    print(f"  Adaptive Defense: {config.enable_adaptive_defense}")
    print(f"  Explainability: {config.explainability_level.value}")
    print(f"  Learning Mode: {config.learning_mode.value}")
    print(f"  Ensemble: {config.enable_ensemble} (size: {config.ensemble_size})")
    print(f"  Advanced Analytics: {config.enable_advanced_analytics}")

    print("\nRunning test with custom configuration...")
    engine = EnhancedAdversarialEngine(config)

    # Run synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            engine.enhanced_adversarial_test(
                content=API_SPECIFICATION,
                content_type="api_spec",
                theorem="Secure user creation API",
                max_iterations=5
            )
        )

        print(f"\n[OK] Testing completed!")
        print(f"  Final Robustness: {result['final_robustness']:.2%}")

        # Show adaptations
        if result['adaptations']:
            print(f"\n  Adaptations Performed: {len(result['adaptations'])}")
            for i, adaptation in enumerate(result['adaptations'], 1):
                print(f"    {i}. Threat Level: {adaptation['threat_level']:.2%}")
                print(f"       Recommended: {', '.join(adaptation['recommended_defenses'][:3])}")

        return result

    finally:
        loop.close()


def demo_3_explainability():
    """Demo 3: Explainability framework"""
    print_section("DEMO 3: Explainability Framework")

    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[SKIP] Enhanced adversarial system not available")
        return None

    # Test with different explainability levels
    for level in ["basic", "detailed", "full"]:
        print(f"\nTesting with {level.upper()} explainability...")

        config = create_enhanced_config(
            explainability_level=level,
            explain_to_user=True,
            max_iterations=3
        )

        engine = EnhancedAdversarialEngine(config)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                engine.enhanced_adversarial_test(
                    content=DOCUMENTATION,
                    content_type="document_general",
                    theorem="Security documentation",
                    max_iterations=3
                )
            )

            explanations = result.get('explanations', [])
            print(f"  Explanations generated: {len(explanations)}")

            if explanations:
                print(f"  Sample explanation:")
                exp = explanations[0]
                print(f"    Decision: {exp['decision']}")
                print(f"    Strategy: {exp['strategy']}")

                if level == "basic":
                    print(f"    Level: {exp['level']}")
                elif level == "detailed":
                    print(f"    Reasoning: {exp.get('reasoning', 'N/A')[:80]}...")
                    print(f"    User-friendly: {exp.get('user_friendly', 'N/A')[:80]}...")
                elif level == "full":
                    print(f"    Reasoning: {exp.get('reasoning', 'N/A')[:60]}...")
                    print(f"    Confidence: {exp.get('confidence', 0):.2%}")
                    print(f"    Internal States: {len(exp.get('internal_states', []))} entries")

        finally:
            loop.close()

    print("\n[OK] Explainability demo completed!")
    return True


def demo_4_continuous_learning():
    """Demo 4: Continuous learning system"""
    print_section("DEMO 4: Continuous Learning")

    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[SKIP] Enhanced adversarial system not available")
        return None

    # First run - establish baseline
    print("Run 1: Establishing baseline...")
    config1 = create_enhanced_config(
        learning_mode="online",
        experience_buffer_size=100,
        max_iterations=3
    )

    engine1 = EnhancedAdversarialEngine(config1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result1 = loop.run_until_complete(
            engine1.enhanced_adversarial_test(
                content=COMPLEX_LOGIC,
                content_type="code_python",
                theorem="Thread-safe transaction processing",
                max_iterations=3
            )
        )

        print(f"  Run 1 Robustness: {result1['final_robustness']:.2%}")

        # Show learning insights
        insights = result1.get('learning_insights', {})
        if insights:
            print(f"\n  Learning Insights (Run 1):")
            print(f"    Total experiences: {insights.get('total_experiences', 0)}")
            print(f"    Overall success rate: {insights.get('overall_success_rate', 0):.2%}")

            most_successful = insights.get('most_successful_attacks', [])
            if most_successful:
                print(f"\n    Most Successful Attacks:")
                for attack, rate in most_successful[:3]:
                    print(f"      - {attack}: {rate:.2%}")

            most_effective = insights.get('most_effective_defenses', [])
            if most_effective:
                print(f"\n    Most Effective Defenses:")
                for defense, effectiveness in most_effective[:3]:
                    print(f"      - {defense}: {effectiveness:.2%}")

    finally:
        loop.close()

    print("\n[OK] Learning demo completed!")
    return True


def demo_5_ensemble_attacks():
    """Demo 5: Ensemble attack system"""
    print_section("DEMO 5: Ensemble Attack System")

    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[SKIP] Enhanced adversarial system not available")
        return None

    # Compare ensemble vs single strategy
    print("Comparing Ensemble vs Single Strategy...")

    # Single strategy
    config_single = create_enhanced_config(
        enable_ensemble=False,
        max_iterations=5
    )

    engine_single = EnhancedAdversarialEngine(config_single)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_single = loop.run_until_complete(
            engine_single.enhanced_adversarial_test(
                content=SECURE_CODE,
                content_type="code_python",
                theorem="Secure authentication",
                max_iterations=5
            )
        )

        print(f"\n  Single Strategy:")
        print(f"    Robustness: {result_single['final_robustness']:.2%}")
        print(f"    Attacks: {result_single['metrics']['total_attacks']}")
        print(f"    Success Rate: {result_single['metrics']['attack_success_rate']:.2%}")

    finally:
        loop.close()

    # Ensemble strategy
    config_ensemble = create_enhanced_config(
        enable_ensemble=True,
        ensemble_size=5,
        voting_strategy="weighted",
        max_iterations=5
    )

    engine_ensemble = EnhancedAdversarialEngine(config_ensemble)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_ensemble = loop.run_until_complete(
            engine_ensemble.enhanced_adversarial_test(
                content=SECURE_CODE,
                content_type="code_python",
                theorem="Secure authentication",
                max_iterations=5
            )
        )

        print(f"\n  Ensemble Strategy:")
        print(f"    Robustness: {result_ensemble['final_robustness']:.2%}")
        print(f"    Attacks: {result_ensemble['metrics']['total_attacks']}")
        print(f"    Success Rate: {result_ensemble['metrics']['attack_success_rate']:.2%}")

        # Calculate improvement
        improvement = result_ensemble['final_robustness'] - result_single['final_robustness']
        print(f"\n  Ensemble Improvement: {improvement:+.2%}")

    finally:
        loop.close()

    print("\n[OK] Ensemble demo completed!")
    return True


def demo_6_performance_comparison():
    """Demo 6: Performance vs quality trade-offs"""
    print_section("DEMO 6: Performance Comparison")

    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[SKIP] Enhanced adversarial system not available")
        return None

    # Test different configurations
    configs = {
        "Fast": create_enhanced_config(
            max_iterations=3,
            ensemble_size=3,
            explainability_level="basic",
            enable_llm_attacks=False
        ),
        "Balanced": create_enhanced_config(
            max_iterations=7,
            ensemble_size=5,
            explainability_level="detailed",
            enable_llm_attacks=True
        ),
        "Thorough": create_enhanced_config(
            max_iterations=10,
            ensemble_size=7,
            explainability_level="full",
            enable_llm_attacks=True,
            enable_adaptive_defense=True
        )
    }

    results = {}

    for name, config in configs.items():
        print(f"\nTesting {name} configuration...")
        engine = EnhancedAdversarialEngine(config)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                engine.enhanced_adversarial_test(
                    content=VULNERABLE_CODE,
                    content_type="code_python",
                    theorem="Authentication function",
                    max_iterations=config.max_iterations if hasattr(config, 'max_iterations') else 5
                )
            )

            results[name] = {
                "duration": result['duration'],
                "robustness": result['final_robustness'],
                "iterations": result['iterations_completed'],
                "attacks": result['metrics']['total_attacks']
            }

            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Robustness: {result['final_robustness']:.2%}")

        finally:
            loop.close()

    # Summary table
    print("\n" + "-" * 60)
    print("Configuration Summary:")
    print("-" * 60)
    print(f"{'Mode':<12} {'Time':<10} {'Robustness':<12} {'Attacks':<10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(f"{name:<12} {metrics['duration']:>6.2f}s   {metrics['robustness']:>10.2%}   {metrics['attacks']:>8}")

    print("\n[OK] Performance comparison completed!")
    return results


def demo_7_adaptive_defense():
    """Demo 7: Adaptive defense system"""
    print_section("DEMO 7: Adaptive Defense System")

    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[SKIP] Enhanced adversarial system not available")
        return None

    config = create_enhanced_config(
        enable_adaptive_defense=True,
        enable_realtime_adaptation=True,
        adaptation_interval=2,  # Adapt every 2 iterations
        max_iterations=6
    )

    engine = EnhancedAdversarialEngine(config)

    print("Testing with adaptive defense enabled...")
    print("The system will adapt its defense strategy based on attack patterns.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            engine.enhanced_adversarial_test(
                content=COMPLEX_LOGIC,
                content_type="code_python",
                theorem="Thread-safe transaction processing",
                max_iterations=6
            )
        )

        print(f"\n[OK] Testing completed!")
        print(f"  Final Robustness: {result['final_robustness']:.2%}")

        # Show adaptations
        adaptations = result.get('adaptations', [])
        print(f"\n  Adaptations Performed: {len(adaptations)}")

        for i, adaptation in enumerate(adaptations, 1):
            print(f"\n  Adaptation {i}:")
            print(f"    Threat Level: {adaptation['threat_level']:.2%}")
            print(f"    Most Common Attacks:")
            for attack_type, count in adaptation.get('most_common_attacks', [])[:3]:
                print(f"      - {attack_type}: {count} occurrences")
            print(f"    Recommended Defenses:")
            for defense in adaptation.get('recommended_defenses', [])[:3]:
                print(f"      - {defense}")

        return result

    finally:
        loop.close()


# =============================================================================
# MAIN MENU
# =============================================================================

def main():
    """Main demo menu"""
    if not ENHANCED_ADVERSARIAL_AVAILABLE:
        print("[FAIL] Enhanced adversarial system not available!")
        print("Please ensure adversarial_advanced.py is in the Python path.")
        return

    print("\n" + "=" * 80)
    print("  ENHANCED ADVERSARIAL TESTING - DEMONSTRATION")
    print("  Advanced Features Showcase")
    print("=" * 80 + "\n")

    demos = [
        ("Basic Enhanced Testing", demo_1_basic_enhanced_testing),
        ("Custom Configuration", demo_2_custom_configuration),
        ("Explainability Framework", demo_3_explainability),
        ("Continuous Learning", demo_4_continuous_learning),
        ("Ensemble Attack System", demo_5_ensemble_attacks),
        ("Performance Comparison", demo_6_performance_comparison),
        ("Adaptive Defense System", demo_7_adaptive_defense),
    ]

    print("Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all demos")
    print(f"  q. Quit")
    print("")

    try:
        choice = input("Select demo (0-7, or q to quit): ").strip().lower()

        if choice == 'q':
            print("Goodbye!")
            return

        choice_num = int(choice) if choice not in ['q', 'Q'] else 0

        if choice_num == 0:
            # Run all demos
            print("\nRunning all demos sequentially...\n")
            for name, demo_func in demos:
                try:
                    print(f"\n{'=' * 80}")
                    print(f"Starting: {name}")
                    print('=' * 80)
                    result = demo_func()
                    if result is None:
                        logger.info(f"Demo {name} was skipped")
                except Exception as e:  # TODO: Catch specific exception instead of Exception
                    logger.error(f"Demo {name} failed: {e}", exc_info=True)
                    print(f"\n[FAIL] Demo failed: {e}")

        elif 1 <= choice_num <= len(demos):
            # Run selected demo
            name, demo_func = demos[choice_num - 1]
            try:
                result = demo_func()
                if result is None:
                    print("\nDemo was skipped")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Demo {name} failed: {e}", exc_info=True)
                print(f"\n[FAIL] Demo failed: {e}")
        else:
            print("Invalid choice")

    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[FAIL] Unexpected error: {e}")

    print("\n" + "=" * 80)
    print("  DEMO COMPLETED")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - ENHANCED_ADVERSARIAL_GUIDE.md")
    print("  - adversarial_advanced.py")
    print("")


if __name__ == "__main__":
    main()
