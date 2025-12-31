"""
MAKER/MDAP-Enhanced Adversarial Testing - Demo

This script demonstrates the integration of MAKER (arXiv:2511.09030) and MDAP
into the adversarial testing workflow.

Features demonstrated:
1. MAKER-enhanced red team: Voting-based attack generation
2. MDAP-enhanced blue team: Decomposed defense strategies
3. Co-evolutionary testing: Attack/defense arms race
4. Zero-error guarantees: Statistical convergence

Usage:
    python demo_adversarial_maker.py
"""

import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_1_basic_maker_adversarial():
    """Demo 1: Basic MAKER-enhanced adversarial testing"""
    print_section("DEMO 1: Basic MAKER-Enhanced Adversarial Testing")

    from adversarial import run_maker_enhanced_adversarial_testing

    # Sample content to test
    sample_code = '''
def authenticate(username, password):
    """Authenticate user with username and password."""
    # Check if user exists
    user = database.get_user(username)

    # Verify password
    if user.password == password:
        return True
    return False
'''

    print("Testing code for security vulnerabilities...")
    print(f"Content length: {len(sample_code)} characters")

    # Run MAKER-enhanced adversarial testing
    result = run_maker_enhanced_adversarial_testing(
        content=sample_code,
        content_type="code",
        coevolution_rounds=2,
        k_ahead=3
    )

    # Display results
    print("\n[OK] Adversarial testing completed!")
    print(f"  - Method: {result.get('method', 'unknown')}")
    print(f"  - Attacks found: {len(result.get('final_attacks', []))}")
    print(f"  - Defenses generated: {len(result.get('final_defenses', []))}")
    print(f"  - Co-evolution rounds: {result.get('total_rounds', 0)}")

    # Show sample attacks
    attacks = result.get('final_attacks', [])
    if attacks:
        print("\n[Sample Attacks]")
        for i, attack in enumerate(attacks[:3], 1):
            print(f"  {i}. {attack.get('title', 'Untitled')}")
            print(f"     Severity: {attack.get('severity', 'UNKNOWN')}")
            print(f"     Category: {attack.get('category', 'UNKNOWN')}")

    return result


def demo_2_maker_voting_only():
    """Demo 2: MAKER voting without MDAP"""
    print_section("DEMO 2: MAKER Voting (Red Team Only)")

    from adversarial import run_maker_enhanced_adversarial_testing

    sample_content = """
API Endpoint: POST /api/users/create

Parameters:
- username: string (required)
- password: string (required)
- email: string (optional)

Authentication: None
Rate Limiting: None
"""

    print("Testing API endpoint for vulnerabilities...")
    print("MAKER voting: ENABLED")
    print("MDAP decomposition: DISABLED")

    # Run with MAKER voting only
    result = run_maker_enhanced_adversarial_testing(
        content=sample_content,
        content_type="api_spec",
        enable_maker_voting=True,
        enable_mdap_decomposition=False,
        coevolution_rounds=2,
        k_ahead=3
    )

    print("\n[OK] Testing completed!")
    print(f"  - Attacks found: {len(result.get('final_attacks', []))}")
    print(f"  - Voting threshold: k=3")

    return result


def demo_3_mdap_decomposition_only():
    """Demo 3: MDAP decomposition without MAKER"""
    print_section("DEMO 3: MDAP Decomposition (Blue Team Only)")

    from adversarial import run_maker_enhanced_adversarial_testing

    sample_attacks = [
        "SQL Injection in login form",
        "XSS vulnerability in search",
        "CSRF in password change"
    ]

    sample_content = f"""
Known vulnerabilities:
{chr(10).join(f'- {attack}' for attack in sample_attacks)}

Generate defense strategies for each.
"""

    print("Generating defense strategies...")
    print("MAKER voting: DISABLED")
    print("MDAP decomposition: ENABLED")

    # Run with MDAP only
    result = run_maker_enhanced_adversarial_testing(
        content=sample_content,
        content_type="document_general",
        enable_maker_voting=False,
        enable_mdap_decomposition=True,
        coevolution_rounds=1
    )

    print("\n[OK] Defense generation completed!")
    print(f"  - Defenses generated: {len(result.get('final_defenses', []))}")

    return result


def demo_4_coevolution():
    """Demo 4: Full co-evolution with multiple rounds"""
    print_section("DEMO 4: Co-Evolutionary Adversarial Testing")

    from adversarial import run_maker_enhanced_adversarial_testing

    sample_content = """
function processPayment(user, amount, cardDetails):
    # Process payment
    transaction = PaymentGateway.charge(cardDetails, amount)

    if transaction.success:
        user.addFunds(amount)
        return {"status": "success"}

    return {"status": "failed"}
"""

    print("Running full co-evolutionary testing...")
    print("Rounds: 5")
    print("Voting threshold: k=3")

    # Run full co-evolution
    result = run_maker_enhanced_adversarial_testing(
        content=sample_content,
        content_type="code",
        coevolution_rounds=5,
        k_ahead=3
    )

    # Show evolution history
    evolution_history = result.get('evolution_history', [])
    if evolution_history:
        print("\n[Evolution History]")
        for round_data in evolution_history:
            print(f"  Round {round_data.get('round', '?')}:")
            print(f"    - Attacks: {round_data.get('num_attacks', 0)}")
            print(f"    - Defenses: {round_data.get('num_defenses', 0)}")
            print(f"    - Effectiveness: {round_data.get('effectiveness', 0):.2%}")

    return result


def demo_5_varying_k_values():
    """Demo 5: Compare different voting thresholds"""
    print_section("DEMO 5: Voting Threshold Comparison")

    from adversarial import run_maker_enhanced_adversarial_testing

    sample_content = """
# Authentication System
class AuthSystem:
    def login(self, username, password):
        user = self.db.query(f"SELECT * FROM users WHERE username='{username}'")
        if user and user.password == password:
            return self.generate_token(username)
        return None
"""

    k_values = [2, 3, 5]
    results = []

    print("Testing with different voting thresholds (k values)...")

    for k in k_values:
        print(f"\n  Testing with k={k}...")

        result = run_maker_enhanced_adversarial_testing(
            content=sample_content,
            content_type="code",
            coevolution_rounds=2,
            k_ahead=k
        )

        num_attacks = len(result.get('final_attacks', []))
        num_defenses = len(result.get('final_defenses', []))

        results.append({
            "k": k,
            "attacks": num_attacks,
            "defenses": num_defenses
        })

        print(f"    Attacks: {num_attacks}, Defenses: {num_defenses}")

    print("\n[Summary]")
    print("  k   | Attacks | Defenses")
    print("  ----|---------|----------")
    for r in results:
        print(f"  {r['k']:>3} | {r['attacks']:>7} | {r['defenses']:>9}")

    return results


def demo_6_capabilities():
    """Demo 6: Check MAKER/MDAP adversarial capabilities"""
    print_section("DEMO 6: MAKER/MDAP Capabilities Check")

    from adversarial import get_maker_adversarial_capabilities

    capabilities = get_maker_adversarial_capabilities()

    print("MAKER/MDAP Adversarial Capabilities:")
    print(f"  - MAKER enabled: {capabilities.get('maker_enabled', False)}")
    print(f"  - MDAP enabled: {capabilities.get('mdap_enabled', False)}")
    print(f"  - Integration status: {capabilities.get('integration_status', 'unknown')}")

    print("\n  Adversarial Modes:")
    for mode in capabilities.get('modes', []):
        print(f"    - {mode}")

    print("\n  Algorithms from Paper:")
    for algo in capabilities.get('algorithms', []):
        print(f"    - {algo}")

    if 'paper' in capabilities:
        paper = capabilities['paper']
        print(f"\n  Paper Reference:")
        print(f"    - Title: {paper.get('title', 'N/A')}")
        print(f"    - arXiv: {paper.get('arxiv', 'N/A')}")
        print(f"    - URL: {paper.get('url', 'N/A')}")

    return capabilities


def main():
    """Run all demos."""
    print("\n")
    print("=" * 80)
    print("  MAKER/MDAP-ENHANCED ADVERSARIAL TESTING - DEMONSTRATION")
    print("  Paper: arXiv:2511.09030 (Solving a Million-Step LLM Task with Zero Errors)")
    print("=" * 80)
    print("")

    demos = [
        ("Basic MAKER-Enhanced Testing", demo_1_basic_maker_adversarial),
        ("MAKER Voting Only", demo_2_maker_voting_only),
        ("MDAP Decomposition Only", demo_3_mdap_decomposition_only),
        ("Full Co-Evolution", demo_4_coevolution),
        ("Voting Threshold Comparison", demo_5_varying_k_values),
        ("Capabilities Check", demo_6_capabilities),
    ]

    print("Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all demos")
    print("")

    try:
        choice = input("Select demo (0-6, or press Enter for all): ").strip()
        if not choice:
            choice = "0"

        choice_num = int(choice)

        if choice_num == 0:
            # Run all demos
            for name, demo_func in demos:
                try:
                    demo_func()
                except Exception as e:
                    logger.error(f"Demo {name} failed: {e}", exc_info=True)
        elif 1 <= choice_num <= len(demos):
            # Run selected demo
            name, demo_func = demos[choice_num - 1]
            demo_func()
        else:
            print("Invalid choice")

    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    print("\n" + "=" * 80)
    print("  DEMO COMPLETED")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - adversarial_maker_integration.py")
    print("  - MAKER_ADVERSARIAL_INTEGRATION_GUIDE.md")
    print("  - Paper: https://arxiv.org/abs/2511.09030")
    print("")


if __name__ == "__main__":
    main()
