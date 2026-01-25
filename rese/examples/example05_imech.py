"""
Example 5: Isomorphism Validation (I_mech)

This example demonstrates how to use I_mech to validate mechanistic similarity
between domains for knowledge transfer.
"""

import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

from phase2.imech import IMechValidator, Domain

def main():
    print("=" * 60)
    print("Example 5: Isomorphism Validation (I_mech)")
    print("=" * 60)
    print()

    # Create source domain (TSP - Traveling Salesman Problem)
    print("Creating Source Domain (TSP):")
    print("-" * 60)

    source_domain = Domain(
        id="tsp",
        name="Traveling Salesman Problem",
        variables={
            "num_cities": 10,
            "coordinates": [(i, i*2) for i in range(10)]
        },
        constraints=[
            {
                "id": "visit_all",
                "type": "hard",
                "description": "Visit all cities exactly once",
                "variables": ["cities", "route"]
            },
            {
                "id": "minimize_distance",
                "type": "soft",
                "description": "Minimize total distance",
                "variables": ["route", "distances"]
            }
        ]
    )

    print(f"Domain: {source_domain.name}")
    print(f"Variables: {list(source_domain.variables.keys())}")
    print(f"Constraints: {len(source_domain.constraints)}")
    print()

    # Create target domain 1 (VRP - Vehicle Routing Problem)
    print("Creating Target Domain 1 (VRP - Similar to TSP):")
    print("-" * 60)

    target_domain_similar = Domain(
        id="vrp",
        name="Vehicle Routing Problem",
        variables={
            "num_vehicles": 3,
            "num_customers": 10,
            "coordinates": [(i, i*2) for i in range(10)]
        },
        constraints=[
            {
                "id": "visit_all_customers",
                "type": "hard",
                "description": "Visit all customers exactly once",
                "variables": ["customers", "routes"]
            },
            {
                "id": "minimize_distance",
                "type": "soft",
                "description": "Minimize total distance across all vehicles",
                "variables": ["routes", "distances"]
            }
        ]
    )

    print(f"Domain: {target_domain_similar.name}")
    print(f"Variables: {list(target_domain_similar.variables.keys())}")
    print(f"Constraints: {len(target_domain_similar.constraints)}")
    print()

    # Create target domain 2 (Knapsack - Different from TSP)
    print("Creating Target Domain 2 (Knapsack - Different from TSP):")
    print("-" * 60)

    target_domain_different = Domain(
        id="knapsack",
        name="Knapsack Problem",
        variables={
            "num_items": 10,
            "capacity": 100,
            "weights": [10, 20, 15, 25, 30, 12, 18, 22, 28, 16],
            "values": [60, 100, 120, 80, 90, 70, 110, 85, 95, 75]
        },
        constraints=[
            {
                "id": "capacity_constraint",
                "type": "hard",
                "description": "Total weight must not exceed capacity",
                "variables": ["items", "capacity"]
            },
            {
                "id": "maximize_value",
                "type": "soft",
                "description": "Maximize total value",
                "variables": ["items", "values"]
            }
        ]
    )

    print(f"Domain: {target_domain_different.name}")
    print(f"Variables: {list(target_domain_different.variables.keys())}")
    print(f"Constraints: {len(target_domain_different.constraints)}")
    print()

    # Compare domains
    print("=" * 60)
    print("Comparing Domains:")
    print("=" * 60)
    print()

    validator = IMechValidator()

    # Comparison 1: TSP vs VRP (similar)
    print("Comparison 1: TSP → VRP")
    print("-" * 60)
    result_similar = validator.compare_domains(source_domain, target_domain_similar)

    print(f"Similarity Score: {result_similar.score:.2f}")
    print(f"Confidence: {result_similar.confidence:.2f}")
    print(f"Transfer Recommended: {result_similar.transfer_recommended}")
    print()

    if result_similar.shared_structure:
        print("Shared Structure:")
        for element in result_similar.shared_structure:
            print(f"  - {element}")
    print()

    if result_similar.differences:
        print("Key Differences:")
        for diff in result_similar.differences:
            print(f"  - {diff}")
    print()

    # Comparison 2: TSP vs Knapsack (different)
    print("Comparison 2: TSP → Knapsack")
    print("-" * 60)
    result_different = validator.compare_domains(source_domain, target_domain_different)

    print(f"Similarity Score: {result_different.score:.2f}")
    print(f"Confidence: {result_different.confidence:.2f}")
    print(f"Transfer Recommended: {result_different.transfer_recommended}")
    print()

    # Analysis
    print("=" * 60)
    print("Analysis:")
    print("-" * 60)

    if result_similar.score > result_different.score:
        print("✓ As expected, TSP and VRP are more similar than TSP and Knapsack")
        print(f"  TSP-VRP similarity: {result_similar.score:.2f}")
        print(f"  TSP-Knapsack similarity: {result_different.score:.2f}")
    print()

    # Knowledge transfer example
    if result_similar.transfer_recommended:
        print("=" * 60)
        print("Knowledge Transfer Example:")
        print("-" * 60)
        print("Transferring knowledge from TSP to VRP...")

        transferred = validator.transfer_knowledge(
            source_domain,
            target_domain_similar
        )

        print(f"Transferred {len(transferred)} constraints/solutions")
        for i, item in enumerate(transferred, 1):
            print(f"  {i}. {item}")
    print()

    print("=" * 60)
    print("Example 5 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
