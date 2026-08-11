#!/usr/bin/env python3
"""
SOP Facet Evolution CLI - Evolve Specific Parts of SOP

Usage:
    # Evolve entire SOP
    python evolve_sop_facets.py --input SOP.txt --output SOP_v16.2.txt

    # Evolve specific facets only
    python evolve_sop_facets.py --input SOP.txt --output SOP_v16.2.txt --facets environmental equipment

    # Evolve single facet
    python evolve_sop_facets.py --input SOP.txt --output SOP_v16.2.txt --facet safety

    # List available facets
    python evolve_sop_facets.py --list-facets
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import template system
from sop_templates import (
    SOPTemplateRegistry,
    SOPFacet,
    evolve_environmental_conditions,
    evolve_equipment_specifications,
    evolve_materials,
    evolve_execution_protocols,
    evolve_quality_control,
    evolve_safety_protocols,
    evolve_validation_scalability
)


FACET_MAPPING = {
    "environmental": SOPFacet.ENVIRONMENTAL,
    "equipment": SOPFacet.EQUIPMENT,
    "materials": SOPFacet.MATERIALS,
    "execution": SOPFacet.EXECUTION_PHASES,
    "quality": SOPFacet.QUALITY_CONTROL,
    "safety": SOPFacet.SAFETY,
    "validation": SOPFacet.VALIDATION
}

FACET_DESCRIPTIONS = {
    "environmental": "Part 0: Environmental Conditions (temperature, humidity, pressure, vibration)",
    "equipment": "Part 1: Equipment Specifications (magnetic field, UV curing, thermal stage)",
    "materials": "Part 2: Materials (resins, nanoclusters, liquid crystals)",
    "execution": "Part 3: Execution Protocols (4 phases of assembly)",
    "quality": "Part 4: Quality Control (acceptance criteria, documentation)",
    "safety": "Part 5: Safety Protocols (emergency procedures, PPE, training)",
    "validation": "Part 6: Validation and Scalability (scaling laws, batch specifications)"
}


def list_facets():
    """Print available facets"""
    print("\nAvailable SOP Facets:\n")
    for key, description in FACET_DESCRIPTIONS.items():
        print(f"  {key:15s} - {description}")
    print()


def parse_facets(facet_args):
    """Parse facet arguments from CLI"""
    if not facet_args:
        return None  # Evolve all facets

    facets = []
    for facet_arg in facet_args:
        facet_lower = facet_arg.lower()
        if facet_lower in FACET_MAPPING:
            facets.append(FACET_MAPPING[facet_lower])
        else:
            print(f"Warning: Unknown facet '{facet_arg}', skipping...")
            print(f"Use --list-facets to see available facets")

    return facets if facets else None


def evolve_sop_with_facets(
    input_path: str,
    output_path: str,
    facets: list,
    api_key: str,
    num_models: int = 7,
    save_metadata: bool = True
):
    """
    Evolve SOP with specified facets

    Args:
        input_path: Path to input SOP
        output_path: Path to output evolved SOP
        facets: List of facets to evolve (None = all)
        api_key: OpenAI API key
        num_models: Number of ensemble models
        save_metadata: Save evolution metadata to JSON
    """
    print(f"\n{'='*70}")
    print("SOP FACET EVOLUTION")
    print(f"{'='*70}\n")

    # Read input SOP
    print(f"Input SOP: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        sop_content = f.read()

    print(f"SOP length: {len(sop_content)} characters")

    if facets:
        print(f"\nEvolving {len(facets)} specific facets:")
        for facet in facets:
            print(f"  - {facet.value}")
    else:
        print(f"\nEvolving all facets (7 total)")

    print(f"Ensemble size: {num_models} models\n")

    # Create registry and evolve
    registry = SOPTemplateRegistry(api_key)

    try:
        results = registry.evolve_entire_sop(
            sop_content=sop_content,
            facets_to_evolve=facets,
            num_models=num_models
        )

        # Save evolved SOP
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results["evolved_sop"])

        print(f"\n[OK] Evolved SOP saved: {output_path}")

        # Save metadata if requested
        if save_metadata:
            metadata_path = output_path + ".facet_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"[OK] Metadata saved: {metadata_path}")

        # Print summary
        print(f"\n{'='*70}")
        print("EVOLUTION SUMMARY")
        print(f"{'='*70}\n")

        print(f"Status: {results['overall_status']}")
        print(f"Total vulnerabilities found: {results['total_vulnerabilities_found']}")
        print(f"Total fixes applied: {results['total_fixes_applied']}")

        print(f"\nFacet Results:")
        for facet_name, facet_result in results["facets"].items():
            status = facet_result.get("status", "UNKNOWN")
            if status == "EVOLVED":
                vulns = facet_result.get("vulnerabilities_found", 0)
                fixes = facet_result.get("fixes_applied", 0)
                score = facet_result.get("quality_score", 0)
                print(f"  {facet_name}:")
                print(f"    Vulnerabilities: {vulns}")
                print(f"    Fixes: {fixes}")
                print(f"    Quality Score: {score:.3f}")
            elif status == "NO_VULNERABILITIES":
                print(f"  {facet_name}: No vulnerabilities found [OK]")
            elif status == "ERROR":
                error = facet_result.get("error", "Unknown error")
                print(f"  {facet_name}: ERROR - {error}")

        return results

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"\n[FAIL] Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evolve_single_facet(
    input_path: str,
    output_path: str,
    facet_name: str,
    api_key: str,
    num_models: int = 7
):
    """
    Evolve a single facet of the SOP

    Args:
        input_path: Path to input SOP
        output_path: Path to output (will contain only evolved facet)
        facet_name: Name of facet to evolve
        api_key: OpenAI API key
        num_models: Number of ensemble models
    """
    facet_lower = facet_name.lower()
    if facet_lower not in FACET_MAPPING:
        print(f"Error: Unknown facet '{facet_name}'")
        print("Use --list-facets to see available facets")
        return None

    facet = FACET_MAPPING[facet_lower]

    print(f"\n{'='*70}")
    print(f"SINGLE FACET EVOLUTION: {facet.value}")
    print(f"{'='*70}\n")

    # Read input SOP
    with open(input_path, 'r', encoding='utf-8') as f:
        sop_content = f.read()

    # Evolve facet
    registry = SOPTemplateRegistry(api_key)

    try:
        result = registry.evolve_facet(
            sop_content=sop_content,
            facet=facet,
            num_models=num_models
        )

        # Save evolved facet
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result["evolved_content"])

        print(f"\n[OK] Evolved facet saved: {output_path}")

        # Save metadata
        metadata_path = output_path + ".metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[OK] Metadata saved: {metadata_path}")

        # Print summary
        print(f"\n{'='*70}")
        print("FACET EVOLUTION SUMMARY")
        print(f"{'='*70}\n")

        print(f"Facet: {result['facet']}")
        print(f"Status: {result['status']}")
        print(f"Vulnerabilities Found: {result['vulnerabilities_found']}")
        print(f"Fixes Applied: {result['fixes_applied']}")
        print(f"Quality Score: {result['quality_score']:.3f}")
        print(f"Consensus Reached: {result['consensus_reached']}")
        print(f"All Validators Passed: {result['all_validators_passed']}")

        print(f"\nValidation Results:")
        for vr in result["validation_results"]:
            status = "[OK]" if vr["passed"] else "[FAIL]"
            print(f"  {status} {vr['validator']}")

        return result

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"\n[FAIL] Facet evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Evolve specific facets of a Standard Operating Procedure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evolve entire SOP
  python evolve_sop_facets.py --input SOP.txt --output SOP_v16.2.txt

  # Evolve specific facets only
  python evolve_sop_facets.py --input SOP.txt --output SOP_v16.2.txt --facets environmental equipment

  # Evolve single facet (save just that facet)
  python evolve_sop_facets.py --input SOP.txt --output facet_part0.txt --facet environmental --single

  # List available facets
  python evolve_sop_facets.py --list-facets

Available Facets:
  environmental  - Part 0: Environmental Conditions
  equipment      - Part 1: Equipment Specifications
  materials      - Part 2: Materials
  execution      - Part 3: Execution Protocols
  quality        - Part 4: Quality Control
  safety         - Part 5: Safety Protocols
  validation     - Part 6: Validation and Scalability
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input SOP file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output evolved SOP (or evolved facet if --single)"
    )

    parser.add_argument(
        "--facets",
        nargs="+",
        choices=list(FACET_MAPPING.keys()),
        help="Specific facets to evolve (default: all)"
    )

    parser.add_argument(
        "--facet",
        choices=list(FACET_MAPPING.keys()),
        help="Single facet to evolve (use with --single)"
    )

    parser.add_argument(
        "--single",
        action="store_true",
        help="Evolve single facet and save just that facet (not entire SOP)"
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=7,
        help="Number of ensemble models (default: 7)"
    )

    parser.add_argument(
        "--list-facets",
        action="store_true",
        help="List available facets and exit"
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save metadata JSON file"
    )

    args = parser.parse_args()

    # Handle --list-facets
    if args.list_facets:
        list_facets()
        return 0

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1

    # Handle --single facet mode
    if args.single:
        if not args.facet:
            print("Error: --facet required when using --single")
            return 1

        result = evolve_single_facet(
            input_path=args.input,
            output_path=args.output,
            facet_name=args.facet,
            api_key=api_key,
            num_models=args.ensemble_size
        )

        return 0 if result else 1

    # Handle regular multi-facet mode
    if args.facet:
        print("Warning: --facet is ignored without --single, use --facets instead")

    facets = parse_facets(args.facets)

    result = evolve_sop_with_facets(
        input_path=args.input,
        output_path=args.output,
        facets=facets,
        api_key=api_key,
        num_models=args.ensemble_size,
        save_metadata=not args.no_metadata
    )

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
