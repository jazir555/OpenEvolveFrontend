"""
Adversarial System Migration Utility

This utility helps migrate from the old adversarial testing system to the enhanced version.

Features:
1. Compatibility check - Verify old system works with new
2. Gradual migration - Migrate incrementally
3. Result conversion - Convert old results to new format
4. Configuration mapping - Map old configs to new
5. Validation - Test new system against old

Usage:
    python migrate_adversarial.py --check
    python migrate_adversarial.py --migrate
    python migrate_adversarial.py --validate
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# IMPORT CHECKS
# =============================================================================

def check_imports() -> Dict[str, bool]:
    """Check which modules are available"""
    available = {}

    # Check old adversarial
    try:
        from adversarial import (
            run_comprehensive_adversarial_testing,
            AdversarialConfig,
            AttackResult,
            DefenseResult
        )
        available["old_adversarial"] = True
        logger.info("✓ Old adversarial module available")
    except ImportError as e:
        available["old_adversarial"] = False
        logger.warning(f"✗ Old adversarial module not available: {e}")

    # Check new adversarial
    try:
        from adversarial_advanced import (
            EnhancedAdversarialEngine,
            AdvancedAdversarialConfig,
            create_enhanced_config,
            quick_enhanced_test
        )
        available["new_adversarial"] = True
        logger.info("✓ New enhanced adversarial module available")
    except ImportError as e:
        available["new_adversarial"] = False
        logger.warning(f"✗ New enhanced adversarial module not available: {e}")

    # Check unified adversarial
    try:
        from adversarial_unified import (
            AdversarialEngine,
            UnifiedAdversarialConfig
        )
        available["unified_adversarial"] = True
        logger.info("✓ Unified adversarial module available")
    except ImportError as e:
        available["unified_adversarial"] = False
        logger.warning(f"✗ Unified adversarial module not available: {e}")

    return available


# =============================================================================
# CONFIGURATION MAPPING
# =============================================================================

def map_old_config_to_new(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map old configuration to new configuration

    Args:
        old_config: Old configuration dictionary

    Returns:
        New configuration dictionary
    """
    new_config = {}

    # Basic mappings
    if "adversarial_rounds" in old_config:
        new_config["max_iterations"] = old_config["adversarial_rounds"]

    if "temperature" in old_config:
        new_config["llm_attack_temperature"] = old_config["temperature"]

    if "max_tokens" in old_config:
        new_config["llm_attack_max_tokens"] = old_config["max_tokens"]

    # Team size mappings
    if "red_team_size" in old_config:
        new_config["ensemble_size"] = min(old_config["red_team_size"], 7)

    # Strategy mappings
    if "enable_mutation" in old_config and old_config["enable_mutation"]:
        # Mutation is always enabled in new system
        pass

    if "enable_llm_attacks" in old_config:
        new_config["enable_llm_attacks"] = old_config["enable_llm_attacks"]

    # Feature flags
    if "enable_caching" in old_config:
        new_config["enable_caching"] = old_config["enable_caching"]

    if "enable_analytics" in old_config:
        new_config["enable_advanced_analytics"] = old_config["enable_analytics"]

    # New features (set to defaults if not specified)
    new_config.setdefault("enable_adaptive_defense", True)
    new_config.setdefault("explainability_level", "detailed")
    new_config.setdefault("learning_mode", "online")
    new_config.setdefault("enable_ensemble", True)

    return new_config


def create_new_config_from_old(old_config_path: str) -> Optional[Dict[str, Any]]:
    """
    Create new configuration from old configuration file

    Args:
        old_config_path: Path to old configuration file (JSON)

    Returns:
        New configuration dictionary
    """
    try:
        with open(old_config_path, 'r') as f:
            old_config = json.load(f)

        new_config = map_old_config_to_new(old_config)

        logger.info(f"Migrated configuration from {old_config_path}")
        return new_config

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {old_config_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        return None
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Error migrating configuration: {e}")
        return None


# =============================================================================
# RESULT CONVERSION
# =============================================================================

def convert_old_result_to_new(old_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert old result format to new result format

    Args:
        old_result: Old result dictionary

    Returns:
        New result dictionary
    """
    new_result = {
        "success": old_result.get("success", False),
        "content": old_result.get("content", ""),
        "final_content": old_result.get("final_content", old_result.get("content", "")),
        "content_type": old_result.get("content_type", "unknown"),
        "theorem": old_result.get("theorem", ""),
        "iterations_completed": old_result.get("iterations_completed", old_result.get("total_rounds", 0)),
        "final_robustness": old_result.get("final_robustness", old_result.get("robustness", 0.0)),
        "duration": old_result.get("duration", 0.0),

        # Attacks
        "attacks": old_result.get("attacks", []),
        "metrics": old_result.get("metrics", {}),

        # New fields (defaults)
        "defenses": old_result.get("defenses", []),
        "explanations": [],
        "adaptations": [],
        "learning_insights": {},
        "explainability_summary": {}
    }

    # Convert metrics if needed
    if "metrics" not in new_result or not new_result["metrics"]:
        new_result["metrics"] = {
            "total_attacks": len(old_result.get("attacks", [])),
            "successful_attacks": sum(1 for a in old_result.get("attacks", []) if a.get("success", False)),
            "total_defenses": len(old_result.get("defenses", [])),
            "successful_defenses": sum(1 for d in old_result.get("defenses", []) if d.get("attack_blocked", False)),
        }

    return new_result


# =============================================================================
# VALIDATION
# =============================================================================

async def validate_migration(
    sample_content: str,
    content_type: str = "code_python",
    theorem: str = "Test theorem"
) -> Dict[str, Any]:
    """
    Validate migration by running both old and new systems

    Args:
        sample_content: Sample content to test
        content_type: Type of content
        theorem: Theorem statement

    Returns:
        Validation results
    """
    validation = {
        "old_system": {"available": False, "result": None, "error": None},
        "new_system": {"available": False, "result": None, "error": None},
        "comparison": None
    }

    # Test old system
    try:
        from adversarial import run_comprehensive_adversarial_testing

        old_result = await asyncio.to_thread(
            run_comprehensive_adversarial_testing,
            content=sample_content,
            content_type=content_type,
            red_team_models=["gpt-4"],
            blue_team_models=["gpt-4"],
            evaluator_models=["gpt-4"],
            api_key="test",
            max_iterations=3
        )

        validation["old_system"] = {
            "available": True,
            "result": old_result,
            "error": None
        }

        logger.info("✓ Old system executed successfully")

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        validation["old_system"]["error"] = str(e)
        logger.warning(f"Old system failed: {e}")

    # Test new system
    try:
        from adversarial_advanced import EnhancedAdversarialEngine, create_enhanced_config

        config = create_enhanced_config(
            max_iterations=3,
            enable_llm_attacks=False,  # Disable for validation (no API key)
            ensemble_size=3
        )

        engine = EnhancedAdversarialEngine(config)

        new_result = await engine.enhanced_adversarial_test(
            content=sample_content,
            content_type=content_type,
            theorem=theorem,
            max_iterations=3
        )

        validation["new_system"] = {
            "available": True,
            "result": new_result,
            "error": None
        }

        logger.info("✓ New system executed successfully")

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        validation["new_system"]["error"] = str(e)
        logger.warning(f"New system failed: {e}")

    # Compare results if both succeeded
    if validation["old_system"]["available"] and validation["new_system"]["available"]:
        old_robustness = validation["old_system"]["result"].get("final_robustness", 0)
        new_robustness = validation["new_system"]["result"].get("final_robustness", 0)

        validation["comparison"] = {
            "old_robustness": old_robustness,
            "new_robustness": new_robustness,
            "robustness_difference": new_robustness - old_robustness,
            "improvement": new_robustness > old_robustness
        }

        logger.info(f"Robustness comparison: {old_robustness:.2%} (old) vs {new_robustness:.2%} (new)")

    return validation


# =============================================================================
# MIGRATION STEPS
# =============================================================================

def step_1_backup_old_code() -> bool:
    """Step 1: Backup old adversarial code"""
    print("\n[Step 1] Backing up old adversarial code...")

    adversarial_files = [
        "adversarial.py",
        "adversarial_unified.py",
        "adversarial_testing.py"
    ]

    backup_dir = Path("./adversarial_backup")

    try:
        backup_dir.mkdir(exist_ok=True)

        for filename in adversarial_files:
            src = Path(filename)
            if src.exists():
                dst = backup_dir / filename
                import shutil
                shutil.copy2(src, dst)
                print(f"  ✓ Backed up {filename}")

        print(f"\n✓ Backup completed: {backup_dir}")
        return True

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"  ✗ Backup failed: {e}")
        return False


def step_2_check_compatibility() -> Dict[str, bool]:
    """Step 2: Check compatibility"""
    print("\n[Step 2] Checking compatibility...")

    available = check_imports()

    print("\nCompatibility Summary:")
    for module, status in available.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {module}: {'Available' if status else 'Not Available'}")

    if not available.get("new_adversarial"):
        print("\n✗ Enhanced adversarial system not found!")
        print("  Please ensure adversarial_advanced.py is installed.")

    return available


def step_3_create_new_config(old_config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Step 3: Create new configuration"""
    print("\n[Step 3] Creating new configuration...")

    if old_config_path:
        print(f"  Migrating from: {old_config_path}")
        new_config = create_new_config_from_old(old_config_path)
    else:
        print("  Creating default configuration")
        from adversarial_advanced import create_enhanced_config
        new_config = create_enhanced_config(enable_advanced_features=True).to_dict()

    if new_config:
        print("  ✓ Configuration created")
        return new_config
    else:
        print("  ✗ Configuration creation failed")
        return None


def step_4_validate_systems() -> Dict[str, Any]:
    """Step 4: Validate both systems"""
    print("\n[Step 4] Validating systems...")

    sample_content = """
    def example_function(x, y):
        return x + y
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        validation = loop.run_until_complete(
            validate_migration(sample_content, "code_python", "Example theorem")
        )

        print("\nValidation Results:")
        if validation["old_system"]["available"]:
            print("  ✓ Old system: Working")
        else:
            print(f"  ✗ Old system: {validation['old_system']['error']}")

        if validation["new_system"]["available"]:
            print("  ✓ New system: Working")
        else:
            print(f"  ✗ New system: {validation['new_system']['error']}")

        if validation["comparison"]:
            comp = validation["comparison"]
            print(f"\n  Robustness Comparison:")
            print(f"    Old: {comp['old_robustness']:.2%}")
            print(f"    New: {comp['new_robustness']:.2%}")
            print(f"    Difference: {comp['robustness_difference']:+.2%}")
            print(f"    Improvement: {'Yes' if comp['improvement'] else 'No'}")

        return validation

    finally:
        loop.close()


def step_5_generate_migration_guide(config: Dict[str, Any], output_path: str = "./MIGRATION_GUIDE.md"):
    """Step 5: Generate migration guide"""
    print(f"\n[Step 5] Generating migration guide...")

    guide = f"""# Adversarial System Migration Guide

## Overview
This guide helps you migrate from the old adversarial testing system to the enhanced version.

## Migration Date
{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Configuration

### Old Configuration
Your old configuration has been mapped to the new system.

### New Configuration
```json
{json.dumps(config, indent=2)}
```

## API Changes

### Old Way
```python
from adversarial import run_comprehensive_adversarial_testing

result = run_comprehensive_adversarial_testing(
    content=content,
    content_type=content_type,
    red_team_models=["gpt-4"],
    blue_team_models=["gpt-4"],
    evaluator_models=["gpt-4"],
    api_key=api_key,
    max_iterations=10
)
```

### New Way
```python
from adversarial_advanced import EnhancedAdversarialEngine, create_enhanced_config

config = create_enhanced_config(**{json.dumps(config, indent=4)})
engine = EnhancedAdversarialEngine(config)

result = await engine.enhanced_adversarial_test(
    content=content,
    content_type=content_type,
    theorem=theorem,
    max_iterations=10
)
```

## New Features Available

1. **AI-Driven Attack Generation**: {config.get('enable_llm_attacks', False)}
2. **Adaptive Defense**: {config.get('enable_adaptive_defense', False)}
3. **Explainability**: {config.get('explainability_level', 'basic')}
4. **Continuous Learning**: {config.get('learning_mode', 'offline')}
5. **Ensemble Attacks**: {config.get('enable_ensemble', False)}
6. **Advanced Analytics**: {config.get('enable_advanced_analytics', False)}

## Result Structure Changes

The new result structure includes additional fields:
- `defenses`: Defense strategies applied
- `explanations`: Explainability data
- `adaptations`: Adaptive defense adjustments
- `learning_insights`: Learning system analysis
- `explainability_summary`: Summary of explanations

See `ENHANCED_ADVERSARIAL_GUIDE.md` for complete documentation.

## Next Steps

1. Review the new configuration above
2. Test with sample content
3. Update your code to use new API
4. Enable additional features as needed
5. Monitor improvements in robustness scores

## Support

For questions or issues, refer to:
- ENHANCED_ADVERSARIAL_GUIDE.md
- adversarial_advanced.py
- demo_enhanced_adversarial.py
"""

    try:
        with open(output_path, 'w') as f:
            f.write(guide)

        print(f"  ✓ Migration guide generated: {output_path}")
        return True

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"  ✗ Failed to generate guide: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate to enhanced adversarial testing system"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check system compatibility"
    )

    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run full migration"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate migration"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to old configuration file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./MIGRATION_GUIDE.md",
        help="Output path for migration guide"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  ADVERSARIAL SYSTEM MIGRATION UTILITY")
    print("=" * 80)

    if args.check:
        # Just check compatibility
        step_2_check_compatibility()

    elif args.validate:
        # Validate systems
        step_4_validate_systems()

    elif args.migrate:
        # Full migration
        print("\nStarting migration process...\n")

        # Step 1: Backup
        if not step_1_backup_old_code():
            print("\n✗ Migration failed: Backup step failed")
            return

        # Step 2: Compatibility check
        available = step_2_check_compatibility()
        if not available.get("new_adversarial"):
            print("\n✗ Migration failed: New system not available")
            return

        # Step 3: Create new config
        config = step_3_create_new_config(args.config)
        if not config:
            print("\n✗ Migration failed: Configuration creation failed")
            return

        # Step 4: Validate
        validation = step_4_validate_systems()

        # Step 5: Generate guide
        step_5_generate_migration_guide(config, args.output)

        print("\n" + "=" * 80)
        print("  MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review the migration guide: " + args.output)
        print("  2. Test with your content")
        print("  3. Update your code")
        print("  4. Enable additional features as needed")
        print("")

    else:
        parser.print_help()


if __name__ == "__main__":
    import datetime  # Import here for migration guide
    main()
