#!/usr/bin/env python3
"""
Fix formatting issues in security-hardened files
"""

import re
from pathlib import Path

def fix_file_formatting(file_path: Path) -> bool:
    """Fix formatting issues in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Fix 1: Correct the header comment - add missing /* before security fixes
        # Pattern: */ followed by * Security Fixes
        content = re.sub(
            r'\*/\n \* Security Fixes Applied \(Wave 5\):',
            r*/\n *\n * Security Fixes Applied (Wave 5):,
            content
        )

        # Fix 2: Remove duplicate environment validation if it appears before class
        # Pattern: validateEnvironment({ ... }); right before export class
        # We want to keep only one, right after imports

        # First, let's remove the env validation before the class
        content = re.sub(
            r'\n// Security: Environment variable validation at startup\nvalidateEnvironment\(\{[^}]+\}\);\n\n',
            '\n',
            content
        )

        # Now add it properly after the security imports
        # Find the security imports
        security_import_pattern = r"(import \{[^}]+\} from '\.\./\.\./templates/security-utils';)"
        security_import_match = re.search(security_import_pattern, content)

        if security_import_match:
            # Extract the env validation that we need to add
            # Look for it later in the file
            env_validation_pattern = r'// Security: Environment variable validation at startup\nvalidateEnvironment\(\{([^}]+)\}\);'
            env_validation_match = re.search(env_validation_pattern, content, re.DOTALL)

            if env_validation_match:
                env_validation = env_validation_match.group(0)
                # Insert it after security imports
                insert_pos = security_import_match.end()
                content = content[:insert_pos] + '\n\n' + env_validation + content[insert_pos:]

                # Remove the duplicate that might be elsewhere
                content = re.sub(
                    env_validation_pattern,
                    '',
                    content,
                    count=1
                )

        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [FIXED] {file_path.name}")
            return True
        else:
            print(f"  [OK] {file_path.name}")
            return True

    except Exception as e:
        print(f"  [ERROR] {file_path.name}: {e}")
        return False

def main():
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("Fixing Security Formatting Issues")
    print("=" * 80)
    print()

    categories = {
        "infrastructure-automation": [
            "container-autohealing.ts",
            "log-anomaly-detection.ts",
            "database-backup-scheduled.ts",
            "service-scaling-automation.ts",
            "certificate-renewal.ts",
            "health-check-dashboard.ts",
            "resource-cleanup.ts",
            "incident-response.ts",
        ],
        "development-automation": [
            "pr-automation.ts",
            "dependency-update.ts",
            "deployment-pipeline.ts",
            "code-quality-check.ts",
            "documentation-generator.ts",
            "test-orchestration.ts",
            "release-automation.ts",
            "branch-cleanup.ts",
        ],
        "llm-operations": [
            "prompt-testing-suite.ts",
            "model-benchmarking.ts",
            "token-usage-monitor.ts",
            "ai-quality-assessment.ts",
            "model-failover.ts",
            "prompt-optimization.ts",
            "cost-optimization.ts",
            "multi-model-ensemble.ts",
        ],
    }

    fixed_count = 0
    total_count = 0

    for category, files in categories.items():
        print(f"[{category.upper()}]")
        category_dir = base_dir / "examples" / category

        for filename in files:
            file_path = category_dir / filename
            if file_path.exists():
                total_count += 1
                if fix_file_formatting(file_path):
                    fixed_count += 1

    print()
    print("=" * 80)
    print(f"Processed {total_count} files, fixed {fixed_count} files")

if __name__ == "__main__":
    main()
