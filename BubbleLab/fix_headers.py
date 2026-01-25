#!/usr/bin/env python3
"""
Fix header comment formatting in all files
"""

import re
from pathlib import Path

def fix_header(file_path: Path) -> bool:
    """Fix header formatting"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Fix the duplicate header comment pattern
        # Pattern: */ followed by * Security Fixes (should be just one)
        # We need to remove the duplicate */ and * before Security Fixes

        # First, let's fix the specific pattern: */\n *\n * Security Fixes
        content = re.sub(
            r'\*/\n \*\n \* Security Fixes Applied \(Wave 5\):',
            r' *\n * Security Fixes Applied (Wave 5):',
            content
        )

        # Also fix: */\n * Security Fixes Applied (no newline)
        content = re.sub(
            r'\*/\n \* Security Fixes Applied \(Wave 5\):',
            r' *\n * Security Fixes Applied (Wave 5):',
            content
        )

        # Remove extra blank lines after security notice
        content = re.sub(
            r' \* URL validation for all endpoints\n \*\n\n\n',
            r' * URL validation for all endpoints\n */',
            content
        )

        # Fix trailing blank lines before imports
        content = re.sub(r'\n\n\nimport', '\n\nimport', content)

        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"  [ERROR] {file_path.name}: {e}")
        return False

def main():
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("Fixing Header Formatting")
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

    total = 0
    fixed = 0

    for category, files in categories.items():
        print(f"[{category.upper()}]")
        category_dir = base_dir / "examples" / category

        for filename in files:
            file_path = category_dir / filename
            if file_path.exists():
                total += 1
                if fix_header(file_path):
                    fixed += 1
                    print(f"  [FIXED] {filename}")
                else:
                    print(f"  [OK] {filename}")

    print()
    print("=" * 80)
    print(f"Processed {total} files, fixed {fixed} files")
    print()

if __name__ == "__main__":
    main()
