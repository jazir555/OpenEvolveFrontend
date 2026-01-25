#!/usr/bin/env python3
"""
Batch fix all security-hardened files
"""

import re
from pathlib import Path

def fix_security_file(file_path: Path) -> bool:
    """Fix a single security-hardened file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix 1: Correct the header comment format
        # Pattern: */ followed by * Security Fixes (missing opening /*)
        content = re.sub(
            r'\*/\n \* Security Fixes Applied \(Wave 5\):',
            r'*/\n *\n * Security Fixes Applied (Wave 5):',
            content
        )

        # Fix 2: Move environment validation to correct position (after imports)
        # First, extract any env validation that exists
        env_validation_pattern = r'// Security: Environment variable validation at startup\nvalidateEnvironment\(\{[^}]+\}\);'
        env_validation_match = re.search(env_validation_pattern, content, re.DOTALL)

        if env_validation_match:
            env_validation = env_validation_match.group(0)

            # Remove all instances of env validation
            content = re.sub(env_validation_pattern, '', content, flags=re.DOTALL)

            # Find where to insert it (after security imports)
            security_import_pattern = r"(from '\.\./\.\./templates/security-utils';)"
            security_import_match = re.search(security_import_pattern, content)

            if security_import_match:
                insert_pos = security_import_match.end()
                content = content[:insert_pos] + '\n\n' + env_validation + '\n' + content[insert_pos:]

        # Fix 3: Remove extra blank lines
        content = re.sub(r'\n\n\n+', '\n\n\n', content)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"  [ERROR] {file_path.name}: {e}")
        return False

def main():
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("Batch Fix Security-Hardened Files")
    print("=" * 80)
    print()

    categories = {
        "infrastructure-automation": [
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
                if fix_security_file(file_path):
                    fixed += 1
                    print(f"  [OK] {filename}")

    print()
    print(f"Fixed {fixed}/{total} files")

if __name__ == "__main__":
    main()
