#!/usr/bin/env python3
"""
Final comprehensive fix for all security-hardened files
"""

import re
from pathlib import Path

def fix_file(file_path: Path) -> bool:
    """Comprehensive fix for a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Fix 1: Remove any duplicate/misplaced security notices in header
        # Pattern to match: */ followed by * Security Fixes (incorrect)
        content = re.sub(
            r'\*/\n \* Security Fixes Applied \(Wave 5\):.*?\*/',
            '*/',
            content,
            flags=re.DOTALL
        )

        # Fix 2: Ensure security notice is properly formatted before imports
        # Check if security notice exists properly
        if 'Security Fixes Applied (Wave 5):' not in content:
            # Add it before the closing */
            security_notice = """ *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting (60 requests/minute)
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints"""
            content = re.sub(
                r'(\* \})\n \*/',
                r'\1' + security_notice + '\n */',
                content
            )

        # Fix 3: Extract and move environment validation to correct position
        env_validation_pattern = r'// Security: Environment variable validation at startup\nvalidateEnvironment\(\{[^}]+\}\);'
        env_validation_match = re.search(env_validation_pattern, content, re.DOTALL)

        if env_validation_match:
            env_validation = env_validation_match.group(0)

            # Remove all instances
            content = re.sub(env_validation_pattern, '', content, flags=re.DOTALL)

            # Find security imports
            security_import_pattern = r"(from '\.\./\.\./templates/security-utils';)"
            security_import_match = re.search(security_import_pattern, content)

            if security_import_match:
                insert_pos = security_import_match.end()
                content = content[:insert_pos] + '\n\n' + env_validation + '\n' + content[insert_pos:]

        # Fix 4: Clean up extra blank lines
        content = re.sub(r'\n\n\n+', '\n\n\n', content)

        # Fix 5: Remove trailing whitespace
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)

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
    print("Final Comprehensive Security Fix")
    print("=" * 80)
    print()

    # Get all example files
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
                if fix_file(file_path):
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
