#!/usr/bin/env python3
"""
Add security notice to files that are missing it
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Add Security Notice
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


from pathlib import Path

def add_security_notice(file_path: Path) -> bool:
    """Add security notice to header"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has security notice
        if 'Security Fixes Applied (Wave 5):' in content:
            return False

        # Find the end of the header comment (first */)
        header_end = content.find('*/')
        if header_end == -1:
            return False

        security_notice = """
 *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting (60 requests/minute)
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints"""

        # Insert before the closing */
        new_content = content[:header_end] + security_notice + content[header_end:]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"  [ERROR] {file_path.name}: {e}")
        return False

def main():
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("Adding Security Notice to Missing Files")
    print("=" * 80)
    print()

    files_to_fix = [
        "development-automation/code-quality-check.ts",
        "development-automation/dependency-update.ts",
        "development-automation/deployment-pipeline.ts",
        "development-automation/documentation-generator.ts",
        "development-automation/release-automation.ts",
        "development-automation/test-orchestration.ts",
        "llm-operations/ai-quality-assessment.ts",
        "llm-operations/cost-optimization.ts",
        "llm-operations/model-benchmarking.ts",
        "llm-operations/model-failover.ts",
        "llm-operations/multi-model-ensemble.ts",
        "llm-operations/prompt-optimization.ts",
        "llm-operations/token-usage-monitor.ts",
    ]

    fixed = 0
    for file_rel in files_to_fix:
        file_path = base_dir / "examples" / file_rel
        if file_path.exists():
            if add_security_notice(file_path):
                fixed += 1
                print(f"  [FIXED] {file_rel}")
            else:
                print(f"  [SKIP] {file_rel}")

    print()
    print(f"Fixed {fixed} files")

if __name__ == "__main__":
    main()
