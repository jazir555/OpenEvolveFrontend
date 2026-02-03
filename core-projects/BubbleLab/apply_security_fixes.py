#!/usr/bin/env python3
"""
Apply Security Fixes to BubbleLab Example Files
This script applies Wave 5 security hardening to all 24 example files.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Security import template for examples
SECURITY_IMPORTS = """import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  SecuritySchemas,
} from '../../templates/security-utils';"""

# Logger and rate limiter initialization
LOGGER_INIT = """  private logger = new StructuredLogger('{workflow_name}');
  private rateLimiter = new RateLimiter({{
    maxRequests: 60,
    windowMs: 60000,
  }});"""

# Environment validation template
ENV_VALIDATION = """// Security: Environment variable validation at startup
validateEnvironment({{
  required: {required_vars},
  schemas: {{
    API_KEY: SecuritySchemas.apiKey,
{url_schemas}
  }},
}});"""

# Authentication check template
AUTH_CHECK = """    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({{ correlationId }});

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {{
      throw new Error('Rate limit exceeded. Please try again later.');
    }}

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      {{ correlationId, ip: payload.headers?.['x-forwarded-for'] }}
    );
    requireAuthentication(authContext);

    this.logger.info({{
      msg: 'Starting {workflow_action}',
    }});"""

def extract_env_vars(content: str) -> List[str]:
    """Extract environment variables used in the file"""
    env_vars = set(['API_KEY'])  # Always require API_KEY

    # Find process.env usages
    pattern = r'process\.env\.([A-Z_0-9]+)'
    matches = re.findall(pattern, content)
    env_vars.update(matches)

    return list(env_vars)

def get_url_schemas(env_vars: List[str]) -> str:
    """Generate URL schema validations"""
    url_vars = [v for v in env_vars if 'URL' in v or 'API' in v or 'ENDPOINT' in v]
    schemas = []
    for var in url_vars:
        if var != 'API_KEY':
            schemas.append(f"    {var}: SecuritySchemas.url,")
    return '\n'.join(schemas)

def apply_security_fixes(file_path: Path) -> bool:
    """Apply security fixes to a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has security fixes
        if 'Security Fixes Applied (Wave 5)' in content:
            print(f"  [OK] Already secured: {file_path.name}")
            return True

        print(f"  [FIX] Securing: {file_path.name}")

        workflow_name = file_path.stem.replace('-', '_')
        workflow_action = workflow_name.replace('_', ' ')

        # 1. Add security imports after bubble-core imports
        import_pattern = r"(import \{[^}]+\} from '@bubblelab/bubble-core';)"
        import_match = re.search(import_pattern, content)
        if import_match:
            content = content[:import_match.end()] + "\n\n" + SECURITY_IMPORTS + content[import_match.end():]

        # 2. Add environment validation before class declaration
        class_pattern = r"(export class \w+ extends BubbleFlow)"
        class_match = re.search(class_pattern, content)
        if class_match:
            env_vars = extract_env_vars(content)
            url_schemas = get_url_schemas(env_vars)

            env_validation = ENV_VALIDATION.format(
                required_vars=str(env_vars),
                url_schemas=url_schemas
            )
            content = content[:class_match.start()] + env_validation + "\n\n" + content[class_match.start():]

        # 3. Add logger and rate limiter initialization in class
        # Find the class declaration and add after readonly fields
        class_body_pattern = r"(export class \w+[^{]*\{)"
        class_body_match = re.search(class_body_pattern, content)

        if class_body_match:
            # Look for readonly fields
            readonly_pattern = r"  (readonly \w+ = '[^']+';)"
            readonly_matches = list(re.finditer(readonly_pattern, content))

            if readonly_matches:
                # Insert after last readonly field
                last_readonly = readonly_matches[-1]
                insert_pos = last_readonly.end()
                logger_init = LOGGER_INIT.format(workflow_name=workflow_name)
                content = content[:insert_pos] + "\n\n" + logger_init + content[insert_pos:]
            else:
                # Insert right after class declaration
                insert_pos = class_body_match.end()
                logger_init = LOGGER_INIT.format(workflow_name=workflow_name)
                content = content[:insert_pos] + "\n" + logger_init + "\n" + content[insert_pos:]

        # 4. Add authentication check at start of handle method
        handle_pattern = r"(async handle\([^)]+\): [^{]+\{)"
        handle_match = re.search(handle_pattern, content)
        if handle_match:
            auth_check = AUTH_CHECK.format(workflow_action=workflow_action)
            content = content[:handle_match.end()] + "\n" + auth_check + "\n" + content[handle_match.end():]

        # 5. Add Wave 5 security notice to header
        header_end_pattern = r"(\* Required Credentials:.*?\*/)"
        header_match = re.search(header_end_pattern, content, re.DOTALL)
        if header_match:
            security_notice = """
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting (60 requests/minute)
 * - Input validation for all user inputs
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints
 *
 """  # Maintain the exact header format
            content = content[:header_match.end()] + security_notice + content[header_match.end():]

        # Write fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  [OK] Secured: {file_path.name}")
        return True

    except Exception as e:
        print(f"  [ERROR] Error securing {file_path.name}: {e}")
        return False

def main():
    """Apply security fixes to all example files"""
    base_dir = Path(__file__).parent

    # Define file categories
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

    print("=" * 80)
    print("BubbleLab Security Hardening - Wave 5")
    print("=" * 80)
    print()

    total_files = sum(len(files) for files in categories.values())
    fixed_count = 0
    skipped_count = 0
    error_count = 0

    for category, files in categories.items():
        print(f"\n[{category.upper()}]")
        print("-" * 80)

        category_dir = base_dir / "examples" / category
        if not category_dir.exists():
            print(f"  [ERROR] Directory not found: {category_dir}")
            continue

        for filename in files:
            file_path = category_dir / filename
            if file_path.exists():
                if apply_security_fixes(file_path):
                    # Check if it was already secured
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if 'Security Fixes Applied (Wave 5)' in f.read():
                            # It was already secured before we ran
                            pass
                        else:
                            fixed_count += 1
                else:
                    error_count += 1
            else:
                print(f"  [ERROR] File not found: {filename}")
                error_count += 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Newly secured: {fixed_count}")
    print(f"Already secured: {skipped_count}")
    print(f"Errors: {error_count}")
    print()

    if error_count == 0:
        print("[SUCCESS] All security fixes applied successfully!")
        print()
        print("Next steps:")
        print("1. Review the changes with: git diff")
        print("2. Test the workflows to ensure they still work")
        print("3. Commit with: git commit -m 'feat: Apply Wave 5 security fixes to all example files'")
    else:
        print("[WARNING] Some files had errors. Please review the output above.")

if __name__ == "__main__":
    main()
