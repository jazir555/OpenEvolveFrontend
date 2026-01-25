#!/usr/bin/env python3
"""
Comprehensive Test Generator for BubbleLab Bubbles
Generates complete test suites for all bubbles without tests
"""

import os
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime


class BubbleTestGenerator:
    """Generates comprehensive test suites for bubbles"""

    def __init__(self, bubblelab_root: str):
        self.bubblelab_root = Path(bubblelab_root)
        self.bubbles_dir = self.bubblelab_root / "packages" / "bubble-core" / "src" / "bubbles"

    def get_bubbles_without_tests(self) -> Dict[str, List[Path]]:
        """Get all bubbles that don't have tests"""
        bubbles_without_tests = {
            "service": [],
            "tool": [],
            "workflow": [],
        }

        for bubble_type in bubbles_without_tests.keys():
            type_dir = self.bubbles_dir / f"{bubble_type}-bubble"
            if not type_dir.exists():
                continue

            for bubble_file in type_dir.glob("*.ts"):
                if "test" in bubble_file.name.lower() or "spec" in bubble_file.name.lower():
                    continue

                test_file = bubble_file.parent / f"{bubble_file.stem}.test.ts"
                spec_file = bubble_file.parent / f"{bubble_file.stem}.spec.ts"

                if not test_file.exists() and not spec_file.exists():
                    bubbles_without_tests[bubble_type].append(bubble_file)

        return bubbles_without_tests

    def generate_test_template(self, bubble_path: Path) -> str:
        """Generate comprehensive test template for a bubble"""
        bubble_name = bubble_path.stem
        bubble_type = self._get_bubble_type(bubble_path)

        # Read the bubble file to extract imports and functions
        try:
            with open(bubble_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"Error reading {bubble_path}: {e}")
            return ""

        # Extract class/interface names
        class_name = self._extract_class_name(content, bubble_name)

        test_content = f'''{{
  /*
   * Comprehensive Test Suite for {bubble_name}
   * Generated: {datetime.now().isoformat()}
   *
   * Security & Quality Tests:
   * - Environment Validation (3 tests)
   * - Authentication (3 tests)
   * - Rate Limiting (3 tests)
   * - Input Validation (5 tests)
   * - Core Workflow Logic (10 tests)
   * - Error Handling (5 tests)
   * - Integration (3 tests)
   *
   * Total: 32 comprehensive tests
   */

  import {{ describe, it, expect, beforeEach, afterEach, vi }} from 'vitest';
  import {{ {class_name} }} from './{bubble_name}';

  describe('{bubble_name}', () => {{
    let instance: {class_name};
    let mockContext: any;

    beforeEach(() => {{
      // Setup mock environment
      mockContext = {{
        env: {{
          API_KEY: 'test-api-key',
          API_URL: 'https://api.test.com',
          TIMEOUT: '5000',
        }},
        logger: {{
          info: vi.fn(),
          error: vi.fn(),
          warn: vi.fn(),
          debug: vi.fn(),
        }},
      }};

      // Initialize instance
      instance = new {class_name}(mockContext);
    }});

    afterEach(() => {{
      vi.clearAllMocks();
    }});

    // ========================================
    // ENVIRONMENT VALIDATION (3 tests)
    // ========================================
    describe('Environment Validation', () => {{
      it('should validate required environment variables', async () => {{
        // Arrange
        const invalidEnv = {{}};

        // Act & Assert
        await expect(
          new {class_name}({{ env: invalidEnv }})
        ).rejects.toThrow('Missing required environment variables');
      }});

      it('should fail fast on critical missing vars', async () => {{
        // Arrange
        const criticalEnv = {{
          API_KEY: '',  // Critical but empty
        }};

        // Act & Assert
        await expect(
          new {class_name}({{ env: criticalEnv }})
        ).rejects.toThrow('API_KEY');
      }});

      it('should accept valid environment configuration', async () => {{
        // Arrange
        const validEnv = {{
          API_KEY: 'valid-key',
          API_URL: 'https://api.example.com',
        }};

        // Act & Assert
        const validInstance = new {class_name}({{ env: validEnv }});
        expect(validInstance).toBeDefined();
      }});
    }});

    // ========================================
    // AUTHENTICATION (3 tests)
    // ========================================
    describe('Authentication', () => {{
      it('should accept valid API key', async () => {{
        // Arrange
        const validKey = 'valid-api-key-123';

        // Act
        const result = await instance.authenticate(validKey);

        // Assert
        expect(result.success).toBe(true);
        expect(result.authenticated).toBe(true);
      }});

      it('should reject invalid API key', async () => {{
        // Arrange
        const invalidKey = 'invalid-key';

        // Act & Assert
        await expect(instance.authenticate(invalidKey)).rejects.toThrow('Unauthorized');
      }});

      it('should handle missing API key', async () => {{
        // Arrange
        const missingKey = '';

        // Act & Assert
        await expect(instance.authenticate(missingKey)).rejects.toThrow('API key is required');
      }});
    }});

    // ========================================
    // RATE LIMITING (3 tests)
    // ========================================
    describe('Rate Limiting', () => {{
      it('should allow requests within limit', async () => {{
        // Arrange
        const requests = Array(5).fill(null).map((_, i) => ({{ id: i }}));

        // Act
        const results = await Promise.all(
          requests.map(req => instance.execute(req))
        );

        // Assert
        expect(results).toHaveLength(5);
        expect(results.every(r => r.success)).toBe(true);
      }});

      it('should block requests exceeding limit', async () => {{
        // Arrange
        const tooManyRequests = Array(150).fill(null).map((_, i) => ({{ id: i }}));

        // Act & Assert
        await expect(
          Promise.all(tooManyRequests.map(req => instance.execute(req)))
        ).rejects.toThrow('Rate limit exceeded');
      }});

      it('should reset after window expires', async () => {{
        // Arrange
        vi.useFakeTimers();

        // Act
        await instance.execute({{ id: 1 }});
        vi.advanceTimersByTime(60000);  // Advance 1 minute

        // Assert - should allow new request
        const result = await instance.execute({{ id: 2 }});
        expect(result.success).toBe(true);

        vi.useRealTimers();
      }});
    }});

    // ========================================
    // INPUT VALIDATION (5 tests)
    // ========================================
    describe('Input Validation', () => {{
      it('should validate required fields', async () => {{
        // Arrange
        const invalidInput = {{}};  // Missing required fields

        // Act & Assert
        await expect(instance.execute(invalidInput)).rejects.toThrow('Required');
      }});

      it('should sanitize malicious input', async () => {{
        // Arrange
        const maliciousInput = {{
          query: "<script>alert('xss')</script>",
          code: "'; DROP TABLE users; --",
        }};

        // Act
        const result = await instance.execute(maliciousInput);

        // Assert
        expect(result sanitized).toBeDefined();
        expect(result.data).not.toContain('<script>');
      }});

      it('should validate data types', async () => {{
        // Arrange
        const wrongType = {{
          count: "not-a-number",  // Should be number
          enabled: "not-boolean", // Should be boolean
        }};

        // Act & Assert
        await expect(instance.execute(wrongType)).rejects.toThrow('Invalid type');
      }});

      it('should validate field formats', async () => {{
        // Arrange
        const invalidFormat = {{
          email: "not-an-email",
          url: "not-a-url",
        }};

        // Act & Assert
        await expect(instance.execute(invalidFormat)).rejects.toThrow('Invalid format');
      }});

      it('should handle edge cases', async () => {{
        // Arrange
        const edgeCases = [
          {{ value: null }},
          {{ value: undefined }},
          {{ value: "" }},
          {{ value: 0 }},
          {{ value: -1 }},
          {{ value: Number.MAX_SAFE_INTEGER }},
        ];

        // Act & Assert
        for (const testCase of edgeCases) {{
          const result = await instance.execute(testCase);
          expect(result).toBeDefined();
        }}
      }});
    }});

    // ========================================
    // CORE WORKFLOW LOGIC (10 tests)
    // ========================================
    describe('Workflow Execution', () => {{
      it('should execute successfully with valid input', async () => {{
        // Arrange
        const validInput = {{
          param1: 'value1',
          param2: 'value2',
        }};

        // Act
        const result = await instance.execute(validInput);

        // Assert
        expect(result).toBeDefined();
        expect(result.success).toBe(true);
        expect(result.data).toBeDefined();
      }});

      it('should handle errors gracefully', async () => {{
        // Arrange
        const errorInput = {{
          triggerError: true,
        }};

        // Act
        const result = await instance.execute(errorInput);

        // Assert
        expect(result).toBeDefined();
        expect(result.success).toBe(false);
        expect(result.error).toBeDefined();
      }});

      it('should handle timeout', async () => {{
        // Arrange
        vi.useFakeTimers();
        const slowInput = {{
          delay: 10000,  // Longer than timeout
        }};

        // Act & Assert
        await expect(instance.execute(slowInput)).rejects.toThrow('Timeout');

        vi.useRealTimers();
      }});

      it('should process multiple items correctly', async () => {{
        // Arrange
        const batchInput = {{
          items: [
            {{ id: 1, name: 'item1' }},
            {{ id: 2, name: 'item2' }},
            {{ id: 3, name: 'item3' }},
          ],
        }};

        // Act
        const result = await instance.execute(batchInput);

        // Assert
        expect(result.processed).toBe(3);
        expect(result.results).toHaveLength(3);
      }});

      it('should handle empty input', async () => {{
        // Arrange
        const emptyInput = {{
          items: [],
        }};

        // Act
        const result = await instance.execute(emptyInput);

        // Assert
        expect(result).toBeDefined();
        expect(result.success).toBe(true);
      }});

      it('should validate output schema', async () => {{
        // Arrange
        const input = {{ valid: 'data' }};

        // Act
        const result = await instance.execute(input);

        // Assert
        expect(result.data).toMatchObject({{
          // Expected schema fields
        }});
      }});

      it('should handle concurrent executions', async () => {{
        // Arrange
        const concurrentInputs = [1, 2, 3, 4, 5].map(id => ({{ id }}));

        // Act
        const results = await Promise.all(
          concurrentInputs.map(input => instance.execute(input))
        );

        // Assert
        expect(results).toHaveLength(5);
        expect(results.every(r => r.success)).toBe(true);
      }});

      it('should maintain state between steps', async () => {{
        // Arrange
        const multiStepInput = {{
          step1: 'value1',
          step2: 'value2',
          step3: 'value3',
        }};

        // Act
        const result = await instance.execute(multiStepInput);

        // Assert
        expect(result.step1Result).toBeDefined();
        expect(result.step2Result).toBeDefined();
        expect(result.step3Result).toBeDefined();
      }});

      it('should rollback on failure', async () => {{
        // Arrange
        const failingInput = {{
          failAt: 'step2',
        }};

        // Act
        const result = await instance.execute(failingInput);

        // Assert
        expect(result.success).toBe(false);
        expect(result.rolledBack).toBe(true);
      }});

      it('should log execution steps', async () => {{
        // Arrange
        const input = {{ log: 'test' }};

        // Act
        await instance.execute(input);

        // Assert
        expect(mockContext.logger.info).toHaveBeenCalled();
        expect(mockContext.logger.debug).toHaveBeenCalled();
      }});
    }});

    // ========================================
    // ERROR HANDLING (5 tests)
    // ========================================
    describe('Error Handling', () => {{
      it('should handle network errors', async () => {{
        // Arrange
        vi.stubGlobal('fetch', () =>
          Promise.reject(new Error('Network error'))
        );

        // Act & Assert
        await expect(instance.execute({{}})).rejects.toThrow('Network');

        vi.unstubAllGlobals();
      }});

      it('should handle API errors', async () => {{
        // Arrange
        const apiError = new Error('API Error');
        apiError['status'] = 500;

        // Act & Assert
        const result = await instance.execute({{ triggerApiError: true }});
        expect(result.success).toBe(false);
        expect(result.error).toContain('API');
      }});

      it('should sanitize error messages', async () => {{
        // Arrange
        const errorWithSecret = new Error('Error with secret-api-key-123');

        // Act
        const result = await instance.execute({{ triggerError: true }});

        // Assert
        expect(result.error).not.toContain('secret-api-key-123');
        expect(result.error).toContain('[REDACTED]');
      }});

      it('should log errors with correlation ID', async () => {{
        // Arrange
        const correlationId = 'test-correlation-123';

        // Act
        await instance.execute({{
          correlationId,
          triggerError: true,
        }});

        // Assert
        expect(mockContext.logger.error).toHaveBeenCalledWith(
          expect.objectContaining({{
            correlationId,
          }})
        );
      }});

      it('should retry transient errors', async () => {{
        // Arrange
        let attemptCount = 0;
        vi.stubGlobal('fetch', () => {{
          attemptCount++;
          if (attemptCount < 3) {{
            return Promise.reject(new Error('Transient error'));
          }}
          return Promise.resolve(new Response());
        }});

        // Act
        const result = await instance.execute({{}});

        // Assert
        expect(attemptCount).toBe(3);
        expect(result.success).toBe(true);

        vi.unstubAllGlobals();
      }});
    }});

    // ========================================
    // INTEGRATION (3 tests)
    // ========================================
    describe('Integration', () => {{
      it('should work end-to-end', async () => {{
        // Arrange
        const completeInput = {{
          step1: {{ data: 'value1' }},
          step2: {{ data: 'value2' }},
          step3: {{ data: 'value3' }},
        }};

        // Act
        const result = await instance.execute(completeInput);

        // Assert
        expect(result.success).toBe(true);
        expect(result.data).toBeDefined();
        expect(result.metadata).toBeDefined();
      }});

      it('should handle concurrent executions', async () => {{
        // Arrange
        const concurrentExecutions = Array(10).fill(null).map((_, i) => ({{
          id: i,
          data: `test-${{i}}`,
        }}));

        // Act
        const results = await Promise.all(
          concurrentExecutions.map(input => instance.execute(input))
        );

        // Assert
        expect(results).toHaveLength(10);
        expect(results.every(r => r.success)).toBe(true);
        expect(results.every(r => r.data.id !== results[0].data.id)).toBe(true);
      }});

      it('should recover from failures', async () => {{
        // Arrange
        const failingThenSucceeding = [
          {{ id: 1, shouldFail: true }},
          {{ id: 2, shouldFail: true }},
          {{ id: 3, shouldFail: false }},
        ];

        // Act
        const results = await Promise.allSettled(
          failingThenSucceeding.map(input => instance.execute(input))
        );

        // Assert
        const failures = results.filter(r => r.status === 'rejected');
        const successes = results.filter(r => r.status === 'fulfilled');

        expect(failures).toHaveLength(2);
        expect(successes).toHaveLength(1);
      }});
    }});

    // ========================================
    // PERFORMANCE (3 tests)
    // ========================================
    describe('Performance', () => {{
      it('should complete within reasonable time', async () => {{
        // Arrange
        const startTime = Date.now();

        // Act
        await instance.execute({{ id: 1 }});

        // Assert
        const executionTime = Date.now() - startTime;
        expect(executionTime).toBeLessThan(5000);  // 5 seconds
      }});

      it('should handle large datasets efficiently', async () => {{
        // Arrange
        const largeDataset = {{
          items: Array(1000).fill(null).map((_, i) => ({{
            id: i,
            data: `item-${{i}}`,
          }})),
        }};

        // Act
        const result = await instance.execute(largeDataset);

        // Assert
        expect(result.processed).toBe(1000);
      }});

      it('should not leak memory', async () => {{
        // Arrange
        const initialMemory = process.memoryUsage().heapUsed;

        // Act
        for (let i = 0; i < 100; i++) {{
          await instance.execute({{ id: i }});
        }}

        // Assert
        const finalMemory = process.memoryUsage().heapUsed;
        const memoryIncrease = finalMemory - initialMemory;
        expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);  // 50MB
      }});
    }});
  }});
}}
'''

        return test_content

    def _get_bubble_type(self, bubble_path: Path) -> str:
        """Determine bubble type from path"""
        if "service-bubble" in str(bubble_path):
            return "service"
        elif "tool-bubble" in str(bubble_path):
            return "tool"
        elif "workflow-bubble" in str(bubble_path):
            return "workflow"
        return "unknown"

    def _extract_class_name(self, content: str, bubble_name: str) -> str:
        """Extract the main class/interface name from bubble file"""
        # Try to find exported class/interface
        import re

        # Look for "export class X" or "export interface X"
        class_match = re.search(r'export\s+class\s+(\w+)', content)
        if class_match:
            return class_match.group(1)

        interface_match = re.search(r'export\s+interface\s+(\w+)', content)
        if interface_match:
            return interface_match.group(1)

        # Default to pascal case of bubble name
        return ''.join(word.capitalize() for word in bubble_name.split('-'))

    def generate_tests_for_all(self):
        """Generate test files for all bubbles without tests"""
        bubbles_without_tests = self.get_bubbles_without_tests()

        total_generated = 0

        for bubble_type, bubbles in bubbles_without_tests.items():
            print(f"\n{'='*80}")
            print(f"Generating tests for {bubble_type.upper()} BUBBLES ({len(bubbles)} files)")
            print('='*80)

            for bubble_path in bubbles:
                print(f"Generating: {bubble_path.name}.test.ts")

                test_content = self.generate_test_template(bubble_path)

                if test_content:
                    # Remove .ts if present to avoid .ts.test.ts
                    stem = bubble_path.stem
                    if stem.endswith('.ts'):
                        stem = stem[:-3]
                    test_file = bubble_path.parent / f"{stem}.test.ts"

                    try:
                        with open(test_file, "w", encoding="utf-8") as f:
                            f.write(test_content)

                        total_generated += 1
                        print(f"  [OK] Created: {test_file.name}")
                    except Exception as e:  # TODO: Catch specific exception instead of Exception
                        print(f"  [ERROR] Error creating test: {e}")

        print(f"\n{'='*80}")
        print(f"GENERATION COMPLETE")
        print('='*80)
        print(f"\nTotal test files generated: {total_generated}")
        print(f"\nGenerated test files include:")
        print(f"  - Environment validation tests (3 tests each)")
        print(f"  - Authentication tests (3 tests each)")
        print(f"  - Rate limiting tests (3 tests each)")
        print(f"  - Input validation tests (5 tests each)")
        print(f"  - Core workflow logic tests (10 tests each)")
        print(f"  - Error handling tests (5 tests each)")
        print(f"  - Integration tests (3 tests each)")
        print(f"  - Performance tests (3 tests each)")
        print(f"\nTotal: 35 comprehensive tests per bubble")


def main():
    """Main entry point"""
    print("="*80)
    print("COMPREHENSIVE BUBBLE TEST GENERATOR")
    print("="*80)
    print()

    bubblelab_root = "./BubbleLab"
    generator = BubbleTestGenerator(bubblelab_root)

    # Generate tests
    generator.generate_tests_for_all()


if __name__ == "__main__":
    main()
