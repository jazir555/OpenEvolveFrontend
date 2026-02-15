/**
 * Contract Test for Environment Validator
 *
 * Tests compliance with Federation Constitution Section 5:
 * - Law of Configuration Explicitness
 * - No magic defaults
 * - Crash immediately if required vars are missing
 * - Type validation (URLs, ports, numbers, booleans)
 */

import {
  validateEnv,
  validateEnvWithTypes,
  getEnv,
  EnvVar,
  EnvType
} from './env-validator';

describe('Environment Validator Contract Tests', () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };

    // Clear all environment variables
    delete process.env.TEST_VAR;
    delete process.env.TEST_URL;
    delete process.env.TEST_PORT;
    delete process.env.TEST_NUMBER;
    delete process.env.TEST_BOOLEAN;
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('validateEnv - Basic Validation', () => {
    it('should pass when all required vars are present', () => {
      process.env.REQUIRED_VAR_1 = 'value1';
      process.env.REQUIRED_VAR_2 = 'value2';

      expect(() => {
        validateEnv(['REQUIRED_VAR_1', 'REQUIRED_VAR_2']);
      }).not.toThrow();
    });

    it('should crash loudly when required var is missing', () => {
      expect(() => {
        validateEnv(['MISSING_VAR']);
      }).toThrow('Missing required environment variable: MISSING_VAR');
    });

    it('should crash when multiple required vars are missing', () => {
      expect(() => {
        validateEnv(['MISSING_1', 'MISSING_2', 'MISSING_3']);
      }).toThrow((error: Error) => {
        expect(error.message).toContain('Environment validation failed');
        expect(error.message).toContain('MISSING_1');
        expect(error.message).toContain('MISSING_2');
        expect(error.message).toContain('MISSING_3');
        return true;
      });
    });

    it('should reject empty string values', () => {
      process.env.EMPTY_VAR = '   ';

      expect(() => {
        validateEnv(['EMPTY_VAR']);
      }).toThrow('Missing required environment variable: EMPTY_VAR');
    });
  });

  describe('validateEnvWithTypes - Type Validation', () => {
    describe('String Type', () => {
      it('should accept valid string values', () => {
        process.env.TEST_STRING = 'valid string';

        const result = validateEnvWithTypes([
          { name: 'TEST_STRING', type: 'string', required: true }
        ]);

        expect(result.TEST_STRING).toBe('valid string');
      });

      it('should use default for optional string', () => {
        const result = validateEnvWithTypes([
          {
            name: 'OPTIONAL_STRING',
            type: 'string',
            required: false,
            default: 'default value'
          }
        ]);

        expect(result.OPTIONAL_STRING).toBe('default value');
      });
    });

    describe('Number Type', () => {
      it('should accept valid number values', () => {
        process.env.TEST_NUMBER = '42';

        const result = validateEnvWithTypes([
          { name: 'TEST_NUMBER', type: 'number', required: true }
        ]);

        expect(result.TEST_NUMBER).toBe(42);
        expect(typeof result.TEST_NUMBER).toBe('number');
      });

      it('should reject invalid number values', () => {
        process.env.TEST_NUMBER = 'not-a-number';

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_NUMBER', type: 'number', required: true }
          ]);
        }).toThrow('TEST_NUMBER: "not-a-number" is not a valid number');
      });

      it('should handle decimal numbers', () => {
        process.env.TEST_DECIMAL = '3.14';

        const result = validateEnvWithTypes([
          { name: 'TEST_DECIMAL', type: 'number', required: true }
        ]);

        expect(result.TEST_DECIMAL).toBe(3.14);
      });

      it('should use default for optional number', () => {
        const result = validateEnvWithTypes([
          {
            name: 'OPTIONAL_NUMBER',
            type: 'number',
            required: false,
            default: 100
          }
        ]);

        expect(result.OPTIONAL_NUMBER).toBe(100);
      });
    });

    describe('Boolean Type', () => {
      it('should accept true variations', () => {
        process.env.TEST_BOOL_TRUE1 = 'true';
        process.env.TEST_BOOL_TRUE2 = '1';

        const result = validateEnvWithTypes([
          { name: 'TEST_BOOL_TRUE1', type: 'boolean', required: true },
          { name: 'TEST_BOOL_TRUE2', type: 'boolean', required: true }
        ]);

        expect(result.TEST_BOOL_TRUE1).toBe(true);
        expect(result.TEST_BOOL_TRUE2).toBe(true);
      });

      it('should accept false variations', () => {
        process.env.TEST_BOOL_FALSE1 = 'false';
        process.env.TEST_BOOL_FALSE2 = '0';

        const result = validateEnvWithTypes([
          { name: 'TEST_BOOL_FALSE1', type: 'boolean', required: true },
          { name: 'TEST_BOOL_FALSE2', type: 'boolean', required: true }
        ]);

        expect(result.TEST_BOOL_FALSE1).toBe(false);
        expect(result.TEST_BOOL_FALSE2).toBe(false);
      });

      it('should reject invalid boolean values', () => {
        process.env.TEST_BOOL_INVALID = 'maybe';

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_BOOL_INVALID', type: 'boolean', required: true }
          ]);
        }).toThrow('TEST_BOOL_INVALID: "maybe" is not a valid boolean');
      });

      it('should be case-sensitive for true/false', () => {
        process.env.TEST_BOOL_UPPER = 'TRUE';
        process.env.TEST_BOOL_LOWER = 'FALSE';

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_BOOL_UPPER', type: 'boolean', required: true }
          ]);
        }).toThrow();

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_BOOL_LOWER', type: 'boolean', required: true }
          ]);
        }).toThrow();
      });
    });

    describe('URL Type', () => {
      it('should accept valid URLs', () => {
        process.env.TEST_URL = 'http://example.com:8080';
        process.env.TEST_URL_SECURE = 'https://api.example.com/v1';

        const result = validateEnvWithTypes([
          { name: 'TEST_URL', type: 'url', required: true },
          { name: 'TEST_URL_SECURE', type: 'url', required: true }
        ]);

        expect(result.TEST_URL).toBe('http://example.com:8080');
        expect(result.TEST_URL_SECURE).toBe('https://api.example.com/v1');
      });

      it('should reject invalid URLs', () => {
        process.env.TEST_INVALID_URL = 'not-a-url';

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_INVALID_URL', type: 'url', required: true }
          ]);
        }).toThrow('TEST_INVALID_URL: "not-a-url" is not a valid URL');
      });

      it('should accept URLs with ports', () => {
        process.env.TEST_URL_PORT = 'http://localhost:3000';

        const result = validateEnvWithTypes([
          { name: 'TEST_URL_PORT', type: 'url', required: true }
        ]);

        expect(result.TEST_URL_PORT).toBe('http://localhost:3000');
      });
    });

    describe('Port Type', () => {
      it('should accept valid ports', () => {
        process.env.TEST_PORT = '8080';
        process.env.TEST_PORT_MIN = '1';
        process.env.TEST_PORT_MAX = '65535';

        const result = validateEnvWithTypes([
          { name: 'TEST_PORT', type: 'port', required: true },
          { name: 'TEST_PORT_MIN', type: 'port', required: true },
          { name: 'TEST_PORT_MAX', type: 'port', required: true }
        ]);

        expect(result.TEST_PORT).toBe(8080);
        expect(result.TEST_PORT_MIN).toBe(1);
        expect(result.TEST_PORT_MAX).toBe(65535);
      });

      it('should reject port < 1', () => {
        process.env.TEST_PORT_LOW = '0';

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_PORT_LOW', type: 'port', required: true }
          ]);
        }).toThrow('TEST_PORT_LOW: "0" is not a valid port');
      });

      it('should reject port > 65535', () => {
        process.env.TEST_PORT_HIGH = '65536';

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_PORT_HIGH', type: 'port', required: true }
          ]);
        }).toThrow('TEST_PORT_HIGH: "65536" is not a valid port');
      });

      it('should reject non-numeric ports', () => {
        process.env.TEST_PORT_INVALID = 'abc';

        expect(() => {
          validateEnvWithTypes([
            { name: 'TEST_PORT_INVALID', type: 'port', required: true }
          ]);
        }).toThrow('TEST_PORT_INVALID: "abc" is not a valid port');
      });
    });
  });

  describe('Mixed Type Validation', () => {
    it('should validate multiple types in one call', () => {
      process.env.MY_STRING = 'test';
      process.env.MY_NUMBER = '42';
      process.env.MY_URL = 'http://example.com';
      process.env.MY_BOOLEAN = 'true';

      const result = validateEnvWithTypes([
        { name: 'MY_STRING', type: 'string', required: true },
        { name: 'MY_NUMBER', type: 'number', required: true },
        { name: 'MY_URL', type: 'url', required: true },
        { name: 'MY_BOOLEAN', type: 'boolean', required: true }
      ]);

      expect(result.MY_STRING).toBe('test');
      expect(result.MY_NUMBER).toBe(42);
      expect(result.MY_URL).toBe('http://example.com');
      expect(result.MY_BOOLEAN).toBe(true);
    });

    it('should provide defaults for missing optional vars', () => {
      const result = validateEnvWithTypes([
        { name: 'REQUIRED_VAR', type: 'string', required: true },
        { name: 'OPTIONAL_STRING', type: 'string', required: false, default: 'default' },
        { name: 'OPTIONAL_NUMBER', type: 'number', required: false, default: 100 }
      ]);

      // Should have required var
      expect(result.REQUIRED_VAR).toBeDefined();
      // Should have defaults
      expect(result.OPTIONAL_STRING).toBe('default');
      expect(result.OPTIONAL_NUMBER).toBe(100);
    });
  });

  describe('getEnv - Single Variable Getter', () => {
    it('should return required string variable', () => {
      process.env.MY_VAR = 'value';

      const result = getEnv('MY_VAR', 'string');
      expect(result).toBe('value');
    });

    it('should return and type-coerce number', () => {
      process.env.MY_NUMBER = '42';

      const result = getEnv('MY_NUMBER', 'number');
      expect(result).toBe(42);
      expect(typeof result).toBe('number');
    });

    it('should return and type-coerce boolean', () => {
      process.env.MY_BOOL = 'true';

      const result = getEnv('MY_BOOL', 'boolean');
      expect(result).toBe(true);
      expect(typeof result).toBe('boolean');
    });

    it('should throw for missing variable', () => {
      expect(() => {
        getEnv('MISSING_VAR', 'string');
      }).toThrow('Missing required environment variable: MISSING_VAR');
    });

    it('should validate URL type', () => {
      process.env.MY_URL = 'http://example.com';

      const result = getEnv('MY_URL', 'url');
      expect(result).toBe('http://example.com');
    });

    it('should throw for invalid URL', () => {
      process.env.INVALID_URL = 'not-url';

      expect(() => {
        getEnv('INVALID_URL', 'url');
      }).toThrow('INVALID_URL: "not-url" is not a valid URL');
    });

    it('should validate port type', () => {
      process.env.MY_PORT = '8080';

      const result = getEnv('MY_PORT', 'port');
      expect(result).toBe(8080);
    });

    it('should throw for invalid port', () => {
      process.env.INVALID_PORT = '70000';

      expect(() => {
        getEnv('INVALID_PORT', 'port');
      }).toThrow('INVALID_PORT: "70000" is not a valid port');
    });
  });

  describe('Law of Configuration Explicitness (Law 5)', () => {
    it('should crash loudly - no silent failures', () => {
      expect(() => {
        validateEnv(['COMPLETELY_MISSING_VAR']);
      }).toThrow((error: Error) => {
        expect(error.message).toContain('COMPLETELY_MISSING_VAR');
        return true;
      });
    });

    it('should not allow localhost defaults', () => {
      // This test verifies we're enforcing explicit configuration
      // If code tries to default to localhost, it should fail
      expect(() => {
        const result = validateEnvWithTypes([
          {
            name: 'SERVICE_URL',
            type: 'url',
            required: true
          }
        ]);
      }).toThrow();
    });

    it('should require all configurable values', () => {
      // All ports, URLs, timeouts must be explicit
      const requiredVars = [
        'SERVICE_URL',
        'SERVICE_PORT',
        'TIMEOUT_MS',
        'MAX_RETRIES'
      ];

      requiredVars.forEach(varName => {
        expect(() => {
          getEnv(varName, 'string');
        }).toThrow();
      });
    });
  });

  describe('Error Messages', () => {
    it('should provide clear error messages', () => {
      expect(() => {
        validateEnv(['MISSING_VAR']);
      }).toThrow((error: Error) => {
        expect(error.message).toContain('Environment validation failed');
        expect(error.message).toContain('MISSING_VAR');
        return true;
      });
    });

    it('should list all missing variables', () => {
      expect(() => {
        validateEnv(['MISSING_1', 'MISSING_2', 'MISSING_3']);
      }).toThrow((error: Error) => {
        expect(error.message).toContain('MISSING_1');
        expect(error.message).toContain('MISSING_2');
        expect(error.message).toContain('MISSING_3');
        return true;
      });
    });

    it('should indicate validation type in error', () => {
      process.env.INVALID_PORT = 'abc';

      expect(() => {
        validateEnvWithTypes([
          { name: 'INVALID_PORT', type: 'port', required: true }
        ]);
      }).toThrow((error: Error) => {
        expect(error.message).toContain('port');
        return true;
      });
    });
  });

  describe('Type Conversion Edge Cases', () => {
    it('should handle whitespace in strings', () => {
      process.env.WHITESPACE_STRING = '  value  ';

      const result = validateEnvWithTypes([
        { name: 'WHITESPACE_STRING', type: 'string', required: true }
      ]);

      expect(result.WHITESPACE_STRING).toBe('  value  ');
    });

    it('should handle negative numbers', () => {
      process.env.NEGATIVE_NUMBER = '-42';

      const result = validateEnvWithTypes([
        { name: 'NEGATIVE_NUMBER', type: 'number', required: true }
      ]);

      expect(result.NEGATIVE_NUMBER).toBe(-42);
    });

    it('should handle zero', () => {
      process.env.ZERO_NUMBER = '0';

      const result = validateEnvWithTypes([
        { name: 'ZERO_NUMBER', type: 'number', required: true }
      ]);

      expect(result.ZERO_NUMBER).toBe(0);
    });
  });
});
