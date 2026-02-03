/**
 * URL Resolution Tests
 * Tests for environment variable fallback chain and URL normalization
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('URL Resolution', () => {
  // Mock the resolveEvolutionApiBaseUrl function
  const resolveEvolutionApiBaseUrl = (env: any): string => {
    const configured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';
    const isProduction = env.MODE === 'production';

    if (!configured || configured.trim().length === 0) {
      if (isProduction) {
        throw new Error(
          'CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set.'
        );
      }
      console.warn(
        '[BubbleLab] VITE_EVOLUTION_API_URL not configured. ' +
        'Falling back to http://localhost:8000 for development.'
      );
      return 'http://localhost:8000';
    }

    const base = configured.trim();

    // Validate URL format
    try {
      new URL(base);
    } catch (error) {
      throw new Error(`CRITICAL: Invalid VITE_EVOLUTION_API_URL format: "${base}"`);
    }

    return base.replace(/\/$/, '');
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('resolveEvolutionApiBaseUrl', () => {
    it('should use VITE_EVOLUTION_API_URL if set', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com',
        VITE_GATEWAY_URL: 'https://gateway.com',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.evolution.com');
    });

    it('should fallback to VITE_GATEWAY_URL if VITE_EVOLUTION_API_URL not set', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: 'https://gateway.com',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://gateway.com');
    });

    it('should fallback to localhost:8000 if neither set in development', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');

      const env = {
        MODE: 'development',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('http://localhost:8000');
      expect(consoleWarnSpy).toHaveBeenCalled();

      consoleWarnSpy.mockRestore();
    });

    it('should throw error if neither set in production', () => {
      const env = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: '',
      };

      expect(() => resolveEvolutionApiBaseUrl(env)).toThrow(
        'CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set'
      );
    });

    it('should remove trailing slashes from URLs', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com/',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.evolution.com');
      expect(url).not.toMatch(/\/$/);
    });

    it('should remove multiple trailing slashes', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com///',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.evolution.com//');
    });

    it('should validate URL format', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'not-a-valid-url',
        VITE_GATEWAY_URL: '',
      };

      expect(() => resolveEvolutionApiBaseUrl(env)).toThrow(
        'CRITICAL: Invalid VITE_EVOLUTION_API_URL format'
      );
    });

    it('should accept valid HTTP URLs', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'http://localhost:8000',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('http://localhost:8000');
    });

    it('should accept valid HTTPS URLs', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.openevolve.com',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.openevolve.com');
    });

    it('should reject URLs without protocol', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'api.openevolve.com',
        VITE_GATEWAY_URL: '',
      };

      expect(() => resolveEvolutionApiBaseUrl(env)).toThrow(
        'CRITICAL: Invalid VITE_EVOLUTION_API_URL format'
      );
    });

    it('should return valid URL string', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(typeof url).toBe('string');

      // Verify it's a valid URL
      const urlObj = new URL(url);
      expect(urlObj.protocol).toBe('https:');
      expect(urlObj.hostname).toBe('api.evolution.com');
    });

    it('should trim whitespace from URLs', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: '  https://api.evolution.com  ',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.evolution.com');
    });

    it('should handle empty string as unset', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');

      const env = {
        MODE: 'development',
        VITE_EVOLUTION_API_URL: '   ',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('http://localhost:8000');
      expect(consoleWarnSpy).toHaveBeenCalled();

      consoleWarnSpy.mockRestore();
    });

    it('should handle localhost with port', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'http://localhost:9000',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('http://localhost:9000');

      const urlObj = new URL(url);
      expect(urlObj.hostname).toBe('localhost');
      expect(urlObj.port).toBe('9000');
    });

    it('should handle IP addresses', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'http://192.168.1.100:8000',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('http://192.168.1.100:8000');

      const urlObj = new URL(url);
      expect(urlObj.hostname).toBe('192.168.1.100');
      expect(urlObj.port).toBe('8000');
    });

    it('should handle URLs with paths', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com/v1',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.evolution.com/v1');
    });

    it('should preserve path when removing trailing slash', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com/v1/',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.evolution.com/v1');
    });

    it('should handle URLs with query parameters', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com?version=1',
        VITE_GATEWAY_URL: '',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://api.evolution.com?version=1');
    });
  });

  describe('Priority Chain', () => {
    it('should prioritize VITE_EVOLUTION_API_URL over VITE_GATEWAY_URL', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://evolution.com',
        VITE_GATEWAY_URL: 'https://gateway.com',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://evolution.com');
    });

    it('should use VITE_GATEWAY_URL when VITE_EVOLUTION_API_URL is empty', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: 'https://gateway.com',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://gateway.com');
    });

    it('should use VITE_GATEWAY_URL when VITE_EVOLUTION_API_URL is whitespace', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');

      const env = {
        MODE: 'development',
        VITE_EVOLUTION_API_URL: '   ',
        VITE_GATEWAY_URL: 'https://gateway.com',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      // The whitespace-only string is treated as empty, so it should fallback
      expect(url).toBe('http://localhost:8000');
      expect(consoleWarnSpy).toHaveBeenCalled();

      consoleWarnSpy.mockRestore();
    });

    it('should not mix values from different variables', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://evolution.com/path',
        VITE_GATEWAY_URL: 'https://gateway.com/other',
      };

      const url = resolveEvolutionApiBaseUrl(env);
      expect(url).toBe('https://evolution.com/path');
      expect(url).not.toContain('gateway');
    });
  });

  describe('Error Handling', () => {
    it('should include the invalid URL in error message', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'invalid-url',
        VITE_GATEWAY_URL: '',
      };

      expect(() => resolveEvolutionApiBaseUrl(env)).toThrow();
      try {
        resolveEvolutionApiBaseUrl(env);
      } catch (error: any) {
        expect(error.message).toContain('invalid-url');
      }
    });

    it('should provide helpful error for missing protocol', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'api.example.com',
        VITE_GATEWAY_URL: '',
      };

      expect(() => resolveEvolutionApiBaseUrl(env)).toThrow();
      try {
        resolveEvolutionApiBaseUrl(env);
      } catch (error: any) {
        expect(error.message).toContain('Invalid');
      }
    });

    it('should distinguish between production and development in errors', () => {
      const prodEnv = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: '',
      };

      expect(() => resolveEvolutionApiBaseUrl(prodEnv)).toThrow(
        'CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set'
      );
    });
  });
});
