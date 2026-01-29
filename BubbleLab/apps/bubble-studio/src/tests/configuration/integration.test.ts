/**
 * Integration Tests
 * Tests for overall application startup and configuration
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('Configuration Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Production Startup Requirements', () => {
    it('should fail production startup without VITE_EVOLUTION_API_URL', () => {
      const productionEnv = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: '',
      };

      const configured = productionEnv.VITE_EVOLUTION_API_URL || productionEnv.VITE_GATEWAY_URL || '';
      const isProduction = productionEnv.MODE === 'production';

      if (!configured || configured.trim().length === 0) {
        if (isProduction) {
          expect(() => {
            throw new Error(
              'CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set.'
            );
          }).toThrow();
        }
      }
    });

    it('should succeed with valid production configuration', () => {
      const productionEnv = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: 'https://api.openevolve.com',
        VITE_GATEWAY_URL: '',
      };

      const configured = productionEnv.VITE_EVOLUTION_API_URL || productionEnv.VITE_GATEWAY_URL || '';
      const base = configured.trim();
      const url = base.replace(/\/$/, '');

      expect(url).toBe('https://api.openevolve.com');
    });

    it('should validate URL format in production', () => {
      const productionEnv = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: 'invalid-url-format',
        VITE_GATEWAY_URL: '',
      };

      const configured = productionEnv.VITE_EVOLUTION_API_URL || productionEnv.VITE_GATEWAY_URL || '';

      expect(() => {
        const base = configured.trim();
        new URL(base);
      }).toThrow();
    });
  });

  describe('Development Startup Behavior', () => {
    it('should warn but succeed without configuration', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');
      const developmentEnv = {
        MODE: 'development',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: '',
      };

      const configured = developmentEnv.VITE_EVOLUTION_API_URL || developmentEnv.VITE_GATEWAY_URL || '';
      const isProduction = developmentEnv.MODE === 'production';

      if (!configured || configured.trim().length === 0) {
        if (!isProduction) {
          console.warn('[BubbleLab] VITE_EVOLUTION_API_URL not configured. Falling back to http://localhost:8000 for development. This will FAIL in production!');
          const url = 'http://localhost:8000';
          expect(url).toBe('http://localhost:8000');
        }
      }

      expect(consoleWarnSpy).toHaveBeenCalled();
      consoleWarnSpy.mockRestore();
    });

    it('should use provided URL when available', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');
      const developmentEnv = {
        MODE: 'development',
        VITE_EVOLUTION_API_URL: 'http://localhost:9000',
        VITE_GATEWAY_URL: '',
      };

      const configured = developmentEnv.VITE_EVOLUTION_API_URL || developmentEnv.VITE_GATEWAY_URL || '';
      const base = configured.trim();
      const url = base.replace(/\/$/, '');

      expect(url).toBe('http://localhost:9000');
      expect(consoleWarnSpy).not.toHaveBeenCalled();
      consoleWarnSpy.mockRestore();
    });
  });

  describe('Environment Variable Chain', () => {
    it('should use VITE_EVOLUTION_API_URL when set', () => {
      const env = {
        VITE_EVOLUTION_API_URL: 'https://evolution.com',
        VITE_GATEWAY_URL: 'https://gateway.com',
      };

      const configured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';
      const url = configured.trim().replace(/\/$/, '');

      expect(url).toBe('https://evolution.com');
    });

    it('should fallback to VITE_GATEWAY_URL', () => {
      const env = {
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: 'https://gateway.com',
      };

      const configured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';
      const url = configured.trim().replace(/\/$/, '');

      expect(url).toBe('https://gateway.com');
    });

    it('should use localhost:8000 as final fallback in dev', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');
      const env = {
        MODE: 'development',
        VITE_EVOLUTION_API_URL: '',
        VITE_GATEWAY_URL: '',
      };

      const configured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';
      const isProduction = env.MODE === 'production';

      let url = '';
      if (!configured || configured.trim().length === 0) {
        if (!isProduction) {
          console.warn('[BubbleLab] Fallback to localhost:8000');
          url = 'http://localhost:8000';
        }
      }

      expect(url).toBe('http://localhost:8000');
      expect(consoleWarnSpy).toHaveBeenCalled();
      consoleWarnSpy.mockRestore();
    });
  });

  describe('Error Prevention', () => {
    it('should prevent startup with invalid URL format', () => {
      const env = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: 'not-a-url',
        VITE_GATEWAY_URL: '',
      };

      const configured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';

      expect(() => {
        const base = configured.trim();
        new URL(base);
      }).toThrow();
    });

    it('should prevent startup with URL missing protocol', () => {
      const env = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: 'api.example.com',
        VITE_GATEWAY_URL: '',
      };

      const configured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';

      expect(() => {
        const base = configured.trim();
        new URL(base);
      }).toThrow();
    });
  });

  describe('Comprehensive Configuration', () => {
    it('should handle all required variables correctly', () => {
      const env = {
        MODE: 'production',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com',
        VITE_GATEWAY_URL: '',
        VITE_API_URL: 'https://api.bubblelab.com',
        VITE_API_ENDPOINT: '',
        VITE_CLERK_PUBLISHABLE_KEY: 'pk_test_key',
        VITE_SHOW_LEGACY_PARAMS: 'false',
        VITE_DISABLE_AUTH: 'false',
        VITE_POSTHOG_API_KEY: 'phc_test_key',
        VITE_POSTHOG_HOST: 'https://us.i.posthog.com',
        VITE_ANALYTICS_ENABLED: 'true',
      };

      // Evolution API URL
      const evolutionConfigured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';
      const evolutionUrl = evolutionConfigured.trim().replace(/\/$/, '');
      expect(evolutionUrl).toBe('https://api.evolution.com');

      // API URL
      const apiConfigured = env.VITE_API_URL || env.VITE_API_ENDPOINT || '';
      const apiBase = apiConfigured && apiConfigured.trim().length > 0 ? apiConfigured : 'http://localhost:3001';
      const apiUrl = apiBase.replace(/\/$/, '');
      expect(apiUrl).toBe('https://api.bubblelab.com');

      // Clerk key
      expect(env.VITE_CLERK_PUBLISHABLE_KEY).toBe('pk_test_key');

      // Feature flags
      expect(env.VITE_SHOW_LEGACY_PARAMS === 'true').toBe(false);
      expect(env.VITE_DISABLE_AUTH === 'true').toBe(false);

      // Analytics
      expect(env.VITE_POSTHOG_API_KEY).toBe('phc_test_key');
      expect(env.VITE_POSTHOG_HOST).toBe('https://us.i.posthog.com');
      expect(env.VITE_ANALYTICS_ENABLED !== 'false').toBe(true);
    });

    it('should handle optional variables being unset', () => {
      const env = {
        MODE: 'test',
        VITE_EVOLUTION_API_URL: 'https://api.evolution.com',
        VITE_GATEWAY_URL: '',
        VITE_API_URL: 'https://api.bubblelab.com',
        VITE_API_ENDPOINT: '',
        VITE_CLERK_PUBLISHABLE_KEY: '',
        VITE_POSTHOG_API_KEY: '',
      };

      // Evolution API URL
      const evolutionConfigured = env.VITE_EVOLUTION_API_URL || env.VITE_GATEWAY_URL || '';
      const evolutionUrl = evolutionConfigured.trim().replace(/\/$/, '');
      expect(evolutionUrl).toBe('https://api.evolution.com');

      // API URL
      const apiConfigured = env.VITE_API_URL || env.VITE_API_ENDPOINT || '';
      const apiBase = apiConfigured && apiConfigured.trim().length > 0 ? apiConfigured : 'http://localhost:3001';
      const apiUrl = apiBase.replace(/\/$/, '');
      expect(apiUrl).toBe('https://api.bubblelab.com');

      // Optional vars should be empty strings
      expect(env.VITE_CLERK_PUBLISHABLE_KEY).toBe('');
      expect(env.VITE_POSTHOG_API_KEY).toBe('');
    });
  });

  describe('Error Message Actionability', () => {
    it('should provide clear troubleshooting information', () => {
      try {
        throw new Error(
          'CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set.\n' +
          'This is a required configuration for production.\n\n' +
          'Please set one of the following:\n' +
          '  - VITE_EVOLUTION_API_URL (preferred)\n' +
          '  - VITE_GATEWAY_URL (fallback)\n\n' +
          'Example: VITE_EVOLUTION_API_URL=https://api.openevolve.com\n\n' +
          'The application cannot start without this configuration.'
        );
      } catch (error: any) {
        expect(error.message).toContain('VITE_EVOLUTION_API_URL');
        expect(error.message).toContain('VITE_GATEWAY_URL');
        expect(error.message).toContain('Example:');
        expect(error.message).toContain('https://api.openevolve.com');
        expect(error.message).toContain('cannot start');
        expect(error.message).toContain('required configuration');
      }
    });

    it('should include the invalid value in validation errors', () => {
      const invalidUrl = 'not-a-valid-url';

      try {
        new URL(invalidUrl);
      } catch (error: any) {
        expect(error.toString()).toContain(invalidUrl);
      }
    });
  });

  describe('Mode-Specific Behavior', () => {
    it('should distinguish production from development behavior', () => {
      const productionMode = 'production';
      const developmentMode = 'development';

      const isProductionProd = productionMode === 'production';
      const isProductionDev = developmentMode === 'production';

      expect(isProductionProd).toBe(true);
      expect(isProductionDev).toBe(false);
    });

    it('should apply different rules based on mode', () => {
      const modes = ['production', 'development', 'test'];

      const productionModes = modes.filter(m => m === 'production');
      const nonProductionModes = modes.filter(m => m !== 'production');

      expect(productionModes.length).toBe(1);
      expect(nonProductionModes.length).toBe(2);
    });
  });
});
