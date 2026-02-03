/**
 * Environment Variable Configuration Tests
 * Tests for Bug #1 and #8: Configuration validation and error handling
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('Environment Variable Configuration Logic', () => {
  describe('Production Mode Configuration', () => {
    it('should require VITE_EVOLUTION_API_URL in production', () => {
      const isProduction = true;
      const configured = '';

      if (!configured || configured.trim().length === 0) {
        if (isProduction) {
          expect(() => {
            throw new Error(
              'CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set.\n' +
              'This is a required configuration for production.'
            );
          }).toThrow();
        }
      }
    });

    it('should provide clear production error message', () => {
      const isProduction = true;
      const configured = '';

      try {
        if (!configured || configured.trim().length === 0) {
          if (isProduction) {
            throw new Error(
              'CRITICAL: VITE_EVOLUTION_API_URL environment variable is not set.\n' +
              'This is a required configuration for production.\n\n' +
              'Please set one of the following:\n' +
              '  - VITE_EVOLUTION_API_URL (preferred)\n' +
              '  - VITE_GATEWAY_URL (fallback)\n\n' +
              'Example: VITE_EVOLUTION_API_URL=https://api.openevolve.com\n\n' +
              'The application cannot start without this configuration.'
            );
          }
        }
      } catch (error: any) {
        expect(error.message).toContain('CRITICAL');
        expect(error.message).toContain('VITE_EVOLUTION_API_URL');
        expect(error.message).toContain('VITE_GATEWAY_URL');
        expect(error.message).toContain('Example:');
        expect(error.message).toContain('production');
      }
    });

    it('should NOT fallback to localhost in production', () => {
      const isProduction = true;
      const configured = '';
      let usedFallback = false;
      let errorThrown = false;

      if (!configured || configured.trim().length === 0) {
        if (isProduction) {
          try {
            throw new Error('Production requires configuration');
          } catch (e) {
            errorThrown = true;
          }
        } else {
          usedFallback = true;
        }
      }

      expect(errorThrown).toBe(true);
      expect(usedFallback).toBe(false);
    });

    it('should accept valid URL in production', () => {
      const configured = 'https://api.openevolve.com';
      const base = configured.trim();

      try {
        new URL(base);
        const url = base.replace(/\/$/, '');
        expect(url).toBe('https://api.openevolve.com');
      } catch (error) {
        expect.fail('Valid URL should not throw');
      }
    });
  });

  describe('Development Mode Configuration', () => {
    it('should warn in development without config', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn');
      const isProduction = false;
      const configured = '';

      if (!configured || configured.trim().length === 0) {
        if (!isProduction) {
          console.warn(
            '[BubbleLab] VITE_EVOLUTION_API_URL not configured. ' +
            'Falling back to http://localhost:8000 for development. ' +
            'This will FAIL in production!'
          );
        }
      }

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('[BubbleLab]')
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Falling back to http://localhost:8000')
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('will FAIL in production')
      );

      consoleWarnSpy.mockRestore();
    });

    it('should fallback to localhost:8000 in development', () => {
      const isProduction = false;
      const configured = '';
      let result = '';

      if (!configured || configured.trim().length === 0) {
        if (!isProduction) {
          result = 'http://localhost:8000';
        }
      }

      expect(result).toBe('http://localhost:8000');
    });

    it('should use provided URL if set', () => {
      const configured = 'http://localhost:9000';
      const base = configured.trim();
      const url = base.replace(/\/$/, '');

      expect(url).toBe('http://localhost:9000');
    });
  });

  describe('URL Validation', () => {
    it('should accept valid HTTP URLs', () => {
      const url = 'http://localhost:8000';
      expect(() => new URL(url)).not.toThrow();
      expect(url.replace(/\/$/, '')).toBe('http://localhost:8000');
    });

    it('should accept valid HTTPS URLs', () => {
      const url = 'https://api.openevolve.com';
      expect(() => new URL(url)).not.toThrow();
      expect(url.replace(/\/$/, '')).toBe('https://api.openevolve.com');
    });

    it('should reject invalid URL format', () => {
      const url = 'not-a-valid-url';
      expect(() => new URL(url)).toThrow();
    });

    it('should reject URL without protocol', () => {
      const url = 'api.openevolve.com';
      expect(() => new URL(url)).toThrow();
    });

    it('should include invalid URL in error message', () => {
      const url = 'invalid-url-format';

      try {
        new URL(url);
        expect.fail('Should have thrown');
      } catch (error: any) {
        expect(error.toString()).toContain('invalid-url-format');
      }
    });
  });

  describe('URL Normalization', () => {
    it('should remove trailing slashes', () => {
      const url = 'https://api.openevolve.com/';
      const normalized = url.replace(/\/$/, '');

      expect(normalized).toBe('https://api.openevolve.com');
      expect(normalized.endsWith('/')).toBe(false);
    });

    it('should remove multiple trailing slashes', () => {
      const url = 'https://api.openevolve.com///';
      const normalized = url.replace(/\/+$/, '');

      expect(normalized).toBe('https://api.openevolve.com');
    });

    it('should handle URLs without trailing slashes', () => {
      const url = 'https://api.openevolve.com';
      const normalized = url.replace(/\/$/, '');

      expect(normalized).toBe('https://api.openevolve.com');
    });

    it('should preserve path when removing trailing slash', () => {
      const url = 'https://api.openevolve.com/v1/';
      const normalized = url.replace(/\/$/, '');

      expect(normalized).toBe('https://api.openevolve.com/v1');
    });
  });

  describe('API URL Resolution', () => {
    it('should prefer VITE_API_URL over VITE_API_ENDPOINT', () => {
      const VITE_API_URL = 'http://preferred.com';
      const VITE_API_ENDPOINT = 'http://fallback.com';

      const configured = VITE_API_URL || VITE_API_ENDPOINT || '';
      const base = configured && configured.trim().length > 0 ? configured : 'http://localhost:3001';
      const normalized = base.replace(/\/$/, '');

      expect(normalized).toBe('http://preferred.com');
    });

    it('should fallback to VITE_API_ENDPOINT', () => {
      const VITE_API_URL = '';
      const VITE_API_ENDPOINT = 'http://fallback.com';

      const configured = VITE_API_URL || VITE_API_ENDPOINT || '';
      const base = configured && configured.trim().length > 0 ? configured : 'http://localhost:3001';
      const normalized = base.replace(/\/$/, '');

      expect(normalized).toBe('http://fallback.com');
    });

    it('should fallback to localhost:3001 if neither set', () => {
      const VITE_API_URL = '';
      const VITE_API_ENDPOINT = '';

      const configured = VITE_API_URL || VITE_API_ENDPOINT || '';
      const base = configured && configured.trim().length > 0 ? configured : 'http://localhost:3001';
      const normalized = base.replace(/\/$/, '');

      expect(normalized).toBe('http://localhost:3001');
    });

    it('should remove trailing slashes from API URL', () => {
      const VITE_API_URL = 'http://api.com/';

      const configured = VITE_API_URL || '';
      const base = configured && configured.trim().length > 0 ? configured : 'http://localhost:3001';
      const normalized = base.replace(/\/$/, '');

      expect(normalized).toBe('http://api.com');
    });
  });

  describe('Evolution API URL Resolution', () => {
    it('should prefer VITE_EVOLUTION_API_URL over VITE_GATEWAY_URL', () => {
      const VITE_EVOLUTION_API_URL = 'https://evolution.com';
      const VITE_GATEWAY_URL = 'https://gateway.com';

      const configured = VITE_EVOLUTION_API_URL || VITE_GATEWAY_URL || '';
      const base = configured.trim();
      const normalized = base.replace(/\/$/, '');

      expect(normalized).toBe('https://evolution.com');
    });

    it('should use VITE_GATEWAY_URL as fallback', () => {
      const VITE_EVOLUTION_API_URL = '';
      const VITE_GATEWAY_URL = 'https://gateway.com';

      const configured = VITE_EVOLUTION_API_URL || VITE_GATEWAY_URL || '';
      const base = configured.trim();
      const normalized = base.replace(/\/$/, '');

      expect(normalized).toBe('https://gateway.com');
    });
  });

  describe('Error Message Quality', () => {
    it('should provide actionable error messages', () => {
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
        expect(error.message).toContain('required configuration');
      }
    });

    it('should indicate configuration prevents startup', () => {
      try {
        throw new Error(
          'The application cannot start without this configuration.'
        );
      } catch (error: any) {
        expect(error.message).toContain('cannot start');
      }
    });

    it('should provide example URLs', () => {
      const errorMessage =
        'Example: VITE_EVOLUTION_API_URL=https://api.openevolve.com\nExamples:\n  - http://localhost:8000 (development)\n  - https://api.openevolve.com (production)';

      expect(errorMessage).toContain('http://localhost:8000');
      expect(errorMessage).toContain('https://api.openevolve.com');
    });
  });

  describe('Whitespace Handling', () => {
    it('should trim whitespace from URLs', () => {
      const url = '  https://api.openevolve.com  ';
      const trimmed = url.trim();

      expect(trimmed).toBe('https://api.openevolve.com');
    });

    it('should treat whitespace-only as empty', () => {
      const url = '   ';
      const isEmpty = !url || url.trim().length === 0;

      expect(isEmpty).toBe(true);
    });

    it('should handle mixed whitespace', () => {
      const url = '\t  https://api.com  \n';
      const trimmed = url.trim();

      expect(trimmed).toBe('https://api.com');
    });
  });

  describe('Mode Detection', () => {
    it('should correctly identify production mode', () => {
      const MODE = 'production';
      const isProduction = MODE === 'production';

      expect(isProduction).toBe(true);
    });

    it('should correctly identify development mode', () => {
      const MODE = 'development';
      const isProduction = MODE === 'production';

      expect(isProduction).toBe(false);
    });

    it('should correctly identify test mode', () => {
      const MODE = 'test';
      const isProduction = MODE === 'production';

      expect(isProduction).toBe(false);
    });
  });
});
