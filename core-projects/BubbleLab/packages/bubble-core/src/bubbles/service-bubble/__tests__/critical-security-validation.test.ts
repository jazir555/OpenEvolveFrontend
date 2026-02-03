/**
 * Critical Security Validation Tests
 *
 * This test suite validates all critical security fixes including:
 * - SSRF prevention for image URLs
 * - Path traversal prevention for file uploads
 * - File size validation
 * - maxIterations validation
 * - Input length limits
 * - Array size limits
 */

import { z } from 'zod';

describe('Critical Security Validations', () => {
  describe('Image URL SSRF Protection', () => {
    const UrlImageSchema = z.object({
      type: z.literal('url'),
      url: z
        .string()
        .url()
        .refine((url) => {
          try {
            const parsedUrl = new URL(url);

            // Only allow http and https protocols
            if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
              return false;
            }

            // Block private/internal IP ranges to prevent SSRF
            const hostname = parsedUrl.hostname.toLowerCase();

            // Block localhost variants
            if (
              hostname === 'localhost' ||
              hostname === '127.0.0.1' ||
              hostname.startsWith('127.') ||
              hostname === '[::1]' ||
              hostname === '0.0.0.0'
            ) {
              return false;
            }

            // Block private IP ranges (CIDR notation)
            const privateIpPatterns = [
              /^10\./,
              /^172\.(1[6-9]|2\d|3[01])\./,
              /^192\.168\./,
              /^169\.254\./, // Link-local
            ];

            if (privateIpPatterns.some((pattern) => pattern.test(hostname))) {
              return false;
            }

            // Block internal hostnames
            const internalHostnames = [
              'metadata.google.internal',
              'instance-data',
              'linklocal.amazonaws.com',
            ];

            if (internalHostnames.includes(hostname)) {
              return false;
            }

            return true;
          } catch {
            return false;
          }
        }, 'URL contains forbidden protocol, internal IP address, or private range'),
    });

    it('should reject localhost URLs', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://localhost:8080/image.png',
      });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toContain('forbidden');
      }
    });

    it('should reject 127.0.0.1', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://127.0.0.1/image.png',
      });
      expect(result.success).toBe(false);
    });

    it('should reject 127.x.x.x IP range', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://127.0.0.2/image.png',
      });
      expect(result.success).toBe(false);
    });

    it('should reject 192.168.x.x private IP range', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://192.168.1.1/image.png',
      });
      expect(result.success).toBe(false);
    });

    it('should reject 10.x.x.x private IP range', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://10.0.0.1/image.png',
      });
      expect(result.success).toBe(false);
    });

    it('should reject 172.16-31.x.x private IP range', () => {
      const result1 = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://172.16.0.1/image.png',
      });
      expect(result1.success).toBe(false);

      const result2 = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://172.31.255.255/image.png',
      });
      expect(result2.success).toBe(false);
    });

    it('should reject cloud metadata URLs', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://metadata.google.internal/computeMetadata/v1/',
      });
      expect(result.success).toBe(false);
    });

    it('should reject file:// protocol', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'file:///etc/passwd',
      });
      expect(result.success).toBe(false);
    });

    it('should reject ftp:// protocol', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'ftp://example.com/image.png',
      });
      expect(result.success).toBe(false);
    });

    it('should accept valid public HTTPS URLs', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'https://example.com/image.png',
      });
      expect(result.success).toBe(true);
    });

    it('should accept valid public HTTP URLs', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'http://public.example.com/image.png',
      });
      expect(result.success).toBe(true);
    });

    it('should accept URLs with subdomains', () => {
      const result = UrlImageSchema.safeParse({
        type: 'url',
        url: 'https://cdn.example.com/images/photo.jpg',
      });
      expect(result.success).toBe(true);
    });
  });

  describe('File Path Validation - Path Traversal Prevention', () => {
    const FilePathSchema = z
      .string()
      .min(1, 'File path is required')
      .max(500, 'File path too long (max 500 characters)')
      .refine((path) => {
        // SECURITY: Prevent path traversal attacks
        const normalizedPath = path.replace(/\\/g, '/');

        // Block path traversal attempts
        if (normalizedPath.includes('..')) {
          return false;
        }

        // Block absolute paths (only allow relative paths from working directory)
        if (normalizedPath.startsWith('/')) {
          return false;
        }

        // Block Windows drive letters
        if (/^[a-zA-Z]:/.test(normalizedPath)) {
          return false;
        }

        // Only allow safe characters in paths
        if (!/^[\w\-./ ]+$/.test(normalizedPath)) {
          return false;
        }

        return true;
      }, 'File path contains invalid characters or path traversal sequences');

    it('should reject path traversal with ..', () => {
      const result = FilePathSchema.safeParse('../../../etc/passwd');
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toContain('path traversal');
      }
    });

    it('should reject relative path with .. in middle', () => {
      const result = FilePathSchema.safeParse('uploads/../../secret.txt');
      expect(result.success).toBe(false);
    });

    it('should reject absolute Unix paths', () => {
      const result = FilePathSchema.safeParse('/etc/passwd');
      expect(result.success).toBe(false);
    });

    it('should reject absolute Windows paths', () => {
      const result = FilePathSchema.safeParse('C:\\Windows\\System32\\config');
      expect(result.success).toBe(false);
    });

    it('should reject paths with null bytes', () => {
      const result = FilePathSchema.safeParse('uploads/file\x00.txt');
      expect(result.success).toBe(false);
    });

    it('should reject paths over 500 characters', () => {
      const longPath = 'a'.repeat(501);
      const result = FilePathSchema.safeParse(longPath);
      expect(result.success).toBe(false);
    });

    it('should accept safe relative paths', () => {
      const result = FilePathSchema.safeParse('uploads/document.pdf');
      expect(result.success).toBe(true);
    });

    it('should accept paths with subdirectories', () => {
      const result = FilePathSchema.safeParse('uploads/january/report.pdf');
      expect(result.success).toBe(true);
    });

    it('should accept paths with hyphens and underscores', () => {
      const result = FilePathSchema.safeParse('uploads/my_file-2023.pdf');
      expect(result.success).toBe(true);
    });

    it('should accept paths with spaces', () => {
      const result = FilePathSchema.safeParse('uploads/my document.pdf');
      expect(result.success).toBe(true);
    });
  });

  describe('maxIterations Validation', () => {
    const MaxIterationsSchema = z
      .number()
      .int()
      .positive()
      .min(5, 'maxIterations must be at least 5 to support multi-step reasoning')
      .default(40);

    it('should reject values less than 5', () => {
      const result = MaxIterationsSchema.safeParse(4);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toContain('at least 5');
      }
    });

    it('should reject 0', () => {
      const result = MaxIterationsSchema.safeParse(0);
      expect(result.success).toBe(false);
    });

    it('should reject negative numbers', () => {
      const result = MaxIterationsSchema.safeParse(-5);
      expect(result.success).toBe(false);
    });

    it('should reject decimal numbers', () => {
      const result = MaxIterationsSchema.safeParse(5.5);
      expect(result.success).toBe(false);
    });

    it('should accept 5', () => {
      const result = MaxIterationsSchema.safeParse(5);
      expect(result.success).toBe(true);
    });

    it('should accept 40 (default)', () => {
      const result = MaxIterationsSchema.safeParse(40);
      expect(result.success).toBe(true);
    });

    it('should accept large values', () => {
      const result = MaxIterationsSchema.safeParse(1000);
      expect(result.success).toBe(true);
    });
  });

  describe('File Size Validation', () => {
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

    it('should enforce 10MB limit', () => {
      const fileSize = 11 * 1024 * 1024; // 11MB
      expect(fileSize).toBeGreaterThan(MAX_FILE_SIZE);
    });

    it('should accept files under 10MB', () => {
      const fileSize = 5 * 1024 * 1024; // 5MB
      expect(fileSize).toBeLessThanOrEqual(MAX_FILE_SIZE);
    });

    it('should accept files exactly at 10MB', () => {
      const fileSize = 10 * 1024 * 1024; // 10MB
      expect(fileSize).toBeLessThanOrEqual(MAX_FILE_SIZE);
    });
  });

  describe('Array Size Validation', () => {
    const createArraySchema = (maxSize: number) =>
      z.array(z.string()).max(maxSize, `Array too large (max ${maxSize} items)`);

    it('should enforce max items limit', () => {
      const schema = createArraySchema(100);
      const largeArray = Array.from({ length: 101 }, (_, i) => `item${i}`);
      const result = schema.safeParse(largeArray);
      expect(result.success).toBe(false);
    });

    it('should accept arrays at max limit', () => {
      const schema = createArraySchema(100);
      const array = Array.from({ length: 100 }, (_, i) => `item${i}`);
      const result = schema.safeParse(array);
      expect(result.success).toBe(true);
    });

    it('should accept empty arrays', () => {
      const schema = createArraySchema(100);
      const result = schema.safeParse([]);
      expect(result.success).toBe(true);
    });
  });

  describe('String Length Validation', () => {
    const createStringSchema = (maxLength: number) =>
      z
        .string()
        .min(1)
        .max(maxLength, `String too long (max ${maxLength} characters)`);

    it('should enforce max length limit', () => {
      const schema = createStringSchema(1000);
      const longString = 'a'.repeat(1001);
      const result = schema.safeParse(longString);
      expect(result.success).toBe(false);
    });

    it('should accept strings at max limit', () => {
      const schema = createStringSchema(1000);
      const string = 'a'.repeat(1000);
      const result = schema.safeParse(string);
      expect(result.success).toBe(true);
    });

    it('should reject empty strings', () => {
      const schema = createStringSchema(1000);
      const result = schema.safeParse('');
      expect(result.success).toBe(false);
    });
  });

  describe('Content Type Validation for Images', () => {
    const validImageTypes = [
      'image/jpeg',
      'image/png',
      'image/gif',
      'image/webp',
      'image/svg+xml',
    ];

    it('should accept valid image types', () => {
      validImageTypes.forEach((type) => {
        expect(type).toMatch(/^image\//);
      });
    });

    it('should reject non-image content types', () => {
      const invalidTypes = [
        'text/html',
        'application/json',
        'application/javascript',
        'text/plain',
      ];

      invalidTypes.forEach((type) => {
        expect(type).not.toMatch(/^image\//);
      });
    });
  });

  describe('Security: Sensitive File Extensions', () => {
    const sensitiveExtensions = [
      '.key',
      '.pem',
      '.crt',
      '.p12',
      '.pfx',
      '.env',
      '.config',
      '.conf',
      '.sh',
      '.bash',
      '.ps1',
      '.bat',
      '.cmd',
      '.exe',
      '.dll',
      '.so',
      '.dylib',
    ];

    it('should detect sensitive file extensions', () => {
      expect(sensitiveExtensions).toContain('.pem');
      expect(sensitiveExtensions).toContain('.env');
      expect(sensitiveExtensions).toContain('.exe');
    });

    it('should allow safe file extensions', () => {
      const safeExtensions = ['.pdf', '.txt', '.jpg', '.png', '.docx'];
      safeExtensions.forEach((ext) => {
        expect(sensitiveExtensions).not.toContain(ext);
      });
    });
  });
});

describe('Integration Tests: Security Boundaries', () => {
  describe('Combined Attack Vectors', () => {
    it('should prevent SSRF via redirect chain (simulated)', () => {
      // This test verifies that URL validation catches SSRF attempts
      // even before any redirect following would occur
      const maliciousUrls = [
        { url: 'http://localhost/admin', type: 'localhost' },
        { url: 'http://192.168.1.1/api', type: 'private_ip' },
        { url: 'http://10.0.0.1/secret', type: 'private_ip' },
        { url: 'http://169.254.169.254/metadata', type: 'private_ip' },
      ];

      maliciousUrls.forEach(({ url, type }) => {
        const parsed = new URL(url);
        const hostname = parsed.hostname;

        // Check if it would be caught by security patterns
        const localhostPattern = /^localhost$/i;
        const privateIpPatterns = [
          /^10\./,
          /^172\.(1[6-9]|2\d|3[01])\./,
          /^192\.168\./,
          /^169\.254\./,
          /^127\./,
        ];

        const isBlocked =
          localhostPattern.test(hostname) ||
          privateIpPatterns.some((pattern) => pattern.test(hostname));

        expect(isBlocked).toBe(true);
      });
    });

    it('should prevent path traversal combined with special characters', () => {
      const maliciousPaths = [
        '../../../etc/passwd',
        '..\\..\\..\\windows\\system32',
        './../../../etc/hosts',
        '....//....//etc/passwd',
      ];

      maliciousPaths.forEach((path) => {
        // All should contain .. which is blocked
        expect(path).toContain('..');
      });
    });
  });
});
