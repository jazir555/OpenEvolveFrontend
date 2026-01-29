/**
 * Error Sanitization Service
 * Provides security-focused sanitization of error messages and data
 */

// Define sensitive data patterns
const SENSITIVE_PATTERNS = [
  /password[:=][^&\s]+/gi,
  /token[:=][^&\s]+/gi,
  /secret[:=][^&\s]+/gi,
  /key[:=][^&\s]+/gi,
  /api[_-]?key[:=][^&\s]+/gi,
  /auth[_-]?token[:=][^&\s]+/gi,
  /bearer\s+[a-zA-Z0-9\._-]+/gi,
  /client[_-]?secret[:=][^&\s]+/gi,
  /access[_-]?token[:=][^&\s]+/gi,
  /refresh[_-]?token[:=][^&\s]+/gi,
  /session[:=][^&\s]+/gi,
  /cookie[:=][^&\s]+/gi,
  /authorization[:=][^&\s]+/gi,
  /x[_-]?api[_-]?key[:=][^&\s]+/gi,
  /user[:=][^&\s]+/gi,  // Could be sensitive in some contexts
  /email[:=][^&\s]+/gi,  // Could be sensitive in some contexts
  /phone[:=][^&\s]+/gi,  // Could be sensitive in some contexts
  /ssn[:=][^&\s]+/gi,    // Social Security Number
  /card[:=][^&\s]+/gi,   // Credit card number
  /cvv[:=][^&\s]+/gi,    // Credit card CVV
  /cvv2[:=][^&\s]+/gi,   // Credit card CVV2
  /cvc[:=][^&\s]+/gi,    // Credit card CVC
  /pin[:=][^&\s]+/gi,    // PIN number
  /cv2[:=][^&\s]+/gi,    // Credit card CV2
  /cav2[:=][^&\s]+/gi,   // Credit card CAV2
  /cvc2[:=][^&\s]+/gi,   // Credit card CVC2
  /start_date[:=][^&\s]+/gi,  // Credit card start date
  /issue_number[:=][^&\s]+/gi, // Credit card issue number
  /account[:=][^&\s]+/gi, // Account number
  /routing[:=][^&\s]+/gi, // Routing number
  /iban[:=][^&\s]+/gi,   // International Bank Account Number
  /swift[:=][^&\s]+/gi,  // SWIFT code
  /bic[:=][^&\s]+/gi,    // Bank Identifier Code
  /tax[:=][^&\s]+/gi,    // Tax ID
  /ein[:=][^&\s]+/gi,    // Employer Identification Number
  /itin[:=][^&\s]+/gi,   // Individual Taxpayer Identification Number
  /imei[:=][^&\s]+/gi,   // International Mobile Equipment Identity
  /imsi[:=][^&\s]+/gi,   // International Mobile Subscriber Identity
  /meid[:=][^&\s]+/gi,   // Mobile Equipment Identifier
  /mac[:=][^&\s]+/gi,    // Media Access Control address
  /ip[:=][^&\s]+/gi,     // IP address (in some contexts)
  /address[:=][^&\s]+/gi, // Physical address (in some contexts)
  /location[:=][^&\s]+/gi, // Location data (in some contexts)
];

// Define sensitive headers to redact
const SENSITIVE_HEADERS = [
  'authorization',
  'x-api-key',
  'x-auth-token',
  'x-csrf-token',
  'x-xsrf-token',
  'cookie',
  'set-cookie',
  'authentication',
  'www-authenticate',
  'proxy-authenticate',
  'proxy-authorization',
  'x-access-token',
  'x-refresh-token',
  'x-session-token',
  'x-security-token',
  'x-api-secret',
  'x-api-key-secret',
  'x-client-secret',
  'x-oauth-token',
  'x-bearer-token',
  'x-jwt-token',
  'x-id-token',
  'x-api-password',
  'x-api-username',
  'x-api-user',
  'x-api-client-id',
  'x-api-client-secret',
  'x-api-app-id',
  'x-api-app-secret',
  'x-api-key-id',
  'x-api-key-secret',
  'x-api-access-key',
  'x-api-secret-key',
  'x-api-session-id',
  'x-api-session-token',
  'x-api-auth-token',
  'x-api-bearer-token',
  'x-api-oauth-token',
  'x-api-jwt-token',
  'x-api-id-token',
  'x-api-refresh-token',
  'x-api-access-token',
  'x-api-password',
  'x-api-username',
  'x-api-user',
  'x-api-client-id',
  'x-api-client-secret',
  'x-api-app-id',
  'x-api-app-secret',
  'x-api-key-id',
  'x-api-key-secret',
  'x-api-access-key',
  'x-api-secret-key',
  'x-api-session-id',
  'x-api-session-token',
];

/**
 * Error Sanitization Service
 * Provides security-focused sanitization of error messages and data
 */
export class ErrorSanitizationService {
  private static instance: ErrorSanitizationService;

  private constructor() {}

  /**
   * Get singleton instance
   */
  static getInstance(): ErrorSanitizationService {
    if (!ErrorSanitizationService.instance) {
      ErrorSanitizationService.instance = new ErrorSanitizationService();
    }
    return ErrorSanitizationService.instance;
  }

  /**
   * Sanitize an error object
   */
  sanitizeError(error: any): any {
    if (!error) return error;

    // Create a copy to avoid modifying the original
    const sanitizedError = { ...error };

    // Sanitize message
    if (sanitizedError.message) {
      sanitizedError.message = this.sanitizeString(sanitizedError.message);
    }

    // Sanitize stack trace
    if (sanitizedError.stack) {
      sanitizedError.stack = this.sanitizeString(sanitizedError.stack);
    }

    // Sanitize additional properties
    if (sanitizedError.config) {
      sanitizedError.config = this.sanitizeConfig(sanitizedError.config);
    }

    if (sanitizedError.request) {
      sanitizedError.request = this.sanitizeRequest(sanitizedError.request);
    }

    if (sanitizedError.response) {
      sanitizedError.response = this.sanitizeResponse(sanitizedError.response);
    }

    // Sanitize any additional properties
    for (const key in sanitizedError) {
      if (typeof sanitizedError[key] === 'string') {
        sanitizedError[key] = this.sanitizeString(sanitizedError[key]);
      } else if (typeof sanitizedError[key] === 'object' && sanitizedError[key] !== null) {
        sanitizedError[key] = this.sanitizeObject(sanitizedError[key]);
      }
    }

    return sanitizedError;
  }

  /**
   * Sanitize a string by removing sensitive information
   */
  sanitizeString(str: string): string {
    if (typeof str !== 'string') return str;

    let sanitized = str;

    // Apply all sensitive patterns
    for (const pattern of SENSITIVE_PATTERNS) {
      sanitized = sanitized.replace(pattern, (match) => {
        // Preserve the key but redact the value
        const separatorIndex = Math.max(match.indexOf('='), match.indexOf(':'));
        if (separatorIndex !== -1) {
          const key = match.substring(0, separatorIndex + 1);
          return `${key}[REDACTED]`;
        }
        return '[REDACTED]';
      });
    }

    return sanitized;
  }

  /**
   * Sanitize a configuration object
   */
  sanitizeConfig(config: any): any {
    if (!config || typeof config !== 'object') return config;

    const sanitized = { ...config };

    // Sanitize URL
    if (sanitized.url) {
      sanitized.url = this.sanitizeUrl(sanitized.url);
    }

    // Sanitize headers
    if (sanitized.headers) {
      sanitized.headers = this.sanitizeHeaders(sanitized.headers);
    }

    // Sanitize data
    if (sanitized.data) {
      sanitized.data = this.sanitizeData(sanitized.data);
    }

    // Sanitize params
    if (sanitized.params) {
      sanitized.params = this.sanitizeData(sanitized.params);
    }

    return sanitized;
  }

  /**
   * Sanitize request object
   */
  sanitizeRequest(request: any): any {
    if (!request || typeof request !== 'object') return request;

    const sanitized = { ...request };

    // Sanitize headers
    if (sanitized.headers) {
      sanitized.headers = this.sanitizeHeaders(sanitized.headers);
    }

    // Sanitize URL
    if (sanitized.url) {
      sanitized.url = this.sanitizeUrl(sanitized.url);
    }

    // Sanitize body/data
    if (sanitized.body) {
      sanitized.body = this.sanitizeData(sanitized.body);
    }

    return sanitized;
  }

  /**
   * Sanitize response object
   */
  sanitizeResponse(response: any): any {
    if (!response || typeof response !== 'object') return response;

    const sanitized = { ...response };

    // Sanitize headers
    if (sanitized.headers) {
      sanitized.headers = this.sanitizeHeaders(sanitized.headers);
    }

    // Sanitize data
    if (sanitized.data) {
      sanitized.data = this.sanitizeData(sanitized.data);
    }

    return sanitized;
  }

  /**
   * Sanitize headers object
   */
  sanitizeHeaders(headers: any): any {
    if (!headers || typeof headers !== 'object') return headers;

    const sanitized: any = {};

    for (const [key, value] of Object.entries(headers)) {
      const lowerKey = key.toLowerCase();

      if (SENSITIVE_HEADERS.includes(lowerKey)) {
        sanitized[key] = '[REDACTED]';
      } else {
        sanitized[key] = typeof value === 'string' ? this.sanitizeString(value as string) : value;
      }
    }

    return sanitized;
  }

  /**
   * Sanitize URL by removing sensitive query parameters
   */
  sanitizeUrl(url: string): string {
    if (typeof url !== 'string') return url;

    try {
      const urlObj = new URL(url);
      const searchParams = new URLSearchParams(urlObj.search);

      // Remove sensitive query parameters
      for (const [key] of searchParams.entries()) {
        const lowerKey = key.toLowerCase();
        if (SENSITIVE_PATTERNS.some(pattern => pattern.test(lowerKey))) {
          searchParams.set(key, '[REDACTED]');
        }
      }

      urlObj.search = searchParams.toString();
      return urlObj.toString();
    } catch (e) {
      // If URL parsing fails, sanitize as string
      return this.sanitizeString(url);
    }
  }

  /**
   * Sanitize data object/array
   */
  sanitizeData(data: any): any {
    if (!data) return data;

    if (typeof data === 'string') {
      return this.sanitizeString(data);
    }

    if (Array.isArray(data)) {
      return data.map(item => this.sanitizeData(item));
    }

    if (typeof data === 'object') {
      const sanitized: any = {};

      for (const [key, value] of Object.entries(data)) {
        const lowerKey = key.toLowerCase();

        // Check if key matches sensitive patterns
        if (SENSITIVE_PATTERNS.some(pattern => pattern.test(lowerKey))) {
          sanitized[key] = '[REDACTED]';
        } else {
          sanitized[key] = typeof value === 'string' 
            ? this.sanitizeString(value) 
            : this.sanitizeData(value);
        }
      }

      return sanitized;
    }

    return data;
  }

  /**
   * Sanitize an object recursively
   */
  sanitizeObject(obj: any): any {
    if (!obj || typeof obj !== 'object') return obj;

    if (Array.isArray(obj)) {
      return obj.map(item => this.sanitizeObject(item));
    }

    const sanitized: any = {};

    for (const [key, value] of Object.entries(obj)) {
      sanitized[key] = typeof value === 'string' 
        ? this.sanitizeString(value) 
        : this.sanitizeObject(value);
    }

    return sanitized;
  }

  /**
   * Check if a string contains sensitive information
   */
  containsSensitiveInfo(str: string): boolean {
    if (typeof str !== 'string') return false;

    return SENSITIVE_PATTERNS.some(pattern => pattern.test(str));
  }

  /**
   * Add a custom sensitive pattern
   */
  addSensitivePattern(pattern: RegExp): void {
    SENSITIVE_PATTERNS.push(pattern);
  }

  /**
   * Add a custom sensitive header
   */
  addSensitiveHeader(header: string): void {
    SENSITIVE_HEADERS.push(header.toLowerCase());
  }

  /**
   * Get all sensitive patterns
   */
  getSensitivePatterns(): RegExp[] {
    return [...SENSITIVE_PATTERNS];
  }

  /**
   * Get all sensitive headers
   */
  getSensitiveHeaders(): string[] {
    return [...SENSITIVE_HEADERS];
  }
}

// Create a singleton instance
export const errorSanitizationService = ErrorSanitizationService.getInstance();

/**
 * Helper function to sanitize an error
 */
export function sanitizeError(error: any): any {
  return errorSanitizationService.sanitizeError(error);
}

/**
 * Helper function to sanitize a string
 */
export function sanitizeString(str: string): string {
  return errorSanitizationService.sanitizeString(str);
}

/**
 * Helper function to sanitize data
 */
export function sanitizeData(data: any): any {
  return errorSanitizationService.sanitizeData(data);
}