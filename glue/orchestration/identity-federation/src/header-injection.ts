/**
 * Header Injection Authentication (OAuth2-Proxy Pattern)
 *
 * Federation Constitution - Identity Federation Strategy (ADR-006)
 * Phase 2: Header Injection Fallback
 *
 * When services don't support OIDC natively, use an auth sidecar
 * that validates tokens and injects user information in headers.
 *
 * Architecture:
 *   User → Frontend → OAuth2-Proxy (Sidecar) → Backend Services
 *                    ↓ validates token
 *                    ↓ injects headers:
 *                      X-Remote-User: username
 *                      X-Remote-Email: email
 *                      X-Remote-Groups: groups
 *
 * Security: Services MUST only trust headers from the sidecar
 * (enforced via network isolation)
 */

import { logger, LoggerContext } from '../../lib/logger';

export interface HeaderInjectionConfig {
  // OAuth2-Proxy configuration
  cookieName?: string;
  cookieSecret?: string;
  cookieRefresh?: string;
  cookieExpire?: string;

  // Header names (can be customized)
  userHeader?: string;
  emailHeader?: string;
  groupsHeader?: string;

  // Security
  requireAuth?: boolean;
  whitelistDomains?: string[];
}

export interface InjectedHeaders {
  'X-Remote-User'?: string;
  'X-Remote-Email'?: string;
  'X-Remote-Groups'?: string;
  'X-Remote-Access-Token'?: string;
}

export class HeaderInjectionAuth {
  private config: HeaderInjectionConfig;
  private loggerContext: LoggerContext;

  constructor(config: HeaderInjectionConfig = {}) {
    this.config = {
      cookieName: '_oauth2_proxy',
      cookieSecret: process.env.OAUTH2_PROXY_COOKIE_SECRET,
      cookieRefresh: '5m',
      cookieExpire: '168h', // 7 days
      userHeader: 'X-Remote-User',
      emailHeader: 'X-Remote-Email',
      groupsHeader: 'X-Remote-Groups',
      requireAuth: true,
      whitelistDomains: [],
      ...config,
    };

    this.loggerContext = {
      correlation_id: `header-injection-${Date.now()}`,
      source_service: 'header-injection-auth',
      target_service: 'oauth2-proxy',
    };

    logger.info('Header Injection Auth initialized', {
      ...this.loggerContext,
      require_auth: this.config.requireAuth,
    });
  }

  /**
   * Extract user information from injected headers
   * Called by backend services to get authenticated user context
   */
  extractUserFromHeaders(headers: Record<string, string>): InjectedHeaders {
    const user = headers[this.config.userHeader!];
    const email = headers[this.config.emailHeader!];
    const groups = headers[this.config.groupsHeader!];
    const accessToken = headers['X-Remote-Access-Token'];

    logger.debug('Extracted user from headers', {
      ...this.loggerContext,
      user,
      email,
      has_groups: !!groups,
    });

    return {
      ...(user && { 'X-Remote-User': user }),
      ...(email && { 'X-Remote-Email': email }),
      ...(groups && { 'X-Remote-Groups': groups }),
      ...(accessToken && { 'X-Remote-Access-Token': accessToken }),
    };
  }

  /**
   * Validate that headers are present
   * Returns false if authentication is required but missing
   */
  validateHeaders(headers: Record<string, string>): boolean {
    if (!this.config.requireAuth) {
      return true;
    }

    const user = headers[this.config.userHeader!];
    const isValid = !!user;

    if (!isValid) {
      logger.warn('Missing required auth headers', {
        ...this.loggerContext,
        expected_header: this.config.userHeader,
      });
    }

    return isValid;
  }

  /**
   * Parse groups from header
   * Groups are typically comma-separated
   */
  parseGroups(headers: Record<string, string>): string[] {
    const groupsHeader = headers[this.config.groupsHeader!];

    if (!groupsHeader) {
      return [];
    }

    const groups = groupsHeader.split(',').map(g => g.trim());

    logger.debug('Parsed groups from headers', {
      ...this.loggerContext,
      group_count: groups.length,
    });

    return groups;
  }

  /**
   * Check if user has required group/role
   */
  hasGroup(headers: Record<string, string>, requiredGroup: string): boolean {
    const groups = this.parseGroups(headers);
    const hasGroup = groups.includes(requiredGroup);

    if (!hasGroup) {
      logger.debug('User missing required group', {
        ...this.loggerContext,
        required_group: requiredGroup,
        user_groups: groups,
      });
    }

    return hasGroup;
  }

  /**
   * Check if user has any of the required groups
   */
  hasAnyGroup(headers: Record<string, string>, requiredGroups: string[]): boolean {
    const groups = this.parseGroups(headers);
    const hasAny = requiredGroups.some(g => groups.includes(g));

    if (!hasAny) {
      logger.debug('User missing any required groups', {
        ...this.loggerContext,
        required_groups: requiredGroups,
        user_groups: groups,
      });
    }

    return hasAny;
  }

  /**
   * Middleware wrapper for Express-like frameworks
   * Validates headers and adds user context to request
   */
  createMiddleware(requiredGroups?: string[]) {
    return async (req: any, res: any, next: any) => {
      const headers = req.headers || {};

      // Validate authentication
      if (!this.validateHeaders(headers)) {
        logger.warn('Unauthorized request - missing auth headers', {
          ...this.loggerContext,
          path: req.path,
          method: req.method,
        });

        return res.status(401).json({
          error: 'Unauthorized',
          message: 'Authentication required',
        });
      }

      // Extract user info
      const userInfo = this.extractUserFromHeaders(headers);
      req.user = {
        username: userInfo['X-Remote-User'],
        email: userInfo['X-Remote-Email'],
        groups: this.parseGroups(headers),
        accessToken: userInfo['X-Remote-Access-Token'],
      };

      // Validate groups if required
      if (requiredGroups && requiredGroups.length > 0) {
        if (!this.hasAnyGroup(headers, requiredGroups)) {
          logger.warn('Forbidden request - missing required groups', {
            ...this.loggerContext,
            required_groups: requiredGroups,
            user_groups: req.user.groups,
            path: req.path,
            method: req.method,
          });

          return res.status(403).json({
            error: 'Forbidden',
            message: 'Insufficient permissions',
          });
        }
      }

      logger.debug('Request authenticated via header injection', {
        ...this.loggerContext,
        user: req.user.username,
        path: req.path,
        method: req.method,
      });

      next();
    };
  }

  /**
   * Create auth headers for API proxying
   * Used when frontend makes requests to backend through sidecar
   */
  createProxyHeaders(accessToken: string): Record<string, string> {
    return {
      'Authorization': `Bearer ${accessToken}`,
    };
  }

  /**
   * Validate OAuth2 proxy cookie
   * Used to verify cookie signature (if cookie secret is available)
   */
  validateCookie(cookie: string): boolean {
    if (!this.config.cookieSecret) {
      logger.warn('Cannot validate cookie - no secret configured', this.loggerContext);
      return true; // Allow if no validation configured
    }

    try {
      // OAuth2 proxy cookies are typically: cookie_value|timestamp|hex_signature
      const parts = cookie.split('|');
      if (parts.length !== 3) {
        return false;
      }

      const [value, timestamp, signature] = parts;
      const expectedSignature = this.computeCookieSignature(value, timestamp);

      const isValid = signature === expectedSignature;

      if (!isValid) {
        logger.warn('Cookie signature validation failed', {
          ...this.loggerContext,
          timestamp,
        });
      }

      return isValid;
    } catch (error) {
      logger.error('Cookie validation error', error as Error, this.loggerContext);
      return false;
    }
  }

  /**
   * Compute OAuth2 proxy cookie signature
   * HMAC-SHA256 of value|timestamp with cookie secret
   */
  private computeCookieSignature(value: string, timestamp: string): string {
    // This is a simplified version
    // In production, use crypto.subtle.sign() for proper HMAC
    const data = `${value}|${timestamp}`;
    const combined = data + this.config.cookieSecret;

    // Simple hash (not cryptographically secure, use HMAC in production)
    let hash = 0;
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash &= hash; // Convert to 32bit integer
    }

    return Math.abs(hash).toString(16);
  }

  /**
   * Check if request is from whitelisted domain
   * Used for CORS and security validation
   */
  isWhitelistedDomain(origin: string): boolean {
    if (!this.config.whitelistDomains || this.config.whitelistDomains.length === 0) {
      return true; // No whitelist configured
    }

    const isWhitelisted = this.config.whitelistDomains.some(domain => {
      return origin === domain || origin.endsWith(`.${domain}`);
    });

    if (!isWhitelisted) {
      logger.warn('Origin not in whitelist', {
        ...this.loggerContext,
        origin,
        whitelist: this.config.whitelistDomains,
      });
    }

    return isWhitelisted;
  }
}

/**
 * Example usage:
 *
 * ```typescript
 * const headerAuth = new HeaderInjectionAuth({
 *   requireAuth: true,
 *   userHeader: 'X-Remote-User',
 *   emailHeader: 'X-Remote-Email',
 *   groupsHeader: 'X-Remote-Groups',
 * });
 *
 * // Express middleware
 * app.use(headerAuth.createMiddleware(['admin', 'users']));
 *
 * // Manual validation
 * const headers = request.headers;
 * if (headerAuth.validateHeaders(headers)) {
 *   const user = headerAuth.extractUserFromHeaders(headers);
 *   const groups = headerAuth.parseGroups(headers);
 * }
 * ```
 */
