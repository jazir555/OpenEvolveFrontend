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
export interface HeaderInjectionConfig {
    cookieName?: string;
    cookieSecret?: string;
    cookieRefresh?: string;
    cookieExpire?: string;
    userHeader?: string;
    emailHeader?: string;
    groupsHeader?: string;
    requireAuth?: boolean;
    whitelistDomains?: string[];
}
export interface InjectedHeaders {
    'X-Remote-User'?: string;
    'X-Remote-Email'?: string;
    'X-Remote-Groups'?: string;
    'X-Remote-Access-Token'?: string;
}
export declare class HeaderInjectionAuth {
    private config;
    private loggerContext;
    constructor(config?: HeaderInjectionConfig);
    /**
     * Extract user information from injected headers
     * Called by backend services to get authenticated user context
     */
    extractUserFromHeaders(headers: Record<string, string>): InjectedHeaders;
    /**
     * Validate that headers are present
     * Returns false if authentication is required but missing
     */
    validateHeaders(headers: Record<string, string>): boolean;
    /**
     * Parse groups from header
     * Groups are typically comma-separated
     */
    parseGroups(headers: Record<string, string>): string[];
    /**
     * Check if user has required group/role
     */
    hasGroup(headers: Record<string, string>, requiredGroup: string): boolean;
    /**
     * Check if user has any of the required groups
     */
    hasAnyGroup(headers: Record<string, string>, requiredGroups: string[]): boolean;
    /**
     * Middleware wrapper for Express-like frameworks
     * Validates headers and adds user context to request
     */
    createMiddleware(requiredGroups?: string[]): (req: any, res: any, next: any) => Promise<any>;
    /**
     * Create auth headers for API proxying
     * Used when frontend makes requests to backend through sidecar
     */
    createProxyHeaders(accessToken: string): Record<string, string>;
    /**
     * Validate OAuth2 proxy cookie
     * Used to verify cookie signature (if cookie secret is available)
     */
    validateCookie(cookie: string): boolean;
    /**
     * Compute OAuth2 proxy cookie signature
     * HMAC-SHA256 of value|timestamp with cookie secret
     */
    private computeCookieSignature;
    /**
     * Check if request is from whitelisted domain
     * Used for CORS and security validation
     */
    isWhitelistedDomain(origin: string): boolean;
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
//# sourceMappingURL=header-injection.d.ts.map