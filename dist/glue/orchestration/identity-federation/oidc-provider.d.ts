/**
 * OIDC Provider Configuration & Token Management
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of Configuration Explicitness: All values via environment variables
 * - Law of UTC: All timestamps in UTC
 * - Law of Runtime Truth: Verify configuration via execution
 *
 * @module glue/orchestration/identity-federation/oidc-provider
 */
/**
 * OIDC Configuration loaded from environment variables
 * Crashes immediately if required configuration is missing
 */
export interface OIDCConfig {
    issuer: string;
    authorizationEndpoint: string;
    tokenEndpoint: string;
    jwksUri: string;
    userInfoEndpoint: string;
    clientId: string;
    clientSecret: string;
    redirectUri: string;
    scopes: string[];
    tokenRequestTimeoutMs: number;
    userInfoTimeoutMs: number;
    jwksTimeoutMs: number;
    refreshExpiryThresholdMs: number;
    sessionMaxAgeMs: number;
    cookieName: string;
    cookieDomain?: string;
    cookieSecure: boolean;
    cookieSameSite: 'strict' | 'lax' | 'none';
    postLogoutRedirectUri?: string;
    endSessionEndpoint?: string;
}
/**
 * OIDC Token set returned from token endpoint
 */
export interface TokenSet {
    access_token: string;
    token_type: string;
    id_token?: string;
    refresh_token?: string;
    expires_in: number;
    scope?: string;
    expires_at?: Date;
    refresh_expires_at?: Date;
}
/**
 * OIDC User information from userinfo endpoint
 */
export interface OIDCUserInfo {
    sub: string;
    name: string;
    email: string;
    email_verified?: boolean;
    picture?: string;
    groups?: string[];
    [key: string]: any;
}
/**
 * Parsed ID Token claims
 */
export interface IDTokenClaims {
    iss: string;
    sub: string;
    aud: string | string[];
    exp: number;
    iat: number;
    auth_time?: number;
    nonce?: string;
    acr?: string;
    amr?: string[];
    azp?: string;
    at_hash?: string;
    c_hash?: string;
    name?: string;
    email?: string;
    email_verified?: boolean;
    picture?: string;
    groups?: string[];
    [key: string]: any;
}
/**
 * OIDC Provider class implementing standard OIDC flows
 */
export declare class OIDCProvider {
    private config;
    private jwksCache;
    private jwksCacheExpiry;
    constructor(config: OIDCConfig);
    /**
     * Load OIDC configuration from environment variables
     * Crashes immediately if required vars are missing
     */
    static fromEnv(): OIDCProvider;
    /**
     * Validate configuration values
     */
    private validateConfig;
    /**
     * Generate authorization URL for OIDC authorization flow
     *
     * @param state - Random state value for CSRF protection
     * @param nonce - Random nonce value for replay protection
     * @param redirectPath - Optional path to redirect after auth
     * @returns Authorization URL
     */
    generateAuthorizationUrl(state: string, nonce: string, redirectPath?: string): string;
    /**
     * Exchange authorization code for tokens
     *
     * @param code - Authorization code from callback
     * @param state - State value from callback (should match generated state)
     * @param expectedState - Expected state value for validation
     * @returns Token set
     */
    exchangeCodeForTokens(code: string, state: string, expectedState: string): Promise<TokenSet>;
    /**
     * Refresh access token using refresh token
     *
     * @param refreshToken - Refresh token
     * @returns New token set
     */
    refreshAccessToken(refreshToken: string): Promise<TokenSet>;
    /**
     * Check if access token needs refresh
     *
     * @param tokenSet - Token set to check
     * @returns True if token needs refresh
     */
    needsRefresh(tokenSet: TokenSet): boolean;
    /**
     * Fetch user information from userinfo endpoint
     *
     * @param accessToken - Access token
     * @returns User information
     */
    getUserInfo(accessToken: string): Promise<OIDCUserInfo>;
    /**
     * Validate and decode ID token
     * NOTE: This is a simplified validation. In production, use a proper JWT library
     * like jose, node-jose, or openid-client for full cryptographic validation
     *
     * @param idToken - ID token to validate
     * @returns Decoded claims
     */
    validateIDToken(idToken: string): Promise<IDTokenClaims>;
    /**
     * Generate logout URL
     *
     * @param idToken - ID token for logout validation
     * @param postLogoutRedirectUri - Optional post-logout redirect URI
     * @returns Logout URL
     */
    generateLogoutUrl(idToken?: string, postLogoutRedirectUri?: string): string;
    /**
     * Generate correlation ID for tracing
     */
    private generateCorrelationId;
}
/**
 * Helper function to generate secure random state/nonce values
 */
export declare function generateSecureRandom(length?: number): string;
//# sourceMappingURL=oidc-provider.d.ts.map