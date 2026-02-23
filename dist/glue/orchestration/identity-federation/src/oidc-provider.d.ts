/**
 * OIDC Provider Integration
 *
 * Federation Constitution - Identity Federation Strategy (ADR-006)
 * Phase 1: OIDC First (Preferred)
 *
 * Integrates with OpenID Connect providers for centralized authentication.
 * Supports Keycloak, Auth0, Okta, and other OIDC-compliant providers.
 *
 * Configuration:
 *   - OIDC_ISSUER: OIDC provider URL
 *   - OIDC_CLIENT_ID: Client ID for this application
 *   - OIDC_CLIENT_SECRET: Client secret (for confidential clients)
 *   - OIDC_REDIRECT_URI: Redirect URI after authentication
 *   - OIDC_SCOPES: Scopes to request (default: openid profile email)
 */
export interface OIDCConfig {
    issuer: string;
    clientId: string;
    clientSecret?: string;
    redirectUri: string;
    scopes?: string[];
    timeout?: number;
}
export interface OIDCToken {
    access_token: string;
    id_token?: string;
    refresh_token?: string;
    token_type: string;
    expires_in: number;
    scope?: string;
}
export interface OIDCUserInfo {
    sub: string;
    name?: string;
    email?: string;
    picture?: string;
    groups?: string[];
    preferred_username?: string;
    [key: string]: unknown;
}
export interface OIDCProviderConfig {
    issuer: string;
    authorization_endpoint: string;
    token_endpoint: string;
    userinfo_endpoint?: string;
    jwks_uri: string;
    end_session_endpoint?: string;
    registration_endpoint?: string;
    scopes_supported?: string[];
    response_types_supported?: string[];
    grant_types_supported?: string[];
    client_id?: string;
    client_secret?: string;
}
export declare class OIDCProvider {
    private config;
    private providerConfig;
    private loggerContext;
    constructor(config: OIDCConfig);
    /**
     * Fetch OIDC provider configuration from .well-known endpoint
     * Caches the configuration for subsequent calls
     */
    fetchProviderConfig(): Promise<OIDCProviderConfig>;
    /**
     * Generate authorization URL for login flow
     */
    getAuthorizationUrl(state?: string, nonce?: string): Promise<string>;
    /**
     * Exchange authorization code for tokens
     */
    exchangeCodeForTokens(code: string, state?: string): Promise<OIDCToken>;
    /**
     * Refresh access token using refresh token
     */
    refreshAccessToken(refreshToken: string): Promise<OIDCToken>;
    /**
     * Fetch user info from UserInfo endpoint
     */
    getUserInfo(accessToken: string): Promise<OIDCUserInfo>;
    /**
     * Validate ID token (basic validation)
     * Note: Full JWT signature validation should be done in production
     */
    validateIdToken(idToken: string): Promise<boolean>;
    /**
     * Logout URL for ending session
     */
    getLogoutUrl(idTokenHint?: string, postLogoutRedirectUri?: string): Promise<string>;
    /**
     * Generate random string for state/nonce
     */
    private generateRandomString;
}
/**
 * Example usage:
 *
 * ```typescript
 * const oidc = new OIDCProvider({
 *   issuer: 'https://keycloak.example.com/realms/myrealm',
 *   clientId: 'my-client',
 *   clientSecret: 'my-secret',
 *   redirectUri: 'http://localhost:3000/callback',
 * });
 *
 * // Get login URL
 * const authUrl = await oidc.getAuthorizationUrl();
 * window.location.href = authUrl;
 *
 * // In callback handler
 * const params = new URLSearchParams(window.location.search);
 * const code = params.get('code');
 * const tokens = await oidc.exchangeCodeForTokens(code!);
 * const userInfo = await oidc.getUserInfo(tokens.access_token);
 * ```
 */
//# sourceMappingURL=oidc-provider.d.ts.map