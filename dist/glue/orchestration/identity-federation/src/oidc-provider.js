"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.OIDCProvider = void 0;
const logger_1 = require("../../lib/logger");
class OIDCProvider {
    constructor(config) {
        this.providerConfig = null;
        this.config = {
            scopes: ['openid', 'profile', 'email'],
            timeout: 30000,
            ...config,
        };
        this.loggerContext = {
            correlation_id: `oidc-${Date.now()}`,
            source_service: 'oidc-provider',
            target_service: this.config.issuer,
        };
        logger_1.logger.info('OIDC Provider initialized', {
            ...this.loggerContext,
            issuer: this.config.issuer,
            client_id: this.config.clientId,
        });
    }
    /**
     * Fetch OIDC provider configuration from .well-known endpoint
     * Caches the configuration for subsequent calls
     */
    async fetchProviderConfig() {
        if (this.providerConfig) {
            return this.providerConfig;
        }
        const wellKnownUrl = `${this.config.issuer}/.well-known/openid-configuration`;
        logger_1.logger.info('Fetching OIDC provider configuration', {
            ...this.loggerContext,
            url: wellKnownUrl,
        });
        try {
            const response = await fetch(wellKnownUrl, {
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`Failed to fetch OIDC config: ${response.status} ${response.statusText}`);
            }
            const config = await response.json();
            // Validate required fields
            if (!config.issuer || !config.authorization_endpoint || !config.token_endpoint) {
                throw new Error('Invalid OIDC provider configuration: missing required fields');
            }
            this.providerConfig = config;
            logger_1.logger.info('OIDC provider configuration fetched successfully', {
                ...this.loggerContext,
                issuer: config.issuer,
                authorization_endpoint: config.authorization_endpoint,
                token_endpoint: config.token_endpoint,
            });
            return config;
        }
        catch (error) {
            logger_1.logger.error('Failed to fetch OIDC provider configuration', error, this.loggerContext);
            throw error;
        }
    }
    /**
     * Generate authorization URL for login flow
     */
    async getAuthorizationUrl(state, nonce) {
        const providerConfig = await this.fetchProviderConfig();
        const params = new URLSearchParams({
            response_type: 'code',
            client_id: this.config.clientId,
            redirect_uri: this.config.redirectUri,
            scope: this.config.scopes.join(' '),
            state: state || this.generateRandomString(32),
            nonce: nonce || this.generateRandomString(32),
        });
        const authUrl = `${providerConfig.authorization_endpoint}?${params.toString()}`;
        logger_1.logger.info('Generated authorization URL', {
            ...this.loggerContext,
            state: state || 'generated',
            nonce: nonce || 'generated',
        });
        return authUrl;
    }
    /**
     * Exchange authorization code for tokens
     */
    async exchangeCodeForTokens(code, state) {
        const providerConfig = await this.fetchProviderConfig();
        logger_1.logger.info('Exchanging authorization code for tokens', {
            ...this.loggerContext,
            state,
        });
        const params = new URLSearchParams({
            grant_type: 'authorization_code',
            code: code,
            redirect_uri: this.config.redirectUri,
            client_id: this.config.clientId,
        });
        // Add client secret for confidential clients
        if (this.config.clientSecret) {
            params.append('client_secret', this.config.clientSecret);
        }
        try {
            const response = await fetch(providerConfig.token_endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: params.toString(),
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(`Token exchange failed: ${error.error || response.statusText}`);
            }
            const token = await response.json();
            logger_1.logger.info('Token exchange successful', {
                ...this.loggerContext,
                token_type: token.token_type,
                expires_in: token.expires_in,
                scopes: token.scope,
            });
            return token;
        }
        catch (error) {
            logger_1.logger.error('Token exchange failed', error, this.loggerContext);
            throw error;
        }
    }
    /**
     * Refresh access token using refresh token
     */
    async refreshAccessToken(refreshToken) {
        const providerConfig = await this.fetchProviderConfig();
        logger_1.logger.info('Refreshing access token', this.loggerContext);
        const params = new URLSearchParams({
            grant_type: 'refresh_token',
            refresh_token: refreshToken,
            client_id: this.config.clientId,
        });
        if (this.config.clientSecret) {
            params.append('client_secret', this.config.clientSecret);
        }
        try {
            const response = await fetch(providerConfig.token_endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: params.toString(),
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(`Token refresh failed: ${error.error || response.statusText}`);
            }
            const token = await response.json();
            logger_1.logger.info('Token refresh successful', this.loggerContext);
            return token;
        }
        catch (error) {
            logger_1.logger.error('Token refresh failed', error, this.loggerContext);
            throw error;
        }
    }
    /**
     * Fetch user info from UserInfo endpoint
     */
    async getUserInfo(accessToken) {
        const providerConfig = await this.fetchProviderConfig();
        if (!providerConfig.userinfo_endpoint) {
            throw new Error('UserInfo endpoint not available');
        }
        logger_1.logger.info('Fetching user info', this.loggerContext);
        try {
            const response = await fetch(providerConfig.userinfo_endpoint, {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                },
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`UserInfo request failed: ${response.status} ${response.statusText}`);
            }
            const userInfo = await response.json();
            logger_1.logger.info('User info fetched successfully', {
                ...this.loggerContext,
                sub: userInfo.sub,
                email: userInfo.email,
                name: userInfo.name,
            });
            return userInfo;
        }
        catch (error) {
            logger_1.logger.error('Failed to fetch user info', error, this.loggerContext);
            throw error;
        }
    }
    /**
     * Validate ID token (basic validation)
     * Note: Full JWT signature validation should be done in production
     */
    async validateIdToken(idToken) {
        logger_1.logger.info('Validating ID token', this.loggerContext);
        try {
            // Split token into parts
            const parts = idToken.split('.');
            if (parts.length !== 3) {
                throw new Error('Invalid ID token format');
            }
            // Decode payload (no signature verification for now)
            const payload = JSON.parse(atob(parts[1]));
            // Validate standard claims
            const now = Math.floor(Date.now() / 1000);
            if (payload.exp && payload.exp < now) {
                throw new Error('ID token expired');
            }
            if (payload.nbf && payload.nbf > now) {
                throw new Error('ID token not yet valid');
            }
            if (payload.iss !== this.config.issuer) {
                throw new Error('ID token issuer mismatch');
            }
            if (payload.aud && payload.aud !== this.config.clientId) {
                throw new Error('ID token audience mismatch');
            }
            logger_1.logger.info('ID token validated successfully', {
                ...this.loggerContext,
                sub: payload.sub,
                exp: payload.exp,
            });
            return true;
        }
        catch (error) {
            logger_1.logger.error('ID token validation failed', error, this.loggerContext);
            return false;
        }
    }
    /**
     * Logout URL for ending session
     */
    async getLogoutUrl(idTokenHint, postLogoutRedirectUri) {
        const providerConfig = await this.fetchProviderConfig();
        if (!providerConfig.end_session_endpoint) {
            throw new Error('End session endpoint not available');
        }
        const params = new URLSearchParams();
        if (idTokenHint) {
            params.append('id_token_hint', idTokenHint);
        }
        if (postLogoutRedirectUri) {
            params.append('post_logout_redirect_uri', postLogoutRedirectUri);
        }
        const logoutUrl = `${providerConfig.end_session_endpoint}?${params.toString()}`;
        logger_1.logger.info('Generated logout URL', this.loggerContext);
        return logoutUrl;
    }
    /**
     * Generate random string for state/nonce
     */
    generateRandomString(length) {
        const charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        const randomValues = new Uint32Array(length);
        if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
            crypto.getRandomValues(randomValues);
            for (let i = 0; i < length; i++) {
                result += charset[randomValues[i] % charset.length];
            }
        }
        else {
            // Fallback for environments without crypto
            for (let i = 0; i < length; i++) {
                result += charset[Math.floor(Math.random() * charset.length)];
            }
        }
        return result;
    }
}
exports.OIDCProvider = OIDCProvider;
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
//# sourceMappingURL=oidc-provider.js.map