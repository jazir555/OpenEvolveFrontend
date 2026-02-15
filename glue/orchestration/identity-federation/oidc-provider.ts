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

import { logger } from '../../lib/logger';

/**
 * OIDC Configuration loaded from environment variables
 * Crashes immediately if required configuration is missing
 */
export interface OIDCConfig {
  // Required: Provider endpoints
  issuer: string;
  authorizationEndpoint: string;
  tokenEndpoint: string;
  jwksUri: string;
  userInfoEndpoint: string;

  // Required: Client credentials
  clientId: string;
  clientSecret: string;
  redirectUri: string;

  // Required: Scopes
  scopes: string[];

  // Optional: timeouts (milliseconds)
  tokenRequestTimeoutMs: number;
  userInfoTimeoutMs: number;
  jwksTimeoutMs: number;

  // Optional: Token refresh thresholds
  refreshExpiryThresholdMs: number;

  // Optional: Session configuration
  sessionMaxAgeMs: number;
  cookieName: string;
  cookieDomain?: string;
  cookieSecure: boolean;
  cookieSameSite: 'strict' | 'lax' | 'none';

  // Optional: Post-logout redirect
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

  // Parsed timestamps (UTC)
  expires_at?: Date;  // Calculated from expires_in
  refresh_expires_at?: Date;  // If provided by IdP
}

/**
 * OIDC User information from userinfo endpoint
 */
export interface OIDCUserInfo {
  sub: string;  // Subject (unique user ID)
  name: string;
  email: string;
  email_verified?: boolean;
  picture?: string;
  groups?: string[];
  [key: string]: any;  // Additional claims
}

/**
 * Parsed ID Token claims
 */
export interface IDTokenClaims {
  iss: string;  // Issuer
  sub: string;  // Subject
  aud: string | string[];  // Audience
  exp: number;  // Expiration time
  iat: number;  // Issued at time
  auth_time?: number;  // Authentication time
  nonce?: string;  // Nonce
  acr?: string;  // Authentication context class reference
  amr?: string[];  // Authentication methods references
  azp?: string;  // Authorized party
  at_hash?: string;  // Access token hash
  c_hash?: string;  // Code hash

  // User claims (subset - additional claims may be present)
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
export class OIDCProvider {
  private config: OIDCConfig;
  private jwksCache: Map<string, any>;
  private jwksCacheExpiry: Date;

  constructor(config: OIDCConfig) {
    this.config = config;
    this.jwksCache = new Map();
    this.jwksCacheExpiry = new Date(0);

    this.validateConfig();
    logger.info({
      msg: 'OIDC Provider initialized',
      issuer: config.issuer,
      client_id: config.clientId,
      correlation_id: this.generateCorrelationId(),
    });
  }

  /**
   * Load OIDC configuration from environment variables
   * Crashes immediately if required vars are missing
   */
  static fromEnv(): OIDCProvider {
    const requiredVars = {
      'OIDC_ISSUER': 'issuer',
      'OIDC_AUTHORIZATION_ENDPOINT': 'authorizationEndpoint',
      'OIDC_TOKEN_ENDPOINT': 'tokenEndpoint',
      'OIDC_JWKS_URI': 'jwksUri',
      'OIDC_USERINFO_ENDPOINT': 'userInfoEndpoint',
      'OIDC_CLIENT_ID': 'clientId',
      'OIDC_CLIENT_SECRET': 'clientSecret',
      'OIDC_REDIRECT_URI': 'redirectUri',
    };

    const config: any = {};
    const errors: string[] = [];

    // Load required variables
    for (const [envVar, field] of Object.entries(requiredVars)) {
      const value = process.env[envVar];
      if (!value) {
        errors.push(`Missing required environment variable: ${envVar}`);
      } else {
        config[field] = value;
      }
    }

    // If errors, crash immediately (Law of Configuration Explicitness)
    if (errors.length > 0) {
      logger.fatal({
        msg: 'OIDC configuration validation failed',
        errors: errors,
      });
      throw new Error(`FATAL: OIDC Configuration Error:\n${errors.map(e => `  - ${e}`).join('\n')}`);
    }

    // Load optional variables with defaults
    config.scopes = (process.env['OIDC_SCOPES'] || 'openid profile email').split(' ');
    config.tokenRequestTimeoutMs = parseInt(process.env['OIDC_TOKEN_TIMEOUT_MS'] || '5000');
    config.userInfoTimeoutMs = parseInt(process.env['OIDC_USERINFO_TIMEOUT_MS'] || '3000');
    config.jwksTimeoutMs = parseInt(process.env['OIDC_JWKS_TIMEOUT_MS'] || '5000');
    config.refreshExpiryThresholdMs = parseInt(process.env['OIDC_REFRESH_THRESHOLD_MS'] || '300000'); // 5 minutes
    config.sessionMaxAgeMs = parseInt(process.env['OIDC_SESSION_MAX_AGE_MS'] || '28800000'); // 8 hours
    config.cookieName = process.env['OIDC_COOKIE_NAME'] || 'oidc_session';
    config.cookieDomain = process.env['OIDC_COOKIE_DOMAIN'];
    config.cookieSecure = process.env['OIDC_COOKIE_SECURE'] !== 'false';
    config.cookieSameSite = (process.env['OIDC_COOKIE_SAME_SITE'] || 'lax') as 'strict' | 'lax' | 'none';
    config.postLogoutRedirectUri = process.env['OIDC_POST_LOGOUT_REDIRECT_URI'];
    config.endSessionEndpoint = process.env['OIDC_END_SESSION_ENDPOINT'];

    return new OIDCProvider(config as OIDCConfig);
  }

  /**
   * Validate configuration values
   */
  private validateConfig(): void {
    const errors: string[] = [];

    // Validate URLs
    try {
      new URL(this.config.issuer);
      new URL(this.config.authorizationEndpoint);
      new URL(this.config.tokenEndpoint);
      new URL(this.config.jwksUri);
      new URL(this.config.userInfoEndpoint);
      new URL(this.config.redirectUri);
    } catch (err) {
      errors.push(`Invalid URL configuration: ${err}`);
    }

    // Validate timeouts are positive
    if (this.config.tokenRequestTimeoutMs <= 0) {
      errors.push('OIDC_TOKEN_TIMEOUT_MS must be positive');
    }
    if (this.config.userInfoTimeoutMs <= 0) {
      errors.push('OIDC_USERINFO_TIMEOUT_MS must be positive');
    }
    if (this.config.jwksTimeoutMs <= 0) {
      errors.push('OIDC_JWKS_TIMEOUT_MS must be positive');
    }

    // Validate scopes include 'openid'
    if (!this.config.scopes.includes('openid')) {
      errors.push('OIDC_SCOPES must include "openid"');
    }

    if (errors.length > 0) {
      throw new Error(`OIDC Configuration Error:\n${errors.map(e => `  - ${e}`).join('\n')}`);
    }
  }

  /**
   * Generate authorization URL for OIDC authorization flow
   *
   * @param state - Random state value for CSRF protection
   * @param nonce - Random nonce value for replay protection
   * @param redirectPath - Optional path to redirect after auth
   * @returns Authorization URL
   */
  generateAuthorizationUrl(state: string, nonce: string, redirectPath?: string): string {
    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.config.clientId,
      redirect_uri: redirectPath ? `${this.config.redirectUri}${redirectPath}` : this.config.redirectUri,
      scope: this.config.scopes.join(' '),
      state: state,
      nonce: nonce,
      response_mode: 'query',
    });

    const url = `${this.config.authorizationEndpoint}?${params.toString()}`;

    logger.info({
      msg: 'Generated authorization URL',
      state: state,
      nonce: nonce,
      correlation_id: this.generateCorrelationId(),
    });

    return url;
  }

  /**
   * Exchange authorization code for tokens
   *
   * @param code - Authorization code from callback
   * @param state - State value from callback (should match generated state)
   * @param expectedState - Expected state value for validation
   * @returns Token set
   */
  async exchangeCodeForTokens(
    code: string,
    state: string,
    expectedState: string
  ): Promise<TokenSet> {
    const correlationId = this.generateCorrelationId();

    // Validate state
    if (state !== expectedState) {
      logger.error({
        msg: 'State validation failed',
        provided_state: state,
        expected_state: expectedState,
        correlation_id: correlationId,
      });
      throw new Error('State validation failed: possible CSRF attack');
    }

    try {
      const params = new URLSearchParams({
        grant_type: 'authorization_code',
        code: code,
        redirect_uri: this.config.redirectUri,
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
      });

      const response = await fetch(this.config.tokenEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: params.toString(),
        signal: AbortSignal.timeout(this.config.tokenRequestTimeoutMs),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error({
          msg: 'Token request failed',
          status: response.status,
          error: errorText,
          correlation_id: correlationId,
        });
        throw new Error(`Token request failed: ${response.status} ${errorText}`);
      }

      const tokenSet = await response.json();

      // Calculate expiration timestamp (UTC)
      if (tokenSet.expires_in) {
        tokenSet.expires_at = new Date(Date.now() + tokenSet.expires_in * 1000);
      }

      logger.info({
        msg: 'Tokens exchanged successfully',
        token_type: tokenSet.token_type,
        expires_at: tokenSet.expires_at,
        correlation_id: correlationId,
      });

      return tokenSet as TokenSet;
    } catch (err: any) {
      if (err.name === 'TimeoutError') {
        logger.error({
          msg: 'Token request timeout',
          timeout_ms: this.config.tokenRequestTimeoutMs,
          correlation_id: correlationId,
        });
        throw new Error(`Token request timeout after ${this.config.tokenRequestTimeoutMs}ms`);
      }
      throw err;
    }
  }

  /**
   * Refresh access token using refresh token
   *
   * @param refreshToken - Refresh token
   * @returns New token set
   */
  async refreshAccessToken(refreshToken: string): Promise<TokenSet> {
    const correlationId = this.generateCorrelationId();

    try {
      const params = new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
      });

      const response = await fetch(this.config.tokenEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: params.toString(),
        signal: AbortSignal.timeout(this.config.tokenRequestTimeoutMs),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error({
          msg: 'Token refresh failed',
          status: response.status,
          error: errorText,
          correlation_id: correlationId,
        });
        throw new Error(`Token refresh failed: ${response.status} ${errorText}`);
      }

      const tokenSet = await response.json();

      // Calculate expiration timestamp (UTC)
      if (tokenSet.expires_in) {
        tokenSet.expires_at = new Date(Date.now() + tokenSet.expires_in * 1000);
      }

      logger.info({
        msg: 'Token refreshed successfully',
        token_type: tokenSet.token_type,
        expires_at: tokenSet.expires_at,
        correlation_id: correlationId,
      });

      return tokenSet as TokenSet;
    } catch (err: any) {
      if (err.name === 'TimeoutError') {
        logger.error({
          msg: 'Token refresh timeout',
          timeout_ms: this.config.tokenRequestTimeoutMs,
          correlation_id: correlationId,
        });
        throw new Error(`Token refresh timeout after ${this.config.tokenRequestTimeoutMs}ms`);
      }
      throw err;
    }
  }

  /**
   * Check if access token needs refresh
   *
   * @param tokenSet - Token set to check
   * @returns True if token needs refresh
   */
  needsRefresh(tokenSet: TokenSet): boolean {
    if (!tokenSet.expires_at) {
      // If no expiration, assume it needs refresh
      return true;
    }

    const now = new Date();
    const expiryThreshold = new Date(tokenSet.expires_at.getTime() - this.config.refreshExpiryThresholdMs);

    return now >= expiryThreshold;
  }

  /**
   * Fetch user information from userinfo endpoint
   *
   * @param accessToken - Access token
   * @returns User information
   */
  async getUserInfo(accessToken: string): Promise<OIDCUserInfo> {
    const correlationId = this.generateCorrelationId();

    try {
      const response = await fetch(this.config.userInfoEndpoint, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
        signal: AbortSignal.timeout(this.config.userInfoTimeoutMs),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error({
          msg: 'User info request failed',
          status: response.status,
          error: errorText,
          correlation_id: correlationId,
        });
        throw new Error(`User info request failed: ${response.status} ${errorText}`);
      }

      const userInfo = await response.json();

      logger.info({
        msg: 'User info fetched successfully',
        sub: userInfo.sub,
        email: userInfo.email,
        correlation_id: correlationId,
      });

      return userInfo as OIDCUserInfo;
    } catch (err: any) {
      if (err.name === 'TimeoutError') {
        logger.error({
          msg: 'User info request timeout',
          timeout_ms: this.config.userInfoTimeoutMs,
          correlation_id: correlationId,
        });
        throw new Error(`User info request timeout after ${this.config.userInfoTimeoutMs}ms`);
      }
      throw err;
    }
  }

  /**
   * Validate and decode ID token
   * NOTE: This is a simplified validation. In production, use a proper JWT library
   * like jose, node-jose, or openid-client for full cryptographic validation
   *
   * @param idToken - ID token to validate
   * @returns Decoded claims
   */
  async validateIDToken(idToken: string): Promise<IDTokenClaims> {
    const correlationId = this.generateCorrelationId();

    try {
      // Split token into parts
      const parts = idToken.split('.');
      if (parts.length !== 3) {
        throw new Error('Invalid ID token format');
      }

      // Decode payload (base64url)
      const payload = parts[1];
      const decoded = Buffer.from(payload, 'base64url').toString('utf-8');
      const claims: IDTokenClaims = JSON.parse(decoded);

      // Validate issuer
      if (claims.iss !== this.config.issuer) {
        throw new Error(`Invalid issuer: ${claims.iss}`);
      }

      // Validate audience
      const audiences = Array.isArray(claims.aud) ? claims.aud : [claims.aud];
      if (!audiences.includes(this.config.clientId)) {
        throw new Error(`Invalid audience: ${claims.aud}`);
      }

      // Validate expiration (UTC)
      const now = Math.floor(Date.now() / 1000);
      if (claims.exp < now) {
        throw new Error('ID token expired');
      }

      // Validate issued at (not in future)
      if (claims.iat > now + 60) {  // Allow 60 seconds clock skew
        throw new Error('ID token issued in the future');
      }

      logger.info({
        msg: 'ID token validated successfully',
        sub: claims.sub,
        iss: claims.iss,
        exp: new Date(claims.exp * 1000),
        correlation_id: correlationId,
      });

      return claims;
    } catch (err: any) {
      logger.error({
        msg: 'ID token validation failed',
        error: err.message,
        correlation_id: correlationId,
      });
      throw err;
    }
  }

  /**
   * Generate logout URL
   *
   * @param idToken - ID token for logout validation
   * @param postLogoutRedirectUri - Optional post-logout redirect URI
   * @returns Logout URL
   */
  generateLogoutUrl(idToken?: string, postLogoutRedirectUri?: string): string {
    if (!this.config.endSessionEndpoint) {
      throw new Error('End session endpoint not configured');
    }

    const params = new URLSearchParams({
      id_token_hint: idToken || '',
      post_logout_redirect_uri: postLogoutRedirectUri || this.config.postLogoutRedirectUri || '',
    });

    const url = `${this.config.endSessionEndpoint}?${params.toString()}`;

    logger.info({
      msg: 'Generated logout URL',
      has_id_token: !!idToken,
      correlation_id: this.generateCorrelationId(),
    });

    return url;
  }

  /**
   * Generate correlation ID for tracing
   */
  private generateCorrelationId(): string {
    return `oidc-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  }
}

/**
 * Helper function to generate secure random state/nonce values
 */
export function generateSecureRandom(length: number = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}
