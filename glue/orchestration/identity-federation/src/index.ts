/**
 * Identity Federation Module
 *
 * Main entry point for identity federation functionality.
 * Exports OIDC, header injection, and user sync components.
 */

export { OIDCProvider, type OIDCConfig, type OIDCToken, type OIDCUserInfo, type OIDCProviderConfig } from './oidc-provider';
export { HeaderInjectionAuth, type HeaderInjectionConfig, type InjectedHeaders } from './header-injection';
export { ShadowAccountSync, type ServiceAdapter, type CentralUser, type ShadowAccount, type SyncOptions, type SyncResult } from './user-sync';
