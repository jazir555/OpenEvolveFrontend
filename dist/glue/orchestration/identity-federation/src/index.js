"use strict";
/**
 * Identity Federation Module
 *
 * Main entry point for identity federation functionality.
 * Exports OIDC, header injection, and user sync components.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ShadowAccountSync = exports.HeaderInjectionAuth = exports.OIDCProvider = void 0;
var oidc_provider_1 = require("./oidc-provider");
Object.defineProperty(exports, "OIDCProvider", { enumerable: true, get: function () { return oidc_provider_1.OIDCProvider; } });
var header_injection_1 = require("./header-injection");
Object.defineProperty(exports, "HeaderInjectionAuth", { enumerable: true, get: function () { return header_injection_1.HeaderInjectionAuth; } });
var user_sync_1 = require("./user-sync");
Object.defineProperty(exports, "ShadowAccountSync", { enumerable: true, get: function () { return user_sync_1.ShadowAccountSync; } });
//# sourceMappingURL=index.js.map