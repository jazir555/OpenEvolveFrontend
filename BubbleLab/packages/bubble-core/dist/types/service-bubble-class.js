import { BaseBubble } from './base-bubble-class.js';
export class ServiceBubble extends BaseBubble {
    type = 'service';
    authType;
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    /**
     * Abstract method to get the metadata of the credential
     * Must be implemented by all service bubbles
     */
    // Optional method, only used for database bubbles
    async getCredentialMetadata() {
        return undefined;
    }
    /**
     * Get the current parameters (credentials are excluded for security)
     * Use chooseCredential() method to access credentials in a controlled way
     */
    get currentParams() {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { credentials, ...sanitized } = this.params;
        return sanitized;
    }
    setParam(paramName, paramValue) {
        this.params[paramName] = paramValue;
    }
    /**
     * Get the current context
     */
    get currentContext() {
        return this.context;
    }
}
//# sourceMappingURL=service-bubble-class.js.map