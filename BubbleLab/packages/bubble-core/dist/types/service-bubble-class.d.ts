import type { BubbleOperationResult, ServiceBubbleParams, BubbleContext } from '@bubblelab/bubble-core';
import { BaseBubble } from './base-bubble-class.js';
import type { DatabaseMetadata } from '@bubblelab/shared-schemas';
export declare abstract class ServiceBubble<TParams extends ServiceBubbleParams = ServiceBubbleParams, TResult extends BubbleOperationResult = BubbleOperationResult> extends BaseBubble<TParams, TResult> {
    readonly type: "service";
    authType?: 'oauth' | 'apikey' | 'none' | 'connection-string';
    constructor(params: unknown, context?: BubbleContext, instanceId?: string);
    abstract testCredential(): Promise<boolean>;
    /**
     * Abstract method to choose the appropriate credential based on bubble parameters
     * Should examine this.params to determine which credential to use from the injected credentials
     * Must be implemented by all service bubbles
     */
    protected abstract chooseCredential(): string | undefined;
    /**
     * Abstract method to get the metadata of the credential
     * Must be implemented by all service bubbles
     */
    getCredentialMetadata(): Promise<DatabaseMetadata | undefined>;
    /**
     * Get the current parameters (credentials are excluded for security)
     * Use chooseCredential() method to access credentials in a controlled way
     */
    get currentParams(): Omit<TParams, 'credentials'>;
    setParam<K extends keyof TParams>(paramName: K, paramValue: TParams[K]): void;
    /**
     * Get the current context
     */
    get currentContext(): BubbleContext | undefined;
}
//# sourceMappingURL=service-bubble-class.d.ts.map