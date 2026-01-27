import { CredentialType, ParsedBubbleWithInfo } from '@bubblelab/shared-schemas';
import { BubbleScript } from '../parse/BubbleScript';
export interface UserCredentialWithId {
    /** The variable id of the bubble */
    bubbleVarId: number | string;
    secret: string;
    credentialType: CredentialType;
    credentialId?: number;
    metadata?: Record<string, unknown>;
}
export interface CredentialInjectionResult {
    success: boolean;
    parsedBubbles?: Record<string, ParsedBubbleWithInfo>;
    code?: string;
    errors?: string[];
    injectedCredentials?: Record<number, {
        isUserCredential: boolean;
        credentialType: CredentialType;
        credentialValue: string;
    }>;
}
export declare class BubbleInjector {
    private bubbleScript;
    private loggerInjector;
    constructor(bubbleScript: BubbleScript);
    /**
     * Extracts required credential types from parsed bubble parameters
     * Returns a map of variableId to the list of credentials required by that bubble
     * @param bubbleParameters - Parsed bubble parameters with info
     * @returns Record mapping bubble variable IDs to their required credential types (excluding system credentials)
     */
    findCredentials(): Record<string, CredentialType[]>;
    /**
     * Extracts the required credential types from AI agent model parameter (including backup model)
     * @param bubble - The parsed bubble to extract model from
     * @returns Array of credential types needed for the models, or null if dynamic (needs all)
     */
    private extractModelCredentialType;
    /**
     * Maps a model string to its credential type
     * @param modelString - Model string in format "provider/model-name"
     * @returns The credential type for the provider, or null if unknown
     */
    private getCredentialTypeForProvider;
    /**
     * Extracts tool credential requirements from AI agent bubble parameters
     * @param bubble - The parsed bubble to extract tool requirements from
     * @returns Array of credential types required by the bubble's tools
     */
    private extractToolCredentials;
    /**
     * Injects credentials into bubble parameters
     * @param bubbleParameters - Parsed bubble parameters with info
     * @param userCredentials - User-provided credentials
     * @param systemCredentials - System-provided credentials (environment variables)
     * @returns Result of credential injection
     */
    injectCredentials(userCredentials?: UserCredentialWithId[], systemCredentials?: Partial<Record<CredentialType, string>>): CredentialInjectionResult;
    /**
     * Injects credentials into a specific bubble's parameters
     */
    private injectCredentialsIntoBubble;
    /**
     * Escapes a string for safe injection into TypeScript code
     */
    private escapeString;
    /**
     * Masks a credential value for debugging/logging
     */
    private maskCredential;
    private getBubble;
    /**
     * Reapply bubble instantiations by normalizing them to single-line format
     * and deleting old multi-line parameters. Processes bubbles in order and
     * tracks line shifts to adjust subsequent bubble locations.
     */
    private reapplyBubbleInstantiations;
    /**
     * Refresh bubble parameters that are sourced from code (like customTools arrays)
     * by re-reading the relevant portion from the updated lines array.
     * @param bubble - The bubble to refresh parameters for
     * @param lines - The current lines array (after nested bubble processing)
     * @param adjustedBubbleEndLine - The bubble's end line adjusted for lines deleted during nested processing
     */
    private refreshBubbleParametersFromSource;
    private buildInvocationDependencyGraphLiteral;
    private injectInvocationDependencyGraphMap;
    /**
     * Apply new bubble parameters by converting them back to code and injecting in place
     * Injects logger to the bubble instantiations
     */
    injectBubbleLoggingAndReinitializeBubbleParameters(loggingEnabled?: boolean): void;
    /** Takes in bubbleId and key, value pair and changes the parameter in the bubble script */
    changeBubbleParameters(bubbleId: number, key: string, value: string | number | boolean | Record<string, unknown> | unknown[]): void;
    /** Changes the credentials field inside the bubble parameters by modifying the value to add ore replace new credentials */
    changeCredentials(bubbleId: number, credentials: Record<CredentialType, string>): void;
}
//# sourceMappingURL=BubbleInjector.d.ts.map