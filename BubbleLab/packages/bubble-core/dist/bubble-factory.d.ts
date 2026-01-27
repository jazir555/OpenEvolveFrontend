import { z } from 'zod';
import type { IBubble, BubbleContext } from './types/bubble.js';
import { CredentialType, type BubbleName, type BubbleNodeType } from '@bubblelab/shared-schemas';
type BubbleDependencySpec = {
    name: BubbleName;
    tools?: BubbleName[];
    instances?: Array<{
        variableName: string;
        isAnonymous: boolean;
        startLine?: number;
        endLine?: number;
    }>;
};
import type { LangGraphTool } from './types/tool-bubble-class.js';
export type BubbleClassWithMetadata<TResult extends object = object> = {
    new (params: unknown, context?: BubbleContext): IBubble<{
        success: boolean;
        error: string;
    } & TResult>;
    readonly bubbleName: BubbleName;
    readonly schema: z.ZodObject<z.ZodRawShape> | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape>[]>;
    readonly resultSchema?: z.ZodObject<z.ZodRawShape> | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape>[]>;
    readonly shortDescription: string;
    readonly longDescription: string;
    readonly alias?: string;
    readonly type: BubbleNodeType;
    readonly credentialOptions?: CredentialType[];
    readonly bubbleDependencies?: BubbleName[];
    toolAgent?: (credentials: Partial<Record<CredentialType, string>>, config?: Record<string, unknown>, context?: BubbleContext) => LangGraphTool;
};
export declare class BubbleFactory {
    private registry;
    private static dependenciesPopulated;
    private static detailedDepsCache;
    private detailedDeps;
    constructor(autoRegisterDefaults?: boolean);
    /**
     * Register a bubble class with the factory
     */
    register(name: BubbleName, bubbleClass: BubbleClassWithMetadata<any>): void;
    /**
     * Get a bubble class from the registry
     */
    get(name: BubbleName): BubbleClassWithMetadata<any> | undefined;
    /**
     * Create a bubble instance
     */
    createBubble<T extends IBubble = IBubble>(name: BubbleName, params?: unknown, context?: BubbleContext): T;
    getDetailedDependencies(name: BubbleName): BubbleDependencySpec[];
    /**
     * List all registered bubble names
     */
    list(): BubbleName[];
    listBubblesForCodeGenerator(): BubbleName[];
    registerDefaults(): Promise<void>;
    /**
     * Get all registered bubble classes
     */
    getAll(): BubbleClassWithMetadata[];
    /**
     * Get metadata for a bubble without instantiating it
     */
    getMetadata(name: BubbleName): {
        bubbleDependenciesDetailed: BubbleDependencySpec[] | undefined;
        name: BubbleName;
        shortDescription: string;
        longDescription: string;
        alias: string | undefined;
        credentialOptions: CredentialType[] | undefined;
        bubbleDependencies: BubbleName[] | undefined;
        schema: z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }> | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }>[]>;
        resultSchema: z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }> | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }>[]> | undefined;
        type: BubbleNodeType;
        params: z.ZodRawShape | undefined;
    } | undefined;
    /**
     * Get all bubble metadata
     */
    getAllMetadata(): ({
        bubbleDependenciesDetailed: BubbleDependencySpec[] | undefined;
        name: BubbleName;
        shortDescription: string;
        longDescription: string;
        alias: string | undefined;
        credentialOptions: CredentialType[] | undefined;
        bubbleDependencies: BubbleName[] | undefined;
        schema: z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }> | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }>[]>;
        resultSchema: z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }> | z.ZodDiscriminatedUnion<string, z.ZodObject<z.ZodRawShape, z.UnknownKeysParam, z.ZodTypeAny, {
            [x: string]: any;
        }, {
            [x: string]: any;
        }>[]> | undefined;
        type: BubbleNodeType;
        params: z.ZodRawShape | undefined;
    } | undefined)[];
    /**
     * Scan bubble source modules to infer direct dependencies between bubbles by
     * inspecting ES module import statements, then attach the resulting
     * `bubbleDependencies` array onto the corresponding registered classes.
     *
     * Notes:
     * - Works in both dev (src) and build (dist) because it resolves paths
     *   relative to this module at runtime.
     * - Only imports under ./bubbles/** that themselves define a bubble class are
     *   considered dependencies; all other imports are ignored.
     */
    private populateBubbleDependenciesFromSource;
    private listModuleFilesRecursively;
    private extractBubbleNamesFromContent;
    /**
     * Get credential to bubble name mapping from registered bubbles
     * Provides type-safe mapping based on actual registered bubbles
     */
    getCredentialToBubbleMapping(): Partial<Record<CredentialType, BubbleName>>;
    /**
     * Get bubble name for a specific credential type
     */
    getBubbleNameForCredential(credentialType: CredentialType): BubbleName | undefined;
    /**
     * Check if a credential type is supported by any registered bubble
     */
    isCredentialSupported(credentialType: CredentialType): boolean;
    /**
     * Generate comprehensive BubbleFlow boilerplate template with all imports
     * This template includes ALL available bubble classes and types
     * Perfect for AI code generation and testing
     */
    generateBubbleFlowBoilerplate(options?: {
        className?: string;
    }): string;
}
export {};
//# sourceMappingURL=bubble-factory.d.ts.map