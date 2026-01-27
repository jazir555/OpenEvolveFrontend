export interface VariableTypeInfo {
    name: string;
    type: string;
    line: number;
    column: number;
}
type DiagnosticsResult = {
    success: boolean;
    errors?: Record<number, string>;
    variableTypes?: VariableTypeInfo[];
};
/**
 * Minimal warmed TypeScript checker for validating string scripts.
 * - Uses LanguageService for fast incremental diagnostics
 * - Keeps shared libs/graphs in memory across requests
 * - Backend callers pass only a code string; imports resolve via tsconfig
 */
declare class LanguageServiceTypechecker {
    private readonly projectDir;
    private readonly options;
    private readonly sanitizedOptions;
    private readonly configFileNames;
    private readonly snapshots;
    private readonly documentRegistry;
    private readonly moduleResolutionCache;
    private languageService;
    constructor(configPath?: string);
    private normalize;
    private getVirtualFiles;
    /**
     * Validate an in-memory script and return diagnostics.
     * fileName affects caching identity and relative module resolution.
     */
    checkCode(code: string, fileName?: string): DiagnosticsResult;
    prewarm(): void;
    removeVirtualFile(fileName: string): void;
    private createFormatHost;
    /**
     * Expands a type to show its full structure, not just the type name
     */
    private expandTypeString;
    /**
     * Extracts type definitions for all variables in the source file
     */
    private extractVariableTypes;
}
export declare function getWarmChecker(configPath?: string): LanguageServiceTypechecker;
/**
 * Convenience API for backend use: pass code string and receive diagnostics.
 * - fileName only influences cache identity and relative module resolution.
 * - configPath selects the tsconfig used to resolve libs and types.
 */
export declare function validateScript(code: string, options?: {
    fileName?: string;
    configPath?: string;
}): DiagnosticsResult;
export {};
//# sourceMappingURL=BubbleValidator.d.ts.map