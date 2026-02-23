import { LeanAideClient, LeanAideTaskResponse } from '../lib/leanaideClient';
export declare function getLeanAideClient(): LeanAideClient;
export declare function initializeLeanAideClient(config: {
    serverUrl?: string;
    apiKey?: string;
}): void;
export declare function translateTheorem(theoremStatement: string, context?: string): Promise<LeanAideTaskResponse>;
export declare function translateDefinition(definitionStatement: string, context?: string): Promise<LeanAideTaskResponse>;
export declare function verifySolution(problem: string, solution: string, context?: string): Promise<LeanAideTaskResponse>;
export declare function elaborateCode(leanCode: string, context?: string): Promise<LeanAideTaskResponse>;
export declare function mathQuery(query: string, context?: string): Promise<LeanAideTaskResponse>;
export declare function isLeanAideAvailable(): boolean;
//# sourceMappingURL=leanaideService.d.ts.map