interface LeanAideConfig {
    serverUrl: string;
    apiKey?: string;
}
export interface LeanAideTaskRequest {
    taskType: string;
    input: string;
    context?: string;
    timeout?: number;
}
export interface LeanAideTaskResponse {
    success: boolean;
    task: string;
    data?: any;
    error?: string;
    logs?: string;
}
export declare class LeanAideClient {
    private config;
    private baseURL;
    constructor(config: LeanAideConfig);
    private request;
    translateTheorem(theoremStatement: string, context?: string): Promise<LeanAideTaskResponse>;
    translateDefinition(definitionStatement: string, context?: string): Promise<LeanAideTaskResponse>;
    verifySolution(problem: string, solution: string, context?: string): Promise<LeanAideTaskResponse>;
    elaborateCode(leanCode: string, context?: string): Promise<LeanAideTaskResponse>;
    mathQuery(query: string, context?: string): Promise<LeanAideTaskResponse>;
}
export {};
