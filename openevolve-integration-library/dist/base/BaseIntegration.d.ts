import type { Integration, ValidationResult, ParameterSchema, ProgressUpdate } from '../types';
import { ApiClient } from '../api/client';
export declare abstract class BaseIntegration<TInputs, TResult> implements Integration<TInputs, TResult> {
    abstract name: string;
    abstract version: string;
    description?: string;
    protected client: ApiClient;
    protected endpoint: string;
    constructor(client: ApiClient, endpoint: string);
    abstract execute(inputs: TInputs): Promise<TResult>;
    abstract getSchema(): ParameterSchema;
    validate(inputs: TInputs): ValidationResult;
    executeStream(inputs: TInputs, onUpdate: (update: ProgressUpdate) => void): Promise<TResult>;
    protected request(method: 'GET' | 'POST' | 'PUT' | 'DELETE', data?: any): Promise<TResult>;
    protected handleError(error: unknown): never;
    getMetadata(): {
        name: string;
        version: string;
        description: string | undefined;
        endpoint: string;
    };
}
//# sourceMappingURL=BaseIntegration.d.ts.map