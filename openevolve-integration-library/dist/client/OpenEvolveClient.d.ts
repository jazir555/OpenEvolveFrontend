import type { OpenEvolveConfig, Integrations } from '../types';
export declare class OpenEvolveClient {
    private apiClient;
    readonly integrations: Integrations;
    readonly config: OpenEvolveConfig;
    constructor(config: OpenEvolveConfig);
    updateConfig(config: Partial<OpenEvolveConfig>): void;
    getConfig(): OpenEvolveConfig;
    disconnect(): void;
    healthCheck(): Promise<Record<string, boolean>>;
    getVersions(): Promise<Record<string, string>>;
    batchExecute<T extends Record<string, any>>(operations: {
        integration: keyof Integrations;
        inputs: any;
    }[]): Promise<T[]>;
    static create(config: OpenEvolveConfig): OpenEvolveClient;
}
//# sourceMappingURL=OpenEvolveClient.d.ts.map