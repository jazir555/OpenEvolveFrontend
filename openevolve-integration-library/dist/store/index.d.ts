import { OpenEvolveClient } from '../api/client';
import { IntegrationName } from '../api/client';
import { IntegrationError } from '../api/errors';
export interface OpenEvolveState {
    client: OpenEvolveClient | null;
    results: Record<string, any>;
    loading: Record<string, boolean>;
    errors: Record<string, IntegrationError | null>;
    versions: Record<string, number>;
    initialize: (client: OpenEvolveClient) => void;
    execute: (integration: IntegrationName | string, inputs: any) => Promise<any>;
    clearResult: (integration: string) => void;
    reset: () => void;
}
export declare const createOpenEvolveStore: () => import("zustand").UseBoundStore<import("zustand").StoreApi<OpenEvolveState>>;
//# sourceMappingURL=index.d.ts.map