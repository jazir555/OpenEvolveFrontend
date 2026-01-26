import { OpenEvolveClient } from '../api/client';
import { IntegrationName } from '../api/client';
type MockResponses = Partial<Record<IntegrationName | string, any>>;
type MockErrors = Partial<Record<IntegrationName | string, Error>>;
export declare function createMockClient(mockResponses?: MockResponses, mockErrors?: MockErrors): OpenEvolveClient;
export {};
//# sourceMappingURL=index.d.ts.map