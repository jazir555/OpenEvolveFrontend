import { OpenEvolveClient } from '../api/client';
import { IntegrationName } from '../api/client';

// Mock types
type MockResponses = Partial<Record<IntegrationName | string, any>>;
type MockErrors = Partial<Record<IntegrationName | string, Error>>;

/**
 * Creates a fully mocked OpenEvolveClient for testing
 */
export function createMockClient(
  mockResponses: MockResponses = {},
  mockErrors: MockErrors = {}
): OpenEvolveClient {
  const client = new OpenEvolveClient({
    baseUrl: 'http://mock-client',
    enableWebSocket: false
  });

  // Mock the backend client
  const backend = client.getBackend();
  jest.spyOn(backend, 'post').mockImplementation(async (endpoint: string, data: any) => {
    const integration = endpoint.split('/').find(p => 
      Object.values(IntegrationName).includes(p as IntegrationName)
    );
    
    if (integration && mockErrors[integration]) {
      throw mockErrors[integration];
    }
    
    if (integration && mockResponses[integration]) {
      return mockResponses[integration];
    }
    return { success: true, mock: true, endpoint, data };
  });

  jest.spyOn(backend, 'get').mockImplementation(async (endpoint: string) => {
    // Check if any integration matches in the endpoint
    const integration = endpoint.split('/').find(p => 
      Object.values(IntegrationName).includes(p as IntegrationName)
    );
    
    if (integration && mockErrors[integration]) {
      throw mockErrors[integration];
    }

    return { success: true, mock: true, endpoint };
  });

  jest.spyOn(backend, 'ping').mockResolvedValue(true);
  jest.spyOn(backend, 'getStatus').mockResolvedValue({
    online: true,
    version: '1.0.0-mock',
    uptime: 1000,
    activeConnections: 0,
    memory: { used: 0, total: 0, percentage: 0 },
    cpu: 0
  });

  // Mock the execute method
  jest.spyOn(client, 'execute').mockImplementation(async (integration: string, inputs: any, options?: any) => {
    if (mockErrors[integration]) {
      throw mockErrors[integration];
    }
    
    if (mockResponses[integration]) {
      return mockResponses[integration];
    }
    return { success: true, mock: true, integration, inputs, executionId: options?.executionId };
  });

  // Mock executeStream
  jest.spyOn(client, 'executeStream').mockImplementation(
    async (integration: string, inputs: any, onProgress: any, options?: any) => {
      const executionId = options?.executionId || 'mock-id';
      
      if (mockErrors[integration]) {
        throw mockErrors[integration];
      }

      onProgress({
        integration,
        executionId,
        progress: 100,
        message: 'Complete',
        timestamp: new Date().toISOString()
      });
      
      if (mockResponses[integration]) {
        return mockResponses[integration];
      }
      return { success: true, mock: true, integration, inputs, executionId };
    }
  );

  return client;
}

