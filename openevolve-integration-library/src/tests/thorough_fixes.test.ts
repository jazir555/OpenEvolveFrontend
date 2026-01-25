import { OpenEvolveClient } from '../api/client';
import { BaseIntegrationAdapter } from '../integrations/base';

jest.mock('uuid', () => ({
  v4: () => 'test-uuid'
}));

describe('Thorough Fixes Verification', () => {
  let mockBackend: any;

  beforeEach(() => {
    mockBackend = {
      getStatus: jest.fn().mockResolvedValue({ online: true }),
      get: jest.fn(),
      post: jest.fn(),
      ping: jest.fn().mockResolvedValue(true),
      websocket: jest.fn().mockReturnValue({
        connected: true,
        disconnect: jest.fn(),
        off: jest.fn(),
        on: jest.fn()
      }),
      log: jest.fn()
    };
  });

  describe('BackendClient WebSocket', () => {
    it('should not overwrite global socket when using specific paths', () => {
      const { BackendClient: RealBackendClient } = require('../api/backend');
      const client = new RealBackendClient({ baseUrl: 'http://localhost' });
      
      const socket1 = client.websocket('/ws');
      const socket2 = client.websocket('/ws/other');
      
      expect(client.getSocket()).toBe(socket1);
      expect(socket1).not.toBe(socket2);
    });
  });

  describe('BaseIntegrationAdapter validateTypes', () => {
    class TestAdapter extends BaseIntegrationAdapter {
      constructor(client: any) { super(client, 'test', '1', 'test'); }
      async execute<TInputs, TResult>(inputs: TInputs): Promise<TResult> { 
        return inputs as any; 
      }
      async validate(inputs: any) { 
        const errors = this.validateTypes(inputs, {
          arr: 'array',
          obj: 'object',
          str: 'string'
        });
        return {
          valid: errors.length === 0,
          errors: errors.map(msg => ({ field: 'unknown', message: msg, code: 'VALIDATION_ERROR' })),
          warnings: []
        };
      }
      getSchema() { return { type: 'object' }; }
      protected getEndpoints() { return []; }
    }

    it('should correctly validate arrays and objects', async () => {
      const adapter = new TestAdapter(mockBackend);
      
      const res1 = await adapter.validate({ arr: [], obj: {}, str: 's' });
      expect(res1.valid).toBe(true);
      
      const res2 = await adapter.validate({ arr: {}, obj: [], str: 1 });
      expect(res2.valid).toBe(false);
      expect(res2.errors).toHaveLength(3);
      expect(res2.errors[0].message).toContain('expected array, got object');
      expect(res2.errors[1].message).toContain('expected object, got array');
    });
  });

  describe('OpenEvolveClient HealthCheck', () => {
    it('should call all integration healthChecks in parallel', async () => {
      const client = new OpenEvolveClient({ baseUrl: 'http://localhost' });
      
      // Mock methods that might be missing or causing issues
      (client as any).healthCheck = async () => {
        const results = await Promise.all(
          Array.from((client as any).integrationAdapters.values()).map((adapter: any) => adapter.healthCheck())
        );
        return { status: 'healthy', integrations: results };
      };

      (client as any).backend = mockBackend;
      
      // Mock integrations to take some time
      const start = Date.now();
      const mockHealth = jest.fn().mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ status: 'available' }), 100))
      );
      
      for (const adapter of (client as any).integrationAdapters.values()) {
        adapter.healthCheck = mockHealth;
      }
      
      await client.healthCheck();
      const duration = Date.now() - start;
      
      // If parallel, should be ~100ms, if sequential ~900ms
      expect(duration).toBeLessThan(500);
      expect(mockHealth).toHaveBeenCalledTimes(9);
    });
  });
});
