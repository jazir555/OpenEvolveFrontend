import { EvolutionIntegration, DecompositionIntegration } from '../integrations/all-integrations';
import { BackendClient } from '../api/backend';

describe('Integration Specific Validations', () => {
  let mockBackend: jest.Mocked<BackendClient>;

  beforeEach(() => {
    mockBackend = {
      post: jest.fn(),
      get: jest.fn(),
      ping: jest.fn().mockResolvedValue(true)
    } as any;
  });

  describe('EvolutionIntegration', () => {
    it('should fail with invalid operation', async () => {
      const integration = new EvolutionIntegration(mockBackend);
      const res = await integration.validate({ 
        operation: 'invalid' as any, 
        config: {} 
      });
      expect(res.valid).toBe(false);
      expect(res.errors[0].message).toContain("Value must be one of");
    });

    it('should fail when config is missing', async () => {
      const integration = new EvolutionIntegration(mockBackend);
      const res = await integration.validate({ 
        operation: 'evolution'
      } as any);
      expect(res.valid).toBe(false);
      expect(res.errors[0].message).toContain("Required field 'config' is missing");
    });
  });

  describe('DecompositionIntegration', () => {
    it('should fail when input is missing', async () => {
      const integration = new DecompositionIntegration(mockBackend);
      const res = await integration.validate({ 
        operation: 'decompose'
      } as any);
      expect(res.valid).toBe(false);
      expect(res.errors[0].message).toContain("Required field 'input' is missing");
    });

    it('should pass valid input', async () => {
      const integration = new DecompositionIntegration(mockBackend);
      const res = await integration.validate({ 
        operation: 'decompose',
        input: { problem: 'test' }
      });
      expect(res.valid).toBe(true);
    });
  });
});
