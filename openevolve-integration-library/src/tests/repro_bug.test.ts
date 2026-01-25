import { LeanAideIntegration } from '../integrations/all-integrations';
import { BackendClient } from '../api/backend';

describe('Bug Reproduction', () => {
  let mockBackend: jest.Mocked<BackendClient>;

  beforeEach(() => {
    mockBackend = {
      post: jest.fn(),
    } as any;
  });

  it('should throw ValidationError and not show [object Object]', async () => {
    const integration = new LeanAideIntegration(mockBackend);
    try {
      // Missing 'operation' and 'input'
      await integration.execute({} as any, { executionId: 'test-id' });
      throw new Error('Should have thrown an error');
    } catch (error: any) {
      expect(error.name).toBe('ValidationError');
      expect(error.message).toContain('Validation failed with 2 error(s)');
      expect(error.message).not.toContain('[object Object]');
      expect(error.errors).toHaveLength(2);
      expect(error.errors[0].field).toBe('operation');
    }
  });
});
