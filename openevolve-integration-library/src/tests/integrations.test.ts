import { BaseIntegrationAdapter } from '../integrations/base';
import { BackendClient } from '../api/backend';
import { ParameterSchema, ValidationResult } from '../api/types';

class TestIntegration extends BaseIntegrationAdapter {
  constructor(client: BackendClient) {
    super(client, 'test', '1.0.0', 'Test Integration');
  }

  async execute(inputs: any): Promise<any> {
    return this.executeBackend('/test', inputs);
  }

  async validate(inputs: any): Promise<ValidationResult> {
    const errors = [
      ...this.validateRequired(inputs, ['req']),
      ...this.validateTypes(inputs, { num: 'number' }),
      ...this.validateEnum(inputs, { choice: ['a', 'b'] })
    ];
    return {
      valid: errors.length === 0,
      errors: errors.map(msg => ({ field: 'unknown', message: msg, code: 'VALIDATION_ERROR' })),
      warnings: []
    };
  }

  getSchema(): ParameterSchema {
    return { type: 'object', properties: {} };
  }

  protected getEndpoints(): string[] {
    return ['/test'];
  }
}

describe('BaseIntegrationAdapter', () => {
  let adapter: TestIntegration;
  let mockBackend: jest.Mocked<BackendClient>;

  beforeEach(() => {
    mockBackend = {
      post: jest.fn(),
      ping: jest.fn().mockResolvedValue(true)
    } as any;
    adapter = new TestIntegration(mockBackend);
  });

  describe('Validation Helpers', () => {
    it('should validate required fields', async () => {
      const res = await adapter.validate({});
      expect(res.valid).toBe(false);
      expect(res.errors[0].message).toContain("Required field 'req'");
    });

    it('should validate types', async () => {
      const res = await adapter.validate({ req: 1, num: 'string' });
      expect(res.valid).toBe(false);
      expect(res.errors[0].message).toContain("invalid type");
    });

    it('should validate enums', async () => {
      const res = await adapter.validate({ req: 1, choice: 'c' });
      expect(res.valid).toBe(false);
      expect(res.errors[0].message).toContain("invalid value");
    });

    it('should pass valid inputs', async () => {
      const res = await adapter.validate({ req: 1, num: 10, choice: 'a' });
      expect(res.valid).toBe(true);
    });
  });

  describe('Backend Execution', () => {
    it('should call backend and transform response', async () => {
      mockBackend.post.mockResolvedValue({ data: 123 });
      const res = await adapter.execute({ test: true });
      expect(res).toEqual({ data: 123 });
      expect(mockBackend.post).toHaveBeenCalledWith(
        '/test', 
        { test: true }, 
        expect.objectContaining({ signal: expect.anything() })
      );
    });
  });
});
