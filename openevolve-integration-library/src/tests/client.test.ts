import { OpenEvolveClient } from '../api/client';
import { IntegrationName } from '../api/client';

// Mock uuid
jest.mock('uuid', () => ({
  v4: () => 'test-uuid'
}));

// Mock BackendClient to avoid network requests
jest.mock('../api/backend', () => {
  return {
    BackendClient: jest.fn().mockImplementation(() => ({
      post: jest.fn().mockResolvedValue({ success: true }),
      get: jest.fn().mockResolvedValue({ success: true }),
      websocket: jest.fn(),
      ping: jest.fn().mockResolvedValue(true),
      isWebSocketConnected: jest.fn().mockReturnValue(true)
    }))
  };
});

describe('OpenEvolveClient', () => {
  let client: OpenEvolveClient;

  beforeEach(() => {
    client = new OpenEvolveClient({
      baseUrl: 'http://localhost:8000'
    });
  });

  it('should initialize with correct configuration', () => {
    expect(client).toBeDefined();
    expect(client.getBackend()).toBeDefined();
  });

  it('should load all integration adapters', () => {
    expect(client.integrations.decomposition).toBeDefined();
    expect(client.integrations.leanaide).toBeDefined();
    expect(client.integrations.evolution).toBeDefined();
    expect(client.integrations.knowledge).toBeDefined();
    expect(client.integrations.maker).toBeDefined();
    expect(client.integrations.crewai).toBeDefined();
  });

  it('should execute integration successfully', async () => {
    const result = await client.execute(IntegrationName.DECOMPOSITION, {
      operation: 'decompose',
      input: { problem: 'test' }
    });
    expect(result).toEqual({ success: true });
  });

  it('should allow access via integration property', async () => {
    jest.spyOn(client.integrations.decomposition, 'execute').mockResolvedValue({ success: true } as any);
    const result = await client.integrations.decomposition.execute({
      operation: 'decompose',
      input: { problem: 'test' }
    }, { executionId: 'test-id' });
    expect(result).toEqual({ success: true });
  });
});
