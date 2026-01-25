jest.mock('uuid', () => ({
  v4: () => 'mock-uuid'
}));

import { OpenEvolveClient } from '../api/client';
import { loggingMiddleware, createCachingMiddleware } from '../api/middleware';
import { BackendClient } from '../api/backend';

jest.mock('../api/backend');

describe('Middleware Pipeline', () => {
  let mockBackend: jest.Mocked<BackendClient>;

  beforeEach(() => {
    jest.clearAllMocks();
    mockBackend = new (BackendClient as any)();
    (mockBackend as any).post = jest.fn().mockResolvedValue({ success: true, data: 'real-data' });
    (mockBackend as any).ping = jest.fn().mockResolvedValue(true);
  });

  it('should run logging middleware', async () => {
    const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
    const client = new OpenEvolveClient({
      baseUrl: 'http://localhost:8000',
      middleware: [loggingMiddleware]
    });
    
    // Inject mock backend
    (client as any).backend = mockBackend;

    await client.execute('leanaide', { operation: 'query', input: {} });

    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Executing leanaide...'), expect.anything());
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('leanaide completed in'));
    consoleSpy.mockRestore();
  });

  it('should cache results and respect bypassCache', async () => {
    const cachingMiddleware = createCachingMiddleware();
    const client = new OpenEvolveClient({
      baseUrl: 'http://localhost:8000',
      middleware: [cachingMiddleware]
    });
    
    (client as any).backend = mockBackend;
    const adapter = (client as any).integrationAdapters.get('knowledge');
    const executeSpy = jest.spyOn(adapter, 'execute').mockResolvedValue({ success: true, data: 'real-data' });

    // First call - should hit backend (adapter)
    await client.execute('knowledge', { operation: 'query', input: { q: 'test' } });
    expect(executeSpy).toHaveBeenCalledTimes(1);

    // Second call - should hit cache
    await client.execute('knowledge', { operation: 'query', input: { q: 'test' } });
    expect(executeSpy).toHaveBeenCalledTimes(1);

    // Third call with bypassCache - should hit backend (adapter)
    await client.execute('knowledge', { operation: 'query', input: { q: 'test' } }, { bypassCache: true });
    expect(executeSpy).toHaveBeenCalledTimes(2);
  });

  it('should prevent multiple next() calls', async () => {
    const badMiddleware = async (_ctx: any, next: any) => {
      await next();
      await next();
    };

    const client = new OpenEvolveClient({
      baseUrl: 'http://localhost:8000',
      middleware: [badMiddleware]
    });
    
    (client as any).backend = mockBackend;

    await expect(client.execute('leanaide', { operation: 'query', input: {} }))
      .rejects.toThrow('next() called multiple times');
  });
});