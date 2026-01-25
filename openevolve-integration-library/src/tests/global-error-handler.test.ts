
import { OpenEvolveClient } from '../api/client';
import { IntegrationError } from '../api/errors';

import axios from 'axios';

// Mock uuid
jest.mock('uuid', () => ({
  v4: () => 'test-uuid'
}));

jest.mock('axios', () => {
  const mAxios = {
    create: jest.fn().mockReturnThis(),
    interceptors: {
      request: { use: jest.fn(), eject: jest.fn() },
      response: { use: jest.fn(), eject: jest.fn() },
    },
    defaults: { headers: { common: {} } },
    post: jest.fn(),
    get: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    patch: jest.fn(),
  };
  return mAxios;
});
const mockedAxios = axios as any;

describe('Global Error Handler', () => {
  let client: OpenEvolveClient;
  let globalErrorHandler: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    globalErrorHandler = jest.fn();
    client = new OpenEvolveClient({
      baseUrl: 'http://localhost:8000',
      onError: globalErrorHandler,
      debug: true,
      retryConfig: {

        maxAttempts: 1,
        initialDelay: 0,
        maxDelay: 0,
        backoffMultiplier: 1,
        retryOn4xx: true,
        retryOn5xx: true,
        retryableStatusCodes: [500]
      }
    });
  });


  it('should call the global error handler when an execution fails', async () => {
    mockedAxios.post.mockRejectedValue({
      response: { status: 500, data: { message: 'Server Error' } }
    });

    try {
      await client.integrations.decomposition.execute({
        operation: 'decompose',
        input: { problem: 'test' }
      });
    } catch (e) {
      // Expected
    }

    expect(globalErrorHandler).toHaveBeenCalledTimes(1);
    const error = globalErrorHandler.mock.calls[0][0];
    expect(error).toBeInstanceOf(IntegrationError);
    expect(error.integration).toBe('decomposition');
    expect(error.code).toBe('EXECUTION_ERROR');
  });

  it('should support adding multiple error handlers', async () => {
    const secondHandler = jest.fn();
    client.addErrorHandler(secondHandler);

    mockedAxios.post.mockRejectedValue(new Error('Network Fail'));

    try {
      await client.integrations.leanaide.execute({
        operation: 'prove',
        input: { theorem: 'theorem' }
      });
    } catch (e) {}

    expect(globalErrorHandler).toHaveBeenCalledTimes(1);
    expect(secondHandler).toHaveBeenCalledTimes(1);
    expect(secondHandler.mock.calls[0][0].integration).toBe('leanaide');
  });


  it('should support removing error handlers', async () => {
    client.removeErrorHandler(globalErrorHandler);

    mockedAxios.post.mockRejectedValue(new Error('Fail'));

    try {
      await client.integrations.knowledge.execute({
        operation: 'query',
        input: {}
      } as any);
    } catch (e) {}

    expect(globalErrorHandler).not.toHaveBeenCalled();
  });

  it('should report errors from background health checks', async () => {
    // Force unhealthy response
    jest.spyOn(client, 'healthCheck').mockResolvedValue({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      backend: { online: false } as any,
      integrations: {}
    });

    // We manually trigger the timer logic for testing or shorten the interval
    client.startHealthCheck(10);
    
    // Wait for at least one interval
    await new Promise(resolve => setTimeout(resolve, 50));
    
    expect(globalErrorHandler).toHaveBeenCalled();
    const error = globalErrorHandler.mock.calls[0][0];
    expect(error.integration).toBe('backend');
    
    client.stopHealthCheck();
  });
});

