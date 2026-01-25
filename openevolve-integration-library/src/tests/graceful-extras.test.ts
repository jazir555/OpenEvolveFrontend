
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

describe('Graceful Failure Extras', () => {
  let client: OpenEvolveClient;

  beforeEach(() => {
    jest.clearAllMocks();
    client = new OpenEvolveClient({
      baseUrl: 'http://localhost:8000',
      retryConfig: {
        maxAttempts: 2,
        initialDelay: 0,
        maxDelay: 0
      }
    });
  });

  it('should return fallback data on failure if provided', async () => {
    mockedAxios.post.mockRejectedValue({
      response: { status: 500, data: { message: 'Failed' } }
    });

    const fallback = { status: 'degraded', results: [] };
    
    const result = await client.integrations.knowledge.execute({
      operation: 'query',
      input: {}
    } as any, { fallback });

    expect(result).toBe(fallback);
  });

  it('should call onRetry callback during retry attempts', async () => {
    mockedAxios.post
      .mockRejectedValueOnce({ response: { status: 500, data: { message: 'Server Error' } } })
      .mockResolvedValueOnce({ data: { success: true } });

    const onRetry = jest.fn();

    const result = await client.integrations.decomposition.execute({
      operation: 'decompose',
      input: { problem: 'test' }
    }, { onRetry });

    expect(result).toEqual({ success: true });
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRetry.mock.calls[0][0]).toBeInstanceOf(IntegrationError);
    expect(onRetry.mock.calls[0][1]).toBe(1); // attempt 1
  });

      it('should throw ParseError if response validation fails', async () => {

        mockedAxios.post.mockResolvedValue({ data: { malformed: true } });

        

        // We override validateResponse for this test

        const anyIntegration = client.integrations.decomposition as any;

        const originalValidate = anyIntegration.validateResponse;

        anyIntegration.validateResponse = () => ({ valid: false, errors: ['Invalid schema'] });

    

        try {

          await client.integrations.decomposition.execute({

            operation: 'decompose',

            input: { problem: 'test' }

          });

          fail('Should have thrown');

        } catch (error: any) {

          expect(error.code).toBe('PARSE_ERROR');

          expect(error.message).toContain('validation failed');

        } finally {

          anyIntegration.validateResponse = originalValidate;

        }

      });

    });

    
