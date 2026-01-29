
import { OpenEvolveClient } from '../api/client';

import { CircuitBreakerError } from '../api/errors';

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





describe('Circuit Breaker', () => {
  let client: OpenEvolveClient;

  beforeEach(() => {
    jest.clearAllMocks();
    client = new OpenEvolveClient({
      baseUrl: 'http://localhost:8000',
      circuitBreakerConfig: {
        enabled: true,
        failureThreshold: 2,
        resetTimeout: 100,
        successThreshold: 1
      },
      retryConfig: {
        maxAttempts: 1,
        initialDelay: 0,
        maxDelay: 0,
        backoffMultiplier: 1,
        retryOn4xx: false,
        retryOn5xx: true,
        retryableStatusCodes: [500]
      }
    });
  });


  it('should open the circuit after reaching the failure threshold', async () => {
    // Mock sequential failures
    mockedAxios.post.mockRejectedValue({
      response: { status: 500, data: { message: 'Server Error' } }
    });

    // First failure
    await expect(client.integrations.decomposition.execute({
      operation: 'decompose',
      input: { problem: 'test' }
    })).rejects.toThrow();

    // Second failure - should open the circuit
    await expect(client.integrations.decomposition.execute({
      operation: 'decompose',
      input: { problem: 'test' }
    })).rejects.toThrow();

    // Third call - should fail immediately with CircuitBreakerError
    try {
      await client.integrations.decomposition.execute({
        operation: 'decompose',
        input: { problem: 'test' }
      });
      fail('Should have thrown CircuitBreakerError');
    } catch (error: any) {
      expect(error).toBeInstanceOf(CircuitBreakerError);
      expect(error.code).toBe('CIRCUIT_OPEN_ERROR');
    }

    // Axios should only have been called twice (for the first two failures)
    expect(mockedAxios.post).toHaveBeenCalledTimes(2);
  });

  it('should move to half-open and then close the circuit after a successful call', async () => {
    mockedAxios.post.mockRejectedValue({
      response: { status: 500, data: { message: 'Server Error' } }
    });

    // Open the circuit (threshold is 2)
    await expect(client.integrations.decomposition.execute({ operation: 'decompose', input: {} })).rejects.toThrow();
    await expect(client.integrations.decomposition.execute({ operation: 'decompose', input: {} })).rejects.toThrow();

    // Verify it is open
    await expect(client.integrations.decomposition.execute({ operation: 'decompose', input: {} })).rejects.toThrow(CircuitBreakerError);

    // Wait for reset timeout (100ms)
    await new Promise(resolve => setTimeout(resolve, 150));

    // Next call should be "half-open" - we mock success
    mockedAxios.post.mockResolvedValue({ data: { success: true } });
    mockedAxios.get.mockResolvedValue({ data: { online: true } }); // for health check/ping if needed

    const result = await client.integrations.decomposition.execute({
      operation: 'decompose',
      input: { problem: 'test' }
    });

    expect(result).toEqual({ success: true });
    expect(mockedAxios.post).toHaveBeenCalledTimes(3);

    // Circuit should now be closed again, another call should work
    const result2 = await client.integrations.decomposition.execute({
      operation: 'decompose',
      input: { problem: 'test' }
    });
    expect(result2).toEqual({ success: true });
    expect(mockedAxios.post).toHaveBeenCalledTimes(4);
  });
});
