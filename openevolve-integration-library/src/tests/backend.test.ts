import axios from 'axios';
import { BackendClient } from '../api/backend';
import { 
  ConnectionError, 
  TimeoutError, 
  ExecutionError,
  ValidationError,
  NotFoundError
} from '../api/errors';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('BackendClient', () => {
  let client: BackendClient;
  const config = {
    baseUrl: 'http://api.test',
    apiKey: 'test-key'
  };

  beforeEach(() => {
    mockedAxios.create.mockReturnValue(mockedAxios as any);
    client = new BackendClient(config);
    jest.clearAllMocks();
  });

  describe('HTTP Methods', () => {
    it('should perform GET request', async () => {
      mockedAxios.get.mockResolvedValue({ data: { success: true } });
      const result = await client.get('/test');
      expect(result).toEqual({ success: true });
      expect(mockedAxios.get).toHaveBeenCalledWith('/test', undefined);
    });

    it('should perform POST request', async () => {
      mockedAxios.post.mockResolvedValue({ data: { success: true } });
      const result = await client.post('/test', { payload: 1 });
      expect(result).toEqual({ success: true });
      expect(mockedAxios.post).toHaveBeenCalledWith('/test', { payload: 1 }, undefined);
    });
  });

  describe('Error Handling', () => {
    it('should handle 400 Bad Request as ValidationError', async () => {
      mockedAxios.get.mockRejectedValue({
        response: {
          status: 400,
          data: { errors: [{ field: 'test', message: 'error', code: 'test' }] }
        }
      });
      await expect(client.get('/test')).rejects.toThrow(ValidationError);
    });

    it('should handle 404 Not Found as NotFoundError', async () => {
      mockedAxios.get.mockRejectedValue({
        response: { status: 404, data: { message: 'not found' } },
        config: { url: '/test' }
      });
      await expect(client.get('/test')).rejects.toThrow(NotFoundError);
    });

    it('should handle timeout as TimeoutError', async () => {
      mockedAxios.get.mockRejectedValue({
        code: 'ECONNABORTED',
        config: { timeout: 1000 }
      });
      await expect(client.get('/test')).rejects.toThrow(TimeoutError);
    });

    it('should handle network failure as ConnectionError', async () => {
      mockedAxios.get.mockRejectedValue({
        request: {},
        code: 'ECONNREFUSED'
      });
      await expect(client.get('/test')).rejects.toThrow(ConnectionError);
    });

    it('should handle generic server errors as ExecutionError', async () => {
      mockedAxios.get.mockRejectedValue({
        response: { status: 500, data: { message: 'server error' } }
      });
      await expect(client.get('/test')).rejects.toThrow(ExecutionError);
    });
  });

  describe('Interceptors', () => {
    it('should apply request transform', async () => {
      const transform = jest.fn(data => ({ ...data, transformed: true }));
      new BackendClient({ ...config, requestTransform: transform });
      
      // Manually trigger interceptor
      const interceptor = (mockedAxios.interceptors.request.use as jest.Mock).mock.calls[0][0];
      const result = interceptor({ data: { test: 1 } });
      
      expect(result.data).toEqual({ test: 1, transformed: true });
      expect(transform).toHaveBeenCalled();
    });
  });
});
