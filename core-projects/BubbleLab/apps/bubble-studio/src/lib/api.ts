import { refreshToken } from './token-refresh';
import { toast } from 'react-toastify';
import { API_BASE_URL, OPENEVOLVE_API_KEY } from '../env';

export interface ApiClientConfig {
  baseURL: string;
  timeout?: number;
  enableRetry?: boolean;
  maxRetries?: number;
  retryDelay?: number;
}

const getOpenEvolveApiKey = (): string | undefined => {
  if (OPENEVOLVE_API_KEY && OPENEVOLVE_API_KEY.trim().length > 0) {
    return OPENEVOLVE_API_KEY.trim();
  }
  try {
    const stored = globalThis.localStorage?.getItem('openevolve_api_key');
    if (stored && stored.trim().length > 0) {
      return stored.trim();
    }
  } catch {
    // ignore localStorage access errors
  }
  return undefined;
};

export interface ApiError {
  status: number;
  data: unknown;
}

export class ApiHttpError extends Error implements ApiError {
  status: number;
  data: unknown;

  constructor(status: number, data: unknown) {
    super(`HTTP ${status}: ${JSON.stringify(data)}`);
    this.name = 'ApiHttpError';
    this.status = status;
    this.data = data;
  }
}

export class ApiClient {
  private baseURL: string;
  private timeout: number;
  private enableRetry: boolean;
  private maxRetries: number;
  private retryDelay: number;

  constructor(configOrBaseURL: string | ApiClientConfig) {
    if (typeof configOrBaseURL === 'string') {
      this.baseURL = configOrBaseURL;
      this.timeout = 30000;
      this.enableRetry = false;
      this.maxRetries = 0;
      this.retryDelay = 1000;
      return;
    }

    this.baseURL = configOrBaseURL.baseURL;
    this.timeout = configOrBaseURL.timeout ?? 30000;
    this.enableRetry = configOrBaseURL.enableRetry ?? false;
    this.maxRetries = configOrBaseURL.maxRetries ?? 0;
    this.retryDelay = configOrBaseURL.retryDelay ?? 1000;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private shouldRetryStatus(status: number): boolean {
    return status === 429 || status >= 500;
  }

  private async parseErrorData(response: Response): Promise<unknown> {
    try {
      return await response.json();
    } catch {
      return await response.text();
    }
  }

  // Extract a human-friendly message from various error payload shapes
  private extractErrorMessage(errorData: unknown): string {
    if (typeof errorData === 'string') return errorData;
    if (!errorData || typeof errorData !== 'object') return 'Request failed';
    const data = errorData as Record<string, unknown>;
    const str = (v: unknown) => (typeof v === 'string' ? v : undefined);
    // Direct
    const direct = str(data.message) || str(data.error) || str(data.detail);
    if (direct) return direct;
    // Nested data
    const nestedData = data.data as Record<string, unknown> | undefined;
    if (nestedData && typeof nestedData === 'object') {
      const nestedMsg =
        str(nestedData.message) ||
        str(nestedData.error) ||
        str(nestedData.detail);
      if (nestedMsg) return nestedMsg;
    }
    // Arrays
    const errors = data.errors as unknown;
    if (Array.isArray(errors) && errors.length > 0) {
      const first = errors[0];
      if (typeof first === 'string') return first;
      if (
        first &&
        typeof first === 'object' &&
        typeof first.message === 'string'
      ) {
        return first.message as string;
      }
    }
    try {
      return JSON.stringify(errorData);
    } catch {
      return 'Request failed';
    }
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    let lastError: unknown = null;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      let timeoutId: ReturnType<typeof setTimeout> | undefined;

      try {
        const token = await refreshToken();
        const apiKey = getOpenEvolveApiKey();
        const config: RequestInit = {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            ...(token && { Authorization: `Bearer ${token}` }),
            ...(apiKey && { 'X-API-Key': apiKey }),
            ...options.headers,
          },
        };

        const fetchPromise = fetch(url, config);
        const timeoutPromise = new Promise<Response>((_, reject) => {
          timeoutId = setTimeout(() => reject(new Error('Request timeout')), this.timeout);
        });

        const response = await Promise.race([fetchPromise, timeoutPromise]);
        if (timeoutId) {
          clearTimeout(timeoutId);
        }

        if (response.status === 401) {
          const errorData = await this.parseErrorData(response);
          const backendMessage = this.extractErrorMessage(errorData);
          const message =
            backendMessage || 'Authentication failed - please log in again';

          console.error('Authentication failed - user may need to log in again');
          toast.error(message, {
            autoClose: 4000,
          });
          throw new Error(message);
        }

        if (!response.ok) {
          const errorData = await this.parseErrorData(response);
          const retriable =
            this.enableRetry &&
            attempt < this.maxRetries &&
            this.shouldRetryStatus(response.status);

          if (retriable) {
            await this.sleep(this.retryDelay * (attempt + 1));
            continue;
          }

          const apiError: ApiError = {
            status: response.status,
            data: errorData,
          };

          console.error('API Error:', apiError);
          try {
            if (response.status !== 404) {
              const message = this.extractErrorMessage(errorData);
              toast.error(message || 'Request failed', {
                autoClose: 5000,
              });
            }
          } catch {
            if (response.status !== 404) {
              toast.error('Request failed', { autoClose: 5000 });
            }
          }

          throw new ApiHttpError(response.status, errorData);
        }

        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
          return await response.json();
        }

        return (await response.text()) as T;
      } catch (error) {
        if (timeoutId) {
          clearTimeout(timeoutId);
        }

        const shouldRetryError =
          this.enableRetry &&
          attempt < this.maxRetries &&
          !(error instanceof ApiHttpError) &&
          !(error instanceof Error && /Authentication failed/.test(error.message));

        if (shouldRetryError) {
          lastError = error;
          await this.sleep(this.retryDelay * (attempt + 1));
          continue;
        }

        if (error instanceof Error) {
          if (
            !/Authentication failed/.test(error.message) &&
            !/^HTTP \d+/.test(error.message)
          ) {
            toast.error(error.message || 'Network error', { autoClose: 5000 });
          }
          throw error;
        }

        const msg = `Network error: ${String(error)}`;
        toast.error(msg, { autoClose: 5000 });
        throw new Error(msg);
      }
    }

    if (lastError instanceof Error) {
      throw lastError;
    }

    throw new Error('Request failed');
  }

  // Streaming request for Server-Sent Events
  async makeStreamingRequest(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const token = await refreshToken();
    const apiKey = getOpenEvolveApiKey();

    const url = `${this.baseURL}${endpoint}`;
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...(apiKey && { 'X-API-Key': apiKey }),
        ...options.headers,
      },
    };

    const response = await fetch(url, config);

    if (response.status === 401) {
      toast.error('Authentication failed - please log in again', {
        autoClose: 4000,
      });
      throw new Error('Authentication failed - please log in again');
    }

    if (!response.ok) {
      let errorData: unknown;
      try {
        errorData = await response.json();
      } catch {
        errorData = await response.text();
      }
      try {
        console.log('Error data:', errorData);
        if (response.status !== 404) {
          const message = this.extractErrorMessage(errorData);
          toast.error(message || 'Request failed', {
            autoClose: 5000,
          });
        }
      } catch {
        if (response.status !== 404) {
          toast.error('Request failed', { autoClose: 5000 });
        }
      }
      throw new ApiHttpError(response.status, errorData);
    }

    return response;
  }

  // HTTP methods
  async get<T>(endpoint: string): Promise<T> {
    return this.makeRequest<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.makeRequest<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.makeRequest<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async patch<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.makeRequest<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.makeRequest<T>(endpoint, { method: 'DELETE' });
  }

  // Streaming POST
  async postStream(
    endpoint: string,
    data?: unknown,
    options?: { signal?: AbortSignal }
  ): Promise<Response> {
    return this.makeStreamingRequest(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
      signal: options?.signal,
    });
  }
}

// Create API client instance
export const api = new ApiClient(API_BASE_URL);
