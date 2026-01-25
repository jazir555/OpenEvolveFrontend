import { useAuthStore } from '@/stores/authStore';
import { gracefulErrorHandler, ErrorHandlingOptions } from '@/utils/gracefulErrorHandler';
import { withApiErrorHandling, ApiErrorHandlingOptions } from '@/utils/ApiErrorHandlingMiddleware';
import { createEnhancedErrorContext } from '@/utils/EnhancedErrorContext';

/**
 * API configuration
 */
const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
};

/**
 * API response wrapper
 */
export interface ApiResponse<T = any> {
  data: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

/**
 * API client class
 */
class ApiClient {
  private baseURL: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;

  constructor() {
    this.baseURL = API_CONFIG.baseURL;
    this.timeout = API_CONFIG.timeout;
    this.defaultHeaders = API_CONFIG.headers;
  }

  /**
   * Get authentication headers
   */
  private getAuthHeaders(): Record<string, string> {
    const token = useAuthStore.getState().token;
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  /**
   * Handle API errors
   */
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      let errorData: any = {};

      // Try to parse error response, but handle if it fails
      try {
        errorData = await response.json();
      } catch (parseError) {
        // If JSON parsing fails, create a generic error object
        errorData = {
          error: {
            code: `HTTP_${response.status}`,
            message: `HTTP Error: ${response.status} ${response.statusText}`,
            details: null
          }
        };
      }

      const error = {
        code: errorData.error?.code || `HTTP_${response.status}`,
        message: errorData.error?.message || response.statusText,
        details: errorData.error?.details || null,
        status: response.status,
        url: response.url
      };

      throw error;
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return null as T;
    }

    // Try to parse JSON response, handle if it fails
    try {
      return await response.json();
    } catch (parseError) {
      // If JSON parsing fails, return the raw response text
      console.warn(`Failed to parse JSON response from ${response.url}`, parseError);
      return await response.text() as unknown as T;
    }
  }

  /**
   * Make a request with timeout
   */
  private async requestWithTimeout(
    url: string,
    options: RequestInit
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${this.timeout}ms to ${url}`);
      }
      throw new Error(`Request failed to ${url}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Resolve a base URL to an absolute URL.
   */
  private resolveBaseURL(baseURL: string): string {
    if (/^https?:\/\//i.test(baseURL)) {
      return baseURL;
    }
    if (typeof window !== 'undefined' && window.location?.origin) {
      return new URL(baseURL, window.location.origin).toString();
    }
    return baseURL;
  }

  /**
   * Build full URL
   */
  private buildUrl(endpoint: string, queryParams?: Record<string, any>): string {
    const url = new URL(endpoint, this.resolveBaseURL(this.baseURL));

    if (queryParams) {
      Object.entries(queryParams).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value));
        }
      });
    }

    return url.toString();
  }

  /**
   * GET request
   */
  async get<T>(endpoint: string, queryParams?: Record<string, any>, options?: ApiErrorHandlingOptions): Promise<T> {
    const context = await createEnhancedErrorContext({
      component: 'ApiClient',
      function: 'get',
      operation: `GET ${endpoint}`,
      additionalData: { endpoint, queryParams, method: 'GET' }
    });

    const result = await withApiErrorHandling<T>(async () => {
      const url = this.buildUrl(endpoint, queryParams);
      const response = await this.requestWithTimeout(url, {
        method: 'GET',
        headers: {
          ...this.defaultHeaders,
          ...this.getAuthHeaders(),
        },
      });
      return this.handleResponse<T>(response);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      collectContext: false, // We're already creating enhanced context
      context: {
        component: 'ApiClient',
        function: 'get',
        operation: `GET ${endpoint}`,
        additionalData: { endpoint, queryParams, method: 'GET' }
      },
      ...options
    });

    if (!result.success) {
      throw result.error || new Error('API request failed');
    }

    return result.data!;
  }

  /**
   * POST request
   */
  async post<T>(endpoint: string, data?: any, options?: ApiErrorHandlingOptions): Promise<T> {
    const context = await createEnhancedErrorContext({
      component: 'ApiClient',
      function: 'post',
      operation: `POST ${endpoint}`,
      additionalData: { endpoint, data, method: 'POST' }
    });

    const result = await withApiErrorHandling<T>(async () => {
      const url = this.buildUrl(endpoint);
      const response = await this.requestWithTimeout(url, {
        method: 'POST',
        headers: {
          ...this.defaultHeaders,
          ...this.getAuthHeaders(),
        },
        body: JSON.stringify(data),
      });
      return this.handleResponse<T>(response);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      collectContext: false, // We're already creating enhanced context
      context: {
        component: 'ApiClient',
        function: 'post',
        operation: `POST ${endpoint}`,
        additionalData: { endpoint, data, method: 'POST' }
      },
      ...options
    });

    if (!result.success) {
      throw result.error || new Error('API request failed');
    }

    return result.data!;
  }

  /**
   * PUT request
   */
  async put<T>(endpoint: string, data?: any, options?: ApiErrorHandlingOptions): Promise<T> {
    const context = await createEnhancedErrorContext({
      component: 'ApiClient',
      function: 'put',
      operation: `PUT ${endpoint}`,
      additionalData: { endpoint, data, method: 'PUT' }
    });

    const result = await withApiErrorHandling<T>(async () => {
      const url = this.buildUrl(endpoint);
      const response = await this.requestWithTimeout(url, {
        method: 'PUT',
        headers: {
          ...this.defaultHeaders,
          ...this.getAuthHeaders(),
        },
        body: JSON.stringify(data),
      });
      return this.handleResponse<T>(response);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      collectContext: false, // We're already creating enhanced context
      context: {
        component: 'ApiClient',
        function: 'put',
        operation: `PUT ${endpoint}`,
        additionalData: { endpoint, data, method: 'PUT' }
      },
      ...options
    });

    if (!result.success) {
      throw result.error || new Error('API request failed');
    }

    return result.data!;
  }

  /**
   * PATCH request
   */
  async patch<T>(endpoint: string, data?: any, options?: ApiErrorHandlingOptions): Promise<T> {
    const context = await createEnhancedErrorContext({
      component: 'ApiClient',
      function: 'patch',
      operation: `PATCH ${endpoint}`,
      additionalData: { endpoint, data, method: 'PATCH' }
    });

    const result = await withApiErrorHandling<T>(async () => {
      const url = this.buildUrl(endpoint);
      const response = await this.requestWithTimeout(url, {
        method: 'PATCH',
        headers: {
          ...this.defaultHeaders,
          ...this.getAuthHeaders(),
        },
        body: JSON.stringify(data),
      });
      return this.handleResponse<T>(response);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      collectContext: false, // We're already creating enhanced context
      context: {
        component: 'ApiClient',
        function: 'patch',
        operation: `PATCH ${endpoint}`,
        additionalData: { endpoint, data, method: 'PATCH' }
      },
      ...options
    });

    if (!result.success) {
      throw result.error || new Error('API request failed');
    }

    return result.data!;
  }

  /**
   * DELETE request
   */
  async delete<T>(endpoint: string, options?: ApiErrorHandlingOptions): Promise<T> {
    const context = await createEnhancedErrorContext({
      component: 'ApiClient',
      function: 'delete',
      operation: `DELETE ${endpoint}`,
      additionalData: { endpoint, method: 'DELETE' }
    });

    const result = await withApiErrorHandling<T>(async () => {
      const url = this.buildUrl(endpoint);
      const response = await this.requestWithTimeout(url, {
        method: 'DELETE',
        headers: {
          ...this.defaultHeaders,
          ...this.getAuthHeaders(),
        },
      });
      return this.handleResponse<T>(response);
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      collectContext: false, // We're already creating enhanced context
      context: {
        component: 'ApiClient',
        function: 'delete',
        operation: `DELETE ${endpoint}`,
        additionalData: { endpoint, method: 'DELETE' }
      },
      ...options
    });

    if (!result.success) {
      throw result.error || new Error('API request failed');
    }

    return result.data!;
  }

  /**
   * File upload
   */
  async uploadFile<T>(
    endpoint: string,
    file: File,
    onProgress?: (progress: number) => void,
    options?: ApiErrorHandlingOptions
  ): Promise<T> {
    const context = await createEnhancedErrorContext({
      component: 'ApiClient',
      function: 'uploadFile',
      operation: `UPLOAD ${endpoint}`,
      additionalData: { endpoint, fileName: file.name, fileSize: file.size, method: 'UPLOAD' }
    });

    const result = await withApiErrorHandling<T>(async () => {
      try {
        const url = this.buildUrl(endpoint);
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();

        return new Promise<T>((resolve, reject) => {
          xhr.upload.addEventListener('progress', (event) => {
            try {
              if (event.lengthComputable && onProgress) {
                const progress = (event.loaded / event.total) * 100;
                onProgress(progress);
              }
            } catch (progressError) {
              console.error('Error in upload progress callback:', progressError);
            }
          });

          xhr.addEventListener('load', () => {
            try {
              if (xhr.status >= 200 && xhr.status < 300) {
                try {
                  const response = JSON.parse(xhr.responseText);
                  resolve(response);
                } catch (parseError) {
                  // If JSON parsing fails, resolve with the raw text
                  console.warn('Failed to parse upload response as JSON', parseError);
                  resolve(xhr.responseText as unknown as T);
                }
              } else {
                try {
                  const error = JSON.parse(xhr.responseText);
                  reject(error);
                } catch (parseError) {
                  // If JSON parsing fails, reject with a generic error
                  reject({
                    code: `HTTP_${xhr.status}`,
                    message: `Upload failed: ${xhr.statusText}`,
                    status: xhr.status,
                    url: url
                  });
                }
              }
            } catch (loadError) {
              console.error('Error in upload load handler:', loadError);
              reject(loadError);
            }
          });

          xhr.addEventListener('error', () => {
            try {
              reject(new Error('Upload failed due to network error'));
            } catch (error) {
              console.error('Error in upload error handler:', error);
            }
          });

          xhr.addEventListener('timeout', () => {
            try {
              reject(new Error('Upload timed out'));
            } catch (error) {
              console.error('Error in upload timeout handler:', error);
            }
          });

          xhr.open('POST', url);
          xhr.timeout = this.timeout; // Set timeout for upload
          xhr.setRequestHeader('Authorization', `Bearer ${useAuthStore.getState().token}`);
          xhr.send(formData);
        });
      } catch (error) {
        console.error('Error initiating file upload:', error);
        throw error instanceof Error ? error : new Error(String(error));
      }
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 1500,
      showUserNotification: true,
      logError: true,
      collectContext: false, // We're already creating enhanced context
      context: {
        component: 'ApiClient',
        function: 'uploadFile',
        operation: `UPLOAD ${endpoint}`,
        additionalData: { endpoint, fileName: file.name, fileSize: file.size, method: 'UPLOAD' }
      },
      ...options
    });

    if (!result.success) {
      throw result.error || new Error('File upload failed');
    }

    return result.data!;
  }

  /**
   * File download
   */
  async downloadFile(endpoint: string, filename?: string, options?: ApiErrorHandlingOptions): Promise<void> {
    const context = await createEnhancedErrorContext({
      component: 'ApiClient',
      function: 'downloadFile',
      operation: `DOWNLOAD ${endpoint}`,
      additionalData: { endpoint, filename, method: 'DOWNLOAD' }
    });

    const result = await withApiErrorHandling<void>(async () => {
      try {
        const url = this.buildUrl(endpoint);
        let response: Response;

        try {
          response = await this.requestWithTimeout(url, {
            method: 'GET',
            headers: {
              ...this.getAuthHeaders(),
            },
          });
        } catch (error) {
          throw new Error(`Download request failed: ${error instanceof Error ? error.message : String(error)}`);
        }

        if (!response.ok) {
          throw new Error(`Download failed with status ${response.status}: ${response.statusText}`);
        }

        try {
          const blob = await response.blob();
          const downloadUrl = window.URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = downloadUrl;
          link.download = filename || 'download';

          // Add to document, click, and clean up
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(downloadUrl);
        } catch (error) {
          throw new Error(`Download processing failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      } catch (error) {
        console.error('File download error:', error);
        throw error;
      }
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 1500,
      showUserNotification: true,
      logError: true,
      collectContext: false, // We're already creating enhanced context
      context: {
        component: 'ApiClient',
        function: 'downloadFile',
        operation: `DOWNLOAD ${endpoint}`,
        additionalData: { endpoint, filename, method: 'DOWNLOAD' }
      },
      ...options
    });

    if (!result.success) {
      throw result.error || new Error('File download failed');
    }
  }
}

/**
 * Export singleton instance
 */
export const apiClient = new ApiClient();

/**
 * Export for testing
 */
export { ApiClient };
