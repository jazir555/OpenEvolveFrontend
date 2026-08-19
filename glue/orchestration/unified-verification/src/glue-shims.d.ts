/**
 * Ambient type declaration for axios (not a dependency of this package).
 * Provides compile-time types only.
 */

declare module 'axios' {
  export interface AxiosError extends Error {
    config?: any;
    request?: any;
    response?: {
      status?: number;
      statusText?: string;
      data?: any;
      headers?: any;
    };
  }

  export interface AxiosResponse<T = any> {
    data: T;
    status: number;
    statusText?: string;
    headers?: any;
    config?: any;
  }

  export interface AxiosInstance {
    post(url: string, data?: any, config?: any): Promise<AxiosResponse>;
    get(url: string, config?: any): Promise<AxiosResponse>;
    isAxiosError(error: unknown): error is AxiosError;
  }

  export const axios: AxiosInstance;
  export default axios;
}
