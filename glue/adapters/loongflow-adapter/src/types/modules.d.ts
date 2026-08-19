/**
 * Ambient module declarations for third-party packages whose types are not
 * installed in this adapter's resolution scope.
 *
 * These are intentionally minimal shims so the adapter type-checks without
 * modifying project-level dependencies. The runtime packages are provided by
 * the host service that consumes this adapter.
 */

declare module 'axios' {
  export type AxiosInstance = any;
  const axios: {
    create(config?: any): AxiosInstance;
    get(url: string, config?: any): Promise<any>;
    post(url: string, data?: any, config?: any): Promise<any>;
    put(url: string, data?: any, config?: any): Promise<any>;
    delete(url: string, config?: any): Promise<any>;
    patch(url: string, data?: any, config?: any): Promise<any>;
    head(url: string, config?: any): Promise<any>;
  };
  export default axios;
}

declare module 'uuid' {
  export function v4(): string;
  export function v1(): string;
  export function v3(): string;
  export function v5(): string;
}
