/**
 * Ambient module declarations for dependencies that are not installed in the
 * local node_modules during type-checking. These are type-level only and do not
 * affect runtime resolution (the packages resolve normally at runtime via the
 * package manager / workspace links).
 */

declare module '@openevolve/glue-lib' {
  export class Logger {
    constructor(name: string);
    info(message: any, context?: any): void;
    warn(message: any, context?: any): void;
    error(message: any, error?: any, context?: any): void;
    debug(message: any, context?: any): void;
  }

  export class CircuitBreaker {
    constructor(options?: any);
    execute<T>(fn: () => Promise<T>): Promise<T>;
  }

  export type LogLevel = any;
  export type CircuitState = any;
  export type CircuitBreakerOptions = any;
  export type CircuitBreakerStats = any;
}

declare module 'axios' {
  export type AxiosInstance = any;
  export type AxiosError = any;
  export const axios: any;
  export default axios;
}
