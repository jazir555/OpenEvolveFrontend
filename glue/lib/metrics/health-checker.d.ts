export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';
export interface HealthCheckResult {
    name: string;
    status: HealthStatus;
    message?: string;
    timestamp: string;
    response_time_ms?: number;
    dependencies?: HealthCheckResult[];
    metadata?: Record<string, any>;
}
export interface HealthCheckOptions {
    timeout?: number;
    critical?: boolean;
}
export type HealthCheckFunction = () => Promise<HealthCheckResult> | HealthCheckResult;
export declare class HealthChecker {
    private checks;
    private serviceName;
    constructor(serviceName: string);
    register(name: string, checkFn: HealthCheckFunction): void;
    unregister(name: string): void;
    private executeCheck;
    checkHealth(): Promise<HealthCheckResult>;
    checkSpecific(checkName: string): Promise<HealthCheckResult>;
    private calculateOverallStatus;
    getLiveness(): Promise<HealthCheckResult>;
    getReadiness(): Promise<HealthCheckResult>;
}
export interface HealthEndpointResponse {
    status: number;
    body: HealthCheckResult;
    headers: Record<string, string>;
}
export declare class HealthEndpointHandler {
    private healthChecker;
    constructor(healthChecker: HealthChecker);
    handleHealth(): Promise<HealthEndpointResponse>;
    handleLiveness(): Promise<HealthEndpointResponse>;
    handleReadiness(): Promise<HealthEndpointResponse>;
    handleSpecificCheck(checkName: string): Promise<HealthEndpointResponse>;
    private statusToStatusCode;
}
export declare function createHttpHealthCheck(url: string, options?: {
    timeout?: number;
    expectedStatus?: number;
}): HealthCheckFunction;
export declare function createTcpHealthCheck(host: string, port: number, options?: {
    timeout?: number;
}): HealthCheckFunction;
export declare function createDatabaseHealthCheck(checkFn: () => Promise<void>, options?: {
    timeout?: number;
}): HealthCheckFunction;
//# sourceMappingURL=health-checker.d.ts.map