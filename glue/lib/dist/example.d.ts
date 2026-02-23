/**
 * Complete Example: Using All Glue Lib Utilities
 *
 * This example demonstrates how to use all utilities together
 * in a production adapter following the Federation Constitution.
 */
interface ApiRequest {
    data: any;
    correlation_id?: string;
}
declare function callExternalService(req: ApiRequest): Promise<any>;
declare function syncUser(userId: string): Promise<void>;
declare function healthCheck(): Promise<{
    healthy: boolean;
    details: any;
}>;
declare function shutdown(): Promise<void>;
export { callExternalService, syncUser, healthCheck, shutdown };
//# sourceMappingURL=example.d.ts.map