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
declare class ApiClient {
    private baseURL;
    private timeout;
    private defaultHeaders;
    constructor();
    /**
     * Get authentication headers
     */
    private getAuthHeaders;
    /**
     * Handle API errors
     */
    private handleResponse;
    /**
     * Make a request with timeout
     */
    private requestWithTimeout;
    /**
     * Build full URL
     */
    private buildUrl;
    /**
     * GET request
     */
    get<T>(endpoint: string, queryParams?: Record<string, any>): Promise<T>;
    /**
     * POST request
     */
    post<T>(endpoint: string, data?: any): Promise<T>;
    /**
     * PUT request
     */
    put<T>(endpoint: string, data?: any): Promise<T>;
    /**
     * PATCH request
     */
    patch<T>(endpoint: string, data?: any): Promise<T>;
    /**
     * DELETE request
     */
    delete<T>(endpoint: string): Promise<T>;
    /**
     * File upload
     */
    uploadFile<T>(endpoint: string, file: File, onProgress?: (progress: number) => void): Promise<T>;
    /**
     * File download
     */
    downloadFile(endpoint: string, filename?: string): Promise<void>;
}
/**
 * Export singleton instance
 */
export declare const apiClient: ApiClient;
/**
 * Export for testing
 */
export { ApiClient };
