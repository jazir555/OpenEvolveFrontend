import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const HttpParamsSchema: z.ZodObject<{
    url: z.ZodString;
    method: z.ZodDefault<z.ZodEnum<["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]>>;
    headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    body: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>>;
    timeout: z.ZodDefault<z.ZodNumber>;
    followRedirects: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    url: string;
    method: "DELETE" | "GET" | "POST" | "PUT" | "PATCH" | "HEAD" | "OPTIONS";
    followRedirects: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    body?: string | Record<string, unknown> | undefined;
}, {
    url: string;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    headers?: Record<string, string> | undefined;
    method?: "DELETE" | "GET" | "POST" | "PUT" | "PATCH" | "HEAD" | "OPTIONS" | undefined;
    body?: string | Record<string, unknown> | undefined;
    followRedirects?: boolean | undefined;
}>;
type HttpParamsInput = z.input<typeof HttpParamsSchema>;
type HttpParams = z.output<typeof HttpParamsSchema>;
declare const HttpResultSchema: z.ZodObject<{
    status: z.ZodNumber;
    statusText: z.ZodString;
    headers: z.ZodRecord<z.ZodString, z.ZodString>;
    body: z.ZodString;
    json: z.ZodOptional<z.ZodUnknown>;
    success: z.ZodBoolean;
    error: z.ZodString;
    responseTime: z.ZodNumber;
    size: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    error: string;
    status: number;
    success: boolean;
    size: number;
    headers: Record<string, string>;
    body: string;
    statusText: string;
    responseTime: number;
    json?: unknown;
}, {
    error: string;
    status: number;
    success: boolean;
    size: number;
    headers: Record<string, string>;
    body: string;
    statusText: string;
    responseTime: number;
    json?: unknown;
}>;
type HttpResult = z.output<typeof HttpResultSchema>;
export declare class HttpBubble extends ServiceBubble<HttpParams, HttpResult> {
    static readonly service = "nodex-core";
    static readonly authType: "none";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodObject<{
        url: z.ZodString;
        method: z.ZodDefault<z.ZodEnum<["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]>>;
        headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        body: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodRecord<z.ZodString, z.ZodUnknown>]>>;
        timeout: z.ZodDefault<z.ZodNumber>;
        followRedirects: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        url: string;
        method: "DELETE" | "GET" | "POST" | "PUT" | "PATCH" | "HEAD" | "OPTIONS";
        followRedirects: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        body?: string | Record<string, unknown> | undefined;
    }, {
        url: string;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        headers?: Record<string, string> | undefined;
        method?: "DELETE" | "GET" | "POST" | "PUT" | "PATCH" | "HEAD" | "OPTIONS" | undefined;
        body?: string | Record<string, unknown> | undefined;
        followRedirects?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        status: z.ZodNumber;
        statusText: z.ZodString;
        headers: z.ZodRecord<z.ZodString, z.ZodString>;
        body: z.ZodString;
        json: z.ZodOptional<z.ZodUnknown>;
        success: z.ZodBoolean;
        error: z.ZodString;
        responseTime: z.ZodNumber;
        size: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: number;
        success: boolean;
        size: number;
        headers: Record<string, string>;
        body: string;
        statusText: string;
        responseTime: number;
        json?: unknown;
    }, {
        error: string;
        status: number;
        success: boolean;
        size: number;
        headers: Record<string, string>;
        body: string;
        statusText: string;
        responseTime: number;
        json?: unknown;
    }>;
    static readonly shortDescription = "Makes HTTP requests to external APIs and services";
    static readonly longDescription = "\n    A basic HTTP client bubble for making requests to external APIs and web services.\n    \n    Features:\n    - Support for all major HTTP methods (GET, POST, PUT, PATCH, DELETE, etc.)\n    - Custom headers and request body support\n    - Configurable timeouts and redirect handling\n    - JSON parsing for API responses\n    - Detailed response metadata (status, headers, timing, size)\n    - Error handling with meaningful messages\n    \n    Use cases:\n    - Calling external REST APIs\n    - Webhook requests\n    - Data fetching from web services\n    - Integration with third-party services\n    - Simple web scraping (for public APIs)\n    - Health checks and monitoring\n  ";
    static readonly alias = "fetch";
    constructor(params?: HttpParamsInput, context?: BubbleContext);
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<HttpResult>;
}
export {};
//# sourceMappingURL=http.d.ts.map