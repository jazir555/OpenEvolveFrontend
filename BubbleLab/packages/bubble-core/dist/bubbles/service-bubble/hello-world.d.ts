import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const HelloWorldParamsSchema: z.ZodObject<{
    name: z.ZodString;
    message: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    message: string;
    name: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    name: string;
    message?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>;
type HelloWorldParamsInput = z.input<typeof HelloWorldParamsSchema>;
type HelloWorldParams = z.output<typeof HelloWorldParamsSchema>;
declare const HelloWorldResultSchema: z.ZodObject<{
    greeting: z.ZodString;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    greeting: string;
    success: boolean;
}, {
    error: string;
    greeting: string;
    success: boolean;
}>;
type HelloWorldResult = z.output<typeof HelloWorldResultSchema>;
export declare class HelloWorldBubble extends ServiceBubble<HelloWorldParams, HelloWorldResult> {
    static readonly service = "nodex-core";
    static readonly authType: "none";
    static readonly bubbleName = "hello-world";
    static readonly type: "service";
    static readonly schema: z.ZodObject<{
        name: z.ZodString;
        message: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        name: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        name: string;
        message?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        greeting: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        greeting: string;
        success: boolean;
    }, {
        error: string;
        greeting: string;
        success: boolean;
    }>;
    static readonly shortDescription = "Simple hello world bubble for testing purposes";
    static readonly longDescription = "\n    A basic hello world bubble that demonstrates the NodeX bubble system.\n    Use cases:\n    - Testing the bubble execution system\n    - Validating NodeX integration\n    - Learning bubble development patterns\n  ";
    static readonly alias = "hello";
    constructor(params?: HelloWorldParamsInput, context?: BubbleContext);
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<HelloWorldResult>;
}
export {};
//# sourceMappingURL=hello-world.d.ts.map