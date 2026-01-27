import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// HelloWorld doesn't need credential imports since it doesn't use any
// Define the parameters schema for the hello world bubble
const HelloWorldParamsSchema = z.object({
    name: z
        .string()
        .min(1, 'Name is required')
        .describe('Name to include in the greeting message'),
    message: z
        .string()
        .optional()
        .default('Hello from NodeX!')
        .describe('Custom greeting message'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Object mapping credential types to values (injected at runtime)'),
});
// Define the result schema for validation
const HelloWorldResultSchema = z.object({
    greeting: z.string().describe('The generated greeting message'),
    success: z.boolean().describe('Whether the operation was successful'),
    error: z.string().describe('Error message if operation failed'),
});
export class HelloWorldBubble extends ServiceBubble {
    static service = 'nodex-core';
    static authType = 'none';
    static bubbleName = 'hello-world';
    static type = 'service';
    static schema = HelloWorldParamsSchema;
    static resultSchema = HelloWorldResultSchema;
    static shortDescription = 'Simple hello world bubble for testing purposes';
    static longDescription = `
    A basic hello world bubble that demonstrates the NodeX bubble system.
    Use cases:
    - Testing the bubble execution system
    - Validating NodeX integration
    - Learning bubble development patterns
  `;
    static alias = 'hello';
    constructor(params = {
        name: 'World',
        message: 'Hello from NodeX!',
    }, context) {
        super(params, context);
    }
    chooseCredential() {
        // HelloWorld bubble doesn't need any credentials
        return undefined;
    }
    async testCredential() {
        // HelloWorld bubble doesn't need any credentials
        return true;
    }
    async performAction(context) {
        // Context is available but not currently used in this implementation
        void context;
        // Simulate some processing, random delay between 200- 700ms
        const delay = Math.floor(Math.random() * 500) + 200;
        await new Promise((resolve) => setTimeout(resolve, delay));
        const greeting = `${this.params.message} ${this.params.name}!`;
        return { greeting, success: true, error: '' };
    }
}
//# sourceMappingURL=hello-world.js.map