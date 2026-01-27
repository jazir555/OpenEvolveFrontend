import { z } from 'zod';
import { CredentialType } from './types';
// Bubble parameter type enum
export var BubbleParameterType;
(function (BubbleParameterType) {
    BubbleParameterType["STRING"] = "string";
    BubbleParameterType["NUMBER"] = "number";
    BubbleParameterType["BOOLEAN"] = "boolean";
    BubbleParameterType["OBJECT"] = "object";
    BubbleParameterType["ARRAY"] = "array";
    BubbleParameterType["ENV"] = "env";
    BubbleParameterType["VARIABLE"] = "variable";
    BubbleParameterType["EXPRESSION"] = "expression";
    BubbleParameterType["UNKNOWN"] = "unknown";
})(BubbleParameterType || (BubbleParameterType = {}));
// Credential configuration mappings - defines what configurations are available for each credential type
export const CREDENTIAL_CONFIGURATION_MAP = {
    [CredentialType.DATABASE_CRED]: {
        ignoreSSL: BubbleParameterType.BOOLEAN,
    },
    [CredentialType.TELEGRAM_BOT_TOKEN]: {},
    [CredentialType.AGI_API_KEY]: {},
    [CredentialType.FUB_CRED]: {},
    [CredentialType.OPENAI_CRED]: {},
    [CredentialType.GOOGLE_GEMINI_CRED]: {},
    [CredentialType.ANTHROPIC_CRED]: {},
    [CredentialType.FIRECRAWL_API_KEY]: {},
    [CredentialType.SLACK_CRED]: {},
    [CredentialType.RESEND_CRED]: {},
    [CredentialType.OPENROUTER_CRED]: {},
    [CredentialType.DEEPSEEK_CRED]: {},
    [CredentialType.CLOUDFLARE_R2_ACCESS_KEY]: {},
    [CredentialType.CLOUDFLARE_R2_SECRET_KEY]: {},
    [CredentialType.CLOUDFLARE_R2_ACCOUNT_ID]: {},
    [CredentialType.APIFY_CRED]: {},
    [CredentialType.ELEVENLABS_API_KEY]: {},
    [CredentialType.GOOGLE_DRIVE_CRED]: {},
    [CredentialType.GMAIL_CRED]: {},
    [CredentialType.GOOGLE_SHEETS_CRED]: {},
    [CredentialType.GOOGLE_CALENDAR_CRED]: {},
    [CredentialType.GITHUB_TOKEN]: {},
    [CredentialType.GITHUB_CRED]: {},
    [CredentialType.AIRTABLE_CRED]: {},
    [CredentialType.ELASTICSEARCH_CRED]: {},
    [CredentialType.NOTION_OAUTH_TOKEN]: {},
    [CredentialType.INSFORGE_BASE_URL]: {},
    [CredentialType.INSFORGE_API_KEY]: {},
    [CredentialType.CUSTOM_AUTH_KEY]: {},
    [CredentialType.STRIPE_CRED]: {},
    [CredentialType.SENDGRID_CRED]: {},
    [CredentialType.TWILIO_CRED]: {},
    [CredentialType.POSTGRESQL_CRED]: {},
    [CredentialType.QDRANT_CRED]: {},
    [CredentialType.REDIS_CRED]: {},
    [CredentialType.OAUTH_TOKEN]: {},
};
// Fixed list of bubble names that need context injection
export const BUBBLE_NAMES_WITH_CONTEXT_INJECTION = [
    'database-analyzer',
    'slack-data-assistant',
];
// Zod schemas for validation and type inference
export const BubbleParameterTypeSchema = z.nativeEnum(BubbleParameterType);
export const BubbleParameterSchema = z.object({
    location: z.optional(z.object({
        startLine: z.number(),
        startCol: z.number(),
        endLine: z.number(),
        endCol: z.number(),
    })),
    variableId: z
        .number()
        .optional()
        .describe('The variable id of the parameter'),
    name: z.string().describe('The name of the parameter'),
    value: z
        .union([
        z.string(),
        z.number(),
        z.boolean(),
        z.record(z.unknown()),
        z.array(z.unknown()),
    ])
        .describe('The value of the parameter'),
    type: BubbleParameterTypeSchema,
    /**
     * Source of the parameter - indicates whether it came from an object literal property
     * or represents the entire first argument. Used to determine if spread pattern should be applied.
     * Ex.
     * const abc = '1234567890';
     * new GoogleDriveBubble({
     *   fileId: abc,
     * })
     * source: 'object-property',
     *
     * new GoogleDriveBubble({
     *   url: 'https://www.google.com',
     *   ...args,
     * })
     * source: 'spread',
     *
     * source = 'first-arg'
     * new GoogleDriveBubble(args)
     */
    source: z
        .enum(['object-property', 'first-arg', 'spread'])
        .optional()
        .describe('Source of the parameter - indicates if it came from an object literal property, represents the entire first argument, or came from a spread operator'),
});
export const BubbleNodeTypeSchema = z.enum([
    'service',
    'tool',
    'workflow',
    'unknown',
]);
export const DependencyGraphNodeSchema = z.lazy(() => z.object({
    name: z.string(),
    variableName: z.string().optional(),
    nodeType: BubbleNodeTypeSchema,
    uniqueId: z.string().optional(),
    variableId: z.number().optional(),
    dependencies: z.array(DependencyGraphNodeSchema),
    // Use lazy reference since FunctionCallWorkflowNodeSchema is defined later
    functionCallChildren: z
        .lazy(() => z.array(FunctionCallWorkflowNodeSchema))
        .optional(),
}));
export const ParsedBubbleSchema = z.object({
    variableName: z.string(),
    bubbleName: z.string(),
    className: z.string(),
    parameters: z.array(BubbleParameterSchema),
    hasAwait: z.boolean(),
    hasActionCall: z.boolean(),
    dependencies: z.array(z.string()).optional(),
    dependencyGraph: DependencyGraphNodeSchema.optional(),
});
export const BubbleDependencySpecSchema = z.object({
    name: z.string(),
    tools: z.array(z.string()).optional(),
});
export const ParsedBubbleWithInfoSchema = z.object({
    variableName: z.string(),
    bubbleName: z.string(),
    className: z.string(),
    parameters: z.array(BubbleParameterSchema),
    hasAwait: z.boolean(),
    hasActionCall: z.boolean(),
    dependencies: z.array(z.string()).optional(),
    dependencyGraph: DependencyGraphNodeSchema.optional(),
    variableId: z.number(),
    nodeType: BubbleNodeTypeSchema,
    location: z.object({
        startLine: z.number(),
        startCol: z.number(),
        endLine: z.number(),
        endCol: z.number(),
    }),
    description: z.string().optional(),
    invocationCallSiteKey: z.string().optional(),
    clonedFromVariableId: z.number().optional(),
    isInsideCustomTool: z.boolean().optional(),
    containingCustomToolId: z.string().optional(),
});
// Zod schemas for workflow nodes
export const WorkflowNodeTypeSchema = z.enum([
    'bubble',
    'if',
    'for',
    'while',
    'try_catch',
    'variable_declaration',
    'return',
    'function_call',
    'code_block',
    'parallel_execution',
    'transformation_function',
]);
export const LocationSchema = z.object({
    startLine: z.number(),
    startCol: z.number(),
    endLine: z.number(),
    endCol: z.number(),
});
export const BubbleWorkflowNodeSchema = z.object({
    type: z.literal('bubble'),
    variableId: z.number(),
});
export const ControlFlowWorkflowNodeSchema = z.lazy(() => z.object({
    type: z.enum(['if', 'for', 'while']),
    location: LocationSchema,
    condition: z.string().optional(),
    children: z.array(WorkflowNodeSchema),
    elseBranch: z.array(WorkflowNodeSchema).optional(),
    thenTerminates: z.boolean().optional(),
    elseTerminates: z.boolean().optional(),
}));
export const TryCatchWorkflowNodeSchema = z.lazy(() => z.object({
    type: z.literal('try_catch'),
    location: LocationSchema,
    children: z.array(WorkflowNodeSchema),
    catchBlock: z.array(WorkflowNodeSchema).optional(),
}));
export const CodeBlockWorkflowNodeSchema = z.lazy(() => z.object({
    type: z.literal('code_block'),
    location: LocationSchema,
    code: z.string(),
    children: z.array(WorkflowNodeSchema),
}));
export const VariableDeclarationBlockNodeSchema = z.lazy(() => z.object({
    type: z.literal('variable_declaration'),
    location: LocationSchema,
    code: z.string(),
    variables: z.array(z.object({
        name: z.string(),
        type: z.enum(['const', 'let', 'var']),
        hasInitializer: z.boolean(),
    })),
    children: z.array(WorkflowNodeSchema),
}));
export const ReturnWorkflowNodeSchema = z.lazy(() => z.object({
    type: z.literal('return'),
    location: LocationSchema,
    code: z.string(),
    value: z.string().optional(),
    children: z.array(WorkflowNodeSchema),
}));
export const FunctionCallWorkflowNodeSchema = z.lazy(() => z.object({
    type: z.literal('function_call'),
    location: LocationSchema,
    functionName: z.string(),
    isMethodCall: z.boolean(),
    description: z.string().optional(),
    arguments: z.string().optional(),
    code: z.string(),
    variableId: z.number(),
    variableDeclaration: z
        .object({
        variableName: z.string(),
        variableType: z.enum(['const', 'let', 'var']),
    })
        .optional(),
    methodDefinition: z
        .object({
        location: z.object({
            startLine: z.number(),
            endLine: z.number(),
        }),
        isAsync: z.boolean(),
        parameters: z.array(z.string()),
    })
        .optional(),
    children: z.array(WorkflowNodeSchema),
}));
export const ParallelExecutionWorkflowNodeSchema = z.lazy(() => z.object({
    type: z.literal('parallel_execution'),
    location: LocationSchema,
    code: z.string(),
    variableDeclaration: z
        .object({
        variableNames: z.array(z.string()),
        variableType: z.enum(['const', 'let', 'var']),
    })
        .optional(),
    children: z.array(WorkflowNodeSchema),
}));
export const TransformationFunctionWorkflowNodeSchema = z.object({
    type: z.literal('transformation_function'),
    location: LocationSchema,
    code: z.string(),
    functionName: z.string(),
    isMethodCall: z.boolean(),
    description: z.string().optional(),
    arguments: z.string().optional(),
    variableId: z.number(),
    variableDeclaration: z
        .object({
        variableName: z.string(),
        variableType: z.enum(['const', 'let', 'var']),
    })
        .optional(),
    methodDefinition: z
        .object({
        location: z.object({
            startLine: z.number(),
            endLine: z.number(),
        }),
        isAsync: z.boolean(),
        parameters: z.array(z.string()),
    })
        .optional(),
});
export const WorkflowNodeSchema = z.lazy(() => z.union([
    BubbleWorkflowNodeSchema,
    ControlFlowWorkflowNodeSchema,
    TryCatchWorkflowNodeSchema,
    CodeBlockWorkflowNodeSchema,
    VariableDeclarationBlockNodeSchema,
    ReturnWorkflowNodeSchema,
    FunctionCallWorkflowNodeSchema,
    ParallelExecutionWorkflowNodeSchema,
    TransformationFunctionWorkflowNodeSchema,
]));
export const ParsedWorkflowSchema = z.object({
    root: z.array(WorkflowNodeSchema),
    bubbles: z.record(z.number(), ParsedBubbleWithInfoSchema),
});
//# sourceMappingURL=bubble-definition-schema.js.map