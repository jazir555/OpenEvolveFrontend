import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const AceToolsBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"executeCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    inputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    code: string;
    operation: "executeCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputs?: Record<string, unknown> | undefined;
}, {
    code: string;
    operation: "executeCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputs?: Record<string, unknown> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validateCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    rules: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "validateCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    rules?: string[] | undefined;
}, {
    code: string;
    operation: "validateCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    rules?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"formatCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    style: z.ZodDefault<z.ZodOptional<z.ZodEnum<["prettier", "eslint", "black", "gofmt", "standard"]>>>;
    options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    style: "prettier" | "eslint" | "black" | "gofmt" | "standard";
    operation: "formatCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    options?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    code: string;
    operation: "formatCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    options?: Record<string, unknown> | undefined;
    style?: "prettier" | "eslint" | "black" | "gofmt" | "standard" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"analyzeCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    metrics: z.ZodOptional<z.ZodArray<z.ZodEnum<["complexity", "maintainability", "security", "performance", "duplication"]>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "analyzeCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metrics?: ("complexity" | "maintainability" | "security" | "performance" | "duplication")[] | undefined;
}, {
    code: string;
    operation: "analyzeCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metrics?: ("complexity" | "maintainability" | "security" | "performance" | "duplication")[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"generateTests">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    testFramework: z.ZodOptional<z.ZodEnum<["jest", "mocha", "pytest", "junit", "testing"]>>;
    coverage: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "generateTests";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    testFramework?: "testing" | "jest" | "mocha" | "pytest" | "junit" | undefined;
    coverage?: number | undefined;
}, {
    code: string;
    operation: "generateTests";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    testFramework?: "testing" | "jest" | "mocha" | "pytest" | "junit" | undefined;
    coverage?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"refactorCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    target: z.ZodDefault<z.ZodOptional<z.ZodEnum<["readability", "performance", "maintainability", "security"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "refactorCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    target: "maintainability" | "security" | "performance" | "readability";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    code: string;
    operation: "refactorCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    target?: "maintainability" | "security" | "performance" | "readability" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"documentCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    format: z.ZodOptional<z.ZodEnum<["javadoc", "jsdoc", "pydoc", "godoc"]>>;
    includeTypes: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "documentCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    includeTypes: boolean;
    format?: "javadoc" | "jsdoc" | "pydoc" | "godoc" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    code: string;
    operation: "documentCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    format?: "javadoc" | "jsdoc" | "pydoc" | "godoc" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    includeTypes?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"transformCode">;
    code: z.ZodString;
    sourceLanguage: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    targetLanguage: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    preserveComments: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "transformCode";
    sourceLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    targetLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    preserveComments: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    code: string;
    operation: "transformCode";
    sourceLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    targetLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    preserveComments?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"optimizeCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    focus: z.ZodDefault<z.ZodOptional<z.ZodEnum<["memory", "speed", "both"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "optimizeCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    focus: "memory" | "speed" | "both";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    code: string;
    operation: "optimizeCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    focus?: "memory" | "speed" | "both" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"reviewCode">;
    code: z.ZodString;
    language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
    categories: z.ZodOptional<z.ZodArray<z.ZodEnum<["best-practices", "security", "performance", "maintainability", "readability"]>, "many">>;
    severity: z.ZodOptional<z.ZodEnum<["info", "warning", "error", "critical"]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    code: string;
    operation: "reviewCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    categories?: ("maintainability" | "security" | "performance" | "readability" | "best-practices")[] | undefined;
    severity?: "info" | "error" | "warning" | "critical" | undefined;
}, {
    code: string;
    operation: "reviewCode";
    language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    categories?: ("maintainability" | "security" | "performance" | "readability" | "best-practices")[] | undefined;
    severity?: "info" | "error" | "warning" | "critical" | undefined;
}>]>;
type AceToolsBubbleParams = z.input<typeof AceToolsBubbleParamsSchema>;
declare const AceToolsBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        language: z.ZodOptional<z.ZodString>;
        executionTime: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        executionTime?: number | undefined;
        language?: string | undefined;
    }, {
        operation: string;
        executionTime?: number | undefined;
        language?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        executionTime?: number | undefined;
        language?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        executionTime?: number | undefined;
        language?: string | undefined;
    };
    data?: unknown;
}>;
type AceToolsBubbleResult = z.output<typeof AceToolsBubbleResultSchema>;
export declare class AceToolsBubble extends ServiceBubble<AceToolsBubbleParams, AceToolsBubbleResult> {
    static readonly service = "ace-tools";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"executeCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        inputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        code: string;
        operation: "executeCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputs?: Record<string, unknown> | undefined;
    }, {
        code: string;
        operation: "executeCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputs?: Record<string, unknown> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validateCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        rules: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "validateCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        rules?: string[] | undefined;
    }, {
        code: string;
        operation: "validateCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        rules?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"formatCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        style: z.ZodDefault<z.ZodOptional<z.ZodEnum<["prettier", "eslint", "black", "gofmt", "standard"]>>>;
        options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        style: "prettier" | "eslint" | "black" | "gofmt" | "standard";
        operation: "formatCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        options?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        code: string;
        operation: "formatCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        options?: Record<string, unknown> | undefined;
        style?: "prettier" | "eslint" | "black" | "gofmt" | "standard" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"analyzeCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        metrics: z.ZodOptional<z.ZodArray<z.ZodEnum<["complexity", "maintainability", "security", "performance", "duplication"]>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "analyzeCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metrics?: ("complexity" | "maintainability" | "security" | "performance" | "duplication")[] | undefined;
    }, {
        code: string;
        operation: "analyzeCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metrics?: ("complexity" | "maintainability" | "security" | "performance" | "duplication")[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"generateTests">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        testFramework: z.ZodOptional<z.ZodEnum<["jest", "mocha", "pytest", "junit", "testing"]>>;
        coverage: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "generateTests";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        testFramework?: "testing" | "jest" | "mocha" | "pytest" | "junit" | undefined;
        coverage?: number | undefined;
    }, {
        code: string;
        operation: "generateTests";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        testFramework?: "testing" | "jest" | "mocha" | "pytest" | "junit" | undefined;
        coverage?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"refactorCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        target: z.ZodDefault<z.ZodOptional<z.ZodEnum<["readability", "performance", "maintainability", "security"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "refactorCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        target: "maintainability" | "security" | "performance" | "readability";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        code: string;
        operation: "refactorCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        target?: "maintainability" | "security" | "performance" | "readability" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"documentCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        format: z.ZodOptional<z.ZodEnum<["javadoc", "jsdoc", "pydoc", "godoc"]>>;
        includeTypes: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "documentCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        includeTypes: boolean;
        format?: "javadoc" | "jsdoc" | "pydoc" | "godoc" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        code: string;
        operation: "documentCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        format?: "javadoc" | "jsdoc" | "pydoc" | "godoc" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        includeTypes?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"transformCode">;
        code: z.ZodString;
        sourceLanguage: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        targetLanguage: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        preserveComments: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "transformCode";
        sourceLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        targetLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        preserveComments: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        code: string;
        operation: "transformCode";
        sourceLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        targetLanguage: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        preserveComments?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"optimizeCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        focus: z.ZodDefault<z.ZodOptional<z.ZodEnum<["memory", "speed", "both"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "optimizeCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        focus: "memory" | "speed" | "both";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        code: string;
        operation: "optimizeCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        focus?: "memory" | "speed" | "both" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"reviewCode">;
        code: z.ZodString;
        language: z.ZodEnum<["javascript", "typescript", "python", "java", "go", "rust", "csharp", "php"]>;
        categories: z.ZodOptional<z.ZodArray<z.ZodEnum<["best-practices", "security", "performance", "maintainability", "readability"]>, "many">>;
        severity: z.ZodOptional<z.ZodEnum<["info", "warning", "error", "critical"]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        code: string;
        operation: "reviewCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        categories?: ("maintainability" | "security" | "performance" | "readability" | "best-practices")[] | undefined;
        severity?: "info" | "error" | "warning" | "critical" | undefined;
    }, {
        code: string;
        operation: "reviewCode";
        language: "javascript" | "typescript" | "python" | "java" | "go" | "rust" | "csharp" | "php";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        categories?: ("maintainability" | "security" | "performance" | "readability" | "best-practices")[] | undefined;
        severity?: "info" | "error" | "warning" | "critical" | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            language: z.ZodOptional<z.ZodString>;
            executionTime: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            executionTime?: number | undefined;
            language?: string | undefined;
        }, {
            operation: string;
            executionTime?: number | undefined;
            language?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            executionTime?: number | undefined;
            language?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            executionTime?: number | undefined;
            language?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Advanced code execution, analysis, and transformation tools";
    static readonly longDescription = "\n    ACE Tools Bubble for comprehensive code operations.\n\n    Features:\n    - Execute code in sandboxed environments\n    - Validate code syntax and structure\n    - Format code according to style guides\n    - Analyze code complexity and metrics\n    - Generate unit tests automatically\n    - Refactor code for best practices\n    - Generate comprehensive documentation\n    - Transform code between languages\n    - Optimize for performance and memory\n    - Perform intelligent code reviews\n\n    Use cases:\n    - Code quality checks in CI/CD\n    - Automated test generation\n    - Code refactoring and modernization\n    - Language migration\n    - Performance optimization\n    - Security audits\n  ";
    static readonly alias = "code";
    constructor(params: AceToolsBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<AceToolsBubbleResult>;
    private executeCode;
    /**
     * Validate code for security issues before execution
     */
    private validateCodeSecurity;
    /**
     * Create an isolated sandbox environment for code execution
     */
    private createSandbox;
    /**
     * Execute code in the isolated sandbox with timeout enforcement
     *
     * SECURITY ARCHITECTURE:
     *
     * Current Implementation: Restricted Function Constructor
     * - Uses Function constructor with frozen sandbox object
     * - Pre-validates code for dangerous patterns
     * - Enforces timeout and resource limits
     * - No access to Node.js APIs (require, process, fs, etc.)
     * - Only safe built-ins exposed (Math, Date, JSON, etc.)
     *
     * Production Recommendations:
     * 1. **isolated-vm** (Recommended for Node.js)
     *    - True V8 isolate context
     *    - Separate memory space
     *    - CPU and memory limits
     *    - Install: npm install isolated-vm
     *
     * 2. **VM2** (Deprecated - DO NOT USE)
     *    - Had critical security vulnerabilities (CVE-2021-23449)
     *    - No longer maintained
     *
     * 3. **Worker Threads** (Alternative)
     *    - Separate OS-level process
     *    - True isolation
     *    - Higher overhead
     *
     * 4. **Docker Containers** (Best for untrusted code)
     *    - Process-level isolation
     *    - Network restrictions
     *    - Resource limits via cgroups
     *
     * Migration Path to isolated-vm:
     * ```typescript
     * import { Isolate, Context } from 'isolated-vm';
     *
     * const isolate = new Isolate({ memoryLimit: 128 });
     * const context = await isolate.createContext();
     *
     * // Inject sandbox variables
     * const jail = context.global;
     * jail.setSync('console', sandbox.console);
     * jail.setSync('Math', sandbox.Math);
     *
     * // Execute code
     * const result = await context.eval(code, { timeout });
     * ```
     */
    private executeInSandbox;
    private validateCode;
    private formatCode;
    private analyzeCode;
    private generateTests;
    private refactorCode;
    private documentCode;
    private transformCode;
    private optimizeCode;
    private reviewCode;
    private extractLanguage;
}
export {};
//# sourceMappingURL=ace-tools-bubble.d.ts.map