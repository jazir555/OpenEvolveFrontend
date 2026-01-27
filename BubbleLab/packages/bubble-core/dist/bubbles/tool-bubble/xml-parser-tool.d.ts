/**
 * XML PARSER TOOL
 *
 * A tool bubble for parsing, validating, and manipulating XML data.
 *
 * Features:
 * - Parse XML to JSON/object format
 * - Validate XML against XSD schema
 * - Extract specific nodes and attributes
 * - Query XML with XPath-like expressions
 * - Generate XML from objects
 * - Format and pretty-print XML
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * XML parser parameters schema
 */
declare const XMLParserToolParamsSchema: z.ZodObject<{
    xmlData: z.ZodString;
    operation: z.ZodEnum<["parse", "validate", "extract", "query", "generate", "format"]>;
    preserveOrder: z.ZodDefault<z.ZodBoolean>;
    explicitArray: z.ZodDefault<z.ZodBoolean>;
    nodes: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    attributes: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    queryPath: z.ZodOptional<z.ZodString>;
    xsdSchema: z.ZodOptional<z.ZodString>;
    rootElement: z.ZodDefault<z.ZodString>;
    prettyPrint: z.ZodDefault<z.ZodBoolean>;
    indent: z.ZodDefault<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "format" | "query" | "validate" | "extract" | "parse" | "generate";
    prettyPrint: boolean;
    indent: string;
    xmlData: string;
    preserveOrder: boolean;
    explicitArray: boolean;
    rootElement: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    attributes?: string[] | undefined;
    queryPath?: string | undefined;
    nodes?: string[] | undefined;
    xsdSchema?: string | undefined;
}, {
    operation: "format" | "query" | "validate" | "extract" | "parse" | "generate";
    xmlData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    attributes?: string[] | undefined;
    queryPath?: string | undefined;
    prettyPrint?: boolean | undefined;
    indent?: string | undefined;
    preserveOrder?: boolean | undefined;
    explicitArray?: boolean | undefined;
    nodes?: string[] | undefined;
    xsdSchema?: string | undefined;
    rootElement?: string | undefined;
}>;
/**
 * XML parser result schema
 */
declare const XMLParserToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodOptional<z.ZodUnknown>;
    nodes: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
    queryResults: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    xml: z.ZodOptional<z.ZodString>;
    isValid: z.ZodOptional<z.ZodBoolean>;
    validationErrors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    stats: z.ZodObject<{
        nodeCount: z.ZodOptional<z.ZodNumber>;
        attributeCount: z.ZodOptional<z.ZodNumber>;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        nodeCount?: number | undefined;
        attributeCount?: number | undefined;
    }, {
        processingTime: number;
        nodeCount?: number | undefined;
        attributeCount?: number | undefined;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        nodeCount?: number | undefined;
        attributeCount?: number | undefined;
    };
    xml?: string | undefined;
    data?: unknown;
    queryResults?: unknown[] | undefined;
    isValid?: boolean | undefined;
    validationErrors?: string[] | undefined;
    nodes?: Record<string, unknown>[] | undefined;
}, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        nodeCount?: number | undefined;
        attributeCount?: number | undefined;
    };
    xml?: string | undefined;
    data?: unknown;
    queryResults?: unknown[] | undefined;
    isValid?: boolean | undefined;
    validationErrors?: string[] | undefined;
    nodes?: Record<string, unknown>[] | undefined;
}>;
type XMLParserToolParams = z.output<typeof XMLParserToolParamsSchema>;
type XMLParserToolResult = z.output<typeof XMLParserToolResultSchema>;
type XMLParserToolParamsInput = z.input<typeof XMLParserToolParamsSchema>;
/**
 * XML Parser Tool
 * Parse, validate, and manipulate XML data
 */
export declare class XMLParserTool extends ToolBubble<XMLParserToolParams, XMLParserToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        xmlData: z.ZodString;
        operation: z.ZodEnum<["parse", "validate", "extract", "query", "generate", "format"]>;
        preserveOrder: z.ZodDefault<z.ZodBoolean>;
        explicitArray: z.ZodDefault<z.ZodBoolean>;
        nodes: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        attributes: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        queryPath: z.ZodOptional<z.ZodString>;
        xsdSchema: z.ZodOptional<z.ZodString>;
        rootElement: z.ZodDefault<z.ZodString>;
        prettyPrint: z.ZodDefault<z.ZodBoolean>;
        indent: z.ZodDefault<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "format" | "query" | "validate" | "extract" | "parse" | "generate";
        prettyPrint: boolean;
        indent: string;
        xmlData: string;
        preserveOrder: boolean;
        explicitArray: boolean;
        rootElement: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        attributes?: string[] | undefined;
        queryPath?: string | undefined;
        nodes?: string[] | undefined;
        xsdSchema?: string | undefined;
    }, {
        operation: "format" | "query" | "validate" | "extract" | "parse" | "generate";
        xmlData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        attributes?: string[] | undefined;
        queryPath?: string | undefined;
        prettyPrint?: boolean | undefined;
        indent?: string | undefined;
        preserveOrder?: boolean | undefined;
        explicitArray?: boolean | undefined;
        nodes?: string[] | undefined;
        xsdSchema?: string | undefined;
        rootElement?: string | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodOptional<z.ZodUnknown>;
        nodes: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        queryResults: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        xml: z.ZodOptional<z.ZodString>;
        isValid: z.ZodOptional<z.ZodBoolean>;
        validationErrors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        stats: z.ZodObject<{
            nodeCount: z.ZodOptional<z.ZodNumber>;
            attributeCount: z.ZodOptional<z.ZodNumber>;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            nodeCount?: number | undefined;
            attributeCount?: number | undefined;
        }, {
            processingTime: number;
            nodeCount?: number | undefined;
            attributeCount?: number | undefined;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            nodeCount?: number | undefined;
            attributeCount?: number | undefined;
        };
        xml?: string | undefined;
        data?: unknown;
        queryResults?: unknown[] | undefined;
        isValid?: boolean | undefined;
        validationErrors?: string[] | undefined;
        nodes?: Record<string, unknown>[] | undefined;
    }, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            nodeCount?: number | undefined;
            attributeCount?: number | undefined;
        };
        xml?: string | undefined;
        data?: unknown;
        queryResults?: unknown[] | undefined;
        isValid?: boolean | undefined;
        validationErrors?: string[] | undefined;
        nodes?: Record<string, unknown>[] | undefined;
    }>;
    static readonly shortDescription = "Parse, validate, and manipulate XML data";
    static readonly longDescription = "\n    A comprehensive tool for XML processing operations.\n\n    Features:\n    - PARSE: Convert XML to JavaScript objects\n    - VALIDATE: Validate XML against XSD schema\n    - EXTRACT: Extract specific nodes and attributes\n    - QUERY: Query XML with XPath-like expressions\n    - GENERATE: Generate XML from JavaScript objects\n    - FORMAT: Pretty-print and format XML\n\n    Parse Options:\n    - preserveOrder: Maintain original node and attribute order\n    - explicitArray: Always wrap children in arrays\n\n    Extract Options:\n    - Extract specific nodes by name\n    - Extract specific attributes\n    - Returns structured data\n\n    Query Options:\n    - XPath-like query paths\n    - Navigate nested structures\n    - Query multiple nodes\n\n    Generate Options:\n    - Convert JavaScript objects to XML\n    - Specify root element name\n    - Pretty-print formatting\n\n    Use cases:\n    - API XML response parsing\n    - Configuration file processing\n    - Data transformation (XML to JSON)\n    - XML validation and quality checks\n    - Report generation in XML format\n\n    Note: This tool uses a simple parser implementation.\n    For production use with complex XML, consider using\n    libraries like xml2js, fast-xml-parser, or libxmljs.\n  ";
    static readonly alias = "xml";
    constructor(params: XMLParserToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs XML operation
     */
    performAction(context?: BubbleContext): Promise<XMLParserToolResult>;
    /**
     * Parse XML to JavaScript object
     */
    private parseXML;
    /**
     * Validate XML
     */
    private validateXML;
    /**
     * Extract specific nodes
     */
    private extractNodes;
    /**
     * Query XML
     */
    private queryXML;
    /**
     * Generate XML from object
     */
    private generateXML;
    /**
     * Format XML
     */
    private formatXML;
}
export {};
//# sourceMappingURL=xml-parser-tool.d.ts.map