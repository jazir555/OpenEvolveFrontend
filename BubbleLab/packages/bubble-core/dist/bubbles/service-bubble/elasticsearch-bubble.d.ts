import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const ElasticsearchBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createIndex">;
    indexName: z.ZodString;
    mappings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    settings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createIndex";
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mappings?: Record<string, unknown> | undefined;
    settings?: Record<string, unknown> | undefined;
}, {
    operation: "createIndex";
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mappings?: Record<string, unknown> | undefined;
    settings?: Record<string, unknown> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"indexDocument">;
    indexName: z.ZodString;
    documentId: z.ZodOptional<z.ZodString>;
    document: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "indexDocument";
    document: Record<string, unknown>;
    indexName: string;
    refresh: "true" | "false" | "wait_for";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    documentId?: string | undefined;
}, {
    operation: "indexDocument";
    document: Record<string, unknown>;
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    documentId?: string | undefined;
    refresh?: "true" | "false" | "wait_for" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"bulkIndex">;
    indexName: z.ZodString;
    documents: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
    refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "bulkIndex";
    documents: Record<string, unknown>[];
    indexName: string;
    refresh: "true" | "false" | "wait_for";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "bulkIndex";
    documents: Record<string, unknown>[];
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    refresh?: "true" | "false" | "wait_for" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"search">;
    indexName: z.ZodString;
    query: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    from: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    size: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    sort: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    source: z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>;
    aggs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: Record<string, unknown>;
    operation: "search";
    size: number;
    from: number;
    indexName: string;
    sort?: unknown[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    source?: boolean | string[] | undefined;
    aggs?: Record<string, unknown> | undefined;
}, {
    query: Record<string, unknown>;
    operation: "search";
    indexName: string;
    sort?: unknown[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    size?: number | undefined;
    from?: number | undefined;
    source?: boolean | string[] | undefined;
    aggs?: Record<string, unknown> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDocument">;
    indexName: z.ZodString;
    documentId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getDocument";
    indexName: string;
    documentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getDocument";
    indexName: string;
    documentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateDocument">;
    indexName: z.ZodString;
    documentId: z.ZodString;
    doc: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateDocument";
    indexName: string;
    documentId: string;
    refresh: "true" | "false" | "wait_for";
    doc: Record<string, unknown>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "updateDocument";
    indexName: string;
    documentId: string;
    doc: Record<string, unknown>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    refresh?: "true" | "false" | "wait_for" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteDocument">;
    indexName: z.ZodString;
    documentId: z.ZodString;
    refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteDocument";
    indexName: string;
    documentId: string;
    refresh: "true" | "false" | "wait_for";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteDocument";
    indexName: string;
    documentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    refresh?: "true" | "false" | "wait_for" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteIndex">;
    indexName: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteIndex";
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteIndex";
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"indexExists">;
    indexName: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "indexExists";
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "indexExists";
    indexName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"aggregate">;
    indexName: z.ZodString;
    aggs: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    query: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    size: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "aggregate";
    size: number;
    indexName: string;
    aggs: Record<string, unknown>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: Record<string, unknown> | undefined;
}, {
    operation: "aggregate";
    indexName: string;
    aggs: Record<string, unknown>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: Record<string, unknown> | undefined;
    size?: number | undefined;
}>]>;
type ElasticsearchBubbleParams = z.input<typeof ElasticsearchBubbleParamsSchema>;
declare const ElasticsearchBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        indexName: z.ZodOptional<z.ZodString>;
        took: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        indexName?: string | undefined;
        took?: number | undefined;
    }, {
        operation: string;
        indexName?: string | undefined;
        took?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        indexName?: string | undefined;
        took?: number | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        indexName?: string | undefined;
        took?: number | undefined;
    };
    data?: unknown;
}>;
type ElasticsearchBubbleResult = z.output<typeof ElasticsearchBubbleResultSchema>;
export declare class ElasticsearchBubble extends ServiceBubble<ElasticsearchBubbleParams, ElasticsearchBubbleResult> {
    static readonly service = "elasticsearch";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createIndex">;
        indexName: z.ZodString;
        mappings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        settings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createIndex";
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mappings?: Record<string, unknown> | undefined;
        settings?: Record<string, unknown> | undefined;
    }, {
        operation: "createIndex";
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mappings?: Record<string, unknown> | undefined;
        settings?: Record<string, unknown> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"indexDocument">;
        indexName: z.ZodString;
        documentId: z.ZodOptional<z.ZodString>;
        document: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "indexDocument";
        document: Record<string, unknown>;
        indexName: string;
        refresh: "true" | "false" | "wait_for";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        documentId?: string | undefined;
    }, {
        operation: "indexDocument";
        document: Record<string, unknown>;
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        documentId?: string | undefined;
        refresh?: "true" | "false" | "wait_for" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"bulkIndex">;
        indexName: z.ZodString;
        documents: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
        refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "bulkIndex";
        documents: Record<string, unknown>[];
        indexName: string;
        refresh: "true" | "false" | "wait_for";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "bulkIndex";
        documents: Record<string, unknown>[];
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        refresh?: "true" | "false" | "wait_for" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"search">;
        indexName: z.ZodString;
        query: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        from: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        size: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        sort: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        source: z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>;
        aggs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: Record<string, unknown>;
        operation: "search";
        size: number;
        from: number;
        indexName: string;
        sort?: unknown[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        source?: boolean | string[] | undefined;
        aggs?: Record<string, unknown> | undefined;
    }, {
        query: Record<string, unknown>;
        operation: "search";
        indexName: string;
        sort?: unknown[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        size?: number | undefined;
        from?: number | undefined;
        source?: boolean | string[] | undefined;
        aggs?: Record<string, unknown> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDocument">;
        indexName: z.ZodString;
        documentId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getDocument";
        indexName: string;
        documentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getDocument";
        indexName: string;
        documentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateDocument">;
        indexName: z.ZodString;
        documentId: z.ZodString;
        doc: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateDocument";
        indexName: string;
        documentId: string;
        refresh: "true" | "false" | "wait_for";
        doc: Record<string, unknown>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "updateDocument";
        indexName: string;
        documentId: string;
        doc: Record<string, unknown>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        refresh?: "true" | "false" | "wait_for" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteDocument">;
        indexName: z.ZodString;
        documentId: z.ZodString;
        refresh: z.ZodDefault<z.ZodOptional<z.ZodEnum<["true", "false", "wait_for"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteDocument";
        indexName: string;
        documentId: string;
        refresh: "true" | "false" | "wait_for";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteDocument";
        indexName: string;
        documentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        refresh?: "true" | "false" | "wait_for" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteIndex">;
        indexName: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteIndex";
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteIndex";
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"indexExists">;
        indexName: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "indexExists";
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "indexExists";
        indexName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"aggregate">;
        indexName: z.ZodString;
        aggs: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        query: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        size: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "aggregate";
        size: number;
        indexName: string;
        aggs: Record<string, unknown>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: Record<string, unknown> | undefined;
    }, {
        operation: "aggregate";
        indexName: string;
        aggs: Record<string, unknown>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: Record<string, unknown> | undefined;
        size?: number | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            indexName: z.ZodOptional<z.ZodString>;
            took: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            indexName?: string | undefined;
            took?: number | undefined;
        }, {
            operation: string;
            indexName?: string | undefined;
            took?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            indexName?: string | undefined;
            took?: number | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            indexName?: string | undefined;
            took?: number | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Full-text search and analytics engine with distributed architecture";
    static readonly longDescription = "\n    Elasticsearch Bubble for full-text search, logging, and analytics.\n\n    Features:\n    - Create and manage indices with custom mappings\n    - Index and search documents in near real-time\n    - Powerful query DSL for complex searches\n    - Bulk operations for high-throughput indexing\n    - Aggregations for data analytics\n    - Distributed and scalable architecture\n\n    Use cases:\n    - Full-text search applications\n    - Log analytics and monitoring\n    - Metrics and dashboard data\n    - Autocomplete and typeahead\n    - Geospatial search\n  ";
    static readonly alias = "es";
    private client;
    constructor(params: ElasticsearchBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getClient;
    protected performAction(context?: BubbleContext): Promise<ElasticsearchBubbleResult>;
    private createIndex;
    private indexDocument;
    private bulkIndex;
    private search;
    private getDocument;
    private updateDocument;
    private deleteDocument;
    private deleteIndex;
    private indexExists;
    private aggregate;
    private extractIndexName;
}
export {};
//# sourceMappingURL=elasticsearch-bubble.d.ts.map