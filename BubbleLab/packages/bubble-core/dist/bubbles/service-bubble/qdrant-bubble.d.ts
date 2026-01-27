import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const QdrantBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createCollection">;
    collectionName: z.ZodString;
    vectorSize: z.ZodNumber;
    distance: z.ZodDefault<z.ZodOptional<z.ZodEnum<["Cosine", "Euclid", "Dot", "Manhattan"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createCollection";
    collectionName: string;
    vectorSize: number;
    distance: "Cosine" | "Euclid" | "Dot" | "Manhattan";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "createCollection";
    collectionName: string;
    vectorSize: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    distance?: "Cosine" | "Euclid" | "Dot" | "Manhattan" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteCollection">;
    collectionName: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteCollection";
    collectionName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteCollection";
    collectionName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"collectionExists">;
    collectionName: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "collectionExists";
    collectionName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "collectionExists";
    collectionName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"insertPoints">;
    collectionName: z.ZodString;
    points: z.ZodArray<z.ZodObject<{
        id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        vector: z.ZodArray<z.ZodNumber, "many">;
        payload: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }, {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }>, "many">;
    wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "insertPoints";
    wait: boolean;
    collectionName: string;
    points: {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "insertPoints";
    collectionName: string;
    points: {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    wait?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"searchPoints">;
    collectionName: z.ZodString;
    vector: z.ZodArray<z.ZodNumber, "many">;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    scoreThreshold: z.ZodOptional<z.ZodNumber>;
    withPayload: z.ZodDefault<z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>>;
    withVector: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "searchPoints";
    limit: number;
    collectionName: string;
    vector: number[];
    withPayload: boolean | string[];
    withVector: boolean;
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    scoreThreshold?: number | undefined;
}, {
    operation: "searchPoints";
    collectionName: string;
    vector: number[];
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    scoreThreshold?: number | undefined;
    withPayload?: boolean | string[] | undefined;
    withVector?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"upsertPoints">;
    collectionName: z.ZodString;
    points: z.ZodArray<z.ZodObject<{
        id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        vector: z.ZodArray<z.ZodNumber, "many">;
        payload: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }, {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }>, "many">;
    wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "upsertPoints";
    wait: boolean;
    collectionName: string;
    points: {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "upsertPoints";
    collectionName: string;
    points: {
        id: string | number;
        vector: number[];
        payload?: Record<string, unknown> | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    wait?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deletePoints">;
    collectionName: z.ZodString;
    points: z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber]>, "many">;
    wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deletePoints";
    wait: boolean;
    collectionName: string;
    points: (string | number)[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deletePoints";
    collectionName: string;
    points: (string | number)[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    wait?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getPoint">;
    collectionName: z.ZodString;
    pointId: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    withPayload: z.ZodDefault<z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>>;
    withVector: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getPoint";
    collectionName: string;
    withPayload: boolean | string[];
    withVector: boolean;
    pointId: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getPoint";
    collectionName: string;
    pointId: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    withPayload?: boolean | string[] | undefined;
    withVector?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"scrollPoints">;
    collectionName: z.ZodString;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodNumber]>>;
    filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    withPayload: z.ZodDefault<z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>>;
    withVector: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "scrollPoints";
    limit: number;
    collectionName: string;
    withPayload: boolean | string[];
    withVector: boolean;
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    offset?: string | number | undefined;
}, {
    operation: "scrollPoints";
    collectionName: string;
    filter?: Record<string, unknown> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: string | number | undefined;
    withPayload?: boolean | string[] | undefined;
    withVector?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updatePayload">;
    collectionName: z.ZodString;
    payload: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    points: z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber]>, "many">;
    wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updatePayload";
    payload: Record<string, unknown>;
    wait: boolean;
    collectionName: string;
    points: (string | number)[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "updatePayload";
    payload: Record<string, unknown>;
    collectionName: string;
    points: (string | number)[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    wait?: boolean | undefined;
}>]>;
type QdrantBubbleParams = z.input<typeof QdrantBubbleParamsSchema>;
declare const QdrantBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        collectionName: z.ZodOptional<z.ZodString>;
        time: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        time?: number | undefined;
        collectionName?: string | undefined;
    }, {
        operation: string;
        time?: number | undefined;
        collectionName?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        time?: number | undefined;
        collectionName?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        time?: number | undefined;
        collectionName?: string | undefined;
    };
    data?: unknown;
}>;
type QdrantBubbleResult = z.output<typeof QdrantBubbleResultSchema>;
export declare class QdrantBubble extends ServiceBubble<QdrantBubbleParams, QdrantBubbleResult> {
    static readonly service = "qdrant";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createCollection">;
        collectionName: z.ZodString;
        vectorSize: z.ZodNumber;
        distance: z.ZodDefault<z.ZodOptional<z.ZodEnum<["Cosine", "Euclid", "Dot", "Manhattan"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createCollection";
        collectionName: string;
        vectorSize: number;
        distance: "Cosine" | "Euclid" | "Dot" | "Manhattan";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "createCollection";
        collectionName: string;
        vectorSize: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        distance?: "Cosine" | "Euclid" | "Dot" | "Manhattan" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteCollection">;
        collectionName: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteCollection";
        collectionName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteCollection";
        collectionName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"collectionExists">;
        collectionName: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "collectionExists";
        collectionName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "collectionExists";
        collectionName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"insertPoints">;
        collectionName: z.ZodString;
        points: z.ZodArray<z.ZodObject<{
            id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
            vector: z.ZodArray<z.ZodNumber, "many">;
            payload: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }, {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }>, "many">;
        wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "insertPoints";
        wait: boolean;
        collectionName: string;
        points: {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "insertPoints";
        collectionName: string;
        points: {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        wait?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"searchPoints">;
        collectionName: z.ZodString;
        vector: z.ZodArray<z.ZodNumber, "many">;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        scoreThreshold: z.ZodOptional<z.ZodNumber>;
        withPayload: z.ZodDefault<z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>>;
        withVector: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "searchPoints";
        limit: number;
        collectionName: string;
        vector: number[];
        withPayload: boolean | string[];
        withVector: boolean;
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        scoreThreshold?: number | undefined;
    }, {
        operation: "searchPoints";
        collectionName: string;
        vector: number[];
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        scoreThreshold?: number | undefined;
        withPayload?: boolean | string[] | undefined;
        withVector?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"upsertPoints">;
        collectionName: z.ZodString;
        points: z.ZodArray<z.ZodObject<{
            id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
            vector: z.ZodArray<z.ZodNumber, "many">;
            payload: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }, {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }>, "many">;
        wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "upsertPoints";
        wait: boolean;
        collectionName: string;
        points: {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "upsertPoints";
        collectionName: string;
        points: {
            id: string | number;
            vector: number[];
            payload?: Record<string, unknown> | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        wait?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deletePoints">;
        collectionName: z.ZodString;
        points: z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber]>, "many">;
        wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deletePoints";
        wait: boolean;
        collectionName: string;
        points: (string | number)[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deletePoints";
        collectionName: string;
        points: (string | number)[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        wait?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getPoint">;
        collectionName: z.ZodString;
        pointId: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        withPayload: z.ZodDefault<z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>>;
        withVector: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getPoint";
        collectionName: string;
        withPayload: boolean | string[];
        withVector: boolean;
        pointId: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getPoint";
        collectionName: string;
        pointId: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        withPayload?: boolean | string[] | undefined;
        withVector?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"scrollPoints">;
        collectionName: z.ZodString;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodNumber]>>;
        filter: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        withPayload: z.ZodDefault<z.ZodOptional<z.ZodUnion<[z.ZodBoolean, z.ZodArray<z.ZodString, "many">]>>>;
        withVector: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "scrollPoints";
        limit: number;
        collectionName: string;
        withPayload: boolean | string[];
        withVector: boolean;
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        offset?: string | number | undefined;
    }, {
        operation: "scrollPoints";
        collectionName: string;
        filter?: Record<string, unknown> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: string | number | undefined;
        withPayload?: boolean | string[] | undefined;
        withVector?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updatePayload">;
        collectionName: z.ZodString;
        payload: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        points: z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodNumber]>, "many">;
        wait: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updatePayload";
        payload: Record<string, unknown>;
        wait: boolean;
        collectionName: string;
        points: (string | number)[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "updatePayload";
        payload: Record<string, unknown>;
        collectionName: string;
        points: (string | number)[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        wait?: boolean | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            collectionName: z.ZodOptional<z.ZodString>;
            time: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            time?: number | undefined;
            collectionName?: string | undefined;
        }, {
            operation: string;
            time?: number | undefined;
            collectionName?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            time?: number | undefined;
            collectionName?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            time?: number | undefined;
            collectionName?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "High-performance vector database for similarity search and AI applications";
    static readonly longDescription = "\n    Qdrant Bubble for vector similarity search and management.\n\n    Features:\n    - Create and manage vector collections\n    - Insert and search high-dimensional vectors\n    - Filter searches by payload metadata\n    - Real-time updates with upsert operations\n    - Scalable architecture with sharding support\n    - Multiple distance metrics (Cosine, Euclidean, Dot, Manhattan)\n\n    Use cases:\n    - Semantic search with embeddings\n    - Recommendation systems\n    - Image and document similarity\n    - RAG (Retrieval Augmented Generation)\n    - Duplicate detection\n    - Clustering and classification\n  ";
    static readonly alias = "vector";
    private client;
    constructor(params: QdrantBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getClient;
    protected performAction(context?: BubbleContext): Promise<QdrantBubbleResult>;
    private createCollection;
    private deleteCollection;
    private collectionExists;
    private insertPoints;
    private searchPoints;
    private upsertPoints;
    private deletePoints;
    private getPoint;
    private scrollPoints;
    private updatePayload;
    private extractCollectionName;
}
export {};
//# sourceMappingURL=qdrant-bubble.d.ts.map