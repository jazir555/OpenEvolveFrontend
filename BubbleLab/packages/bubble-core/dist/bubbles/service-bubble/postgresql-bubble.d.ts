import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const PostgresqlBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"executeQuery">;
    query: z.ZodString;
    params: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    operation: "executeQuery";
    params?: any[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    query: string;
    operation: "executeQuery";
    params?: any[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"executeBatch">;
    queries: z.ZodArray<z.ZodObject<{
        query: z.ZodString;
        params: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        params?: any[] | undefined;
    }, {
        query: string;
        params?: any[] | undefined;
    }>, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "executeBatch";
    queries: {
        query: string;
        params?: any[] | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "executeBatch";
    queries: {
        query: string;
        params?: any[] | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"insertRow">;
    table: z.ZodString;
    data: z.ZodRecord<z.ZodString, z.ZodAny>;
    returning: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    data: Record<string, any>;
    operation: "insertRow";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    returning?: string[] | undefined;
}, {
    data: Record<string, any>;
    operation: "insertRow";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    returning?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateRows">;
    table: z.ZodString;
    data: z.ZodRecord<z.ZodString, z.ZodAny>;
    where: z.ZodRecord<z.ZodString, z.ZodAny>;
    returning: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    data: Record<string, any>;
    operation: "updateRows";
    table: string;
    where: Record<string, any>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    returning?: string[] | undefined;
}, {
    data: Record<string, any>;
    operation: "updateRows";
    table: string;
    where: Record<string, any>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    returning?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteRows">;
    table: z.ZodString;
    where: z.ZodRecord<z.ZodString, z.ZodAny>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteRows";
    table: string;
    where: Record<string, any>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteRows";
    table: string;
    where: Record<string, any>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"selectRows">;
    table: z.ZodString;
    columns: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodString, "many">>>;
    where: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    orderBy: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    limit: z.ZodOptional<z.ZodNumber>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "selectRows";
    offset: number;
    columns: string[];
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    orderBy?: string[] | undefined;
    where?: Record<string, any> | undefined;
}, {
    operation: "selectRows";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    columns?: string[] | undefined;
    orderBy?: string[] | undefined;
    where?: Record<string, any> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createTable">;
    table: z.ZodString;
    columns: z.ZodRecord<z.ZodString, z.ZodObject<{
        type: z.ZodString;
        constraints: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        default: z.ZodOptional<z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        default?: any;
        constraints?: string[] | undefined;
    }, {
        type: string;
        default?: any;
        constraints?: string[] | undefined;
    }>>;
    primaryKey: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createTable";
    columns: Record<string, {
        type: string;
        default?: any;
        constraints?: string[] | undefined;
    }>;
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    primaryKey?: string[] | undefined;
}, {
    operation: "createTable";
    columns: Record<string, {
        type: string;
        default?: any;
        constraints?: string[] | undefined;
    }>;
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    primaryKey?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"dropTable">;
    table: z.ZodString;
    ifExists: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    cascade: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "dropTable";
    table: string;
    ifExists: boolean;
    cascade: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "dropTable";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ifExists?: boolean | undefined;
    cascade?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"tableExists">;
    table: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "tableExists";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "tableExists";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"tableInfo">;
    table: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "tableInfo";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "tableInfo";
    table: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type PostgresqlBubbleParams = z.input<typeof PostgresqlBubbleParamsSchema>;
declare const PostgresqlBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        table: z.ZodOptional<z.ZodString>;
        rowsAffected: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        table?: string | undefined;
        rowsAffected?: number | undefined;
    }, {
        operation: string;
        table?: string | undefined;
        rowsAffected?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        table?: string | undefined;
        rowsAffected?: number | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        table?: string | undefined;
        rowsAffected?: number | undefined;
    };
    data?: unknown;
}>;
type PostgresqlBubbleResult = z.output<typeof PostgresqlBubbleResultSchema>;
export declare class PostgresqlBubble extends ServiceBubble<PostgresqlBubbleParams, PostgresqlBubbleResult> {
    static readonly service = "postgresql";
    static readonly authType: "password";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"executeQuery">;
        query: z.ZodString;
        params: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        operation: "executeQuery";
        params?: any[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        query: string;
        operation: "executeQuery";
        params?: any[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"executeBatch">;
        queries: z.ZodArray<z.ZodObject<{
            query: z.ZodString;
            params: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        }, "strip", z.ZodTypeAny, {
            query: string;
            params?: any[] | undefined;
        }, {
            query: string;
            params?: any[] | undefined;
        }>, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "executeBatch";
        queries: {
            query: string;
            params?: any[] | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "executeBatch";
        queries: {
            query: string;
            params?: any[] | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"insertRow">;
        table: z.ZodString;
        data: z.ZodRecord<z.ZodString, z.ZodAny>;
        returning: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        data: Record<string, any>;
        operation: "insertRow";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        returning?: string[] | undefined;
    }, {
        data: Record<string, any>;
        operation: "insertRow";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        returning?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateRows">;
        table: z.ZodString;
        data: z.ZodRecord<z.ZodString, z.ZodAny>;
        where: z.ZodRecord<z.ZodString, z.ZodAny>;
        returning: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        data: Record<string, any>;
        operation: "updateRows";
        table: string;
        where: Record<string, any>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        returning?: string[] | undefined;
    }, {
        data: Record<string, any>;
        operation: "updateRows";
        table: string;
        where: Record<string, any>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        returning?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteRows">;
        table: z.ZodString;
        where: z.ZodRecord<z.ZodString, z.ZodAny>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteRows";
        table: string;
        where: Record<string, any>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteRows";
        table: string;
        where: Record<string, any>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"selectRows">;
        table: z.ZodString;
        columns: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodString, "many">>>;
        where: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        orderBy: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "selectRows";
        offset: number;
        columns: string[];
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        orderBy?: string[] | undefined;
        where?: Record<string, any> | undefined;
    }, {
        operation: "selectRows";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        columns?: string[] | undefined;
        orderBy?: string[] | undefined;
        where?: Record<string, any> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createTable">;
        table: z.ZodString;
        columns: z.ZodRecord<z.ZodString, z.ZodObject<{
            type: z.ZodString;
            constraints: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            default: z.ZodOptional<z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            default?: any;
            constraints?: string[] | undefined;
        }, {
            type: string;
            default?: any;
            constraints?: string[] | undefined;
        }>>;
        primaryKey: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createTable";
        columns: Record<string, {
            type: string;
            default?: any;
            constraints?: string[] | undefined;
        }>;
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        primaryKey?: string[] | undefined;
    }, {
        operation: "createTable";
        columns: Record<string, {
            type: string;
            default?: any;
            constraints?: string[] | undefined;
        }>;
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        primaryKey?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"dropTable">;
        table: z.ZodString;
        ifExists: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        cascade: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "dropTable";
        table: string;
        ifExists: boolean;
        cascade: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "dropTable";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ifExists?: boolean | undefined;
        cascade?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"tableExists">;
        table: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "tableExists";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "tableExists";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"tableInfo">;
        table: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "tableInfo";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "tableInfo";
        table: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            table: z.ZodOptional<z.ZodString>;
            rowsAffected: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            table?: string | undefined;
            rowsAffected?: number | undefined;
        }, {
            operation: string;
            table?: string | undefined;
            rowsAffected?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            table?: string | undefined;
            rowsAffected?: number | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            table?: string | undefined;
            rowsAffected?: number | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Advanced open source relational database";
    static readonly longDescription = "\n    PostgreSQL Bubble for relational data management.\n\n    Features:\n    - ACID compliant transactions\n    - Complex queries with JOINs and subqueries\n    - Support for JSON/JSONB data types\n    - Full-text search capabilities\n    - Extensible with custom functions\n    - Advanced indexing options\n\n    Use cases:\n    - Primary application database\n    - Complex data relationships\n    - Financial transactions\n    - Analytics and reporting\n    - Geospatial data with PostGIS\n  ";
    static readonly alias = "postgres";
    private pool;
    constructor(params: PostgresqlBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getPool;
    protected performAction(context?: BubbleContext): Promise<PostgresqlBubbleResult>;
    private executeQuery;
    private executeBatch;
    private insertRow;
    private updateRows;
    private deleteRows;
    private selectRows;
    private createTable;
    private dropTable;
    private tableExists;
    private tableInfo;
    private extractTableName;
}
export {};
//# sourceMappingURL=postgresql-bubble.d.ts.map