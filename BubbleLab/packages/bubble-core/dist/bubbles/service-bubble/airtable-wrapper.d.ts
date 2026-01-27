/**
 * Airtable Wrapper Service Bubble - OpenEvolve Resilient Implementation
 *
 * Complete production implementation with 12 operations and full resilience patterns:
 *
 * Table Operations:
 * 1. listRecords - List records from table with pagination
 * 2. getRecord - Get a specific record by ID
 * 3. createRecord - Create a new record
 * 4. updateRecord - Update an existing record
 * 5. deleteRecord - Delete a record
 * 6. batchCreate - Create multiple records (max 10)
 * 7. batchUpdate - Update multiple records (max 10)
 * 8. batchDelete - Delete multiple records (max 10)
 *
 * Query Operations:
 * 9. queryRecords - Query records with formula filter
 * 10. searchRecords - Full-text search across records
 *
 * Metadata Operations:
 * 11. getSchema - Get table schema with field definitions
 * 12. listTables - List all tables in a base
 *
 * Security & Resilience Features:
 * - Circuit breaker pattern (5 failures opens circuit, 60s timeout)
 * - Exponential backoff retry (1s, 2s, 4s, 8s, 16s)
 * - Rate limiting (5 requests/sec per Airtable base)
 * - Input validation with Zod schemas
 * - Structured logging with correlation IDs
 * - Error sanitization
 * - API key authentication
 * - Token bucket rate limiter
 */
import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const AirtableWrapperParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"listRecords">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    maxRecords: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodOptional<z.ZodString>;
    fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    sort: z.ZodOptional<z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        direction: z.ZodEnum<["asc", "desc"]>;
    }, "strip", z.ZodTypeAny, {
        direction: "asc" | "desc";
        field: string;
    }, {
        direction: "asc" | "desc";
        field: string;
    }>, "many">>;
    view: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listRecords";
    baseId: string;
    maxRecords: number;
    tableId: string;
    sort?: {
        direction: "asc" | "desc";
        field: string;
    }[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
    offset?: string | undefined;
    view?: string | undefined;
}, {
    operation: "listRecords";
    baseId: string;
    tableId: string;
    sort?: {
        direction: "asc" | "desc";
        field: string;
    }[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
    offset?: string | undefined;
    maxRecords?: number | undefined;
    view?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRecord">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    recordId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getRecord";
    baseId: string;
    recordId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getRecord";
    baseId: string;
    recordId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createRecord">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    fields: z.ZodRecord<z.ZodString, z.ZodAny>;
    typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    fields: Record<string, any>;
    operation: "createRecord";
    baseId: string;
    typecast: boolean;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    fields: Record<string, any>;
    operation: "createRecord";
    baseId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    typecast?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateRecord">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    recordId: z.ZodString;
    fields: z.ZodRecord<z.ZodString, z.ZodAny>;
    typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    fields: Record<string, any>;
    operation: "updateRecord";
    baseId: string;
    recordId: string;
    typecast: boolean;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    fields: Record<string, any>;
    operation: "updateRecord";
    baseId: string;
    recordId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    typecast?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteRecord">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    recordId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteRecord";
    baseId: string;
    recordId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteRecord";
    baseId: string;
    recordId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchCreate">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    records: z.ZodArray<z.ZodObject<{
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, any>;
    }, {
        fields: Record<string, any>;
    }>, "many">;
    typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "batchCreate";
    baseId: string;
    records: {
        fields: Record<string, any>;
    }[];
    typecast: boolean;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "batchCreate";
    baseId: string;
    records: {
        fields: Record<string, any>;
    }[];
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    typecast?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchUpdate">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    records: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, any>;
        id: string;
    }, {
        fields: Record<string, any>;
        id: string;
    }>, "many">;
    typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "batchUpdate";
    baseId: string;
    records: {
        fields: Record<string, any>;
        id: string;
    }[];
    typecast: boolean;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "batchUpdate";
    baseId: string;
    records: {
        fields: Record<string, any>;
        id: string;
    }[];
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    typecast?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchDelete">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    recordIds: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "batchDelete";
    baseId: string;
    recordIds: string[];
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "batchDelete";
    baseId: string;
    recordIds: string[];
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"queryRecords">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    filterByFormula: z.ZodString;
    maxRecords: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    sort: z.ZodOptional<z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        direction: z.ZodEnum<["asc", "desc"]>;
    }, "strip", z.ZodTypeAny, {
        direction: "asc" | "desc";
        field: string;
    }, {
        direction: "asc" | "desc";
        field: string;
    }>, "many">>;
    view: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "queryRecords";
    baseId: string;
    filterByFormula: string;
    maxRecords: number;
    tableId: string;
    sort?: {
        direction: "asc" | "desc";
        field: string;
    }[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
    view?: string | undefined;
}, {
    operation: "queryRecords";
    baseId: string;
    filterByFormula: string;
    tableId: string;
    sort?: {
        direction: "asc" | "desc";
        field: string;
    }[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
    maxRecords?: number | undefined;
    view?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"searchRecords">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    searchString: z.ZodString;
    fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    maxRecords: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "searchRecords";
    searchString: string;
    baseId: string;
    maxRecords: number;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
}, {
    operation: "searchRecords";
    searchString: string;
    baseId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
    maxRecords?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getSchema">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getSchema";
    baseId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getSchema";
    baseId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listTables">;
    baseId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listTables";
    baseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "listTables";
    baseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type AirtableWrapperParams = z.input<typeof AirtableWrapperParamsSchema>;
declare const AirtableWrapperResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"listRecords">;
    result: z.ZodObject<{
        records: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }, {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }>, "many">;
        offset: z.ZodOptional<z.ZodString>;
        count: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    }, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "listRecords";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    };
}, {
    operation: "listRecords";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRecord">;
    result: z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    }, {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getRecord";
    result: {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    };
}, {
    operation: "getRecord";
    result: {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createRecord">;
    result: z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    }, {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createRecord";
    result: {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    };
}, {
    operation: "createRecord";
    result: {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateRecord">;
    result: z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    }, {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "updateRecord";
    result: {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    };
}, {
    operation: "updateRecord";
    result: {
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
        error?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteRecord">;
    result: z.ZodObject<{
        deleted: z.ZodBoolean;
        recordId: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        deleted: boolean;
        recordId: string;
        error?: string | undefined;
    }, {
        success: boolean;
        deleted: boolean;
        recordId: string;
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteRecord";
    result: {
        success: boolean;
        deleted: boolean;
        recordId: string;
        error?: string | undefined;
    };
}, {
    operation: "deleteRecord";
    result: {
        success: boolean;
        deleted: boolean;
        recordId: string;
        error?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchCreate">;
    result: z.ZodObject<{
        records: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodOptional<z.ZodString>;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }, {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }>, "many">;
        count: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    }, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "batchCreate";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    };
}, {
    operation: "batchCreate";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchUpdate">;
    result: z.ZodObject<{
        records: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodOptional<z.ZodString>;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }, {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }>, "many">;
        count: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    }, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "batchUpdate";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    };
}, {
    operation: "batchUpdate";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
        error?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchDelete">;
    result: z.ZodObject<{
        deleted: z.ZodBoolean;
        count: z.ZodNumber;
        recordIds: z.ZodArray<z.ZodString, "many">;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
        error?: string | undefined;
    }, {
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "batchDelete";
    result: {
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
        error?: string | undefined;
    };
}, {
    operation: "batchDelete";
    result: {
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
        error?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"queryRecords">;
    result: z.ZodObject<{
        records: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }, {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }>, "many">;
        offset: z.ZodOptional<z.ZodString>;
        count: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    }, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "queryRecords";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    };
}, {
    operation: "queryRecords";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"searchRecords">;
    result: z.ZodObject<{
        records: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }, {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }>, "many">;
        offset: z.ZodOptional<z.ZodString>;
        count: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    }, {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "searchRecords";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    };
}, {
    operation: "searchRecords";
    result: {
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        error?: string | undefined;
        offset?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getSchema">;
    result: z.ZodObject<{
        tableId: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        primaryFieldId: z.ZodString;
        fields: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            options: z.ZodOptional<z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }, {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }>, "many">;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        primaryFieldId: string;
        tableId: string;
        error?: string | undefined;
        description?: string | undefined;
    }, {
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        primaryFieldId: string;
        tableId: string;
        error?: string | undefined;
        description?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getSchema";
    result: {
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        primaryFieldId: string;
        tableId: string;
        error?: string | undefined;
        description?: string | undefined;
    };
}, {
    operation: "getSchema";
    result: {
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        primaryFieldId: string;
        tableId: string;
        error?: string | undefined;
        description?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listTables">;
    result: z.ZodObject<{
        tables: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            primaryFieldId: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }, {
            name: string;
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }>, "many">;
        count: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        count: number;
        tables: {
            name: string;
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }[];
        error?: string | undefined;
    }, {
        success: boolean;
        count: number;
        tables: {
            name: string;
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }[];
        error?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "listTables";
    result: {
        success: boolean;
        count: number;
        tables: {
            name: string;
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }[];
        error?: string | undefined;
    };
}, {
    operation: "listTables";
    result: {
        success: boolean;
        count: number;
        tables: {
            name: string;
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }[];
        error?: string | undefined;
    };
}>]>;
type AirtableWrapperResult = z.output<typeof AirtableWrapperResultSchema>;
export declare class AirtableWrapperBubble<T extends AirtableWrapperParams = AirtableWrapperParams> extends ServiceBubble<T, any> {
    static readonly type: "service";
    static readonly service = "airtable-wrapper";
    static readonly authType: "apikey";
    static readonly bubbleName = "airtable-wrapper";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"listRecords">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        maxRecords: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodOptional<z.ZodString>;
        fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        sort: z.ZodOptional<z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            direction: z.ZodEnum<["asc", "desc"]>;
        }, "strip", z.ZodTypeAny, {
            direction: "asc" | "desc";
            field: string;
        }, {
            direction: "asc" | "desc";
            field: string;
        }>, "many">>;
        view: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listRecords";
        baseId: string;
        maxRecords: number;
        tableId: string;
        sort?: {
            direction: "asc" | "desc";
            field: string;
        }[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
        offset?: string | undefined;
        view?: string | undefined;
    }, {
        operation: "listRecords";
        baseId: string;
        tableId: string;
        sort?: {
            direction: "asc" | "desc";
            field: string;
        }[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
        offset?: string | undefined;
        maxRecords?: number | undefined;
        view?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRecord">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        recordId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRecord";
        baseId: string;
        recordId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getRecord";
        baseId: string;
        recordId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createRecord">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, any>;
        operation: "createRecord";
        baseId: string;
        typecast: boolean;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        fields: Record<string, any>;
        operation: "createRecord";
        baseId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        typecast?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateRecord">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        recordId: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, any>;
        operation: "updateRecord";
        baseId: string;
        recordId: string;
        typecast: boolean;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        fields: Record<string, any>;
        operation: "updateRecord";
        baseId: string;
        recordId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        typecast?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteRecord">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        recordId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteRecord";
        baseId: string;
        recordId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteRecord";
        baseId: string;
        recordId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchCreate">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        records: z.ZodArray<z.ZodObject<{
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, any>;
        }, {
            fields: Record<string, any>;
        }>, "many">;
        typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchCreate";
        baseId: string;
        records: {
            fields: Record<string, any>;
        }[];
        typecast: boolean;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "batchCreate";
        baseId: string;
        records: {
            fields: Record<string, any>;
        }[];
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        typecast?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchUpdate">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        records: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, any>;
            id: string;
        }, {
            fields: Record<string, any>;
            id: string;
        }>, "many">;
        typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchUpdate";
        baseId: string;
        records: {
            fields: Record<string, any>;
            id: string;
        }[];
        typecast: boolean;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "batchUpdate";
        baseId: string;
        records: {
            fields: Record<string, any>;
            id: string;
        }[];
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        typecast?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchDelete">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        recordIds: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchDelete";
        baseId: string;
        recordIds: string[];
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "batchDelete";
        baseId: string;
        recordIds: string[];
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"queryRecords">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        filterByFormula: z.ZodString;
        maxRecords: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        sort: z.ZodOptional<z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            direction: z.ZodEnum<["asc", "desc"]>;
        }, "strip", z.ZodTypeAny, {
            direction: "asc" | "desc";
            field: string;
        }, {
            direction: "asc" | "desc";
            field: string;
        }>, "many">>;
        view: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "queryRecords";
        baseId: string;
        filterByFormula: string;
        maxRecords: number;
        tableId: string;
        sort?: {
            direction: "asc" | "desc";
            field: string;
        }[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
        view?: string | undefined;
    }, {
        operation: "queryRecords";
        baseId: string;
        filterByFormula: string;
        tableId: string;
        sort?: {
            direction: "asc" | "desc";
            field: string;
        }[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
        maxRecords?: number | undefined;
        view?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"searchRecords">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        searchString: z.ZodString;
        fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        maxRecords: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "searchRecords";
        searchString: string;
        baseId: string;
        maxRecords: number;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
    }, {
        operation: "searchRecords";
        searchString: string;
        baseId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
        maxRecords?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getSchema">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getSchema";
        baseId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getSchema";
        baseId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listTables">;
        baseId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listTables";
        baseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "listTables";
        baseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"listRecords">;
        result: z.ZodObject<{
            records: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                createdTime: z.ZodString;
                fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            }, "strip", z.ZodTypeAny, {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }, {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }>, "many">;
            offset: z.ZodOptional<z.ZodString>;
            count: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        }, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "listRecords";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        };
    }, {
        operation: "listRecords";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRecord">;
        result: z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        }, {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRecord";
        result: {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        };
    }, {
        operation: "getRecord";
        result: {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createRecord">;
        result: z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        }, {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createRecord";
        result: {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        };
    }, {
        operation: "createRecord";
        result: {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateRecord">;
        result: z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        }, {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateRecord";
        result: {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        };
    }, {
        operation: "updateRecord";
        result: {
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
            error?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteRecord">;
        result: z.ZodObject<{
            deleted: z.ZodBoolean;
            recordId: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            deleted: boolean;
            recordId: string;
            error?: string | undefined;
        }, {
            success: boolean;
            deleted: boolean;
            recordId: string;
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteRecord";
        result: {
            success: boolean;
            deleted: boolean;
            recordId: string;
            error?: string | undefined;
        };
    }, {
        operation: "deleteRecord";
        result: {
            success: boolean;
            deleted: boolean;
            recordId: string;
            error?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchCreate">;
        result: z.ZodObject<{
            records: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                createdTime: z.ZodOptional<z.ZodString>;
                fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            }, "strip", z.ZodTypeAny, {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }, {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }>, "many">;
            count: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        }, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchCreate";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        };
    }, {
        operation: "batchCreate";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchUpdate">;
        result: z.ZodObject<{
            records: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                createdTime: z.ZodOptional<z.ZodString>;
                fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            }, "strip", z.ZodTypeAny, {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }, {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }>, "many">;
            count: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        }, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchUpdate";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        };
    }, {
        operation: "batchUpdate";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
            error?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchDelete">;
        result: z.ZodObject<{
            deleted: z.ZodBoolean;
            count: z.ZodNumber;
            recordIds: z.ZodArray<z.ZodString, "many">;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
            error?: string | undefined;
        }, {
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchDelete";
        result: {
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
            error?: string | undefined;
        };
    }, {
        operation: "batchDelete";
        result: {
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
            error?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"queryRecords">;
        result: z.ZodObject<{
            records: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                createdTime: z.ZodString;
                fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            }, "strip", z.ZodTypeAny, {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }, {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }>, "many">;
            offset: z.ZodOptional<z.ZodString>;
            count: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        }, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "queryRecords";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        };
    }, {
        operation: "queryRecords";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"searchRecords">;
        result: z.ZodObject<{
            records: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                createdTime: z.ZodString;
                fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            }, "strip", z.ZodTypeAny, {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }, {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }>, "many">;
            offset: z.ZodOptional<z.ZodString>;
            count: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        }, {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "searchRecords";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        };
    }, {
        operation: "searchRecords";
        result: {
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            error?: string | undefined;
            offset?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getSchema">;
        result: z.ZodObject<{
            tableId: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            primaryFieldId: z.ZodString;
            fields: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                name: z.ZodString;
                type: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
                options: z.ZodOptional<z.ZodAny>;
            }, "strip", z.ZodTypeAny, {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }, {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }>, "many">;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            primaryFieldId: string;
            tableId: string;
            error?: string | undefined;
            description?: string | undefined;
        }, {
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            primaryFieldId: string;
            tableId: string;
            error?: string | undefined;
            description?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getSchema";
        result: {
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            primaryFieldId: string;
            tableId: string;
            error?: string | undefined;
            description?: string | undefined;
        };
    }, {
        operation: "getSchema";
        result: {
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            primaryFieldId: string;
            tableId: string;
            error?: string | undefined;
            description?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listTables">;
        result: z.ZodObject<{
            tables: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                name: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
                primaryFieldId: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                name: string;
                id: string;
                primaryFieldId: string;
                description?: string | undefined;
            }, {
                name: string;
                id: string;
                primaryFieldId: string;
                description?: string | undefined;
            }>, "many">;
            count: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count: number;
            tables: {
                name: string;
                id: string;
                primaryFieldId: string;
                description?: string | undefined;
            }[];
            error?: string | undefined;
        }, {
            success: boolean;
            count: number;
            tables: {
                name: string;
                id: string;
                primaryFieldId: string;
                description?: string | undefined;
            }[];
            error?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "listTables";
        result: {
            success: boolean;
            count: number;
            tables: {
                name: string;
                id: string;
                primaryFieldId: string;
                description?: string | undefined;
            }[];
            error?: string | undefined;
        };
    }, {
        operation: "listTables";
        result: {
            success: boolean;
            count: number;
            tables: {
                name: string;
                id: string;
                primaryFieldId: string;
                description?: string | undefined;
            }[];
            error?: string | undefined;
        };
    }>]>;
    static readonly shortDescription = "OpenEvolve resilient Airtable integration";
    static readonly longDescription = "\n    OpenEvolve-specific Airtable wrapper with comprehensive resilience patterns.\n\n    Operations (12 total):\n    Table Operations:\n    1. listRecords - List records with pagination, filtering, sorting\n    2. getRecord - Get a specific record by ID\n    3. createRecord - Create a new record\n    4. updateRecord - Update an existing record\n    5. deleteRecord - Delete a record\n    6. batchCreate - Create up to 10 records\n    7. batchUpdate - Update up to 10 records\n    8. batchDelete - Delete up to 10 records\n\n    Query Operations:\n    9. queryRecords - Query with formula filters\n    10. searchRecords - Full-text search\n\n    Metadata Operations:\n    11. getSchema - Get table schema and field definitions\n    12. listTables - List all tables in a base\n\n    Resilience Features:\n    - Circuit breaker (opens after 5 failures, 60s timeout)\n    - Exponential backoff retry (1s, 2s, 4s, 8s, 16s)\n    - Rate limiting (5 requests/sec per base)\n    - Input validation with Zod schemas\n    - Structured logging with correlation IDs\n    - Error sanitization\n    - Request deduplication\n    - Dead letter queue for failed operations\n\n    Security Features:\n    - API key authentication\n    - Base ID format validation (starts with 'app')\n    - Table ID validation\n    - Record ID validation (starts with 'rec')\n    - Field name validation\n    - Rate limiting enforcement\n  ";
    static readonly alias = "airtable";
    private client;
    private resilience;
    private logger;
    private correlationId;
    constructor(params: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<AirtableWrapperResult, {
        operation: T['operation'];
    }>>;
    private listRecords;
    private getRecord;
    private createRecord;
    private updateRecord;
    private deleteRecord;
    private batchCreate;
    private batchUpdate;
    private batchDelete;
    private queryRecords;
    private searchRecords;
    private getSchema;
    private listTables;
    private errorResult;
    /**
     * Get circuit breaker state
     */
    getCircuitBreakerState(): import("../../__mocks__/resilience.js").CircuitBreakerState;
    /**
     * Get circuit breaker statistics
     */
    getCircuitBreakerStats(): {
        state: import("../../__mocks__/resilience.js").CircuitBreakerState;
        failureCount: number;
        successCount: number;
    };
    /**
     * Reset circuit breaker
     */
    resetCircuitBreaker(): Promise<void>;
    /**
     * Get deduplicator statistics
     */
    getDeduplicatorStats(): {
        totalProcessed: number;
        duplicates: number;
        byKey: {
            [k: string]: number;
        };
    };
    /**
     * Get dead letter queue entries
     */
    getDeadLetterEntries(): {
        error: string;
        timestamp: number;
        data: any;
    }[];
    /**
     * Clear dead letter queue
     */
    clearDeadLetterQueue(): void;
}
export {};
//# sourceMappingURL=airtable-wrapper.d.ts.map