import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const AirtableBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
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
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getTable">;
    baseId: z.ZodString;
    tableId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getTable";
    baseId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getTable";
    baseId: string;
    tableId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type AirtableBubbleParams = z.input<typeof AirtableBubbleParamsSchema>;
declare const AirtableBubbleResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
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
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    }, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "listRecords";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    };
}, {
    operation: "listRecords";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRecord">;
    result: z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    }, {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getRecord";
    result: {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    };
}, {
    operation: "getRecord";
    result: {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createRecord">;
    result: z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    }, {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createRecord";
    result: {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    };
}, {
    operation: "createRecord";
    result: {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateRecord">;
    result: z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodAny>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    }, {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "updateRecord";
    result: {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    };
}, {
    operation: "updateRecord";
    result: {
        error: string;
        success: boolean;
        fields: Record<string, any>;
        id: string;
        createdTime: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteRecord">;
    result: z.ZodObject<{
        deleted: z.ZodBoolean;
        recordId: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        deleted: boolean;
        recordId: string;
    }, {
        error: string;
        success: boolean;
        deleted: boolean;
        recordId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteRecord";
    result: {
        error: string;
        success: boolean;
        deleted: boolean;
        recordId: string;
    };
}, {
    operation: "deleteRecord";
    result: {
        error: string;
        success: boolean;
        deleted: boolean;
        recordId: string;
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
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
    }, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "batchCreate";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
    };
}, {
    operation: "batchCreate";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
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
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
    }, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "batchUpdate";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
    };
}, {
    operation: "batchUpdate";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime?: string | undefined;
        }[];
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"batchDelete">;
    result: z.ZodObject<{
        deleted: z.ZodBoolean;
        count: z.ZodNumber;
        recordIds: z.ZodArray<z.ZodString, "many">;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
    }, {
        error: string;
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "batchDelete";
    result: {
        error: string;
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
    };
}, {
    operation: "batchDelete";
    result: {
        error: string;
        success: boolean;
        count: number;
        deleted: boolean;
        recordIds: string[];
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
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    }, {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "queryRecords";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    };
}, {
    operation: "queryRecords";
    result: {
        error: string;
        success: boolean;
        count: number;
        records: {
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }[];
        offset?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getTable">;
    result: z.ZodObject<{
        tableId: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
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
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        tableId: string;
        description?: string | undefined;
    }, {
        error: string;
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        tableId: string;
        description?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getTable";
    result: {
        error: string;
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        tableId: string;
        description?: string | undefined;
    };
}, {
    operation: "getTable";
    result: {
        error: string;
        name: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: any;
            description?: string | undefined;
        }[];
        tableId: string;
        description?: string | undefined;
    };
}>]>;
type AirtableBubbleResult = z.output<typeof AirtableBubbleResultSchema>;
export declare class AirtableBubble<T extends AirtableBubbleParams = AirtableBubbleParams> extends ServiceBubble<T, any> {
    static readonly type: "service";
    static readonly service = "airtable";
    static readonly authType: "apikey";
    static readonly bubbleName = "airtable";
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
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getTable">;
        baseId: z.ZodString;
        tableId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getTable";
        baseId: string;
        tableId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getTable";
        baseId: string;
        tableId: string;
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
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        }, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "listRecords";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        };
    }, {
        operation: "listRecords";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRecord">;
        result: z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }, {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRecord";
        result: {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        };
    }, {
        operation: "getRecord";
        result: {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createRecord">;
        result: z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }, {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createRecord";
        result: {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        };
    }, {
        operation: "createRecord";
        result: {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateRecord">;
        result: z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodAny>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }, {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateRecord";
        result: {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        };
    }, {
        operation: "updateRecord";
        result: {
            error: string;
            success: boolean;
            fields: Record<string, any>;
            id: string;
            createdTime: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteRecord">;
        result: z.ZodObject<{
            deleted: z.ZodBoolean;
            recordId: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            deleted: boolean;
            recordId: string;
        }, {
            error: string;
            success: boolean;
            deleted: boolean;
            recordId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteRecord";
        result: {
            error: string;
            success: boolean;
            deleted: boolean;
            recordId: string;
        };
    }, {
        operation: "deleteRecord";
        result: {
            error: string;
            success: boolean;
            deleted: boolean;
            recordId: string;
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
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
        }, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchCreate";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
        };
    }, {
        operation: "batchCreate";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
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
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
        }, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchUpdate";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
        };
    }, {
        operation: "batchUpdate";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime?: string | undefined;
            }[];
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"batchDelete">;
        result: z.ZodObject<{
            deleted: z.ZodBoolean;
            count: z.ZodNumber;
            recordIds: z.ZodArray<z.ZodString, "many">;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
        }, {
            error: string;
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "batchDelete";
        result: {
            error: string;
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
        };
    }, {
        operation: "batchDelete";
        result: {
            error: string;
            success: boolean;
            count: number;
            deleted: boolean;
            recordIds: string[];
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
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        }, {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "queryRecords";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        };
    }, {
        operation: "queryRecords";
        result: {
            error: string;
            success: boolean;
            count: number;
            records: {
                fields: Record<string, any>;
                id: string;
                createdTime: string;
            }[];
            offset?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getTable">;
        result: z.ZodObject<{
            tableId: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
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
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            tableId: string;
            description?: string | undefined;
        }, {
            error: string;
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            tableId: string;
            description?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getTable";
        result: {
            error: string;
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            tableId: string;
            description?: string | undefined;
        };
    }, {
        operation: "getTable";
        result: {
            error: string;
            name: string;
            success: boolean;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: any;
                description?: string | undefined;
            }[];
            tableId: string;
            description?: string | undefined;
        };
    }>]>;
    static readonly shortDescription = "Complete Airtable integration for database operations";
    static readonly longDescription = "\n    Comprehensive Airtable service bubble for all database operations.\n\n    Operations:\n    1. listRecords - List records with pagination and sorting\n    2. getRecord - Get a specific record by ID\n    3. createRecord - Create a new record\n    4. updateRecord - Update an existing record\n    5. deleteRecord - Delete a record\n    6. batchCreate - Create multiple records (up to 10)\n    7. batchUpdate - Update multiple records (up to 10)\n    8. batchDelete - Delete multiple records (up to 10)\n    9. queryRecords - Query with formula filters\n    10. getTable - Get table schema and field definitions\n\n    Features:\n    - Full CRUD operations\n    - Batch operations for efficiency\n    - Formula-based querying\n    - Field type conversion\n    - Pagination support\n    - Sorting capabilities\n    - Resilience patterns\n  ";
    static readonly alias = "airtable";
    private client;
    private resilience;
    constructor(params: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<AirtableBubbleResult, {
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
    private getTable;
    private errorResult;
}
export {};
//# sourceMappingURL=airtable-bubble.d.ts.map