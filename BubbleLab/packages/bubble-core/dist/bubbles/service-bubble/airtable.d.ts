import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const AirtableParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"list_records">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    filterByFormula: z.ZodOptional<z.ZodString>;
    maxRecords: z.ZodOptional<z.ZodNumber>;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    sort: z.ZodOptional<z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
    }, "strip", z.ZodTypeAny, {
        direction: "asc" | "desc";
        field: string;
    }, {
        field: string;
        direction?: "asc" | "desc" | undefined;
    }>, "many">>;
    view: z.ZodOptional<z.ZodString>;
    cellFormat: z.ZodDefault<z.ZodOptional<z.ZodEnum<["json", "string"]>>>;
    timeZone: z.ZodOptional<z.ZodString>;
    userLocale: z.ZodOptional<z.ZodString>;
    offset: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_records";
    pageSize: number;
    baseId: string;
    tableIdOrName: string;
    cellFormat: "string" | "json";
    sort?: {
        direction: "asc" | "desc";
        field: string;
    }[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
    timeZone?: string | undefined;
    offset?: string | undefined;
    filterByFormula?: string | undefined;
    maxRecords?: number | undefined;
    view?: string | undefined;
    userLocale?: string | undefined;
}, {
    operation: "list_records";
    baseId: string;
    tableIdOrName: string;
    sort?: {
        field: string;
        direction?: "asc" | "desc" | undefined;
    }[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string[] | undefined;
    timeZone?: string | undefined;
    offset?: string | undefined;
    pageSize?: number | undefined;
    filterByFormula?: string | undefined;
    maxRecords?: number | undefined;
    view?: string | undefined;
    cellFormat?: "string" | "json" | undefined;
    userLocale?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_record">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    recordId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_record";
    baseId: string;
    tableIdOrName: string;
    recordId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_record";
    baseId: string;
    tableIdOrName: string;
    recordId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_records">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    records: z.ZodArray<z.ZodObject<{
        fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
    }, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
    }>, "many">;
    typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_records";
    baseId: string;
    tableIdOrName: string;
    records: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
    }[];
    typecast: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "create_records";
    baseId: string;
    tableIdOrName: string;
    records: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    typecast?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_records">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    records: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
    }, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
    }>, "many">;
    typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_records";
    baseId: string;
    tableIdOrName: string;
    records: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
    }[];
    typecast: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "update_records";
    baseId: string;
    tableIdOrName: string;
    records: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    typecast?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_records">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    recordIds: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_records";
    baseId: string;
    tableIdOrName: string;
    recordIds: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_records";
    baseId: string;
    tableIdOrName: string;
    recordIds: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_bases">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_bases";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "list_bases";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_base_schema">;
    baseId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_base_schema";
    baseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_base_schema";
    baseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_table">;
    baseId: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    fields: z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        type: z.ZodEnum<["singleLineText", "multilineText", "richText", "email", "url", "phoneNumber", "number", "percent", "currency", "rating", "duration", "singleSelect", "multipleSelects", "singleCollaborator", "multipleCollaborators", "date", "dateTime", "checkbox", "multipleRecordLinks", "multipleAttachments", "barcode", "button", "formula", "createdTime", "lastModifiedTime", "createdBy", "lastModifiedBy", "autoNumber", "externalSyncSource", "count", "lookup", "rollup"]>;
        description: z.ZodOptional<z.ZodString>;
        options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
        name: string;
        options?: Record<string, unknown> | undefined;
        description?: string | undefined;
    }, {
        type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
        name: string;
        options?: Record<string, unknown> | undefined;
        description?: string | undefined;
    }>, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    fields: {
        type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
        name: string;
        options?: Record<string, unknown> | undefined;
        description?: string | undefined;
    }[];
    operation: "create_table";
    baseId: string;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    name: string;
    fields: {
        type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
        name: string;
        options?: Record<string, unknown> | undefined;
        description?: string | undefined;
    }[];
    operation: "create_table";
    baseId: string;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_table">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    name: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_table";
    baseId: string;
    tableIdOrName: string;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "update_table";
    baseId: string;
    tableIdOrName: string;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_field">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    name: z.ZodString;
    type: z.ZodEnum<["singleLineText", "multilineText", "richText", "email", "url", "phoneNumber", "number", "percent", "currency", "rating", "duration", "singleSelect", "multipleSelects", "singleCollaborator", "multipleCollaborators", "date", "dateTime", "checkbox", "multipleRecordLinks", "multipleAttachments", "barcode", "button", "formula", "createdTime", "lastModifiedTime", "createdBy", "lastModifiedBy", "autoNumber", "externalSyncSource", "count", "lookup", "rollup"]>;
    description: z.ZodOptional<z.ZodString>;
    options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
    name: string;
    operation: "create_field";
    baseId: string;
    tableIdOrName: string;
    options?: Record<string, unknown> | undefined;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
    name: string;
    operation: "create_field";
    baseId: string;
    tableIdOrName: string;
    options?: Record<string, unknown> | undefined;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_field">;
    baseId: z.ZodString;
    tableIdOrName: z.ZodString;
    fieldIdOrName: z.ZodString;
    name: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_field";
    baseId: string;
    tableIdOrName: string;
    fieldIdOrName: string;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "update_field";
    baseId: string;
    tableIdOrName: string;
    fieldIdOrName: string;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
declare const AirtableResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"list_records">;
    ok: z.ZodBoolean;
    records: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }>, "many">>;
    offset: z.ZodOptional<z.ZodString>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_records";
    ok: boolean;
    offset?: string | undefined;
    records?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }[] | undefined;
}, {
    success: boolean;
    operation: "list_records";
    ok: boolean;
    error?: string | undefined;
    offset?: string | undefined;
    records?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_record">;
    ok: z.ZodBoolean;
    record: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }>>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_record";
    ok: boolean;
    record?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    } | undefined;
}, {
    success: boolean;
    operation: "get_record";
    ok: boolean;
    error?: string | undefined;
    record?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_records">;
    ok: z.ZodBoolean;
    records: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }>, "many">>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_records";
    ok: boolean;
    records?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }[] | undefined;
}, {
    success: boolean;
    operation: "create_records";
    ok: boolean;
    error?: string | undefined;
    records?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_records">;
    ok: z.ZodBoolean;
    records: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        createdTime: z.ZodString;
        fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
    }, "strip", z.ZodTypeAny, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }, {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }>, "many">>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_records";
    ok: boolean;
    records?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }[] | undefined;
}, {
    success: boolean;
    operation: "update_records";
    ok: boolean;
    error?: string | undefined;
    records?: {
        fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        id: string;
        createdTime: string;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_records">;
    ok: z.ZodBoolean;
    records: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        deleted: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        id: string;
        deleted: boolean;
    }, {
        id: string;
        deleted: boolean;
    }>, "many">>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_records";
    ok: boolean;
    records?: {
        id: string;
        deleted: boolean;
    }[] | undefined;
}, {
    success: boolean;
    operation: "delete_records";
    ok: boolean;
    error?: string | undefined;
    records?: {
        id: string;
        deleted: boolean;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_bases">;
    ok: z.ZodBoolean;
    bases: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        permissionLevel: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        permissionLevel: string;
    }, {
        name: string;
        id: string;
        permissionLevel: string;
    }>, "many">>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_bases";
    ok: boolean;
    bases?: {
        name: string;
        id: string;
        permissionLevel: string;
    }[] | undefined;
}, {
    success: boolean;
    operation: "list_bases";
    ok: boolean;
    error?: string | undefined;
    bases?: {
        name: string;
        id: string;
        permissionLevel: string;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_base_schema">;
    ok: z.ZodBoolean;
    tables: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        primaryFieldId: z.ZodString;
        fields: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            id: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }, {
            type: string;
            name: string;
            id: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }>, "many">;
        views: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            id: string;
        }, {
            type: string;
            name: string;
            id: string;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
        views?: {
            type: string;
            name: string;
            id: string;
        }[] | undefined;
    }, {
        name: string;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
        views?: {
            type: string;
            name: string;
            id: string;
        }[] | undefined;
    }>, "many">>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_base_schema";
    ok: boolean;
    tables?: {
        name: string;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
        views?: {
            type: string;
            name: string;
            id: string;
        }[] | undefined;
    }[] | undefined;
}, {
    success: boolean;
    operation: "get_base_schema";
    ok: boolean;
    error?: string | undefined;
    tables?: {
        name: string;
        fields: {
            type: string;
            name: string;
            id: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
        views?: {
            type: string;
            name: string;
            id: string;
        }[] | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_table">;
    ok: z.ZodBoolean;
    table: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        primaryFieldId: z.ZodString;
        fields: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodEnum<["singleLineText", "multilineText", "richText", "email", "url", "phoneNumber", "number", "percent", "currency", "rating", "duration", "singleSelect", "multipleSelects", "singleCollaborator", "multipleCollaborators", "date", "dateTime", "checkbox", "multipleRecordLinks", "multipleAttachments", "barcode", "button", "formula", "createdTime", "lastModifiedTime", "createdBy", "lastModifiedBy", "autoNumber", "externalSyncSource", "count", "lookup", "rollup"]>;
        }, "strip", z.ZodTypeAny, {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            id: string;
        }, {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            id: string;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        name: string;
        fields: {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            id: string;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
    }, {
        name: string;
        fields: {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            id: string;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
    }>>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_table";
    ok: boolean;
    table?: {
        name: string;
        fields: {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            id: string;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
    } | undefined;
}, {
    success: boolean;
    operation: "create_table";
    ok: boolean;
    error?: string | undefined;
    table?: {
        name: string;
        fields: {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            id: string;
        }[];
        id: string;
        primaryFieldId: string;
        description?: string | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_table">;
    ok: z.ZodBoolean;
    table: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        description?: string | undefined;
    }, {
        name: string;
        id: string;
        description?: string | undefined;
    }>>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_table";
    ok: boolean;
    table?: {
        name: string;
        id: string;
        description?: string | undefined;
    } | undefined;
}, {
    success: boolean;
    operation: "update_table";
    ok: boolean;
    error?: string | undefined;
    table?: {
        name: string;
        id: string;
        description?: string | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_field">;
    ok: z.ZodBoolean;
    field: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        type: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    }, {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    }>>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_field";
    ok: boolean;
    field?: {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    } | undefined;
}, {
    success: boolean;
    operation: "create_field";
    ok: boolean;
    error?: string | undefined;
    field?: {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_field">;
    ok: z.ZodBoolean;
    field: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        type: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    }, {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    }>>;
    error: z.ZodDefault<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_field";
    ok: boolean;
    field?: {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    } | undefined;
}, {
    success: boolean;
    operation: "update_field";
    ok: boolean;
    error?: string | undefined;
    field?: {
        type: string;
        name: string;
        id: string;
        description?: string | undefined;
    } | undefined;
}>]>;
type AirtableResult = z.output<typeof AirtableResultSchema>;
type AirtableParams = z.input<typeof AirtableParamsSchema>;
export type AirtableParamsInput = z.input<typeof AirtableParamsSchema>;
export type AirtableOperationResult<T extends AirtableParams['operation']> = Extract<AirtableResult, {
    operation: T;
}>;
export declare class AirtableBubble<T extends AirtableParams = AirtableParams> extends ServiceBubble<T, Extract<AirtableResult, {
    operation: T['operation'];
}>> {
    testCredential(): Promise<boolean>;
    static readonly type: "service";
    static readonly service = "airtable";
    static readonly authType: "apikey";
    static readonly bubbleName = "airtable";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"list_records">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        fields: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        filterByFormula: z.ZodOptional<z.ZodString>;
        maxRecords: z.ZodOptional<z.ZodNumber>;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        sort: z.ZodOptional<z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
        }, "strip", z.ZodTypeAny, {
            direction: "asc" | "desc";
            field: string;
        }, {
            field: string;
            direction?: "asc" | "desc" | undefined;
        }>, "many">>;
        view: z.ZodOptional<z.ZodString>;
        cellFormat: z.ZodDefault<z.ZodOptional<z.ZodEnum<["json", "string"]>>>;
        timeZone: z.ZodOptional<z.ZodString>;
        userLocale: z.ZodOptional<z.ZodString>;
        offset: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_records";
        pageSize: number;
        baseId: string;
        tableIdOrName: string;
        cellFormat: "string" | "json";
        sort?: {
            direction: "asc" | "desc";
            field: string;
        }[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
        timeZone?: string | undefined;
        offset?: string | undefined;
        filterByFormula?: string | undefined;
        maxRecords?: number | undefined;
        view?: string | undefined;
        userLocale?: string | undefined;
    }, {
        operation: "list_records";
        baseId: string;
        tableIdOrName: string;
        sort?: {
            field: string;
            direction?: "asc" | "desc" | undefined;
        }[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string[] | undefined;
        timeZone?: string | undefined;
        offset?: string | undefined;
        pageSize?: number | undefined;
        filterByFormula?: string | undefined;
        maxRecords?: number | undefined;
        view?: string | undefined;
        cellFormat?: "string" | "json" | undefined;
        userLocale?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_record">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        recordId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_record";
        baseId: string;
        tableIdOrName: string;
        recordId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_record";
        baseId: string;
        tableIdOrName: string;
        recordId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_records">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        records: z.ZodArray<z.ZodObject<{
            fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        }, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        }>, "many">;
        typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_records";
        baseId: string;
        tableIdOrName: string;
        records: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        }[];
        typecast: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "create_records";
        baseId: string;
        tableIdOrName: string;
        records: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        typecast?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_records">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        records: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
        }, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
        }>, "many">;
        typecast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_records";
        baseId: string;
        tableIdOrName: string;
        records: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
        }[];
        typecast: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "update_records";
        baseId: string;
        tableIdOrName: string;
        records: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        typecast?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_records">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        recordIds: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_records";
        baseId: string;
        tableIdOrName: string;
        recordIds: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_records";
        baseId: string;
        tableIdOrName: string;
        recordIds: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_bases">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_bases";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "list_bases";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_base_schema">;
        baseId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_base_schema";
        baseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_base_schema";
        baseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_table">;
        baseId: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        fields: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            type: z.ZodEnum<["singleLineText", "multilineText", "richText", "email", "url", "phoneNumber", "number", "percent", "currency", "rating", "duration", "singleSelect", "multipleSelects", "singleCollaborator", "multipleCollaborators", "date", "dateTime", "checkbox", "multipleRecordLinks", "multipleAttachments", "barcode", "button", "formula", "createdTime", "lastModifiedTime", "createdBy", "lastModifiedBy", "autoNumber", "externalSyncSource", "count", "lookup", "rollup"]>;
            description: z.ZodOptional<z.ZodString>;
            options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }, {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }>, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        fields: {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }[];
        operation: "create_table";
        baseId: string;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        name: string;
        fields: {
            type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
            name: string;
            options?: Record<string, unknown> | undefined;
            description?: string | undefined;
        }[];
        operation: "create_table";
        baseId: string;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_table">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        name: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_table";
        baseId: string;
        tableIdOrName: string;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "update_table";
        baseId: string;
        tableIdOrName: string;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_field">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        name: z.ZodString;
        type: z.ZodEnum<["singleLineText", "multilineText", "richText", "email", "url", "phoneNumber", "number", "percent", "currency", "rating", "duration", "singleSelect", "multipleSelects", "singleCollaborator", "multipleCollaborators", "date", "dateTime", "checkbox", "multipleRecordLinks", "multipleAttachments", "barcode", "button", "formula", "createdTime", "lastModifiedTime", "createdBy", "lastModifiedBy", "autoNumber", "externalSyncSource", "count", "lookup", "rollup"]>;
        description: z.ZodOptional<z.ZodString>;
        options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
        name: string;
        operation: "create_field";
        baseId: string;
        tableIdOrName: string;
        options?: Record<string, unknown> | undefined;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
        name: string;
        operation: "create_field";
        baseId: string;
        tableIdOrName: string;
        options?: Record<string, unknown> | undefined;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_field">;
        baseId: z.ZodString;
        tableIdOrName: z.ZodString;
        fieldIdOrName: z.ZodString;
        name: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_field";
        baseId: string;
        tableIdOrName: string;
        fieldIdOrName: string;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "update_field";
        baseId: string;
        tableIdOrName: string;
        fieldIdOrName: string;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"list_records">;
        ok: z.ZodBoolean;
        records: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }>, "many">>;
        offset: z.ZodOptional<z.ZodString>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_records";
        ok: boolean;
        offset?: string | undefined;
        records?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }[] | undefined;
    }, {
        success: boolean;
        operation: "list_records";
        ok: boolean;
        error?: string | undefined;
        offset?: string | undefined;
        records?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_record">;
        ok: z.ZodBoolean;
        record: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }>>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_record";
        ok: boolean;
        record?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        } | undefined;
    }, {
        success: boolean;
        operation: "get_record";
        ok: boolean;
        error?: string | undefined;
        record?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_records">;
        ok: z.ZodBoolean;
        records: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }>, "many">>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_records";
        ok: boolean;
        records?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }[] | undefined;
    }, {
        success: boolean;
        operation: "create_records";
        ok: boolean;
        error?: string | undefined;
        records?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_records">;
        ok: z.ZodBoolean;
        records: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            createdTime: z.ZodString;
            fields: z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodArray<z.ZodUnknown, "many">, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodNull]>>;
        }, "strip", z.ZodTypeAny, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }, {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }>, "many">>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_records";
        ok: boolean;
        records?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }[] | undefined;
    }, {
        success: boolean;
        operation: "update_records";
        ok: boolean;
        error?: string | undefined;
        records?: {
            fields: Record<string, string | number | boolean | unknown[] | Record<string, unknown> | null>;
            id: string;
            createdTime: string;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_records">;
        ok: z.ZodBoolean;
        records: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            deleted: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            id: string;
            deleted: boolean;
        }, {
            id: string;
            deleted: boolean;
        }>, "many">>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_records";
        ok: boolean;
        records?: {
            id: string;
            deleted: boolean;
        }[] | undefined;
    }, {
        success: boolean;
        operation: "delete_records";
        ok: boolean;
        error?: string | undefined;
        records?: {
            id: string;
            deleted: boolean;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_bases">;
        ok: z.ZodBoolean;
        bases: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            permissionLevel: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            permissionLevel: string;
        }, {
            name: string;
            id: string;
            permissionLevel: string;
        }>, "many">>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_bases";
        ok: boolean;
        bases?: {
            name: string;
            id: string;
            permissionLevel: string;
        }[] | undefined;
    }, {
        success: boolean;
        operation: "list_bases";
        ok: boolean;
        error?: string | undefined;
        bases?: {
            name: string;
            id: string;
            permissionLevel: string;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_base_schema">;
        ok: z.ZodBoolean;
        tables: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            primaryFieldId: z.ZodString;
            fields: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                name: z.ZodString;
                type: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
                options: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            }, "strip", z.ZodTypeAny, {
                type: string;
                name: string;
                id: string;
                options?: Record<string, unknown> | undefined;
                description?: string | undefined;
            }, {
                type: string;
                name: string;
                id: string;
                options?: Record<string, unknown> | undefined;
                description?: string | undefined;
            }>, "many">;
            views: z.ZodOptional<z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                name: z.ZodString;
                type: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type: string;
                name: string;
                id: string;
            }, {
                type: string;
                name: string;
                id: string;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: Record<string, unknown> | undefined;
                description?: string | undefined;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
            views?: {
                type: string;
                name: string;
                id: string;
            }[] | undefined;
        }, {
            name: string;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: Record<string, unknown> | undefined;
                description?: string | undefined;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
            views?: {
                type: string;
                name: string;
                id: string;
            }[] | undefined;
        }>, "many">>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_base_schema";
        ok: boolean;
        tables?: {
            name: string;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: Record<string, unknown> | undefined;
                description?: string | undefined;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
            views?: {
                type: string;
                name: string;
                id: string;
            }[] | undefined;
        }[] | undefined;
    }, {
        success: boolean;
        operation: "get_base_schema";
        ok: boolean;
        error?: string | undefined;
        tables?: {
            name: string;
            fields: {
                type: string;
                name: string;
                id: string;
                options?: Record<string, unknown> | undefined;
                description?: string | undefined;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
            views?: {
                type: string;
                name: string;
                id: string;
            }[] | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_table">;
        ok: z.ZodBoolean;
        table: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            primaryFieldId: z.ZodString;
            fields: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                name: z.ZodString;
                type: z.ZodEnum<["singleLineText", "multilineText", "richText", "email", "url", "phoneNumber", "number", "percent", "currency", "rating", "duration", "singleSelect", "multipleSelects", "singleCollaborator", "multipleCollaborators", "date", "dateTime", "checkbox", "multipleRecordLinks", "multipleAttachments", "barcode", "button", "formula", "createdTime", "lastModifiedTime", "createdBy", "lastModifiedBy", "autoNumber", "externalSyncSource", "count", "lookup", "rollup"]>;
            }, "strip", z.ZodTypeAny, {
                type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
                name: string;
                id: string;
            }, {
                type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
                name: string;
                id: string;
            }>, "many">;
        }, "strip", z.ZodTypeAny, {
            name: string;
            fields: {
                type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
                name: string;
                id: string;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }, {
            name: string;
            fields: {
                type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
                name: string;
                id: string;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        }>>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_table";
        ok: boolean;
        table?: {
            name: string;
            fields: {
                type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
                name: string;
                id: string;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        } | undefined;
    }, {
        success: boolean;
        operation: "create_table";
        ok: boolean;
        error?: string | undefined;
        table?: {
            name: string;
            fields: {
                type: "number" | "date" | "email" | "url" | "duration" | "count" | "createdTime" | "dateTime" | "rating" | "currency" | "singleLineText" | "multilineText" | "richText" | "phoneNumber" | "percent" | "singleSelect" | "multipleSelects" | "singleCollaborator" | "multipleCollaborators" | "checkbox" | "multipleRecordLinks" | "multipleAttachments" | "barcode" | "button" | "formula" | "lastModifiedTime" | "createdBy" | "lastModifiedBy" | "autoNumber" | "externalSyncSource" | "lookup" | "rollup";
                name: string;
                id: string;
            }[];
            id: string;
            primaryFieldId: string;
            description?: string | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_table">;
        ok: z.ZodBoolean;
        table: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            description?: string | undefined;
        }, {
            name: string;
            id: string;
            description?: string | undefined;
        }>>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_table";
        ok: boolean;
        table?: {
            name: string;
            id: string;
            description?: string | undefined;
        } | undefined;
    }, {
        success: boolean;
        operation: "update_table";
        ok: boolean;
        error?: string | undefined;
        table?: {
            name: string;
            id: string;
            description?: string | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_field">;
        ok: z.ZodBoolean;
        field: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        }, {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        }>>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_field";
        ok: boolean;
        field?: {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        } | undefined;
    }, {
        success: boolean;
        operation: "create_field";
        ok: boolean;
        error?: string | undefined;
        field?: {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_field">;
        ok: z.ZodBoolean;
        field: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        }, {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        }>>;
        error: z.ZodDefault<z.ZodString>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_field";
        ok: boolean;
        field?: {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        } | undefined;
    }, {
        success: boolean;
        operation: "update_field";
        ok: boolean;
        error?: string | undefined;
        field?: {
            type: string;
            name: string;
            id: string;
            description?: string | undefined;
        } | undefined;
    }>]>;
    static readonly shortDescription = "Airtable integration for managing records in bases and tables";
    static readonly longDescription = "\n    Comprehensive Airtable integration bubble for managing bases, tables, fields, and records.\n    Use cases:\n    - List records with filtering, sorting, and pagination\n    - Retrieve individual records by ID\n    - Create, update, and delete records\n    - List all accessible bases\n    - Get base schema with all tables and fields\n    - Create and update tables\n    - Create and update fields\n    - Support for all Airtable field types (text, number, attachments, links, etc.)\n    \n    Security Features:\n    - Personal Access Token authentication\n    - Parameter validation and sanitization\n    - Rate limiting awareness (5 requests per second per base)\n    - Comprehensive error handling\n  ";
    static readonly alias = "airtable";
    constructor(params?: T, context?: BubbleContext, instanceId?: string);
    protected performAction(context?: BubbleContext): Promise<Extract<AirtableResult, {
        operation: T['operation'];
    }>>;
    private listRecords;
    private getRecord;
    private createRecords;
    private updateRecords;
    private deleteRecords;
    private listBases;
    private getBaseSchema;
    /**
     * Normalizes field definitions by adding required default options for field types that need them.
     * This provides a better UX by auto-fixing common configuration issues.
     */
    private normalizeFieldOptions;
    private createTable;
    private updateTable;
    private createField;
    private updateField;
    private formatAirtableError;
    protected chooseCredential(): string | undefined;
    private makeAirtableApiCall;
}
export {};
//# sourceMappingURL=airtable.d.ts.map