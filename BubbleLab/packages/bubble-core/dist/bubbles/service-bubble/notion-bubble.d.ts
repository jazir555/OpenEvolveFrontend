import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const NotionBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createPage">;
    parentPageId: z.ZodString;
    title: z.ZodString;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    icon: z.ZodOptional<z.ZodString>;
    cover: z.ZodOptional<z.ZodString>;
    children: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    operation: "createPage";
    parentPageId: string;
    properties?: Record<string, any> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    children?: any[] | undefined;
    icon?: string | undefined;
    cover?: string | undefined;
}, {
    title: string;
    operation: "createPage";
    parentPageId: string;
    properties?: Record<string, any> | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    children?: any[] | undefined;
    icon?: string | undefined;
    cover?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getPage">;
    pageId: z.ZodString;
    includeChildren: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getPage";
    pageId: string;
    includeChildren: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getPage";
    pageId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    includeChildren?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updatePage">;
    pageId: z.ZodString;
    properties: z.ZodRecord<z.ZodString, z.ZodAny>;
    archived: z.ZodOptional<z.ZodBoolean>;
    icon: z.ZodOptional<z.ZodString>;
    cover: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    properties: Record<string, any>;
    operation: "updatePage";
    pageId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    icon?: string | undefined;
    cover?: string | undefined;
    archived?: boolean | undefined;
}, {
    properties: Record<string, any>;
    operation: "updatePage";
    pageId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    icon?: string | undefined;
    cover?: string | undefined;
    archived?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deletePage">;
    pageId: z.ZodString;
    archived: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deletePage";
    archived: boolean;
    pageId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deletePage";
    pageId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    archived?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"queryDatabase">;
    databaseId: z.ZodString;
    filter: z.ZodOptional<z.ZodAny>;
    sorts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
    startCursor: z.ZodOptional<z.ZodString>;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "queryDatabase";
    pageSize: number;
    databaseId: string;
    filter?: any;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    sorts?: any[] | undefined;
    startCursor?: string | undefined;
}, {
    operation: "queryDatabase";
    databaseId: string;
    filter?: any;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    sorts?: any[] | undefined;
    pageSize?: number | undefined;
    startCursor?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createDatabaseEntry">;
    databaseId: z.ZodString;
    properties: z.ZodRecord<z.ZodString, z.ZodAny>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    properties: Record<string, any>;
    operation: "createDatabaseEntry";
    databaseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    properties: Record<string, any>;
    operation: "createDatabaseEntry";
    databaseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateDatabaseEntry">;
    pageId: z.ZodString;
    properties: z.ZodRecord<z.ZodString, z.ZodAny>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    properties: Record<string, any>;
    operation: "updateDatabaseEntry";
    pageId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    properties: Record<string, any>;
    operation: "updateDatabaseEntry";
    pageId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDatabase">;
    databaseId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getDatabase";
    databaseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getDatabase";
    databaseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"appendBlocks">;
    blockId: z.ZodString;
    blocks: z.ZodArray<z.ZodAny, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "appendBlocks";
    blocks: any[];
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "appendBlocks";
    blocks: any[];
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getBlocks">;
    blockId: z.ZodString;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    startCursor: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getBlocks";
    pageSize: number;
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    startCursor?: string | undefined;
}, {
    operation: "getBlocks";
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pageSize?: number | undefined;
    startCursor?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getBlock">;
    blockId: z.ZodString;
    includeChildren: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getBlock";
    includeChildren: boolean;
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getBlock";
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    includeChildren?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateBlock">;
    blockId: z.ZodString;
    type: z.ZodString;
    content: z.ZodAny;
    archived: z.ZodOptional<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    type: string;
    operation: "updateBlock";
    blockId: string;
    content?: any;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    archived?: boolean | undefined;
}, {
    type: string;
    operation: "updateBlock";
    blockId: string;
    content?: any;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    archived?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteBlock">;
    blockId: z.ZodString;
    archived: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteBlock";
    archived: boolean;
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteBlock";
    blockId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    archived?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"search">;
    query: z.ZodString;
    filter: z.ZodOptional<z.ZodObject<{
        value: z.ZodEnum<["page", "database"]>;
        property: z.ZodOptional<z.ZodEnum<["object"]>>;
    }, "strip", z.ZodTypeAny, {
        value: "page" | "database";
        property?: "object" | undefined;
    }, {
        value: "page" | "database";
        property?: "object" | undefined;
    }>>;
    sort: z.ZodOptional<z.ZodObject<{
        direction: z.ZodOptional<z.ZodEnum<["ascending", "descending"]>>;
        timestamp: z.ZodOptional<z.ZodEnum<["last_edited_time"]>>;
    }, "strip", z.ZodTypeAny, {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    }, {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    }>>;
    startCursor: z.ZodOptional<z.ZodString>;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    operation: "search";
    pageSize: number;
    sort?: {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    } | undefined;
    filter?: {
        value: "page" | "database";
        property?: "object" | undefined;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    startCursor?: string | undefined;
}, {
    query: string;
    operation: "search";
    sort?: {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    } | undefined;
    filter?: {
        value: "page" | "database";
        property?: "object" | undefined;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pageSize?: number | undefined;
    startCursor?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"searchPages">;
    query: z.ZodString;
    filter: z.ZodOptional<z.ZodObject<{
        value: z.ZodEnum<["page", "database"]>;
        property: z.ZodOptional<z.ZodEnum<["object"]>>;
    }, "strip", z.ZodTypeAny, {
        value: "page" | "database";
        property?: "object" | undefined;
    }, {
        value: "page" | "database";
        property?: "object" | undefined;
    }>>;
    sort: z.ZodOptional<z.ZodObject<{
        direction: z.ZodOptional<z.ZodEnum<["ascending", "descending"]>>;
        timestamp: z.ZodOptional<z.ZodEnum<["last_edited_time"]>>;
    }, "strip", z.ZodTypeAny, {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    }, {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    }>>;
    startCursor: z.ZodOptional<z.ZodString>;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    operation: "searchPages";
    pageSize: number;
    sort?: {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    } | undefined;
    filter?: {
        value: "page" | "database";
        property?: "object" | undefined;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    startCursor?: string | undefined;
}, {
    query: string;
    operation: "searchPages";
    sort?: {
        timestamp?: "last_edited_time" | undefined;
        direction?: "ascending" | "descending" | undefined;
    } | undefined;
    filter?: {
        value: "page" | "database";
        property?: "object" | undefined;
    } | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pageSize?: number | undefined;
    startCursor?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDatabaseEntries">;
    databaseId: z.ZodString;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    startCursor: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getDatabaseEntries";
    pageSize: number;
    databaseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    startCursor?: string | undefined;
}, {
    operation: "getDatabaseEntries";
    databaseId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pageSize?: number | undefined;
    startCursor?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createDatabase">;
    parentId: z.ZodString;
    title: z.ZodString;
    properties: z.ZodRecord<z.ZodString, z.ZodAny>;
    description: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
    icon: z.ZodOptional<z.ZodString>;
    cover: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    properties: Record<string, any>;
    title: string;
    operation: "createDatabase";
    parentId: string;
    description?: any[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    icon?: string | undefined;
    cover?: string | undefined;
}, {
    properties: Record<string, any>;
    title: string;
    operation: "createDatabase";
    parentId: string;
    description?: any[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    icon?: string | undefined;
    cover?: string | undefined;
}>]>;
type NotionBubbleParams = z.input<typeof NotionBubbleParamsSchema>;
declare const NotionBubbleResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createPage">;
    result: z.ZodObject<{
        pageId: z.ZodString;
        title: z.ZodOptional<z.ZodString>;
        url: z.ZodString;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        createdTime: z.ZodOptional<z.ZodString>;
        lastEditedTime: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    }, {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createPage";
    result: {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    };
}, {
    operation: "createPage";
    result: {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getPage">;
    result: z.ZodObject<{
        pageId: z.ZodString;
        title: z.ZodOptional<z.ZodString>;
        parent: z.ZodOptional<z.ZodAny>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        children: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        icon: z.ZodOptional<z.ZodString>;
        cover: z.ZodOptional<z.ZodString>;
        createdTime: z.ZodOptional<z.ZodString>;
        lastEditedTime: z.ZodOptional<z.ZodString>;
        archived: z.ZodOptional<z.ZodBoolean>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        parent?: any;
        children?: any[] | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    }, {
        error: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        parent?: any;
        children?: any[] | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getPage";
    result: {
        error: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        parent?: any;
        children?: any[] | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    };
}, {
    operation: "getPage";
    result: {
        error: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        parent?: any;
        children?: any[] | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updatePage">;
    result: z.ZodObject<{
        pageId: z.ZodString;
        title: z.ZodOptional<z.ZodString>;
        url: z.ZodString;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        createdTime: z.ZodOptional<z.ZodString>;
        lastEditedTime: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    }, {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "updatePage";
    result: {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    };
}, {
    operation: "updatePage";
    result: {
        error: string;
        url: string;
        success: boolean;
        pageId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
        createdTime?: string | undefined;
        lastEditedTime?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deletePage">;
    result: z.ZodObject<{
        pageId: z.ZodString;
        archived: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        archived: boolean;
        pageId: string;
    }, {
        error: string;
        success: boolean;
        archived: boolean;
        pageId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deletePage";
    result: {
        error: string;
        success: boolean;
        archived: boolean;
        pageId: string;
    };
}, {
    operation: "deletePage";
    result: {
        error: string;
        success: boolean;
        archived: boolean;
        pageId: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"queryDatabase">;
    result: z.ZodObject<{
        results: z.ZodArray<z.ZodAny, "many">;
        nextCursor: z.ZodOptional<z.ZodString>;
        hasMore: z.ZodBoolean;
        totalCount: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    }, {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "queryDatabase";
    result: {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    };
}, {
    operation: "queryDatabase";
    result: {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createDatabase">;
    result: z.ZodObject<{
        databaseId: z.ZodString;
        title: z.ZodOptional<z.ZodString>;
        url: z.ZodString;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
    }, {
        error: string;
        url: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createDatabase";
    result: {
        error: string;
        url: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
    };
}, {
    operation: "createDatabase";
    result: {
        error: string;
        url: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        title?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"appendBlock">;
    result: z.ZodObject<{
        blockId: z.ZodString;
        appendedBlocks: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        blockId: string;
        appendedBlocks: number;
    }, {
        error: string;
        success: boolean;
        blockId: string;
        appendedBlocks: number;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "appendBlock";
    result: {
        error: string;
        success: boolean;
        blockId: string;
        appendedBlocks: number;
    };
}, {
    operation: "appendBlock";
    result: {
        error: string;
        success: boolean;
        blockId: string;
        appendedBlocks: number;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getBlock">;
    result: z.ZodObject<{
        blockId: z.ZodString;
        type: z.ZodString;
        content: z.ZodOptional<z.ZodAny>;
        children: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        createdTime: z.ZodOptional<z.ZodString>;
        lastEditedTime: z.ZodOptional<z.ZodString>;
        archived: z.ZodOptional<z.ZodBoolean>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        content?: any;
        createdTime?: string | undefined;
        children?: any[] | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    }, {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        content?: any;
        createdTime?: string | undefined;
        children?: any[] | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getBlock";
    result: {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        content?: any;
        createdTime?: string | undefined;
        children?: any[] | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    };
}, {
    operation: "getBlock";
    result: {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        content?: any;
        createdTime?: string | undefined;
        children?: any[] | undefined;
        archived?: boolean | undefined;
        lastEditedTime?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateBlock">;
    result: z.ZodObject<{
        blockId: z.ZodString;
        type: z.ZodString;
        content: z.ZodOptional<z.ZodAny>;
        hasChildren: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        hasChildren: boolean;
        content?: any;
    }, {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        hasChildren: boolean;
        content?: any;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "updateBlock";
    result: {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        hasChildren: boolean;
        content?: any;
    };
}, {
    operation: "updateBlock";
    result: {
        error: string;
        type: string;
        success: boolean;
        blockId: string;
        hasChildren: boolean;
        content?: any;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteBlock">;
    result: z.ZodObject<{
        blockId: z.ZodString;
        archived: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        archived: boolean;
        blockId: string;
    }, {
        error: string;
        success: boolean;
        archived: boolean;
        blockId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteBlock";
    result: {
        error: string;
        success: boolean;
        archived: boolean;
        blockId: string;
    };
}, {
    operation: "deleteBlock";
    result: {
        error: string;
        success: boolean;
        archived: boolean;
        blockId: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"searchPages">;
    result: z.ZodObject<{
        results: z.ZodArray<z.ZodAny, "many">;
        nextCursor: z.ZodOptional<z.ZodString>;
        hasMore: z.ZodBoolean;
        totalCount: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    }, {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "searchPages";
    result: {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    };
}, {
    operation: "searchPages";
    result: {
        error: string;
        success: boolean;
        results: any[];
        totalCount: number;
        hasMore: boolean;
        nextCursor?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDatabase">;
    result: z.ZodObject<{
        databaseId: z.ZodString;
        title: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        parent: z.ZodOptional<z.ZodAny>;
        icon: z.ZodOptional<z.ZodString>;
        cover: z.ZodOptional<z.ZodString>;
        url: z.ZodOptional<z.ZodString>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        description?: any[] | undefined;
        title?: string | undefined;
        url?: string | undefined;
        parent?: any;
        icon?: string | undefined;
        cover?: string | undefined;
    }, {
        error: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        description?: any[] | undefined;
        title?: string | undefined;
        url?: string | undefined;
        parent?: any;
        icon?: string | undefined;
        cover?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getDatabase";
    result: {
        error: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        description?: any[] | undefined;
        title?: string | undefined;
        url?: string | undefined;
        parent?: any;
        icon?: string | undefined;
        cover?: string | undefined;
    };
}, {
    operation: "getDatabase";
    result: {
        error: string;
        success: boolean;
        databaseId: string;
        properties?: Record<string, any> | undefined;
        description?: any[] | undefined;
        title?: string | undefined;
        url?: string | undefined;
        parent?: any;
        icon?: string | undefined;
        cover?: string | undefined;
    };
}>]>;
type NotionBubbleResult = z.output<typeof NotionBubbleResultSchema>;
export declare class NotionBubble<T extends NotionBubbleParams = NotionBubbleParams> extends ServiceBubble<T, any> {
    static readonly type: "service";
    static readonly service = "notion";
    static readonly authType: "apikey";
    static readonly bubbleName = "notion";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createPage">;
        parentPageId: z.ZodString;
        title: z.ZodString;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        icon: z.ZodOptional<z.ZodString>;
        cover: z.ZodOptional<z.ZodString>;
        children: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        operation: "createPage";
        parentPageId: string;
        properties?: Record<string, any> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        children?: any[] | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
    }, {
        title: string;
        operation: "createPage";
        parentPageId: string;
        properties?: Record<string, any> | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        children?: any[] | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getPage">;
        pageId: z.ZodString;
        includeChildren: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getPage";
        pageId: string;
        includeChildren: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getPage";
        pageId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        includeChildren?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updatePage">;
        pageId: z.ZodString;
        properties: z.ZodRecord<z.ZodString, z.ZodAny>;
        archived: z.ZodOptional<z.ZodBoolean>;
        icon: z.ZodOptional<z.ZodString>;
        cover: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        properties: Record<string, any>;
        operation: "updatePage";
        pageId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
        archived?: boolean | undefined;
    }, {
        properties: Record<string, any>;
        operation: "updatePage";
        pageId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
        archived?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deletePage">;
        pageId: z.ZodString;
        archived: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deletePage";
        archived: boolean;
        pageId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deletePage";
        pageId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        archived?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"queryDatabase">;
        databaseId: z.ZodString;
        filter: z.ZodOptional<z.ZodAny>;
        sorts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        startCursor: z.ZodOptional<z.ZodString>;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "queryDatabase";
        pageSize: number;
        databaseId: string;
        filter?: any;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        sorts?: any[] | undefined;
        startCursor?: string | undefined;
    }, {
        operation: "queryDatabase";
        databaseId: string;
        filter?: any;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        sorts?: any[] | undefined;
        pageSize?: number | undefined;
        startCursor?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createDatabaseEntry">;
        databaseId: z.ZodString;
        properties: z.ZodRecord<z.ZodString, z.ZodAny>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        properties: Record<string, any>;
        operation: "createDatabaseEntry";
        databaseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        properties: Record<string, any>;
        operation: "createDatabaseEntry";
        databaseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateDatabaseEntry">;
        pageId: z.ZodString;
        properties: z.ZodRecord<z.ZodString, z.ZodAny>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        properties: Record<string, any>;
        operation: "updateDatabaseEntry";
        pageId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        properties: Record<string, any>;
        operation: "updateDatabaseEntry";
        pageId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDatabase">;
        databaseId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getDatabase";
        databaseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getDatabase";
        databaseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"appendBlocks">;
        blockId: z.ZodString;
        blocks: z.ZodArray<z.ZodAny, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "appendBlocks";
        blocks: any[];
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "appendBlocks";
        blocks: any[];
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getBlocks">;
        blockId: z.ZodString;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        startCursor: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getBlocks";
        pageSize: number;
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        startCursor?: string | undefined;
    }, {
        operation: "getBlocks";
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pageSize?: number | undefined;
        startCursor?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getBlock">;
        blockId: z.ZodString;
        includeChildren: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getBlock";
        includeChildren: boolean;
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getBlock";
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        includeChildren?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateBlock">;
        blockId: z.ZodString;
        type: z.ZodString;
        content: z.ZodAny;
        archived: z.ZodOptional<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        operation: "updateBlock";
        blockId: string;
        content?: any;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        archived?: boolean | undefined;
    }, {
        type: string;
        operation: "updateBlock";
        blockId: string;
        content?: any;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        archived?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteBlock">;
        blockId: z.ZodString;
        archived: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteBlock";
        archived: boolean;
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteBlock";
        blockId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        archived?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"search">;
        query: z.ZodString;
        filter: z.ZodOptional<z.ZodObject<{
            value: z.ZodEnum<["page", "database"]>;
            property: z.ZodOptional<z.ZodEnum<["object"]>>;
        }, "strip", z.ZodTypeAny, {
            value: "page" | "database";
            property?: "object" | undefined;
        }, {
            value: "page" | "database";
            property?: "object" | undefined;
        }>>;
        sort: z.ZodOptional<z.ZodObject<{
            direction: z.ZodOptional<z.ZodEnum<["ascending", "descending"]>>;
            timestamp: z.ZodOptional<z.ZodEnum<["last_edited_time"]>>;
        }, "strip", z.ZodTypeAny, {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        }, {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        }>>;
        startCursor: z.ZodOptional<z.ZodString>;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        operation: "search";
        pageSize: number;
        sort?: {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        } | undefined;
        filter?: {
            value: "page" | "database";
            property?: "object" | undefined;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        startCursor?: string | undefined;
    }, {
        query: string;
        operation: "search";
        sort?: {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        } | undefined;
        filter?: {
            value: "page" | "database";
            property?: "object" | undefined;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pageSize?: number | undefined;
        startCursor?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"searchPages">;
        query: z.ZodString;
        filter: z.ZodOptional<z.ZodObject<{
            value: z.ZodEnum<["page", "database"]>;
            property: z.ZodOptional<z.ZodEnum<["object"]>>;
        }, "strip", z.ZodTypeAny, {
            value: "page" | "database";
            property?: "object" | undefined;
        }, {
            value: "page" | "database";
            property?: "object" | undefined;
        }>>;
        sort: z.ZodOptional<z.ZodObject<{
            direction: z.ZodOptional<z.ZodEnum<["ascending", "descending"]>>;
            timestamp: z.ZodOptional<z.ZodEnum<["last_edited_time"]>>;
        }, "strip", z.ZodTypeAny, {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        }, {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        }>>;
        startCursor: z.ZodOptional<z.ZodString>;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        operation: "searchPages";
        pageSize: number;
        sort?: {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        } | undefined;
        filter?: {
            value: "page" | "database";
            property?: "object" | undefined;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        startCursor?: string | undefined;
    }, {
        query: string;
        operation: "searchPages";
        sort?: {
            timestamp?: "last_edited_time" | undefined;
            direction?: "ascending" | "descending" | undefined;
        } | undefined;
        filter?: {
            value: "page" | "database";
            property?: "object" | undefined;
        } | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pageSize?: number | undefined;
        startCursor?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDatabaseEntries">;
        databaseId: z.ZodString;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        startCursor: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getDatabaseEntries";
        pageSize: number;
        databaseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        startCursor?: string | undefined;
    }, {
        operation: "getDatabaseEntries";
        databaseId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pageSize?: number | undefined;
        startCursor?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createDatabase">;
        parentId: z.ZodString;
        title: z.ZodString;
        properties: z.ZodRecord<z.ZodString, z.ZodAny>;
        description: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        icon: z.ZodOptional<z.ZodString>;
        cover: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        properties: Record<string, any>;
        title: string;
        operation: "createDatabase";
        parentId: string;
        description?: any[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
    }, {
        properties: Record<string, any>;
        title: string;
        operation: "createDatabase";
        parentId: string;
        description?: any[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        icon?: string | undefined;
        cover?: string | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createPage">;
        result: z.ZodObject<{
            pageId: z.ZodString;
            title: z.ZodOptional<z.ZodString>;
            url: z.ZodString;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            createdTime: z.ZodOptional<z.ZodString>;
            lastEditedTime: z.ZodOptional<z.ZodString>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        }, {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createPage";
        result: {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        };
    }, {
        operation: "createPage";
        result: {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getPage">;
        result: z.ZodObject<{
            pageId: z.ZodString;
            title: z.ZodOptional<z.ZodString>;
            parent: z.ZodOptional<z.ZodAny>;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            children: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            icon: z.ZodOptional<z.ZodString>;
            cover: z.ZodOptional<z.ZodString>;
            createdTime: z.ZodOptional<z.ZodString>;
            lastEditedTime: z.ZodOptional<z.ZodString>;
            archived: z.ZodOptional<z.ZodBoolean>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            parent?: any;
            children?: any[] | undefined;
            icon?: string | undefined;
            cover?: string | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        }, {
            error: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            parent?: any;
            children?: any[] | undefined;
            icon?: string | undefined;
            cover?: string | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getPage";
        result: {
            error: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            parent?: any;
            children?: any[] | undefined;
            icon?: string | undefined;
            cover?: string | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        };
    }, {
        operation: "getPage";
        result: {
            error: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            parent?: any;
            children?: any[] | undefined;
            icon?: string | undefined;
            cover?: string | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updatePage">;
        result: z.ZodObject<{
            pageId: z.ZodString;
            title: z.ZodOptional<z.ZodString>;
            url: z.ZodString;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            createdTime: z.ZodOptional<z.ZodString>;
            lastEditedTime: z.ZodOptional<z.ZodString>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        }, {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "updatePage";
        result: {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        };
    }, {
        operation: "updatePage";
        result: {
            error: string;
            url: string;
            success: boolean;
            pageId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
            createdTime?: string | undefined;
            lastEditedTime?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deletePage">;
        result: z.ZodObject<{
            pageId: z.ZodString;
            archived: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            archived: boolean;
            pageId: string;
        }, {
            error: string;
            success: boolean;
            archived: boolean;
            pageId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deletePage";
        result: {
            error: string;
            success: boolean;
            archived: boolean;
            pageId: string;
        };
    }, {
        operation: "deletePage";
        result: {
            error: string;
            success: boolean;
            archived: boolean;
            pageId: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"queryDatabase">;
        result: z.ZodObject<{
            results: z.ZodArray<z.ZodAny, "many">;
            nextCursor: z.ZodOptional<z.ZodString>;
            hasMore: z.ZodBoolean;
            totalCount: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        }, {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "queryDatabase";
        result: {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        };
    }, {
        operation: "queryDatabase";
        result: {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createDatabase">;
        result: z.ZodObject<{
            databaseId: z.ZodString;
            title: z.ZodOptional<z.ZodString>;
            url: z.ZodString;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            url: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
        }, {
            error: string;
            url: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createDatabase";
        result: {
            error: string;
            url: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
        };
    }, {
        operation: "createDatabase";
        result: {
            error: string;
            url: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            title?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"appendBlock">;
        result: z.ZodObject<{
            blockId: z.ZodString;
            appendedBlocks: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            blockId: string;
            appendedBlocks: number;
        }, {
            error: string;
            success: boolean;
            blockId: string;
            appendedBlocks: number;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "appendBlock";
        result: {
            error: string;
            success: boolean;
            blockId: string;
            appendedBlocks: number;
        };
    }, {
        operation: "appendBlock";
        result: {
            error: string;
            success: boolean;
            blockId: string;
            appendedBlocks: number;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getBlock">;
        result: z.ZodObject<{
            blockId: z.ZodString;
            type: z.ZodString;
            content: z.ZodOptional<z.ZodAny>;
            children: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            createdTime: z.ZodOptional<z.ZodString>;
            lastEditedTime: z.ZodOptional<z.ZodString>;
            archived: z.ZodOptional<z.ZodBoolean>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            content?: any;
            createdTime?: string | undefined;
            children?: any[] | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        }, {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            content?: any;
            createdTime?: string | undefined;
            children?: any[] | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getBlock";
        result: {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            content?: any;
            createdTime?: string | undefined;
            children?: any[] | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        };
    }, {
        operation: "getBlock";
        result: {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            content?: any;
            createdTime?: string | undefined;
            children?: any[] | undefined;
            archived?: boolean | undefined;
            lastEditedTime?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateBlock">;
        result: z.ZodObject<{
            blockId: z.ZodString;
            type: z.ZodString;
            content: z.ZodOptional<z.ZodAny>;
            hasChildren: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            hasChildren: boolean;
            content?: any;
        }, {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            hasChildren: boolean;
            content?: any;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateBlock";
        result: {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            hasChildren: boolean;
            content?: any;
        };
    }, {
        operation: "updateBlock";
        result: {
            error: string;
            type: string;
            success: boolean;
            blockId: string;
            hasChildren: boolean;
            content?: any;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteBlock">;
        result: z.ZodObject<{
            blockId: z.ZodString;
            archived: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            archived: boolean;
            blockId: string;
        }, {
            error: string;
            success: boolean;
            archived: boolean;
            blockId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteBlock";
        result: {
            error: string;
            success: boolean;
            archived: boolean;
            blockId: string;
        };
    }, {
        operation: "deleteBlock";
        result: {
            error: string;
            success: boolean;
            archived: boolean;
            blockId: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"searchPages">;
        result: z.ZodObject<{
            results: z.ZodArray<z.ZodAny, "many">;
            nextCursor: z.ZodOptional<z.ZodString>;
            hasMore: z.ZodBoolean;
            totalCount: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        }, {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "searchPages";
        result: {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        };
    }, {
        operation: "searchPages";
        result: {
            error: string;
            success: boolean;
            results: any[];
            totalCount: number;
            hasMore: boolean;
            nextCursor?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDatabase">;
        result: z.ZodObject<{
            databaseId: z.ZodString;
            title: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            parent: z.ZodOptional<z.ZodAny>;
            icon: z.ZodOptional<z.ZodString>;
            cover: z.ZodOptional<z.ZodString>;
            url: z.ZodOptional<z.ZodString>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            description?: any[] | undefined;
            title?: string | undefined;
            url?: string | undefined;
            parent?: any;
            icon?: string | undefined;
            cover?: string | undefined;
        }, {
            error: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            description?: any[] | undefined;
            title?: string | undefined;
            url?: string | undefined;
            parent?: any;
            icon?: string | undefined;
            cover?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getDatabase";
        result: {
            error: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            description?: any[] | undefined;
            title?: string | undefined;
            url?: string | undefined;
            parent?: any;
            icon?: string | undefined;
            cover?: string | undefined;
        };
    }, {
        operation: "getDatabase";
        result: {
            error: string;
            success: boolean;
            databaseId: string;
            properties?: Record<string, any> | undefined;
            description?: any[] | undefined;
            title?: string | undefined;
            url?: string | undefined;
            parent?: any;
            icon?: string | undefined;
            cover?: string | undefined;
        };
    }>]>;
    static readonly shortDescription = "Production-ready Notion integration for pages, databases, and blocks";
    static readonly longDescription = "\n    Comprehensive Notion service bubble for all workspace operations.\n\n    Operations:\n    1. createPage - Create new pages with properties and content\n    2. getPage - Retrieve page information and content blocks\n    3. updatePage - Update page properties, icon, and cover\n    4. deletePage - Archive or delete pages\n    5. queryDatabase - Query databases with filters and sorting\n    6. createDatabaseEntry - Add entry to database\n    7. updateDatabaseEntry - Update database entry\n    8. getDatabase - Get database schema and configuration\n    9. appendBlocks - Append content blocks to pages\n    10. getBlocks - Get child blocks\n    11. getBlock - Get block content and children\n    12. updateBlock - Update block content\n    13. deleteBlock - Archive or delete blocks\n    14. search - Search across workspace\n    15. searchPages - Legacy search operation\n    16. getDatabaseEntries - List all entries with pagination\n    17. createDatabase - Create new databases with custom schemas\n\n    Features:\n    - Full page and database CRUD\n    - Rich block content support\n    - Property management\n    - Database querying with filters\n    - Search functionality\n    - Rate limiting (3 req/sec)\n    - Input validation and sanitization\n    - Resilience patterns with retry\n  ";
    static readonly alias = "notion";
    private client;
    constructor(params: T, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<NotionBubbleResult, {
        operation: T['operation'];
    }>>;
    private createPage;
    private getPage;
    private updatePage;
    private deletePage;
    private queryDatabase;
    private createDatabase;
    private appendBlocks;
    private getBlocks;
    private getBlock;
    private updateBlock;
    private deleteBlock;
    private searchPages;
    private getDatabase;
    private search;
    private createDatabaseEntry;
    private updateDatabaseEntry;
    private getDatabaseEntries;
    private errorResult;
}
export {};
//# sourceMappingURL=notion-bubble.d.ts.map