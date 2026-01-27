import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const GoogleDriveBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"uploadFile">;
    fileName: z.ZodEffects<z.ZodString, string, string>;
    content: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
    mimeType: z.ZodOptional<z.ZodString>;
    parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    content: string | Buffer<ArrayBufferLike>;
    operation: "uploadFile";
    fileName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mimeType?: string | undefined;
    parents?: string[] | undefined;
}, {
    content: string | Buffer<ArrayBufferLike>;
    operation: "uploadFile";
    fileName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mimeType?: string | undefined;
    parents?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"downloadFile">;
    fileId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "downloadFile";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "downloadFile";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listFiles">;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    pageToken: z.ZodOptional<z.ZodString>;
    query: z.ZodOptional<z.ZodString>;
    orderBy: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listFiles";
    pageSize: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    pageToken?: string | undefined;
    orderBy?: string | undefined;
}, {
    operation: "listFiles";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    pageSize?: number | undefined;
    pageToken?: string | undefined;
    orderBy?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"searchFiles">;
    query: z.ZodString;
    pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    pageToken: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    operation: "searchFiles";
    pageSize: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pageToken?: string | undefined;
}, {
    query: string;
    operation: "searchFiles";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pageSize?: number | undefined;
    pageToken?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteFile">;
    fileId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteFile";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteFile";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createFolder">;
    folderName: z.ZodString;
    parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createFolder";
    folderName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    parents?: string[] | undefined;
}, {
    operation: "createFolder";
    folderName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    parents?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"shareFile">;
    fileId: z.ZodString;
    role: z.ZodEnum<["reader", "writer", "commenter", "owner"]>;
    type: z.ZodEnum<["user", "group", "anyone", "domain"]>;
    emailAddress: z.ZodOptional<z.ZodString>;
    allowFileDiscovery: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    type: "user" | "group" | "domain" | "anyone";
    operation: "shareFile";
    role: "reader" | "writer" | "commenter" | "owner";
    fileId: string;
    allowFileDiscovery: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    emailAddress?: string | undefined;
}, {
    type: "user" | "group" | "domain" | "anyone";
    operation: "shareFile";
    role: "reader" | "writer" | "commenter" | "owner";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    emailAddress?: string | undefined;
    allowFileDiscovery?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getFileInfo">;
    fileId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getFileInfo";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getFileInfo";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateFile">;
    fileId: z.ZodString;
    content: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
    mimeType: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    content: string | Buffer<ArrayBufferLike>;
    operation: "updateFile";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mimeType?: string | undefined;
}, {
    content: string | Buffer<ArrayBufferLike>;
    operation: "updateFile";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mimeType?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"copyFile">;
    fileId: z.ZodString;
    fileName: z.ZodString;
    parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "copyFile";
    fileName: string;
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    parents?: string[] | undefined;
}, {
    operation: "copyFile";
    fileName: string;
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    parents?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getPermissions">;
    fileId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getPermissions";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getPermissions";
    fileId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"revokeAccess">;
    fileId: z.ZodString;
    permissionId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "revokeAccess";
    fileId: string;
    permissionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "revokeAccess";
    fileId: string;
    permissionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateMetadata">;
    fileId: z.ZodString;
    fileName: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    starred: z.ZodOptional<z.ZodBoolean>;
    parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateMetadata";
    fileId: string;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fileName?: string | undefined;
    parents?: string[] | undefined;
    starred?: boolean | undefined;
}, {
    operation: "updateMetadata";
    fileId: string;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fileName?: string | undefined;
    parents?: string[] | undefined;
    starred?: boolean | undefined;
}>]>;
type GoogleDriveBubbleParams = z.input<typeof GoogleDriveBubbleParamsSchema>;
declare const GoogleDriveBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        fileId: z.ZodOptional<z.ZodString>;
        fileName: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        fileName?: string | undefined;
        fileId?: string | undefined;
    }, {
        operation: string;
        fileName?: string | undefined;
        fileId?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        fileName?: string | undefined;
        fileId?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        fileName?: string | undefined;
        fileId?: string | undefined;
    };
    data?: unknown;
}>;
type GoogleDriveBubbleResult = z.output<typeof GoogleDriveBubbleResultSchema>;
export declare class GoogleDriveBubble extends ServiceBubble<GoogleDriveBubbleParams, GoogleDriveBubbleResult> {
    static readonly service = "google-drive";
    static readonly authType: "oauth";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"uploadFile">;
        fileName: z.ZodEffects<z.ZodString, string, string>;
        content: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
        mimeType: z.ZodOptional<z.ZodString>;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        content: string | Buffer<ArrayBufferLike>;
        operation: "uploadFile";
        fileName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mimeType?: string | undefined;
        parents?: string[] | undefined;
    }, {
        content: string | Buffer<ArrayBufferLike>;
        operation: "uploadFile";
        fileName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mimeType?: string | undefined;
        parents?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"downloadFile">;
        fileId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "downloadFile";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "downloadFile";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listFiles">;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        pageToken: z.ZodOptional<z.ZodString>;
        query: z.ZodOptional<z.ZodString>;
        orderBy: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listFiles";
        pageSize: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        pageToken?: string | undefined;
        orderBy?: string | undefined;
    }, {
        operation: "listFiles";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        pageSize?: number | undefined;
        pageToken?: string | undefined;
        orderBy?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"searchFiles">;
        query: z.ZodString;
        pageSize: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        pageToken: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        operation: "searchFiles";
        pageSize: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pageToken?: string | undefined;
    }, {
        query: string;
        operation: "searchFiles";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pageSize?: number | undefined;
        pageToken?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteFile">;
        fileId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteFile";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteFile";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createFolder">;
        folderName: z.ZodString;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createFolder";
        folderName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        parents?: string[] | undefined;
    }, {
        operation: "createFolder";
        folderName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        parents?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"shareFile">;
        fileId: z.ZodString;
        role: z.ZodEnum<["reader", "writer", "commenter", "owner"]>;
        type: z.ZodEnum<["user", "group", "anyone", "domain"]>;
        emailAddress: z.ZodOptional<z.ZodString>;
        allowFileDiscovery: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        type: "user" | "group" | "domain" | "anyone";
        operation: "shareFile";
        role: "reader" | "writer" | "commenter" | "owner";
        fileId: string;
        allowFileDiscovery: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        emailAddress?: string | undefined;
    }, {
        type: "user" | "group" | "domain" | "anyone";
        operation: "shareFile";
        role: "reader" | "writer" | "commenter" | "owner";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        emailAddress?: string | undefined;
        allowFileDiscovery?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getFileInfo">;
        fileId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getFileInfo";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getFileInfo";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateFile">;
        fileId: z.ZodString;
        content: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
        mimeType: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        content: string | Buffer<ArrayBufferLike>;
        operation: "updateFile";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mimeType?: string | undefined;
    }, {
        content: string | Buffer<ArrayBufferLike>;
        operation: "updateFile";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mimeType?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"copyFile">;
        fileId: z.ZodString;
        fileName: z.ZodString;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "copyFile";
        fileName: string;
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        parents?: string[] | undefined;
    }, {
        operation: "copyFile";
        fileName: string;
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        parents?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getPermissions">;
        fileId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getPermissions";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getPermissions";
        fileId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"revokeAccess">;
        fileId: z.ZodString;
        permissionId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "revokeAccess";
        fileId: string;
        permissionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "revokeAccess";
        fileId: string;
        permissionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateMetadata">;
        fileId: z.ZodString;
        fileName: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        starred: z.ZodOptional<z.ZodBoolean>;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateMetadata";
        fileId: string;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fileName?: string | undefined;
        parents?: string[] | undefined;
        starred?: boolean | undefined;
    }, {
        operation: "updateMetadata";
        fileId: string;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fileName?: string | undefined;
        parents?: string[] | undefined;
        starred?: boolean | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            fileId: z.ZodOptional<z.ZodString>;
            fileName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            fileName?: string | undefined;
            fileId?: string | undefined;
        }, {
            operation: string;
            fileName?: string | undefined;
            fileId?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            fileName?: string | undefined;
            fileId?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            fileName?: string | undefined;
            fileId?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Cloud file storage and synchronization service";
    static readonly longDescription = "\n    Google Drive Bubble for cloud storage and file management.\n\n    Features:\n    - Upload and download files (up to 5GB)\n    - Create and manage folders\n    - Share files with specific permissions\n    - Search files by name or content\n    - File metadata management\n    - Copy and update operations\n    - Permission management\n    - Integration with Google Workspace\n    - OAuth2 authentication\n    - Rate limiting and quota management\n\n    Use cases:\n    - Document storage and backup\n    - File sharing and collaboration\n    - Automated report generation\n    - Content management\n    - Integration with other Google services\n\n    Security:\n    - All operations use OAuth2 tokens\n    - File size validation (max 5GB)\n    - Path traversal prevention\n    - Rate limiting (upload: 5/min, others: 50/min)\n    - Input sanitization and validation\n  ";
    static readonly alias = "drive";
    private accessToken;
    private baseUrl;
    private uploadUrl;
    private logger;
    private rateLimitTracker;
    constructor(params: GoogleDriveBubbleParams, context?: BubbleContext, instanceId?: string);
    /**
     * Check rate limit for an operation
     */
    private checkRateLimit;
    /**
     * Validate file size
     */
    private validateFileSize;
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getToken;
    protected performAction(context?: BubbleContext): Promise<GoogleDriveBubbleResult>;
    private makeRequest;
    private uploadFile;
    private downloadFile;
    private listFiles;
    private searchFiles;
    private deleteFile;
    private createFolder;
    private shareFile;
    private getFileInfo;
    private updateFile;
    private copyFile;
    /**
     * Get file permissions
     */
    private getPermissions;
    /**
     * Revoke access to a file
     */
    private revokeAccess;
    /**
     * Update file metadata
     */
    private updateMetadata;
}
export {};
//# sourceMappingURL=google-drive-bubble.d.ts.map