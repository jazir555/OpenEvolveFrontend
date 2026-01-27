import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const StorageParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"getUploadUrl">;
    bucketName: z.ZodString;
    fileName: z.ZodString;
    accountId: z.ZodOptional<z.ZodString>;
    region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    expirationMinutes: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    contentType: z.ZodOptional<z.ZodString>;
    userId: z.ZodEffects<z.ZodOptional<z.ZodString>, string | undefined, string | undefined>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getUploadUrl";
    bucketName: string;
    fileName: string;
    region: string;
    expirationMinutes: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    contentType?: string | undefined;
    userId?: string | undefined;
}, {
    operation: "getUploadUrl";
    bucketName: string;
    fileName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    region?: string | undefined;
    expirationMinutes?: number | undefined;
    contentType?: string | undefined;
    userId?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getFile">;
    bucketName: z.ZodString;
    fileName: z.ZodString;
    accountId: z.ZodOptional<z.ZodString>;
    region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    expirationMinutes: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    userId: z.ZodEffects<z.ZodOptional<z.ZodString>, string | undefined, string | undefined>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getFile";
    bucketName: string;
    fileName: string;
    region: string;
    expirationMinutes: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    userId?: string | undefined;
}, {
    operation: "getFile";
    bucketName: string;
    fileName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    region?: string | undefined;
    expirationMinutes?: number | undefined;
    userId?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteFile">;
    bucketName: z.ZodString;
    fileName: z.ZodString;
    accountId: z.ZodOptional<z.ZodString>;
    region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteFile";
    bucketName: string;
    fileName: string;
    region: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
}, {
    operation: "deleteFile";
    bucketName: string;
    fileName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    region?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateFile">;
    bucketName: z.ZodDefault<z.ZodString>;
    fileName: z.ZodString;
    accountId: z.ZodOptional<z.ZodString>;
    region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    contentType: z.ZodOptional<z.ZodString>;
    fileContent: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateFile";
    bucketName: string;
    fileName: string;
    region: string;
    fileContent: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    contentType?: string | undefined;
}, {
    operation: "updateFile";
    fileName: string;
    fileContent: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    bucketName?: string | undefined;
    accountId?: string | undefined;
    region?: string | undefined;
    contentType?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getMultipleUploadUrls">;
    bucketName: z.ZodString;
    pdfFileName: z.ZodString;
    pageCount: z.ZodNumber;
    accountId: z.ZodOptional<z.ZodString>;
    region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    expirationMinutes: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    userId: z.ZodEffects<z.ZodOptional<z.ZodString>, string | undefined, string | undefined>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getMultipleUploadUrls";
    bucketName: string;
    region: string;
    expirationMinutes: number;
    pdfFileName: string;
    pageCount: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    userId?: string | undefined;
}, {
    operation: "getMultipleUploadUrls";
    bucketName: string;
    pdfFileName: string;
    pageCount: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    accountId?: string | undefined;
    region?: string | undefined;
    expirationMinutes?: number | undefined;
    userId?: string | undefined;
}>]>;
declare const StorageResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"getUploadUrl">;
    success: z.ZodBoolean;
    uploadUrl: z.ZodOptional<z.ZodString>;
    fileName: z.ZodOptional<z.ZodString>;
    contentType: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "getUploadUrl";
    fileName?: string | undefined;
    contentType?: string | undefined;
    uploadUrl?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "getUploadUrl";
    fileName?: string | undefined;
    contentType?: string | undefined;
    uploadUrl?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getFile">;
    success: z.ZodBoolean;
    downloadUrl: z.ZodOptional<z.ZodString>;
    fileUrl: z.ZodOptional<z.ZodString>;
    fileName: z.ZodOptional<z.ZodString>;
    fileSize: z.ZodOptional<z.ZodNumber>;
    contentType: z.ZodOptional<z.ZodString>;
    lastModified: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "getFile";
    fileName?: string | undefined;
    contentType?: string | undefined;
    downloadUrl?: string | undefined;
    fileUrl?: string | undefined;
    fileSize?: number | undefined;
    lastModified?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "getFile";
    fileName?: string | undefined;
    contentType?: string | undefined;
    downloadUrl?: string | undefined;
    fileUrl?: string | undefined;
    fileSize?: number | undefined;
    lastModified?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteFile">;
    success: z.ZodBoolean;
    fileName: z.ZodOptional<z.ZodString>;
    deleted: z.ZodOptional<z.ZodBoolean>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "deleteFile";
    deleted?: boolean | undefined;
    fileName?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "deleteFile";
    deleted?: boolean | undefined;
    fileName?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateFile">;
    success: z.ZodBoolean;
    fileName: z.ZodOptional<z.ZodString>;
    updated: z.ZodOptional<z.ZodBoolean>;
    contentType: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "updateFile";
    updated?: boolean | undefined;
    fileName?: string | undefined;
    contentType?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "updateFile";
    updated?: boolean | undefined;
    fileName?: string | undefined;
    contentType?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getMultipleUploadUrls">;
    success: z.ZodBoolean;
    pdfUploadUrl: z.ZodOptional<z.ZodString>;
    pdfFileName: z.ZodOptional<z.ZodString>;
    pageUploadUrls: z.ZodOptional<z.ZodArray<z.ZodObject<{
        pageNumber: z.ZodNumber;
        uploadUrl: z.ZodString;
        fileName: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        fileName: string;
        uploadUrl: string;
        pageNumber: number;
    }, {
        fileName: string;
        uploadUrl: string;
        pageNumber: number;
    }>, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "getMultipleUploadUrls";
    pdfFileName?: string | undefined;
    pdfUploadUrl?: string | undefined;
    pageUploadUrls?: {
        fileName: string;
        uploadUrl: string;
        pageNumber: number;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "getMultipleUploadUrls";
    pdfFileName?: string | undefined;
    pdfUploadUrl?: string | undefined;
    pageUploadUrls?: {
        fileName: string;
        uploadUrl: string;
        pageNumber: number;
    }[] | undefined;
}>]>;
type StorageResult = z.output<typeof StorageResultSchema>;
type StorageParams = z.input<typeof StorageParamsSchema>;
export type StorageOperationResult<T extends StorageParams['operation']> = Extract<StorageResult, {
    operation: T;
}>;
export declare class StorageBubble<T extends StorageParams = StorageParams> extends ServiceBubble<T, Extract<StorageResult, {
    operation: T['operation'];
}>> {
    static readonly service = "cloudflare-r2";
    static readonly authType: "apikey";
    static readonly bubbleName = "storage";
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"getUploadUrl">;
        bucketName: z.ZodString;
        fileName: z.ZodString;
        accountId: z.ZodOptional<z.ZodString>;
        region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        expirationMinutes: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        contentType: z.ZodOptional<z.ZodString>;
        userId: z.ZodEffects<z.ZodOptional<z.ZodString>, string | undefined, string | undefined>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getUploadUrl";
        bucketName: string;
        fileName: string;
        region: string;
        expirationMinutes: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        contentType?: string | undefined;
        userId?: string | undefined;
    }, {
        operation: "getUploadUrl";
        bucketName: string;
        fileName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        region?: string | undefined;
        expirationMinutes?: number | undefined;
        contentType?: string | undefined;
        userId?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getFile">;
        bucketName: z.ZodString;
        fileName: z.ZodString;
        accountId: z.ZodOptional<z.ZodString>;
        region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        expirationMinutes: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        userId: z.ZodEffects<z.ZodOptional<z.ZodString>, string | undefined, string | undefined>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getFile";
        bucketName: string;
        fileName: string;
        region: string;
        expirationMinutes: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        userId?: string | undefined;
    }, {
        operation: "getFile";
        bucketName: string;
        fileName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        region?: string | undefined;
        expirationMinutes?: number | undefined;
        userId?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteFile">;
        bucketName: z.ZodString;
        fileName: z.ZodString;
        accountId: z.ZodOptional<z.ZodString>;
        region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteFile";
        bucketName: string;
        fileName: string;
        region: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
    }, {
        operation: "deleteFile";
        bucketName: string;
        fileName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        region?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateFile">;
        bucketName: z.ZodDefault<z.ZodString>;
        fileName: z.ZodString;
        accountId: z.ZodOptional<z.ZodString>;
        region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        contentType: z.ZodOptional<z.ZodString>;
        fileContent: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateFile";
        bucketName: string;
        fileName: string;
        region: string;
        fileContent: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        contentType?: string | undefined;
    }, {
        operation: "updateFile";
        fileName: string;
        fileContent: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        bucketName?: string | undefined;
        accountId?: string | undefined;
        region?: string | undefined;
        contentType?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getMultipleUploadUrls">;
        bucketName: z.ZodString;
        pdfFileName: z.ZodString;
        pageCount: z.ZodNumber;
        accountId: z.ZodOptional<z.ZodString>;
        region: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        expirationMinutes: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        userId: z.ZodEffects<z.ZodOptional<z.ZodString>, string | undefined, string | undefined>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getMultipleUploadUrls";
        bucketName: string;
        region: string;
        expirationMinutes: number;
        pdfFileName: string;
        pageCount: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        userId?: string | undefined;
    }, {
        operation: "getMultipleUploadUrls";
        bucketName: string;
        pdfFileName: string;
        pageCount: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        accountId?: string | undefined;
        region?: string | undefined;
        expirationMinutes?: number | undefined;
        userId?: string | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"getUploadUrl">;
        success: z.ZodBoolean;
        uploadUrl: z.ZodOptional<z.ZodString>;
        fileName: z.ZodOptional<z.ZodString>;
        contentType: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "getUploadUrl";
        fileName?: string | undefined;
        contentType?: string | undefined;
        uploadUrl?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "getUploadUrl";
        fileName?: string | undefined;
        contentType?: string | undefined;
        uploadUrl?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getFile">;
        success: z.ZodBoolean;
        downloadUrl: z.ZodOptional<z.ZodString>;
        fileUrl: z.ZodOptional<z.ZodString>;
        fileName: z.ZodOptional<z.ZodString>;
        fileSize: z.ZodOptional<z.ZodNumber>;
        contentType: z.ZodOptional<z.ZodString>;
        lastModified: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "getFile";
        fileName?: string | undefined;
        contentType?: string | undefined;
        downloadUrl?: string | undefined;
        fileUrl?: string | undefined;
        fileSize?: number | undefined;
        lastModified?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "getFile";
        fileName?: string | undefined;
        contentType?: string | undefined;
        downloadUrl?: string | undefined;
        fileUrl?: string | undefined;
        fileSize?: number | undefined;
        lastModified?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteFile">;
        success: z.ZodBoolean;
        fileName: z.ZodOptional<z.ZodString>;
        deleted: z.ZodOptional<z.ZodBoolean>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "deleteFile";
        deleted?: boolean | undefined;
        fileName?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "deleteFile";
        deleted?: boolean | undefined;
        fileName?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateFile">;
        success: z.ZodBoolean;
        fileName: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodBoolean>;
        contentType: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "updateFile";
        updated?: boolean | undefined;
        fileName?: string | undefined;
        contentType?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "updateFile";
        updated?: boolean | undefined;
        fileName?: string | undefined;
        contentType?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getMultipleUploadUrls">;
        success: z.ZodBoolean;
        pdfUploadUrl: z.ZodOptional<z.ZodString>;
        pdfFileName: z.ZodOptional<z.ZodString>;
        pageUploadUrls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            pageNumber: z.ZodNumber;
            uploadUrl: z.ZodString;
            fileName: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }, {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }>, "many">>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "getMultipleUploadUrls";
        pdfFileName?: string | undefined;
        pdfUploadUrl?: string | undefined;
        pageUploadUrls?: {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "getMultipleUploadUrls";
        pdfFileName?: string | undefined;
        pdfUploadUrl?: string | undefined;
        pageUploadUrls?: {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }[] | undefined;
    }>]>;
    static readonly shortDescription = "Cloudflare R2 storage operations for file management";
    static readonly longDescription = "\n    A comprehensive storage bubble for Cloudflare R2 operations.\n    Use cases:\n    - Generate presigned upload URLs for client-side file uploads\n    - Get secure download URLs for file retrieval with authentication  \n    - Delete files from R2 buckets\n    - Update/replace files in R2 buckets (supports base64 encoded content for binary files like images)\n    - Manage file access with time-limited URLs\n  ";
    static readonly alias = "r2";
    private s3Client;
    constructor(params?: T, context?: BubbleContext);
    protected chooseCredential(): string | undefined;
    private initializeS3Client;
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<Extract<StorageResult, {
        operation: T['operation'];
    }>>;
    private getUploadUrl;
    private getFile;
    private deleteFile;
    private updateFile;
    /**
     * Helper method to detect if a string is base64 encoded
     */
    private isBase64;
    private getMultipleUploadUrls;
}
export {};
//# sourceMappingURL=storage.d.ts.map