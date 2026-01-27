import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const GoogleDriveParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"upload_file">;
    name: z.ZodString;
    content: z.ZodString;
    mimeType: z.ZodOptional<z.ZodString>;
    parent_folder_id: z.ZodOptional<z.ZodString>;
    convert_to_google_docs: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    content: string;
    operation: "upload_file";
    convert_to_google_docs: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mimeType?: string | undefined;
    parent_folder_id?: string | undefined;
}, {
    name: string;
    content: string;
    operation: "upload_file";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mimeType?: string | undefined;
    parent_folder_id?: string | undefined;
    convert_to_google_docs?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"download_file">;
    file_id: z.ZodString;
    export_format: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "download_file";
    file_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    export_format?: string | undefined;
}, {
    operation: "download_file";
    file_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    export_format?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_files">;
    folder_id: z.ZodOptional<z.ZodString>;
    query: z.ZodOptional<z.ZodString>;
    max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    include_folders: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    order_by: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_files";
    max_results: number;
    include_folders: boolean;
    order_by: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    folder_id?: string | undefined;
}, {
    operation: "list_files";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    folder_id?: string | undefined;
    max_results?: number | undefined;
    include_folders?: boolean | undefined;
    order_by?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_folder">;
    name: z.ZodString;
    parent_folder_id: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "create_folder";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    parent_folder_id?: string | undefined;
}, {
    name: string;
    operation: "create_folder";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    parent_folder_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_file">;
    file_id: z.ZodString;
    permanent: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_file";
    file_id: string;
    permanent: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_file";
    file_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    permanent?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_file_info">;
    file_id: z.ZodString;
    include_permissions: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_file_info";
    file_id: string;
    include_permissions: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_file_info";
    file_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    include_permissions?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"share_file">;
    file_id: z.ZodString;
    email_address: z.ZodOptional<z.ZodString>;
    role: z.ZodDefault<z.ZodOptional<z.ZodEnum<["reader", "writer", "commenter", "owner"]>>>;
    type: z.ZodDefault<z.ZodOptional<z.ZodEnum<["user", "group", "domain", "anyone"]>>>;
    send_notification: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    type: "user" | "group" | "domain" | "anyone";
    operation: "share_file";
    file_id: string;
    role: "reader" | "writer" | "commenter" | "owner";
    send_notification: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    email_address?: string | undefined;
}, {
    operation: "share_file";
    file_id: string;
    type?: "user" | "group" | "domain" | "anyone" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    email_address?: string | undefined;
    role?: "reader" | "writer" | "commenter" | "owner" | undefined;
    send_notification?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"move_file">;
    file_id: z.ZodString;
    new_parent_folder_id: z.ZodOptional<z.ZodString>;
    remove_parent_folder_id: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "move_file";
    file_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    new_parent_folder_id?: string | undefined;
    remove_parent_folder_id?: string | undefined;
}, {
    operation: "move_file";
    file_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    new_parent_folder_id?: string | undefined;
    remove_parent_folder_id?: string | undefined;
}>]>;
declare const GoogleDriveResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"upload_file">;
    success: z.ZodBoolean;
    file: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        mimeType: z.ZodString;
        size: z.ZodOptional<z.ZodString>;
        createdTime: z.ZodOptional<z.ZodString>;
        modifiedTime: z.ZodOptional<z.ZodString>;
        webViewLink: z.ZodOptional<z.ZodString>;
        webContentLink: z.ZodOptional<z.ZodString>;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        shared: z.ZodOptional<z.ZodBoolean>;
        owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
            displayName: z.ZodOptional<z.ZodString>;
            emailAddress: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "upload_file";
    file?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "upload_file";
    file?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"download_file">;
    success: z.ZodBoolean;
    content: z.ZodOptional<z.ZodString>;
    filename: z.ZodOptional<z.ZodString>;
    mimeType: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "download_file";
    content?: string | undefined;
    mimeType?: string | undefined;
    filename?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "download_file";
    content?: string | undefined;
    mimeType?: string | undefined;
    filename?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_files">;
    success: z.ZodBoolean;
    files: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        mimeType: z.ZodString;
        size: z.ZodOptional<z.ZodString>;
        createdTime: z.ZodOptional<z.ZodString>;
        modifiedTime: z.ZodOptional<z.ZodString>;
        webViewLink: z.ZodOptional<z.ZodString>;
        webContentLink: z.ZodOptional<z.ZodString>;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        shared: z.ZodOptional<z.ZodBoolean>;
        owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
            displayName: z.ZodOptional<z.ZodString>;
            emailAddress: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }>, "many">>;
    total_count: z.ZodOptional<z.ZodNumber>;
    next_page_token: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_files";
    files?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }[] | undefined;
    total_count?: number | undefined;
    next_page_token?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_files";
    files?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }[] | undefined;
    total_count?: number | undefined;
    next_page_token?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_folder">;
    success: z.ZodBoolean;
    folder: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        webViewLink: z.ZodOptional<z.ZodString>;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        webViewLink?: string | undefined;
        parents?: string[] | undefined;
    }, {
        name: string;
        id: string;
        webViewLink?: string | undefined;
        parents?: string[] | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_folder";
    folder?: {
        name: string;
        id: string;
        webViewLink?: string | undefined;
        parents?: string[] | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_folder";
    folder?: {
        name: string;
        id: string;
        webViewLink?: string | undefined;
        parents?: string[] | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_file">;
    success: z.ZodBoolean;
    deleted_file_id: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_file";
    deleted_file_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_file";
    deleted_file_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_file_info">;
    success: z.ZodBoolean;
    file: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        mimeType: z.ZodString;
        size: z.ZodOptional<z.ZodString>;
        createdTime: z.ZodOptional<z.ZodString>;
        modifiedTime: z.ZodOptional<z.ZodString>;
        webViewLink: z.ZodOptional<z.ZodString>;
        webContentLink: z.ZodOptional<z.ZodString>;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        shared: z.ZodOptional<z.ZodBoolean>;
        owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
            displayName: z.ZodOptional<z.ZodString>;
            emailAddress: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }>>;
    permissions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        type: z.ZodString;
        role: z.ZodString;
        emailAddress: z.ZodOptional<z.ZodString>;
        displayName: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        id: string;
        role: string;
        displayName?: string | undefined;
        emailAddress?: string | undefined;
    }, {
        type: string;
        id: string;
        role: string;
        displayName?: string | undefined;
        emailAddress?: string | undefined;
    }>, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_file_info";
    file?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    } | undefined;
    permissions?: {
        type: string;
        id: string;
        role: string;
        displayName?: string | undefined;
        emailAddress?: string | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_file_info";
    file?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    } | undefined;
    permissions?: {
        type: string;
        id: string;
        role: string;
        displayName?: string | undefined;
        emailAddress?: string | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"share_file">;
    success: z.ZodBoolean;
    permission_id: z.ZodOptional<z.ZodString>;
    share_link: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "share_file";
    permission_id?: string | undefined;
    share_link?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "share_file";
    permission_id?: string | undefined;
    share_link?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"move_file">;
    success: z.ZodBoolean;
    file: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        mimeType: z.ZodString;
        size: z.ZodOptional<z.ZodString>;
        createdTime: z.ZodOptional<z.ZodString>;
        modifiedTime: z.ZodOptional<z.ZodString>;
        webViewLink: z.ZodOptional<z.ZodString>;
        webContentLink: z.ZodOptional<z.ZodString>;
        parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        shared: z.ZodOptional<z.ZodBoolean>;
        owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
            displayName: z.ZodOptional<z.ZodString>;
            emailAddress: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }, {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }, {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "move_file";
    file?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "move_file";
    file?: {
        name: string;
        mimeType: string;
        id: string;
        size?: string | undefined;
        createdTime?: string | undefined;
        modifiedTime?: string | undefined;
        webViewLink?: string | undefined;
        webContentLink?: string | undefined;
        parents?: string[] | undefined;
        shared?: boolean | undefined;
        owners?: {
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    } | undefined;
}>]>;
type GoogleDriveResult = z.output<typeof GoogleDriveResultSchema>;
type GoogleDriveParams = z.input<typeof GoogleDriveParamsSchema>;
export type GoogleDriveOperationResult<T extends GoogleDriveParams['operation']> = Extract<GoogleDriveResult, {
    operation: T;
}>;
export type GoogleDriveParamsInput = z.input<typeof GoogleDriveParamsSchema>;
/**
 * Google Drive Bubble - Complete Service Bubble Implementation
 *
 * Provides comprehensive integration with Google Drive API for file and folder management,
 * including upload, download, search, sharing, and organization capabilities.
 *
 * @template T - Google Drive bubble parameters type
 */
export declare class GoogleDriveBubble<T extends GoogleDriveParams = GoogleDriveParams> extends ServiceBubble<T, Extract<GoogleDriveResult, {
    operation: T['operation'];
}>> {
    static readonly type: "service";
    static readonly service = "google-drive";
    static readonly authType: "oauth";
    static readonly bubbleName = "google-drive";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"upload_file">;
        name: z.ZodString;
        content: z.ZodString;
        mimeType: z.ZodOptional<z.ZodString>;
        parent_folder_id: z.ZodOptional<z.ZodString>;
        convert_to_google_docs: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        content: string;
        operation: "upload_file";
        convert_to_google_docs: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mimeType?: string | undefined;
        parent_folder_id?: string | undefined;
    }, {
        name: string;
        content: string;
        operation: "upload_file";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mimeType?: string | undefined;
        parent_folder_id?: string | undefined;
        convert_to_google_docs?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"download_file">;
        file_id: z.ZodString;
        export_format: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "download_file";
        file_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        export_format?: string | undefined;
    }, {
        operation: "download_file";
        file_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        export_format?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_files">;
        folder_id: z.ZodOptional<z.ZodString>;
        query: z.ZodOptional<z.ZodString>;
        max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        include_folders: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        order_by: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_files";
        max_results: number;
        include_folders: boolean;
        order_by: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        folder_id?: string | undefined;
    }, {
        operation: "list_files";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        folder_id?: string | undefined;
        max_results?: number | undefined;
        include_folders?: boolean | undefined;
        order_by?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_folder">;
        name: z.ZodString;
        parent_folder_id: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "create_folder";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        parent_folder_id?: string | undefined;
    }, {
        name: string;
        operation: "create_folder";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        parent_folder_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_file">;
        file_id: z.ZodString;
        permanent: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_file";
        file_id: string;
        permanent: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_file";
        file_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        permanent?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_file_info">;
        file_id: z.ZodString;
        include_permissions: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_file_info";
        file_id: string;
        include_permissions: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_file_info";
        file_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        include_permissions?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"share_file">;
        file_id: z.ZodString;
        email_address: z.ZodOptional<z.ZodString>;
        role: z.ZodDefault<z.ZodOptional<z.ZodEnum<["reader", "writer", "commenter", "owner"]>>>;
        type: z.ZodDefault<z.ZodOptional<z.ZodEnum<["user", "group", "domain", "anyone"]>>>;
        send_notification: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        type: "user" | "group" | "domain" | "anyone";
        operation: "share_file";
        file_id: string;
        role: "reader" | "writer" | "commenter" | "owner";
        send_notification: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        email_address?: string | undefined;
    }, {
        operation: "share_file";
        file_id: string;
        type?: "user" | "group" | "domain" | "anyone" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        email_address?: string | undefined;
        role?: "reader" | "writer" | "commenter" | "owner" | undefined;
        send_notification?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"move_file">;
        file_id: z.ZodString;
        new_parent_folder_id: z.ZodOptional<z.ZodString>;
        remove_parent_folder_id: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "move_file";
        file_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        new_parent_folder_id?: string | undefined;
        remove_parent_folder_id?: string | undefined;
    }, {
        operation: "move_file";
        file_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        new_parent_folder_id?: string | undefined;
        remove_parent_folder_id?: string | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"upload_file">;
        success: z.ZodBoolean;
        file: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            mimeType: z.ZodString;
            size: z.ZodOptional<z.ZodString>;
            createdTime: z.ZodOptional<z.ZodString>;
            modifiedTime: z.ZodOptional<z.ZodString>;
            webViewLink: z.ZodOptional<z.ZodString>;
            webContentLink: z.ZodOptional<z.ZodString>;
            parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            shared: z.ZodOptional<z.ZodBoolean>;
            owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
                displayName: z.ZodOptional<z.ZodString>;
                emailAddress: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "upload_file";
        file?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "upload_file";
        file?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"download_file">;
        success: z.ZodBoolean;
        content: z.ZodOptional<z.ZodString>;
        filename: z.ZodOptional<z.ZodString>;
        mimeType: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "download_file";
        content?: string | undefined;
        mimeType?: string | undefined;
        filename?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "download_file";
        content?: string | undefined;
        mimeType?: string | undefined;
        filename?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_files">;
        success: z.ZodBoolean;
        files: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            mimeType: z.ZodString;
            size: z.ZodOptional<z.ZodString>;
            createdTime: z.ZodOptional<z.ZodString>;
            modifiedTime: z.ZodOptional<z.ZodString>;
            webViewLink: z.ZodOptional<z.ZodString>;
            webContentLink: z.ZodOptional<z.ZodString>;
            parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            shared: z.ZodOptional<z.ZodBoolean>;
            owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
                displayName: z.ZodOptional<z.ZodString>;
                emailAddress: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }>, "many">>;
        total_count: z.ZodOptional<z.ZodNumber>;
        next_page_token: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_files";
        files?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }[] | undefined;
        total_count?: number | undefined;
        next_page_token?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_files";
        files?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }[] | undefined;
        total_count?: number | undefined;
        next_page_token?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_folder">;
        success: z.ZodBoolean;
        folder: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            webViewLink: z.ZodOptional<z.ZodString>;
            parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            webViewLink?: string | undefined;
            parents?: string[] | undefined;
        }, {
            name: string;
            id: string;
            webViewLink?: string | undefined;
            parents?: string[] | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_folder";
        folder?: {
            name: string;
            id: string;
            webViewLink?: string | undefined;
            parents?: string[] | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_folder";
        folder?: {
            name: string;
            id: string;
            webViewLink?: string | undefined;
            parents?: string[] | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_file">;
        success: z.ZodBoolean;
        deleted_file_id: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_file";
        deleted_file_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_file";
        deleted_file_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_file_info">;
        success: z.ZodBoolean;
        file: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            mimeType: z.ZodString;
            size: z.ZodOptional<z.ZodString>;
            createdTime: z.ZodOptional<z.ZodString>;
            modifiedTime: z.ZodOptional<z.ZodString>;
            webViewLink: z.ZodOptional<z.ZodString>;
            webContentLink: z.ZodOptional<z.ZodString>;
            parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            shared: z.ZodOptional<z.ZodBoolean>;
            owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
                displayName: z.ZodOptional<z.ZodString>;
                emailAddress: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }>>;
        permissions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            type: z.ZodString;
            role: z.ZodString;
            emailAddress: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            id: string;
            role: string;
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }, {
            type: string;
            id: string;
            role: string;
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }>, "many">>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_file_info";
        file?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        } | undefined;
        permissions?: {
            type: string;
            id: string;
            role: string;
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_file_info";
        file?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        } | undefined;
        permissions?: {
            type: string;
            id: string;
            role: string;
            displayName?: string | undefined;
            emailAddress?: string | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"share_file">;
        success: z.ZodBoolean;
        permission_id: z.ZodOptional<z.ZodString>;
        share_link: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "share_file";
        permission_id?: string | undefined;
        share_link?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "share_file";
        permission_id?: string | undefined;
        share_link?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"move_file">;
        success: z.ZodBoolean;
        file: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            mimeType: z.ZodString;
            size: z.ZodOptional<z.ZodString>;
            createdTime: z.ZodOptional<z.ZodString>;
            modifiedTime: z.ZodOptional<z.ZodString>;
            webViewLink: z.ZodOptional<z.ZodString>;
            webContentLink: z.ZodOptional<z.ZodString>;
            parents: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            shared: z.ZodOptional<z.ZodBoolean>;
            owners: z.ZodOptional<z.ZodArray<z.ZodObject<{
                displayName: z.ZodOptional<z.ZodString>;
                emailAddress: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }, {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }, {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "move_file";
        file?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "move_file";
        file?: {
            name: string;
            mimeType: string;
            id: string;
            size?: string | undefined;
            createdTime?: string | undefined;
            modifiedTime?: string | undefined;
            webViewLink?: string | undefined;
            webContentLink?: string | undefined;
            parents?: string[] | undefined;
            shared?: boolean | undefined;
            owners?: {
                displayName?: string | undefined;
                emailAddress?: string | undefined;
            }[] | undefined;
        } | undefined;
    }>]>;
    static readonly shortDescription = "Google Drive integration for file management";
    static readonly longDescription = "\n    Google Drive service integration for comprehensive file and folder management.\n    Use cases:\n    - Upload files and documents to Google Drive\n    - Download files with format conversion support\n    - List and search files with advanced filtering\n    - Create and organize folders\n    - Share files and manage permissions\n    - Get detailed file metadata and information\n\n    Security Features:\n    - OAuth 2.0 authentication with Google\n    - Scoped access permissions\n    - Secure file handling and validation\n    - User-controlled sharing and permissions\n  ";
    static readonly alias = "gdrive";
    /**
     * Create a new Google Drive Bubble instance
     * @param params - Operation parameters
     * @param context - Bubble execution context
     */
    constructor(params?: T, context?: BubbleContext);
    /**
     * Test the validity of the Google Drive credentials
     * @returns Promise that resolves to true if credentials are valid, false otherwise
     * @throws AuthenticationError if credentials are missing
     */
    testCredential(): Promise<boolean>;
    /**
     * Make an API request to Google Drive
     * @param endpoint - API endpoint (full URL or relative path)
     * @param method - HTTP method
     * @param body - Request body
     * @param headers - Additional headers
     * @param responseType - Expected response type
     * @returns Promise that resolves with the API response
     * @throws ExternalServiceError if the API request fails
     */
    private makeGoogleApiRequest;
    /**
     * Execute the Google Drive operation specified in params
     * @param context - Bubble execution context (unused)
     * @returns Promise that resolves with the operation result
     */
    protected performAction(context?: BubbleContext): Promise<Extract<GoogleDriveResult, {
        operation: T['operation'];
    }>>;
    private uploadFile;
    private isTextMimeType;
    private downloadFile;
    private listFiles;
    private createFolder;
    private deleteFile;
    private getFileInfo;
    private shareFile;
    private moveFile;
    private isBase64;
    private extractBase64Content;
    private findAndExtractBase64;
    private findBase64InObject;
    private detectMimeTypeFromBase64;
    protected chooseCredential(): string | undefined;
}
export {};
//# sourceMappingURL=google-drive.d.ts.map