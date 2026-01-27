/**
 * PDF FORM OPERATIONS WORKFLOW
 *
 * A unified workflow for PDF form operations including field discovery,
 * form filling, checkbox analysis, and form validation using pdf-lib.
 *
 * Provides multiple operations similar to Resend email service:
 * - discover: Extract all form fields with metadata
 * - fill: Fill form fields with provided values
 * - analyze-checkboxes: Get checkbox states and possible values
 * - validate: Verify form field values and completion
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
/**
 * Combined parameters schema using discriminated union
 */
declare const PDFFormOperationsParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"discover">;
    pdfData: z.ZodString;
    targetPage: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "discover";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    targetPage?: number | undefined;
}, {
    operation: "discover";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    targetPage?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"fill">;
    pdfData: z.ZodString;
    fieldValues: z.ZodRecord<z.ZodString, z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "fill";
    pdfData: string;
    fieldValues: Record<string, string>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "fill";
    pdfData: string;
    fieldValues: Record<string, string>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"analyze-checkboxes">;
    pdfData: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "analyze-checkboxes";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "analyze-checkboxes";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validate">;
    pdfData: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "validate";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "validate";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"convert-to-images">;
    pdfData: z.ZodString;
    format: z.ZodDefault<z.ZodEnum<["png", "jpeg"]>>;
    quality: z.ZodDefault<z.ZodNumber>;
    dpi: z.ZodDefault<z.ZodNumber>;
    pages: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    format: "png" | "jpeg";
    operation: "convert-to-images";
    pdfData: string;
    quality: number;
    dpi: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pages?: number[] | undefined;
}, {
    operation: "convert-to-images";
    pdfData: string;
    format?: "png" | "jpeg" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    quality?: number | undefined;
    dpi?: number | undefined;
    pages?: number[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"convert-to-markdown">;
    pdfData: z.ZodString;
    pages: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    includeFormFields: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "convert-to-markdown";
    pdfData: string;
    includeFormFields: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pages?: number[] | undefined;
}, {
    operation: "convert-to-markdown";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pages?: number[] | undefined;
    includeFormFields?: boolean | undefined;
}>]>;
/**
 * Combined result schema using discriminated union
 */
declare const PDFFormOperationsResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"discover">;
    fields: z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        page: z.ZodNumber;
        name: z.ZodString;
        type: z.ZodString;
        field_type: z.ZodString;
        current_value: z.ZodString;
        choices: z.ZodArray<z.ZodString, "many">;
        rect: z.ZodTuple<[z.ZodNumber, z.ZodNumber, z.ZodNumber, z.ZodNumber], null>;
        x: z.ZodNumber;
        y: z.ZodNumber;
        width: z.ZodNumber;
        height: z.ZodNumber;
        field_flags: z.ZodNumber;
        label: z.ZodString;
        potential_labels: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        type: string;
        name: string;
        id: number;
        label: string;
        width: number;
        height: number;
        page: number;
        field_type: string;
        current_value: string;
        choices: string[];
        rect: [number, number, number, number];
        x: number;
        y: number;
        field_flags: number;
        potential_labels: string[];
    }, {
        type: string;
        name: string;
        id: number;
        label: string;
        width: number;
        height: number;
        page: number;
        field_type: string;
        current_value: string;
        choices: string[];
        rect: [number, number, number, number];
        x: number;
        y: number;
        field_flags: number;
        potential_labels: string[];
    }>, "many">;
    totalFields: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    fields: {
        type: string;
        name: string;
        id: number;
        label: string;
        width: number;
        height: number;
        page: number;
        field_type: string;
        current_value: string;
        choices: string[];
        rect: [number, number, number, number];
        x: number;
        y: number;
        field_flags: number;
        potential_labels: string[];
    }[];
    operation: "discover";
    totalFields: number;
}, {
    error: string;
    success: boolean;
    fields: {
        type: string;
        name: string;
        id: number;
        label: string;
        width: number;
        height: number;
        page: number;
        field_type: string;
        current_value: string;
        choices: string[];
        rect: [number, number, number, number];
        x: number;
        y: number;
        field_flags: number;
        potential_labels: string[];
    }[];
    operation: "discover";
    totalFields: number;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"fill">;
    filledPdfData: z.ZodString;
    filledFields: z.ZodNumber;
    verification: z.ZodRecord<z.ZodString, z.ZodObject<{
        value: z.ZodString;
        type: z.ZodString;
        page: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        value: string;
        type: string;
        page: number;
    }, {
        value: string;
        type: string;
        page: number;
    }>>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "fill";
    filledPdfData: string;
    filledFields: number;
    verification: Record<string, {
        value: string;
        type: string;
        page: number;
    }>;
}, {
    error: string;
    success: boolean;
    operation: "fill";
    filledPdfData: string;
    filledFields: number;
    verification: Record<string, {
        value: string;
        type: string;
        page: number;
    }>;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"analyze-checkboxes">;
    checkboxes: z.ZodRecord<z.ZodString, z.ZodObject<{
        page: z.ZodNumber;
        current_value: z.ZodString;
        possible_values: z.ZodArray<z.ZodString, "many">;
        field_flags: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        page: number;
        current_value: string;
        field_flags: number;
        possible_values: string[];
    }, {
        page: number;
        current_value: string;
        field_flags: number;
        possible_values: string[];
    }>>;
    totalCheckboxes: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "analyze-checkboxes";
    checkboxes: Record<string, {
        page: number;
        current_value: string;
        field_flags: number;
        possible_values: string[];
    }>;
    totalCheckboxes: number;
}, {
    error: string;
    success: boolean;
    operation: "analyze-checkboxes";
    checkboxes: Record<string, {
        page: number;
        current_value: string;
        field_flags: number;
        possible_values: string[];
    }>;
    totalCheckboxes: number;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validate">;
    fields: z.ZodRecord<z.ZodString, z.ZodObject<{
        value: z.ZodString;
        type: z.ZodString;
        page: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        value: string;
        type: string;
        page: number;
    }, {
        value: string;
        type: string;
        page: number;
    }>>;
    totalFields: z.ZodNumber;
    filledFields: z.ZodNumber;
    emptyFields: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    fields: Record<string, {
        value: string;
        type: string;
        page: number;
    }>;
    operation: "validate";
    totalFields: number;
    filledFields: number;
    emptyFields: number;
}, {
    error: string;
    success: boolean;
    fields: Record<string, {
        value: string;
        type: string;
        page: number;
    }>;
    operation: "validate";
    totalFields: number;
    filledFields: number;
    emptyFields: number;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"convert-to-images">;
    images: z.ZodArray<z.ZodObject<{
        pageNumber: z.ZodNumber;
        imageData: z.ZodString;
        format: z.ZodString;
        width: z.ZodNumber;
        height: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        format: string;
        pageNumber: number;
        width: number;
        height: number;
        imageData: string;
    }, {
        format: string;
        pageNumber: number;
        width: number;
        height: number;
        imageData: string;
    }>, "many">;
    totalPages: z.ZodNumber;
    convertedPages: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    images: {
        format: string;
        pageNumber: number;
        width: number;
        height: number;
        imageData: string;
    }[];
    operation: "convert-to-images";
    totalPages: number;
    convertedPages: number;
}, {
    error: string;
    success: boolean;
    images: {
        format: string;
        pageNumber: number;
        width: number;
        height: number;
        imageData: string;
    }[];
    operation: "convert-to-images";
    totalPages: number;
    convertedPages: number;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"convert-to-markdown">;
    markdown: z.ZodString;
    pages: z.ZodArray<z.ZodObject<{
        pageNumber: z.ZodNumber;
        markdown: z.ZodString;
        formFields: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            name: z.ZodString;
            type: z.ZodString;
            value: z.ZodString;
            x: z.ZodNumber;
            y: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type: string;
            name: string;
            id: number;
            x: number;
            y: number;
        }, {
            value: string;
            type: string;
            name: string;
            id: number;
            x: number;
            y: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        pageNumber: number;
        markdown: string;
        formFields?: {
            value: string;
            type: string;
            name: string;
            id: number;
            x: number;
            y: number;
        }[] | undefined;
    }, {
        pageNumber: number;
        markdown: string;
        formFields?: {
            value: string;
            type: string;
            name: string;
            id: number;
            x: number;
            y: number;
        }[] | undefined;
    }>, "many">;
    totalPages: z.ZodNumber;
    convertedPages: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "convert-to-markdown";
    pages: {
        pageNumber: number;
        markdown: string;
        formFields?: {
            value: string;
            type: string;
            name: string;
            id: number;
            x: number;
            y: number;
        }[] | undefined;
    }[];
    totalPages: number;
    convertedPages: number;
    markdown: string;
}, {
    error: string;
    success: boolean;
    operation: "convert-to-markdown";
    pages: {
        pageNumber: number;
        markdown: string;
        formFields?: {
            value: string;
            type: string;
            name: string;
            id: number;
            x: number;
            y: number;
        }[] | undefined;
    }[];
    totalPages: number;
    convertedPages: number;
    markdown: string;
}>]>;
type PDFFormOperationsParams = z.output<typeof PDFFormOperationsParamsSchema>;
type PDFFormOperationsResult = z.output<typeof PDFFormOperationsResultSchema>;
/**
 * PDF Form Operations Workflow
 * Provides unified interface for PDF form operations using pdf-lib
 */
export declare class PDFFormOperationsWorkflow<T extends PDFFormOperationsParams = PDFFormOperationsParams> extends WorkflowBubble<T, Extract<PDFFormOperationsResult, {
    operation: T['operation'];
}>> {
    static readonly type: "workflow";
    static readonly bubbleName = "pdf-form-operations";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"discover">;
        pdfData: z.ZodString;
        targetPage: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "discover";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        targetPage?: number | undefined;
    }, {
        operation: "discover";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        targetPage?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"fill">;
        pdfData: z.ZodString;
        fieldValues: z.ZodRecord<z.ZodString, z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "fill";
        pdfData: string;
        fieldValues: Record<string, string>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "fill";
        pdfData: string;
        fieldValues: Record<string, string>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"analyze-checkboxes">;
        pdfData: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "analyze-checkboxes";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "analyze-checkboxes";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validate">;
        pdfData: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "validate";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "validate";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"convert-to-images">;
        pdfData: z.ZodString;
        format: z.ZodDefault<z.ZodEnum<["png", "jpeg"]>>;
        quality: z.ZodDefault<z.ZodNumber>;
        dpi: z.ZodDefault<z.ZodNumber>;
        pages: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        format: "png" | "jpeg";
        operation: "convert-to-images";
        pdfData: string;
        quality: number;
        dpi: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pages?: number[] | undefined;
    }, {
        operation: "convert-to-images";
        pdfData: string;
        format?: "png" | "jpeg" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        quality?: number | undefined;
        dpi?: number | undefined;
        pages?: number[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"convert-to-markdown">;
        pdfData: z.ZodString;
        pages: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        includeFormFields: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "convert-to-markdown";
        pdfData: string;
        includeFormFields: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pages?: number[] | undefined;
    }, {
        operation: "convert-to-markdown";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pages?: number[] | undefined;
        includeFormFields?: boolean | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"discover">;
        fields: z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            page: z.ZodNumber;
            name: z.ZodString;
            type: z.ZodString;
            field_type: z.ZodString;
            current_value: z.ZodString;
            choices: z.ZodArray<z.ZodString, "many">;
            rect: z.ZodTuple<[z.ZodNumber, z.ZodNumber, z.ZodNumber, z.ZodNumber], null>;
            x: z.ZodNumber;
            y: z.ZodNumber;
            width: z.ZodNumber;
            height: z.ZodNumber;
            field_flags: z.ZodNumber;
            label: z.ZodString;
            potential_labels: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            type: string;
            name: string;
            id: number;
            label: string;
            width: number;
            height: number;
            page: number;
            field_type: string;
            current_value: string;
            choices: string[];
            rect: [number, number, number, number];
            x: number;
            y: number;
            field_flags: number;
            potential_labels: string[];
        }, {
            type: string;
            name: string;
            id: number;
            label: string;
            width: number;
            height: number;
            page: number;
            field_type: string;
            current_value: string;
            choices: string[];
            rect: [number, number, number, number];
            x: number;
            y: number;
            field_flags: number;
            potential_labels: string[];
        }>, "many">;
        totalFields: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: number;
            label: string;
            width: number;
            height: number;
            page: number;
            field_type: string;
            current_value: string;
            choices: string[];
            rect: [number, number, number, number];
            x: number;
            y: number;
            field_flags: number;
            potential_labels: string[];
        }[];
        operation: "discover";
        totalFields: number;
    }, {
        error: string;
        success: boolean;
        fields: {
            type: string;
            name: string;
            id: number;
            label: string;
            width: number;
            height: number;
            page: number;
            field_type: string;
            current_value: string;
            choices: string[];
            rect: [number, number, number, number];
            x: number;
            y: number;
            field_flags: number;
            potential_labels: string[];
        }[];
        operation: "discover";
        totalFields: number;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"fill">;
        filledPdfData: z.ZodString;
        filledFields: z.ZodNumber;
        verification: z.ZodRecord<z.ZodString, z.ZodObject<{
            value: z.ZodString;
            type: z.ZodString;
            page: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type: string;
            page: number;
        }, {
            value: string;
            type: string;
            page: number;
        }>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "fill";
        filledPdfData: string;
        filledFields: number;
        verification: Record<string, {
            value: string;
            type: string;
            page: number;
        }>;
    }, {
        error: string;
        success: boolean;
        operation: "fill";
        filledPdfData: string;
        filledFields: number;
        verification: Record<string, {
            value: string;
            type: string;
            page: number;
        }>;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"analyze-checkboxes">;
        checkboxes: z.ZodRecord<z.ZodString, z.ZodObject<{
            page: z.ZodNumber;
            current_value: z.ZodString;
            possible_values: z.ZodArray<z.ZodString, "many">;
            field_flags: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            page: number;
            current_value: string;
            field_flags: number;
            possible_values: string[];
        }, {
            page: number;
            current_value: string;
            field_flags: number;
            possible_values: string[];
        }>>;
        totalCheckboxes: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "analyze-checkboxes";
        checkboxes: Record<string, {
            page: number;
            current_value: string;
            field_flags: number;
            possible_values: string[];
        }>;
        totalCheckboxes: number;
    }, {
        error: string;
        success: boolean;
        operation: "analyze-checkboxes";
        checkboxes: Record<string, {
            page: number;
            current_value: string;
            field_flags: number;
            possible_values: string[];
        }>;
        totalCheckboxes: number;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validate">;
        fields: z.ZodRecord<z.ZodString, z.ZodObject<{
            value: z.ZodString;
            type: z.ZodString;
            page: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type: string;
            page: number;
        }, {
            value: string;
            type: string;
            page: number;
        }>>;
        totalFields: z.ZodNumber;
        filledFields: z.ZodNumber;
        emptyFields: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        fields: Record<string, {
            value: string;
            type: string;
            page: number;
        }>;
        operation: "validate";
        totalFields: number;
        filledFields: number;
        emptyFields: number;
    }, {
        error: string;
        success: boolean;
        fields: Record<string, {
            value: string;
            type: string;
            page: number;
        }>;
        operation: "validate";
        totalFields: number;
        filledFields: number;
        emptyFields: number;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"convert-to-images">;
        images: z.ZodArray<z.ZodObject<{
            pageNumber: z.ZodNumber;
            imageData: z.ZodString;
            format: z.ZodString;
            width: z.ZodNumber;
            height: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            format: string;
            pageNumber: number;
            width: number;
            height: number;
            imageData: string;
        }, {
            format: string;
            pageNumber: number;
            width: number;
            height: number;
            imageData: string;
        }>, "many">;
        totalPages: z.ZodNumber;
        convertedPages: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        images: {
            format: string;
            pageNumber: number;
            width: number;
            height: number;
            imageData: string;
        }[];
        operation: "convert-to-images";
        totalPages: number;
        convertedPages: number;
    }, {
        error: string;
        success: boolean;
        images: {
            format: string;
            pageNumber: number;
            width: number;
            height: number;
            imageData: string;
        }[];
        operation: "convert-to-images";
        totalPages: number;
        convertedPages: number;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"convert-to-markdown">;
        markdown: z.ZodString;
        pages: z.ZodArray<z.ZodObject<{
            pageNumber: z.ZodNumber;
            markdown: z.ZodString;
            formFields: z.ZodOptional<z.ZodArray<z.ZodObject<{
                id: z.ZodNumber;
                name: z.ZodString;
                type: z.ZodString;
                value: z.ZodString;
                x: z.ZodNumber;
                y: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type: string;
                name: string;
                id: number;
                x: number;
                y: number;
            }, {
                value: string;
                type: string;
                name: string;
                id: number;
                x: number;
                y: number;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            pageNumber: number;
            markdown: string;
            formFields?: {
                value: string;
                type: string;
                name: string;
                id: number;
                x: number;
                y: number;
            }[] | undefined;
        }, {
            pageNumber: number;
            markdown: string;
            formFields?: {
                value: string;
                type: string;
                name: string;
                id: number;
                x: number;
                y: number;
            }[] | undefined;
        }>, "many">;
        totalPages: z.ZodNumber;
        convertedPages: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "convert-to-markdown";
        pages: {
            pageNumber: number;
            markdown: string;
            formFields?: {
                value: string;
                type: string;
                name: string;
                id: number;
                x: number;
                y: number;
            }[] | undefined;
        }[];
        totalPages: number;
        convertedPages: number;
        markdown: string;
    }, {
        error: string;
        success: boolean;
        operation: "convert-to-markdown";
        pages: {
            pageNumber: number;
            markdown: string;
            formFields?: {
                value: string;
                type: string;
                name: string;
                id: number;
                x: number;
                y: number;
            }[] | undefined;
        }[];
        totalPages: number;
        convertedPages: number;
        markdown: string;
    }>]>;
    static readonly shortDescription = "PDF form field operations (discover, fill, analyze, validate, convert-to-images, convert-to-markdown)";
    static readonly longDescription = "\n    Unified PDF form operations workflow providing comprehensive form field manipulation.\n    \n    Operations:\n    - discover: Extract all form fields with coordinates and metadata\n    - fill: Fill form fields with provided values and return filled PDF\n    - analyze-checkboxes: Analyze checkbox fields and their possible values\n    - validate: Verify form field values and completion status\n    - convert-to-images: Convert PDF pages to PNG/JPEG images with customizable quality and DPI\n    - convert-to-markdown: Convert PDF to markdown format using AI analysis of visual content\n    \n    Uses PyMuPDF (fitz) library via Python scripts for all PDF operations.\n    \n    Input: Base64 encoded PDF data\n    Output: Operation-specific results with success/error handling\n  ";
    static readonly alias = "pdf-forms";
    constructor(params?: T, context?: BubbleContext);
    /**
     * Execute a Python script with the given arguments (stdin input)
     */
    private executePythonScript;
    /**
     * Execute a Python script with file input (for optimized memory usage)
     */
    private executePythonFileScript;
    protected performAction(context?: BubbleContext): Promise<Extract<PDFFormOperationsResult, {
        operation: T['operation'];
    }>>;
    /**
     * Discover all form fields in the PDF
     */
    private discoverFields;
    /**
     * Fill form fields with provided values
     */
    private fillFields;
    /**
     * Analyze checkbox fields and their possible values
     */
    private analyzeCheckboxes;
    /**
     * Validate form fields and their values
     */
    private validateFields;
    /**
     * Convert PDF pages to images
     */
    private convertToImages;
    /**
     * Convert PDF to markdown using AI analysis of visual content
     */
    private convertToMarkdown;
}
export {};
//# sourceMappingURL=pdf-form-operations.workflow.d.ts.map