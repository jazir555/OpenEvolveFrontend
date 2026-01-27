/**
 * PDF GENERATOR TOOL
 *
 * A tool bubble for generating PDF documents from various input formats.
 *
 * Features:
 * - Generate PDF from HTML
 * - Generate PDF from Markdown
 * - Generate PDF from text
 * - Add headers and footers
 * - Page numbering
 * - Custom page sizes
 * - Margin configuration
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Page size options
 */
export declare enum PageSize {
    A4 = "A4",
    LETTER = "Letter",
    LEGAL = "Legal",
    TABLOID = "Tabloid"
}
/**
 * Page orientation
 */
export declare enum PageOrientation {
    PORTRAIT = "portrait",
    LANDSCAPE = "landscape"
}
/**
 * PDF generator parameters schema
 */
declare const PDFGeneratorToolParamsSchema: z.ZodObject<{
    content: z.ZodString;
    contentType: z.ZodDefault<z.ZodEnum<["html", "markdown", "text"]>>;
    outputPath: z.ZodOptional<z.ZodString>;
    returnBase64: z.ZodDefault<z.ZodBoolean>;
    pageSize: z.ZodDefault<z.ZodNativeEnum<typeof PageSize>>;
    orientation: z.ZodDefault<z.ZodNativeEnum<typeof PageOrientation>>;
    margins: z.ZodOptional<z.ZodObject<{
        top: z.ZodDefault<z.ZodNumber>;
        right: z.ZodDefault<z.ZodNumber>;
        bottom: z.ZodDefault<z.ZodNumber>;
        left: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        top: number;
        left: number;
        right: number;
        bottom: number;
    }, {
        top?: number | undefined;
        left?: number | undefined;
        right?: number | undefined;
        bottom?: number | undefined;
    }>>;
    header: z.ZodOptional<z.ZodObject<{
        text: z.ZodOptional<z.ZodString>;
        height: z.ZodDefault<z.ZodNumber>;
        showPageNumbers: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        height: number;
        showPageNumbers: boolean;
        text?: string | undefined;
    }, {
        text?: string | undefined;
        height?: number | undefined;
        showPageNumbers?: boolean | undefined;
    }>>;
    footer: z.ZodOptional<z.ZodObject<{
        text: z.ZodOptional<z.ZodString>;
        height: z.ZodDefault<z.ZodNumber>;
        showPageNumbers: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        height: number;
        showPageNumbers: boolean;
        text?: string | undefined;
    }, {
        text?: string | undefined;
        height?: number | undefined;
        showPageNumbers?: boolean | undefined;
    }>>;
    styles: z.ZodOptional<z.ZodString>;
    title: z.ZodOptional<z.ZodString>;
    author: z.ZodOptional<z.ZodString>;
    subject: z.ZodOptional<z.ZodString>;
    keywords: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    content: string;
    contentType: "text" | "html" | "markdown";
    pageSize: PageSize;
    returnBase64: boolean;
    orientation: PageOrientation;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    footer?: {
        height: number;
        showPageNumbers: boolean;
        text?: string | undefined;
    } | undefined;
    subject?: string | undefined;
    author?: string | undefined;
    header?: {
        height: number;
        showPageNumbers: boolean;
        text?: string | undefined;
    } | undefined;
    keywords?: string[] | undefined;
    margins?: {
        top: number;
        left: number;
        right: number;
        bottom: number;
    } | undefined;
    outputPath?: string | undefined;
    styles?: string | undefined;
}, {
    content: string;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    footer?: {
        text?: string | undefined;
        height?: number | undefined;
        showPageNumbers?: boolean | undefined;
    } | undefined;
    subject?: string | undefined;
    contentType?: "text" | "html" | "markdown" | undefined;
    author?: string | undefined;
    header?: {
        text?: string | undefined;
        height?: number | undefined;
        showPageNumbers?: boolean | undefined;
    } | undefined;
    pageSize?: PageSize | undefined;
    keywords?: string[] | undefined;
    margins?: {
        top?: number | undefined;
        left?: number | undefined;
        right?: number | undefined;
        bottom?: number | undefined;
    } | undefined;
    outputPath?: string | undefined;
    returnBase64?: boolean | undefined;
    orientation?: PageOrientation | undefined;
    styles?: string | undefined;
}>;
/**
 * PDF generator result schema
 */
declare const PDFGeneratorToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    pdfData: z.ZodOptional<z.ZodString>;
    pdfPath: z.ZodOptional<z.ZodString>;
    pageCount: z.ZodOptional<z.ZodNumber>;
    fileSize: z.ZodOptional<z.ZodNumber>;
    stats: z.ZodObject<{
        charactersProcessed: z.ZodOptional<z.ZodNumber>;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        charactersProcessed?: number | undefined;
    }, {
        processingTime: number;
        charactersProcessed?: number | undefined;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        charactersProcessed?: number | undefined;
    };
    pageCount?: number | undefined;
    fileSize?: number | undefined;
    pdfData?: string | undefined;
    pdfPath?: string | undefined;
}, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        charactersProcessed?: number | undefined;
    };
    pageCount?: number | undefined;
    fileSize?: number | undefined;
    pdfData?: string | undefined;
    pdfPath?: string | undefined;
}>;
type PDFGeneratorToolParams = z.output<typeof PDFGeneratorToolParamsSchema>;
type PDFGeneratorToolResult = z.output<typeof PDFGeneratorToolResultSchema>;
type PDFGeneratorToolParamsInput = z.input<typeof PDFGeneratorToolParamsSchema>;
/**
 * PDF Generator Tool
 * Generate PDF documents from HTML, Markdown, or text
 */
export declare class PDFGeneratorTool extends ToolBubble<PDFGeneratorToolParams, PDFGeneratorToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        content: z.ZodString;
        contentType: z.ZodDefault<z.ZodEnum<["html", "markdown", "text"]>>;
        outputPath: z.ZodOptional<z.ZodString>;
        returnBase64: z.ZodDefault<z.ZodBoolean>;
        pageSize: z.ZodDefault<z.ZodNativeEnum<typeof PageSize>>;
        orientation: z.ZodDefault<z.ZodNativeEnum<typeof PageOrientation>>;
        margins: z.ZodOptional<z.ZodObject<{
            top: z.ZodDefault<z.ZodNumber>;
            right: z.ZodDefault<z.ZodNumber>;
            bottom: z.ZodDefault<z.ZodNumber>;
            left: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            top: number;
            left: number;
            right: number;
            bottom: number;
        }, {
            top?: number | undefined;
            left?: number | undefined;
            right?: number | undefined;
            bottom?: number | undefined;
        }>>;
        header: z.ZodOptional<z.ZodObject<{
            text: z.ZodOptional<z.ZodString>;
            height: z.ZodDefault<z.ZodNumber>;
            showPageNumbers: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            height: number;
            showPageNumbers: boolean;
            text?: string | undefined;
        }, {
            text?: string | undefined;
            height?: number | undefined;
            showPageNumbers?: boolean | undefined;
        }>>;
        footer: z.ZodOptional<z.ZodObject<{
            text: z.ZodOptional<z.ZodString>;
            height: z.ZodDefault<z.ZodNumber>;
            showPageNumbers: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            height: number;
            showPageNumbers: boolean;
            text?: string | undefined;
        }, {
            text?: string | undefined;
            height?: number | undefined;
            showPageNumbers?: boolean | undefined;
        }>>;
        styles: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        author: z.ZodOptional<z.ZodString>;
        subject: z.ZodOptional<z.ZodString>;
        keywords: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        contentType: "text" | "html" | "markdown";
        pageSize: PageSize;
        returnBase64: boolean;
        orientation: PageOrientation;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        footer?: {
            height: number;
            showPageNumbers: boolean;
            text?: string | undefined;
        } | undefined;
        subject?: string | undefined;
        author?: string | undefined;
        header?: {
            height: number;
            showPageNumbers: boolean;
            text?: string | undefined;
        } | undefined;
        keywords?: string[] | undefined;
        margins?: {
            top: number;
            left: number;
            right: number;
            bottom: number;
        } | undefined;
        outputPath?: string | undefined;
        styles?: string | undefined;
    }, {
        content: string;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        footer?: {
            text?: string | undefined;
            height?: number | undefined;
            showPageNumbers?: boolean | undefined;
        } | undefined;
        subject?: string | undefined;
        contentType?: "text" | "html" | "markdown" | undefined;
        author?: string | undefined;
        header?: {
            text?: string | undefined;
            height?: number | undefined;
            showPageNumbers?: boolean | undefined;
        } | undefined;
        pageSize?: PageSize | undefined;
        keywords?: string[] | undefined;
        margins?: {
            top?: number | undefined;
            left?: number | undefined;
            right?: number | undefined;
            bottom?: number | undefined;
        } | undefined;
        outputPath?: string | undefined;
        returnBase64?: boolean | undefined;
        orientation?: PageOrientation | undefined;
        styles?: string | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        pdfData: z.ZodOptional<z.ZodString>;
        pdfPath: z.ZodOptional<z.ZodString>;
        pageCount: z.ZodOptional<z.ZodNumber>;
        fileSize: z.ZodOptional<z.ZodNumber>;
        stats: z.ZodObject<{
            charactersProcessed: z.ZodOptional<z.ZodNumber>;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            charactersProcessed?: number | undefined;
        }, {
            processingTime: number;
            charactersProcessed?: number | undefined;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            charactersProcessed?: number | undefined;
        };
        pageCount?: number | undefined;
        fileSize?: number | undefined;
        pdfData?: string | undefined;
        pdfPath?: string | undefined;
    }, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            charactersProcessed?: number | undefined;
        };
        pageCount?: number | undefined;
        fileSize?: number | undefined;
        pdfData?: string | undefined;
        pdfPath?: string | undefined;
    }>;
    static readonly shortDescription = "Generate PDF documents from HTML, Markdown, or text";
    static readonly longDescription = "\n    A PDF generation tool for creating documents from various input formats.\n\n    Features:\n    - Generate PDF from HTML content\n    - Generate PDF from Markdown\n    - Generate PDF from plain text\n    - Custom page sizes (A4, Letter, Legal, Tabloid)\n    - Portrait or landscape orientation\n    - Configurable margins\n    - Headers and footers with page numbers\n    - CSS styling support\n    - PDF metadata (title, author, subject, keywords)\n\n    Content Types:\n    - HTML: Full HTML with CSS styling support\n    - Markdown: Markdown conversion to PDF\n    - TEXT: Plain text with basic formatting\n\n    Page Sizes:\n    - A4: 210 x 297 mm (default)\n    - Letter: 8.5 x 11 inches\n    - Legal: 8.5 x 14 inches\n    - Tabloid: 11 x 17 inches\n\n    Use cases:\n    - Report generation\n    - Invoice creation\n    - Document archiving\n    - Certificate generation\n    - E-book creation\n    - Formatted document export\n\n    Note: This tool requires a PDF generation library.\n    Recommended libraries:\n    - Node.js: pdfkit, puppeteer, jsPDF\n    - Browser: jsPDF, pdfmake\n    - Cloud: HTML-to-PDF APIs\n\n    The current implementation provides a placeholder.\n    For production use, integrate with a PDF library.\n  ";
    static readonly alias = "pdf";
    constructor(params: PDFGeneratorToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - generates PDF
     */
    performAction(context?: BubbleContext): Promise<PDFGeneratorToolResult>;
    /**
     * Process content based on type
     */
    private processContent;
    /**
     * Process Markdown content
     */
    private processMarkdown;
    /**
     * Process plain text content
     */
    private processText;
    /**
     * Process HTML content
     * Improved parser that handles nested tags and basic styling
     */
    private processHTML;
    /**
     * Apply CSS styles to elements
     * Basic implementation for common CSS properties
     */
    private applyStyles;
    /**
     * Add processed content to PDF
     * Enhanced with image support and style application
     */
    private addContentToPDF;
    /**
     * Apply element styles from CSS
     */
    private applyElementStyles;
    /**
     * Add image to PDF
     * Supports URLs, base64, and local file paths
     */
    private addImageToPDF;
}
export {};
//# sourceMappingURL=pdf-generator-tool.d.ts.map