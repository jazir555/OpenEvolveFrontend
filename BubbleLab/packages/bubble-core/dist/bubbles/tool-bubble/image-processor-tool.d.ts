/**
 * IMAGE PROCESSOR TOOL
 *
 * A tool bubble for basic image processing operations.
 * Uses Sharp library for image manipulation in Node.js environments.
 *
 * Features:
 * - Image resizing and scaling
 * - Format conversion (JPEG, PNG, WebP, TIFF)
 * - Image metadata extraction
 * - Basic image filters
 * - Compression optimization
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Image format types
 */
export declare enum ImageFormat {
    JPEG = "jpeg",
    PNG = "png",
    WEBP = "webp",
    TIFF = "tiff",
    GIF = "gif"
}
/**
 * Image processor parameters schema
 */
declare const ImageProcessorToolParamsSchema: z.ZodObject<{
    imagePath: z.ZodOptional<z.ZodString>;
    imageData: z.ZodOptional<z.ZodString>;
    imageUrl: z.ZodOptional<z.ZodString>;
    operation: z.ZodEnum<["resize", "convert", "metadata", "optimize", "crop", "rotate", "filter"]>;
    width: z.ZodOptional<z.ZodNumber>;
    height: z.ZodOptional<z.ZodNumber>;
    fit: z.ZodOptional<z.ZodDefault<z.ZodEnum<["cover", "contain", "fill", "inside", "outside"]>>>;
    format: z.ZodOptional<z.ZodNativeEnum<typeof ImageFormat>>;
    quality: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    cropLeft: z.ZodOptional<z.ZodNumber>;
    cropTop: z.ZodOptional<z.ZodNumber>;
    cropWidth: z.ZodOptional<z.ZodNumber>;
    cropHeight: z.ZodOptional<z.ZodNumber>;
    rotation: z.ZodOptional<z.ZodNumber>;
    filter: z.ZodOptional<z.ZodEnum<["grayscale", "blur", "sharpen", "negate", "normalize"]>>;
    outputPath: z.ZodOptional<z.ZodString>;
    returnBase64: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "filter" | "metadata" | "resize" | "convert" | "optimize" | "crop" | "rotate";
    returnBase64: boolean;
    filter?: "normalize" | "grayscale" | "blur" | "sharpen" | "negate" | undefined;
    format?: ImageFormat | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    width?: number | undefined;
    height?: number | undefined;
    imageUrl?: string | undefined;
    quality?: number | undefined;
    imageData?: string | undefined;
    imagePath?: string | undefined;
    fit?: "fill" | "cover" | "contain" | "inside" | "outside" | undefined;
    cropLeft?: number | undefined;
    cropTop?: number | undefined;
    cropWidth?: number | undefined;
    cropHeight?: number | undefined;
    rotation?: number | undefined;
    outputPath?: string | undefined;
}, {
    operation: "filter" | "metadata" | "resize" | "convert" | "optimize" | "crop" | "rotate";
    filter?: "normalize" | "grayscale" | "blur" | "sharpen" | "negate" | undefined;
    format?: ImageFormat | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    width?: number | undefined;
    height?: number | undefined;
    imageUrl?: string | undefined;
    quality?: number | undefined;
    imageData?: string | undefined;
    imagePath?: string | undefined;
    fit?: "fill" | "cover" | "contain" | "inside" | "outside" | undefined;
    cropLeft?: number | undefined;
    cropTop?: number | undefined;
    cropWidth?: number | undefined;
    cropHeight?: number | undefined;
    rotation?: number | undefined;
    outputPath?: string | undefined;
    returnBase64?: boolean | undefined;
}>;
/**
 * Image processor result schema
 */
declare const ImageProcessorToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    imageData: z.ZodOptional<z.ZodString>;
    imagePath: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        width: z.ZodNumber;
        height: z.ZodNumber;
        format: z.ZodString;
        size: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        format: string;
        size: number;
        width: number;
        height: number;
    }, {
        format: string;
        size: number;
        width: number;
        height: number;
    }>>;
    stats: z.ZodObject<{
        originalSize: z.ZodOptional<z.ZodNumber>;
        processedSize: z.ZodOptional<z.ZodNumber>;
        compressionRatio: z.ZodOptional<z.ZodNumber>;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        originalSize?: number | undefined;
        processedSize?: number | undefined;
        compressionRatio?: number | undefined;
    }, {
        processingTime: number;
        originalSize?: number | undefined;
        processedSize?: number | undefined;
        compressionRatio?: number | undefined;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        originalSize?: number | undefined;
        processedSize?: number | undefined;
        compressionRatio?: number | undefined;
    };
    metadata?: {
        format: string;
        size: number;
        width: number;
        height: number;
    } | undefined;
    imageData?: string | undefined;
    imagePath?: string | undefined;
}, {
    error: string;
    success: boolean;
    stats: {
        processingTime: number;
        originalSize?: number | undefined;
        processedSize?: number | undefined;
        compressionRatio?: number | undefined;
    };
    metadata?: {
        format: string;
        size: number;
        width: number;
        height: number;
    } | undefined;
    imageData?: string | undefined;
    imagePath?: string | undefined;
}>;
type ImageProcessorToolParams = z.output<typeof ImageProcessorToolParamsSchema>;
type ImageProcessorToolResult = z.output<typeof ImageProcessorToolResultSchema>;
type ImageProcessorToolParamsInput = z.input<typeof ImageProcessorToolParamsSchema>;
/**
 * Image Processor Tool
 * Basic image processing operations
 */
export declare class ImageProcessorTool extends ToolBubble<ImageProcessorToolParams, ImageProcessorToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        imagePath: z.ZodOptional<z.ZodString>;
        imageData: z.ZodOptional<z.ZodString>;
        imageUrl: z.ZodOptional<z.ZodString>;
        operation: z.ZodEnum<["resize", "convert", "metadata", "optimize", "crop", "rotate", "filter"]>;
        width: z.ZodOptional<z.ZodNumber>;
        height: z.ZodOptional<z.ZodNumber>;
        fit: z.ZodOptional<z.ZodDefault<z.ZodEnum<["cover", "contain", "fill", "inside", "outside"]>>>;
        format: z.ZodOptional<z.ZodNativeEnum<typeof ImageFormat>>;
        quality: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        cropLeft: z.ZodOptional<z.ZodNumber>;
        cropTop: z.ZodOptional<z.ZodNumber>;
        cropWidth: z.ZodOptional<z.ZodNumber>;
        cropHeight: z.ZodOptional<z.ZodNumber>;
        rotation: z.ZodOptional<z.ZodNumber>;
        filter: z.ZodOptional<z.ZodEnum<["grayscale", "blur", "sharpen", "negate", "normalize"]>>;
        outputPath: z.ZodOptional<z.ZodString>;
        returnBase64: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "filter" | "metadata" | "resize" | "convert" | "optimize" | "crop" | "rotate";
        returnBase64: boolean;
        filter?: "normalize" | "grayscale" | "blur" | "sharpen" | "negate" | undefined;
        format?: ImageFormat | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        width?: number | undefined;
        height?: number | undefined;
        imageUrl?: string | undefined;
        quality?: number | undefined;
        imageData?: string | undefined;
        imagePath?: string | undefined;
        fit?: "fill" | "cover" | "contain" | "inside" | "outside" | undefined;
        cropLeft?: number | undefined;
        cropTop?: number | undefined;
        cropWidth?: number | undefined;
        cropHeight?: number | undefined;
        rotation?: number | undefined;
        outputPath?: string | undefined;
    }, {
        operation: "filter" | "metadata" | "resize" | "convert" | "optimize" | "crop" | "rotate";
        filter?: "normalize" | "grayscale" | "blur" | "sharpen" | "negate" | undefined;
        format?: ImageFormat | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        width?: number | undefined;
        height?: number | undefined;
        imageUrl?: string | undefined;
        quality?: number | undefined;
        imageData?: string | undefined;
        imagePath?: string | undefined;
        fit?: "fill" | "cover" | "contain" | "inside" | "outside" | undefined;
        cropLeft?: number | undefined;
        cropTop?: number | undefined;
        cropWidth?: number | undefined;
        cropHeight?: number | undefined;
        rotation?: number | undefined;
        outputPath?: string | undefined;
        returnBase64?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        imageData: z.ZodOptional<z.ZodString>;
        imagePath: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodObject<{
            width: z.ZodNumber;
            height: z.ZodNumber;
            format: z.ZodString;
            size: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            format: string;
            size: number;
            width: number;
            height: number;
        }, {
            format: string;
            size: number;
            width: number;
            height: number;
        }>>;
        stats: z.ZodObject<{
            originalSize: z.ZodOptional<z.ZodNumber>;
            processedSize: z.ZodOptional<z.ZodNumber>;
            compressionRatio: z.ZodOptional<z.ZodNumber>;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            originalSize?: number | undefined;
            processedSize?: number | undefined;
            compressionRatio?: number | undefined;
        }, {
            processingTime: number;
            originalSize?: number | undefined;
            processedSize?: number | undefined;
            compressionRatio?: number | undefined;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            originalSize?: number | undefined;
            processedSize?: number | undefined;
            compressionRatio?: number | undefined;
        };
        metadata?: {
            format: string;
            size: number;
            width: number;
            height: number;
        } | undefined;
        imageData?: string | undefined;
        imagePath?: string | undefined;
    }, {
        error: string;
        success: boolean;
        stats: {
            processingTime: number;
            originalSize?: number | undefined;
            processedSize?: number | undefined;
            compressionRatio?: number | undefined;
        };
        metadata?: {
            format: string;
            size: number;
            width: number;
            height: number;
        } | undefined;
        imageData?: string | undefined;
        imagePath?: string | undefined;
    }>;
    static readonly shortDescription = "Process images with resize, convert, filter operations";
    static readonly longDescription = "\n    A tool for basic image processing operations.\n\n    Features:\n    - RESIZE: Resize images with various fit methods\n    - CONVERT: Convert between image formats (JPEG, PNG, WebP, TIFF)\n    - METADATA: Extract image metadata (dimensions, format, size)\n    - OPTIMIZE: Compress images for web delivery\n    - CROP: Crop images to specific dimensions\n    - ROTATE: Rotate images by specified angle\n    - FILTER: Apply basic filters (grayscale, blur, sharpen)\n\n    Resize Fit Methods:\n    - cover: Cover the area (crop if needed)\n    - contain: Contain within area (letterbox)\n    - fill: Fill the area (stretch)\n    - inside: Fit inside the area\n    - outside: Fit outside the area\n\n    Supported Formats:\n    - JPEG: Best for photographs\n    - PNG: Best for graphics with transparency\n    - WebP: Modern format with good compression\n    - TIFF: High-quality format for printing\n    - GIF: For simple animations\n\n    Filters:\n    - grayscale: Convert to grayscale\n    - blur: Apply blur effect\n    - sharpen: Sharpen image\n    - negate: Invert colors\n    - normalize: Normalize color levels\n\n    Use cases:\n    - Thumbnail generation\n    - Image optimization for web\n    - Format conversion\n    - Batch image processing\n    - Image metadata extraction\n\n    Note: This tool requires the Sharp library for Node.js environments.\n    For browser environments, consider using Canvas API.\n  ";
    static readonly alias = "image";
    constructor(params: ImageProcessorToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs image processing
     */
    performAction(context?: BubbleContext): Promise<ImageProcessorToolResult>;
    /**
     * Apply resize operation
     */
    private applyResize;
    /**
     * Apply format conversion
     */
    private applyConvert;
    /**
     * Apply optimization for web
     */
    private applyOptimize;
    /**
     * Apply crop operation
     */
    private applyCrop;
    /**
     * Apply rotation operation
     */
    private applyRotate;
    /**
     * Apply filter operation
     */
    private applyFilter;
}
export {};
//# sourceMappingURL=image-processor-tool.d.ts.map