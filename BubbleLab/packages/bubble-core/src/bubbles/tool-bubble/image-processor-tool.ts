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
export enum ImageFormat {
  JPEG = 'jpeg',
  PNG = 'png',
  WEBP = 'webp',
  TIFF = 'tiff',
  GIF = 'gif',
}

/**
 * Image processor parameters schema
 */
const ImageProcessorToolParamsSchema = z.object({
  // Input
  imagePath: z
    .string()
    .optional()
    .describe('Path to input image file'),

  imageData: z
    .string()
    .optional()
    .describe('Base64 encoded image data'),

  imageUrl: z
    .string()
    .url()
    .optional()
    .describe('URL of image to process'),

  // Operations
  operation: z
    .enum(['resize', 'convert', 'metadata', 'optimize', 'crop', 'rotate', 'filter'])
    .describe('Image processing operation'),

  // Resize options
  width: z
    .number()
    .int()
    .positive()
    .optional()
    .describe('Target width in pixels'),

  height: z
    .number()
    .int()
    .positive()
    .optional()
    .describe('Target height in pixels'),

  fit: z
    .enum(['cover', 'contain', 'fill', 'inside', 'outside'])
    .default('cover')
    .optional()
    .describe('Fit method for resizing'),

  // Convert options
  format: z
    .nativeEnum(ImageFormat)
    .optional()
    .describe('Target image format'),

  quality: z
    .number()
    .int()
    .min(1)
    .max(100)
    .default(80)
    .optional()
    .describe('Image quality (1-100)'),

  // Crop options
  cropLeft: z
    .number()
    .int()
    .optional()
    .describe('Crop left position'),

  cropTop: z
    .number()
    .int()
    .optional()
    .describe('Crop top position'),

  cropWidth: z
    .number()
    .int()
    .positive()
    .optional()
    .describe('Crop width'),

  cropHeight: z
    .number()
    .int()
    .positive()
    .optional()
    .describe('Crop height'),

  // Rotate options
  rotation: z
    .number()
    .int()
    .optional()
    .describe('Rotation angle in degrees'),

  // Filter options
  filter: z
    .enum(['grayscale', 'blur', 'sharpen', 'negate', 'normalize'])
    .optional()
    .describe('Filter to apply'),

  // Output
  outputPath: z
    .string()
    .optional()
    .describe('Output path for processed image'),

  returnBase64: z
    .boolean()
    .default(true)
    .describe('Return processed image as base64'),

  // Credentials
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for cloud storage'),
});

/**
 * Image metadata schema
 */
const ImageMetadataSchema = z.object({
  width: z.number().describe('Image width in pixels'),
  height: z.number().describe('Image height in pixels'),
  format: z.string().describe('Image format'),
  size: z.number().describe('File size in bytes'),
  colorSpace: z.string().optional().describe('Color space'),
  hasAlpha: z.boolean().optional().describe('Whether image has alpha channel'),
  orientation: z.number().optional().describe('Image orientation'),
});

/**
 * Image processor result schema
 */
const ImageProcessorToolResultSchema = z.object({
  // Result
  success: z.boolean().describe('Whether the operation was successful'),

  // Processed image
  imageData: z
    .string()
    .optional()
    .describe('Base64 encoded processed image'),

  imagePath: z
    .string()
    .optional()
    .describe('Path to saved processed image'),

  // Metadata
  metadata: z
    .object({
      width: z.number(),
      height: z.number(),
      format: z.string(),
      size: z.number(),
    })
    .optional()
    .describe('Image metadata'),

  // Statistics
  stats: z
    .object({
      originalSize: z.number().optional(),
      processedSize: z.number().optional(),
      compressionRatio: z.number().optional(),
      processingTime: z.number(),
    })
    .describe('Processing statistics'),

  error: z.string().describe('Error message if operation failed'),
});

// Type definitions
type ImageProcessorToolParams = z.output<typeof ImageProcessorToolParamsSchema>;
type ImageProcessorToolResult = z.output<typeof ImageProcessorToolResultSchema>;
type ImageProcessorToolParamsInput = z.input<typeof ImageProcessorToolParamsSchema>;

/**
 * Image Processor Tool
 * Basic image processing operations
 */
export class ImageProcessorTool extends ToolBubble<
  ImageProcessorToolParams,
  ImageProcessorToolResult
> {
  /**
   * REQUIRED STATIC METADATA
   */
  static readonly type = 'tool' as const;
  static readonly bubbleName: BubbleName = 'image-processor-tool';
  static readonly schema = ImageProcessorToolParamsSchema;
  static readonly resultSchema = ImageProcessorToolResultSchema;
  static readonly shortDescription =
    'Process images with resize, convert, filter operations';
  static readonly longDescription = `
    A tool for basic image processing operations.

    Features:
    - RESIZE: Resize images with various fit methods
    - CONVERT: Convert between image formats (JPEG, PNG, WebP, TIFF)
    - METADATA: Extract image metadata (dimensions, format, size)
    - OPTIMIZE: Compress images for web delivery
    - CROP: Crop images to specific dimensions
    - ROTATE: Rotate images by specified angle
    - FILTER: Apply basic filters (grayscale, blur, sharpen)

    Resize Fit Methods:
    - cover: Cover the area (crop if needed)
    - contain: Contain within area (letterbox)
    - fill: Fill the area (stretch)
    - inside: Fit inside the area
    - outside: Fit outside the area

    Supported Formats:
    - JPEG: Best for photographs
    - PNG: Best for graphics with transparency
    - WebP: Modern format with good compression
    - TIFF: High-quality format for printing
    - GIF: For simple animations

    Filters:
    - grayscale: Convert to grayscale
    - blur: Apply blur effect
    - sharpen: Sharpen image
    - negate: Invert colors
    - normalize: Normalize color levels

    Use cases:
    - Thumbnail generation
    - Image optimization for web
    - Format conversion
    - Batch image processing
    - Image metadata extraction

    Note: This tool requires the Sharp library for Node.js environments.
    For browser environments, consider using Canvas API.
  `;
  static readonly alias = 'image';

  constructor(
    params: ImageProcessorToolParamsInput,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  /**
   * Main action method - performs image processing
   */
  async performAction(
    context?: BubbleContext
  ): Promise<ImageProcessorToolResult> {
    void context; // Context available but not currently used
    const startTime = Date.now();

    try {
      console.log(`[ImageProcessorTool] Executing operation: ${this.params.operation}`);

      // Dynamic import of sharp to handle optional dependency
      let sharp: any;
      try {
        sharp = (await import('sharp')).default;
      } catch (importError) {
        throw new Error('Sharp library is required for image processing. Install it with: npm install sharp');
      }

      let image: any;
      let originalSize = 0;

      // Load image from different sources
      if (this.params.imagePath) {
        image = sharp(this.params.imagePath);
        const metadata = await image.metadata();
        originalSize = metadata.size || 0;
      } else if (this.params.imageData) {
        const buffer = Buffer.from(this.params.imageData, 'base64');
        originalSize = buffer.length;
        image = sharp(buffer);
      } else if (this.params.imageUrl) {
        try {
          const response = await fetch(this.params.imageUrl);
          if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.statusText}`);
          }
          const buffer = Buffer.from(await response.arrayBuffer());
          originalSize = buffer.length;
          image = sharp(buffer);
        } catch (fetchError) {
          throw new Error(`Failed to load image from URL: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`);
        }
      } else {
        throw new Error('No image input provided. Specify imagePath, imageData, or imageUrl');
      }

      // Get original metadata
      const metadata = await image.metadata();

      // Apply the requested operation
      switch (this.params.operation) {
        case 'resize':
          image = this.applyResize(image);
          break;

        case 'convert':
          image = this.applyConvert(image);
          break;

        case 'metadata':
          // Metadata only, no processing needed
          break;

        case 'optimize':
          image = this.applyOptimize(image);
          break;

        case 'crop':
          image = this.applyCrop(image, metadata);
          break;

        case 'rotate':
          image = this.applyRotate(image);
          break;

        case 'filter':
          image = this.applyFilter(image);
          break;

        default:
          throw new Error(`Unsupported operation: ${this.params.operation}`);
      }

      // Get processed metadata after operation
      const processedMetadata = await image.clone().metadata();

      // Generate output
      let imageData: string | undefined;
      let imagePath: string | undefined;
      let processedSize = 0;

      if (this.params.returnBase64 || !this.params.outputPath) {
        const buffer = await image.toBuffer();
        processedSize = buffer.length;
        imageData = buffer.toString('base64');
      }

      if (this.params.outputPath) {
        await image.toFile(this.params.outputPath);
        imagePath = this.params.outputPath;
        if (!processedSize) {
          const fs = await import('fs/promises');
          const stats = await fs.stat(this.params.outputPath);
          processedSize = stats.size;
        }
      }

      // Calculate statistics
      const processingTime = Date.now() - startTime;
      const compressionRatio = originalSize > 0 ? (1 - processedSize / originalSize) * 100 : 0;

      console.log(`[ImageProcessorTool] Operation completed successfully in ${processingTime}ms`);

      return {
        success: true,
        imageData,
        imagePath,
        metadata: {
          width: processedMetadata.width || metadata.width || 0,
          height: processedMetadata.height || metadata.height || 0,
          format: processedMetadata.format || metadata.format || 'unknown',
          size: processedSize,
        },
        stats: {
          originalSize,
          processedSize,
          compressionRatio: Math.round(compressionRatio * 100) / 100,
          processingTime,
        },
        error: '',
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';

      console.error(`[ImageProcessorTool] Operation failed: ${errorMessage}`);

      return {
        success: false,
        stats: {
          processingTime: Date.now() - startTime,
        },
        error: errorMessage,
      };
    }
  }

  /**
   * Apply resize operation
   */
  private applyResize(image: any): any {
    const { width, height, fit } = this.params;

    if (!width && !height) {
      throw new Error('Resize operation requires at least width or height parameter');
    }

    const resizeOptions: any = {};

    if (fit) {
      resizeOptions.fit = fit;
    }

    // Preserve aspect ratio by default if only one dimension is provided
    if (width && !height) {
      return image.resize(width, null, resizeOptions);
    } else if (!width && height) {
      return image.resize(null, height, resizeOptions);
    } else {
      return image.resize(width, height, resizeOptions);
    }
  }

  /**
   * Apply format conversion
   */
  private applyConvert(image: any): any {
    const { format, quality } = this.params;

    if (!format) {
      throw new Error('Convert operation requires format parameter');
    }

    const formatOptions: any = {};

    if (quality !== undefined) {
      formatOptions.quality = quality;
    }

    return image.toFormat(format, formatOptions);
  }

  /**
   * Apply optimization for web
   */
  private applyOptimize(image: any): any {
    const { quality } = this.params;

    // Convert to WebP with specified quality for best compression
    return image.webp({
      quality: quality || 80,
      effort: 6, // Maximum compression effort
    });
  }

  /**
   * Apply crop operation
   */
  private applyCrop(image: any, metadata: any): any {
    const { cropLeft, cropTop, cropWidth, cropHeight } = this.params;

    if (!cropWidth && !cropHeight) {
      throw new Error('Crop operation requires at least cropWidth or cropHeight');
    }

    const extractOptions: any = {
      left: cropLeft || 0,
      top: cropTop || 0,
      width: cropWidth || metadata.width,
      height: cropHeight || metadata.height,
    };

    // Validate crop dimensions
    if (extractOptions.left + extractOptions.width > (metadata.width || 0)) {
      throw new Error('Crop area extends beyond image width');
    }

    if (extractOptions.top + extractOptions.height > (metadata.height || 0)) {
      throw new Error('Crop area extends beyond image height');
    }

    return image.extract(extractOptions);
  }

  /**
   * Apply rotation operation
   */
  private applyRotate(image: any): any {
    const { rotation } = this.params;

    if (rotation === undefined) {
      throw new Error('Rotate operation requires rotation parameter');
    }

    return image.rotate(rotation);
  }

  /**
   * Apply filter operation
   */
  private applyFilter(image: any): any {
    const { filter } = this.params;

    if (!filter) {
      throw new Error('Filter operation requires filter parameter');
    }

    switch (filter) {
      case 'grayscale':
        return image.grayscale();

      case 'blur':
        return image.blur();

      case 'sharpen':
        return image.sharpen();

      case 'negate':
        return image.negate();

      case 'normalize':
        return image.normalize();

      default:
        throw new Error(`Unsupported filter: ${filter}`);
    }
  }
}
