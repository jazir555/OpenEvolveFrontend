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
export enum PageSize {
  A4 = 'A4',
  LETTER = 'Letter',
  LEGAL = 'Legal',
  TABLOID = 'Tabloid',
}

/**
 * Page orientation
 */
export enum PageOrientation {
  PORTRAIT = 'portrait',
  LANDSCAPE = 'landscape',
}

/**
 * PDF Element interface
 */
interface PDFElement {
  type: string;
  content: any;
  style?: Record<string, any>;
}

/**
 * PDF generator parameters schema
 */
const PDFGeneratorToolParamsSchema = z.object({
  // Input content
  content: z
    .string()
    .describe('Content to convert to PDF'),

  contentType: z
    .enum(['html', 'markdown', 'text'])
    .default('html')
    .describe('Type of content'),

  // Output options
  outputPath: z
    .string()
    .optional()
    .describe('Path to save PDF file'),

  returnBase64: z
    .boolean()
    .default(true)
    .describe('Return PDF as base64 encoded string'),

  // Page configuration
  pageSize: z
    .nativeEnum(PageSize)
    .default(PageSize.A4)
    .describe('Page size'),

  orientation: z
    .nativeEnum(PageOrientation)
    .default(PageOrientation.PORTRAIT)
    .describe('Page orientation'),

  margins: z
    .object({
      top: z.number().default(20),
      right: z.number().default(20),
      bottom: z.number().default(20),
      left: z.number().default(20),
    })
    .optional()
    .describe('Page margins in millimeters'),

  // Header and footer
  header: z
    .object({
      text: z.string().optional(),
      height: z.number().default(20),
      showPageNumbers: z.boolean().default(false),
    })
    .optional()
    .describe('Header configuration'),

  footer: z
    .object({
      text: z.string().optional(),
      height: z.number().default(20),
      showPageNumbers: z.boolean().default(true),
    })
    .optional()
    .describe('Footer configuration'),

  // Styling options
  styles: z
    .string()
    .optional()
    .describe('CSS styles for HTML content'),

  // Metadata
  title: z
    .string()
    .optional()
    .describe('PDF title metadata'),

  author: z
    .string()
    .optional()
    .describe('PDF author metadata'),

  subject: z
    .string()
    .optional()
    .describe('PDF subject metadata'),

  keywords: z
    .array(z.string())
    .optional()
    .describe('PDF keywords metadata'),

  // Credentials
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for external PDF services'),
});

/**
 * PDF generator result schema
 */
const PDFGeneratorToolResultSchema = z.object({
  // Result
  success: z.boolean().describe('Whether the PDF generation was successful'),

  // Generated PDF
  pdfData: z
    .string()
    .optional()
    .describe('Base64 encoded PDF data'),

  pdfPath: z
    .string()
    .optional()
    .describe('Path where PDF was saved'),

  // Metadata
  pageCount: z
    .number()
    .optional()
    .describe('Number of pages in PDF'),

  fileSize: z
    .number()
    .optional()
    .describe('Size of PDF in bytes'),

  // Statistics
  stats: z
    .object({
      charactersProcessed: z.number().optional(),
      processingTime: z.number(),
    })
    .describe('Generation statistics'),

  error: z.string().describe('Error message if generation failed'),
});

// Type definitions
type PDFGeneratorToolParams = z.output<typeof PDFGeneratorToolParamsSchema>;
type PDFGeneratorToolResult = z.output<typeof PDFGeneratorToolResultSchema>;
type PDFGeneratorToolParamsInput = z.input<typeof PDFGeneratorToolParamsSchema>;

/**
 * PDF Generator Tool
 * Generate PDF documents from HTML, Markdown, or text
 */
export class PDFGeneratorTool extends ToolBubble<
  PDFGeneratorToolParams,
  PDFGeneratorToolResult
> {
  /**
   * REQUIRED STATIC METADATA
   */
  static readonly type = 'tool' as const;
  static readonly bubbleName: BubbleName = 'pdf-generator-tool';
  static readonly schema = PDFGeneratorToolParamsSchema;
  static readonly resultSchema = PDFGeneratorToolResultSchema;
  static readonly shortDescription =
    'Generate PDF documents from HTML, Markdown, or text';
  static readonly longDescription = `
    A PDF generation tool for creating documents from various input formats.

    Features:
    - Generate PDF from HTML content
    - Generate PDF from Markdown
    - Generate PDF from plain text
    - Custom page sizes (A4, Letter, Legal, Tabloid)
    - Portrait or landscape orientation
    - Configurable margins
    - Headers and footers with page numbers
    - CSS styling support
    - PDF metadata (title, author, subject, keywords)

    Content Types:
    - HTML: Full HTML with CSS styling support
    - Markdown: Markdown conversion to PDF
    - TEXT: Plain text with basic formatting

    Page Sizes:
    - A4: 210 x 297 mm (default)
    - Letter: 8.5 x 11 inches
    - Legal: 8.5 x 14 inches
    - Tabloid: 11 x 17 inches

    Use cases:
    - Report generation
    - Invoice creation
    - Document archiving
    - Certificate generation
    - E-book creation
    - Formatted document export

    Note: This tool requires a PDF generation library.
    Recommended libraries:
    - Node.js: pdfkit, puppeteer, jsPDF
    - Browser: jsPDF, pdfmake
    - Cloud: HTML-to-PDF APIs

    The current implementation provides a placeholder.
    For production use, integrate with a PDF library.
  `;
  static readonly alias = 'pdf';

  constructor(
    params: PDFGeneratorToolParamsInput,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  /**
   * Main action method - generates PDF
   */
  async performAction(
    context?: BubbleContext
  ): Promise<PDFGeneratorToolResult> {
    void context; // Context available but not currently used
    const startTime = Date.now();

    try {
      console.log(`[PDFGeneratorTool] Generating PDF from ${this.params.contentType}`);

      const { content, contentType } = this.params;

      if (!content) {
        throw new Error('Content is required for PDF generation');
      }

      // Dynamic import of PDFKit
      let PDFKit: any;
      try {
        PDFKit = await import('pdfkit');
      } catch (importError) {
        throw new Error('PDFKit library is required for PDF generation. Install it with: npm install pdfkit');
      }

      // Create PDF document
      const doc = new PDFKit.default({
        size: this.params.pageSize,
        layout: this.params.orientation,
        margins: this.params.margins || { top: 20, right: 20, bottom: 20, left: 20 },
        info: {
          Title: this.params.title || 'Generated PDF',
          Author: this.params.author || 'PDF Generator Tool',
          Subject: this.params.subject,
          Keywords: this.params.keywords?.join(', ') || '',
        },
      });

      // Collect PDF chunks
      const chunks: Buffer[] = [];
      doc.on('data', (chunk: Buffer) => chunks.push(chunk));

      // Process content based on type
      const processedContent = await this.processContent(content, contentType, doc);

      // Add content to PDF
      await this.addContentToPDF(doc, processedContent);

      // Finalize PDF
      doc.end();

      // Wait for PDF generation to complete
      const pdfBuffer = await new Promise<Buffer>((resolve) => {
        doc.on('end', () => {
          resolve(Buffer.concat(chunks));
        });
      });

      const pdfData = pdfBuffer.toString('base64');
      const processingTime = Date.now() - startTime;

      // Save to file if output path specified
      let pdfPath: string | undefined;
      if (this.params.outputPath) {
        const fs = await import('fs/promises');
        await fs.writeFile(this.params.outputPath, pdfBuffer);
        pdfPath = this.params.outputPath;
        console.log(`[PDFGeneratorTool] PDF saved to ${pdfPath}`);
      }

      console.log(`[PDFGeneratorTool] PDF generated successfully in ${processingTime}ms`);

      return {
        success: true,
        pdfData,
        pdfPath,
        pageCount: doc.pageCount,
        fileSize: pdfBuffer.length,
        stats: {
          charactersProcessed: content.length,
          processingTime,
        },
        error: '',
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';

      console.error(`[PDFGeneratorTool] Generation failed: ${errorMessage}`);

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
   * Process content based on type
   */
  private async processContent(content: string, contentType: string, doc: any): Promise<any> {
    switch (contentType) {
      case 'markdown':
        return this.processMarkdown(content, doc);

      case 'text':
        return this.processText(content, doc);

      case 'html':
        return this.processHTML(content, doc);

      default:
        throw new Error(`Unsupported content type: ${contentType}`);
    }
  }

  /**
   * Process Markdown content
   */
  private async processMarkdown(markdown: string, doc: any): Promise<PDFElement[]> {
    const elements: PDFElement[] = [];

    const lines = markdown.split('\n');
    let currentParagraph: (string | any)[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Headers
      if (line.startsWith('### ')) {
        if (currentParagraph.length > 0) {
          elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
          currentParagraph = [];
        }
        elements.push({ type: 'header3', content: line.substring(4) });
      } else if (line.startsWith('## ')) {
        if (currentParagraph.length > 0) {
          elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
          currentParagraph = [];
        }
        elements.push({ type: 'header2', content: line.substring(3) });
      } else if (line.startsWith('# ')) {
        if (currentParagraph.length > 0) {
          elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
          currentParagraph = [];
        }
        elements.push({ type: 'header1', content: line.substring(2) });
      }
      // Bold
      else if (line.startsWith('**') || line.startsWith('__')) {
        const boldMatch = line.match(/[*_]{2}([^*_]+)[*_]{2}/);
        if (boldMatch) {
          currentParagraph.push({ text: boldMatch[1], bold: true });
        }
      }
      // Italic
      else if (line.startsWith('*') || line.startsWith('_')) {
        const italicMatch = line.match(/[*_]([^*_]+)[*_]/);
        if (italicMatch) {
          currentParagraph.push({ text: italicMatch[1], italic: true });
        }
      }
      // Links
      else if (line.includes('[') && line.includes('](')) {
        const linkMatch = line.match(/\[([^\]]+)\]\(([^\)]+)\)/);
        if (linkMatch) {
          currentParagraph.push({ text: linkMatch[1], link: linkMatch[2], underline: true, color: 'blue' });
        }
      }
      // Lists
      else if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        if (currentParagraph.length > 0) {
          elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
          currentParagraph = [];
        }
        elements.push({ type: 'bullet', content: line.trim().substring(2) });
      } else if (line.match(/^\d+\.\s/)) {
        if (currentParagraph.length > 0) {
          elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
          currentParagraph = [];
        }
        elements.push({ type: 'numbered', content: line.trim().substring(line.indexOf('. ') + 2) });
      }
      // Code blocks
      else if (line.startsWith('```')) {
        if (currentParagraph.length > 0) {
          elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
          currentParagraph = [];
        }
        // Skip code lines until closing ```
        i++;
        const codeLines: string[] = [];
        while (i < lines.length && !lines[i].startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        elements.push({ type: 'code', content: codeLines.join('\n') });
      }
      // Empty line
      else if (line.trim() === '') {
        if (currentParagraph.length > 0) {
          elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
          currentParagraph = [];
        }
      }
      // Regular text
      else {
        currentParagraph.push(line);
      }
    }

    // Add remaining paragraph
    if (currentParagraph.length > 0) {
      elements.push({ type: 'paragraph', content: currentParagraph.join(' ') });
    }

    return elements;
  }

  /**
   * Process plain text content
   */
  private async processText(text: string, doc: any): Promise<PDFElement[]> {
    const elements: PDFElement[] = [];
    const paragraphs = text.split(/\n\n+/);

    for (const paragraph of paragraphs) {
      elements.push({ type: 'paragraph', content: paragraph });
    }

    return elements;
  }

  /**
   * Process HTML content
   * Improved parser that handles nested tags and basic styling
   */
  private async processHTML(html: string, doc: any): Promise<PDFElement[]> {
    const elements: PDFElement[] = [];

    // Improved HTML parsing with support for nested structures
    const parseNode = (html: string): void => {
      let remaining = html;

      while (remaining.length > 0) {
        // Extract text before tags
        const textMatch = remaining.match(/^([^<]+)/);
        if (textMatch) {
          const text = textMatch[1].trim();
          if (text) {
            elements.push({ type: 'text', content: text });
          }
          remaining = remaining.substring(textMatch[1].length);
        }

        // Handle self-closing tags (img, br, hr)
        const selfClosingMatch = remaining.match(/^<(img|br|hr)([^>]*)\s*\/>/);
        if (selfClosingMatch) {
          const [fullMatch, tagName, attrs] = selfClosingMatch;

          if (tagName === 'img') {
            const srcMatch = attrs.match(/src=["']([^"']+)["']/);
            if (srcMatch) {
              elements.push({ type: 'image', content: srcMatch[1] });
            }
          } else if (tagName === 'br') {
            elements.push({ type: 'newline', content: '' });
          }

          remaining = remaining.substring(fullMatch.length);
          continue;
        }

        // Handle opening tags
        const openTagMatch = remaining.match(/^<(\w+)([^>]*)>/);
        if (openTagMatch) {
          const [fullMatch, tagName, attrs] = openTagMatch;
          const closingTag = `</${tagName}>`;
          const closeIndex = remaining.indexOf(closingTag);

          if (closeIndex !== -1) {
            // Extract content between tags (may contain nested tags)
            const innerContent = remaining.substring(fullMatch.length, closeIndex);
            const text = innerContent.replace(/<[^>]+>/g, '').trim();

            switch (tagName.toLowerCase()) {
              case 'h1':
                elements.push({ type: 'header1', content: text });
                break;
              case 'h2':
                elements.push({ type: 'header2', content: text });
                break;
              case 'h3':
                elements.push({ type: 'header3', content: text });
                break;
              case 'h4':
                elements.push({ type: 'header4', content: text });
                break;
              case 'h5':
                elements.push({ type: 'header5', content: text });
                break;
              case 'h6':
                elements.push({ type: 'header6', content: text });
                break;
              case 'p':
                elements.push({ type: 'paragraph', content: text });
                break;
              case 'b':
              case 'strong':
                elements.push({ type: 'bold', content: text });
                break;
              case 'i':
              case 'em':
                elements.push({ type: 'italic', content: text });
                break;
              case 'u':
                elements.push({ type: 'underline', content: text });
                break;
              case 'a':
                const hrefMatch = attrs.match(/href=["']([^"']+)["']/);
                if (hrefMatch) {
                  elements.push({ type: 'link', content: { text, url: hrefMatch[1] } });
                } else {
                  elements.push({ type: 'paragraph', content: text });
                }
                break;
              case 'ul':
              case 'ol':
                // Parse list items
                const listItems = innerContent.split(/<li[^>]*>(.*?)<\/li>/).filter((item) => item.trim());
                listItems.forEach((item) => {
                  const itemText = item.replace(/<[^>]+>/g, '').trim();
                  if (itemText) {
                    elements.push({
                      type: tagName === 'ul' ? 'bullet' : 'numbered',
                      content: itemText
                    });
                  }
                });
                break;
              case 'li':
                // Already handled by ul/ol
                break;
              case 'code':
                elements.push({ type: 'code', content: text });
                break;
              case 'pre':
                elements.push({ type: 'code', content: text });
                break;
              case 'blockquote':
                elements.push({ type: 'quote', content: text });
                break;
              case 'img':
                const srcMatch = attrs.match(/src=["']([^"']+)["']/);
                if (srcMatch) {
                  elements.push({ type: 'image', content: srcMatch[1] });
                }
                break;
              default:
                if (text) {
                  elements.push({ type: 'paragraph', content: text });
                }
            }

            remaining = remaining.substring(closeIndex + closingTag.length);
          } else {
            // No closing tag found, skip the opening tag
            remaining = remaining.substring(fullMatch.length);
          }
        } else if (remaining.startsWith('<')) {
          // Skip unclosed or self-closing tags
          const tagEnd = remaining.indexOf('>') + 1;
          if (tagEnd > 0) {
            remaining = remaining.substring(tagEnd);
          } else {
            break;
          }
        } else {
          break;
        }
      }
    };

    parseNode(html);

    // Apply CSS styles if provided
    if (this.params.styles) {
      this.applyStyles(elements, this.params.styles);
    }

    return elements;
  }

  /**
   * Apply CSS styles to elements
   * Basic implementation for common CSS properties
   */
  private applyStyles(elements: PDFElement[], css: string): void {
    // Parse CSS rules (simplified)
    const rules: Record<string, any> = {};

    // Extract style rules
    const styleRegex = /([^{]+)\s*\{\s*([^}]+)\s*\}/g;
    let match;

    while ((match = styleRegex.exec(css)) !== null) {
      const selector = match[1].trim();
      const properties = match[2];

      rules[selector] = {};
      const propRegex = /(\w+)\s*:\s*([^;]+);/g;
      let propMatch;

      while ((propMatch = propRegex.exec(properties)) !== null) {
        rules[selector][propMatch[1]] = propMatch[2].trim();
      }
    }

    // Apply styles to elements (simplified - just for demonstration)
    // In production, you'd need a more sophisticated CSS parser
    elements.forEach((element) => {
      // Apply generic styles
      if (rules['*']) {
        if (!element.style) element.style = {};
        Object.assign(element.style, rules['*']);
      }

      // Apply tag-specific styles
      const tag = element.type;
      if (rules[tag]) {
        if (!element.style) element.style = {};
        Object.assign(element.style, rules[tag]);
      }
    });
  }

  /**
   * Add processed content to PDF
   * Enhanced with image support and style application
   */
  private async addContentToPDF(doc: any, elements: PDFElement[]): Promise<void> {
    for (const element of elements) {
      // Apply element styles if present
      if (element.style) {
        this.applyElementStyles(doc, element.style);
      }

      switch (element.type) {
        case 'header1':
          doc.fontSize(24).font('Helvetica-Bold').text(element.content, { continued: false });
          doc.moveDown(0.5);
          break;

        case 'header2':
          doc.fontSize(20).font('Helvetica-Bold').text(element.content, { continued: false });
          doc.moveDown(0.4);
          break;

        case 'header3':
          doc.fontSize(16).font('Helvetica-Bold').text(element.content, { continued: false });
          doc.moveDown(0.3);
          break;

        case 'header4':
          doc.fontSize(14).font('Helvetica-Bold').text(element.content, { continued: false });
          doc.moveDown(0.3);
          break;

        case 'header5':
          doc.fontSize(12).font('Helvetica-Bold').text(element.content, { continued: false });
          doc.moveDown(0.3);
          break;

        case 'header6':
          doc.fontSize(10).font('Helvetica-Bold').text(element.content, { continued: false });
          doc.moveDown(0.3);
          break;

        case 'paragraph':
        case 'text':
          doc.fontSize(12).font('Helvetica').text(element.content, { continued: false });
          doc.moveDown(0.3);
          break;

        case 'bold':
          doc.fontSize(12).font('Helvetica-Bold').text(element.content, { continued: false });
          doc.moveDown(0.3);
          break;

        case 'italic':
          doc.fontSize(12).font('Helvetica-Oblique').text(element.content, { continued: false });
          doc.moveDown(0.3);
          break;

        case 'underline':
          doc.fontSize(12).font('Helvetica').text(element.content, { continued: false, underline: true });
          doc.moveDown(0.3);
          break;

        case 'link':
          doc.fontSize(12).font('Helvetica').fillColor('blue')
            .text(element.content.text, { continued: false, link: element.content.url, underline: true });
          doc.fillColor('black');
          doc.moveDown(0.3);
          break;

        case 'bullet':
          doc.fontSize(12).font('Helvetica').list([element.content], { bulletRadius: 2 });
          break;

        case 'numbered':
          doc.fontSize(12).font('Helvetica').list([element.content], { listType: 'numbered' });
          break;

        case 'code':
          doc.fontSize(10).font('Courier').fillColor('gray').text(element.content, { continued: false });
          doc.fillColor('black');
          doc.moveDown(0.3);
          break;

        case 'quote':
          doc.fontSize(12).font('Helvetica-Oblique').fillColor('gray')
            .text(`"${element.content}"`, { continued: false });
          doc.fillColor('black');
          doc.moveDown(0.3);
          break;

        case 'image':
          await this.addImageToPDF(doc, element.content);
          break;

        case 'newline':
          doc.moveDown(0.2);
          break;
      }

      // Reset styles after each element
      doc.fontSize(12).font('Helvetica').fillColor('black');
    }

    // Add footer with page numbers if specified
    if (this.params.footer?.showPageNumbers) {
      const pages = doc.bufferedPageRange();
      for (let i = 0; i < pages.count; i++) {
        doc.switchToPage(i);
        doc.fontSize(10).text(
          `Page ${i + 1} of ${pages.count}`,
          0,
          doc.page.height - 50,
          { align: 'center' }
        );
      }
    }
  }

  /**
   * Apply element styles from CSS
   */
  private applyElementStyles(doc: any, styles: any): void {
    if (styles['font-size']) {
      const size = parseInt(styles['font-size']);
      if (!isNaN(size)) {
        doc.fontSize(size);
      }
    }

    if (styles['color']) {
      doc.fillColor(styles['color']);
    }

    if (styles['font-family']) {
      const fontFamily = styles['font-family'].replace(/['"]/g, '');
      if (fontFamily.includes('Courier')) {
        doc.font('Courier');
      } else if (fontFamily.includes('Times')) {
        doc.font('Times-Roman');
      } else {
        doc.font('Helvetica');
      }
    }

    if (styles['font-weight'] === 'bold') {
      doc.font('Helvetica-Bold');
    }

    if (styles['font-style'] === 'italic') {
      doc.font('Helvetica-Oblique');
    }
  }

  /**
   * Add image to PDF
   * Supports URLs, base64, and local file paths
   */
  private async addImageToPDF(doc: any, imageSource: string): Promise<void> {
    try {
      let imageBuffer: Buffer;

      // Check if it's a base64 image
      if (imageSource.startsWith('data:image')) {
        const base64Data = imageSource.split(',')[1];
        imageBuffer = Buffer.from(base64Data, 'base64');
      }
      // Check if it's a URL
      else if (imageSource.startsWith('http://') || imageSource.startsWith('https://')) {
        // For URLs, you would need to fetch the image
        // This requires https module or a library like axios
        console.warn('[PDFGeneratorTool] URL images require HTTP client implementation');
        return;
      }
      // Assume it's a local file path
      else {
        const fs = await import('fs/promises');
        try {
          imageBuffer = await fs.readFile(imageSource);
        } catch (error) {
          console.warn(`[PDFGeneratorTool] Could not read image file: ${imageSource}`);
          return;
        }
      }

      // Add image to PDF with proper sizing
      const maxWidth = doc.page.width - (doc.page.margins.left + doc.page.margins.right);
      const maxHeight = 300; // Max height in points

      doc.image(imageBuffer, {
        fit: [maxWidth, maxHeight],
        align: 'center',
      });

      doc.moveDown(0.5);
    } catch (error) {
      console.error('[PDFGeneratorTool] Failed to add image:', error);
    }
  }

}
