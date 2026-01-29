// Template for Nanobanana Image Pipeline (Google Sheets, AI Agent, Google Drive)

import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';

export const templateCode = `import { z } from 'zod';

import {
  BubbleFlow,
  AIAgentBubble,
  GoogleSheetsBubble,
  GoogleDriveBubble,
  type WebhookEvent,
} from '@bubblelab/bubble-core';

export interface Output {
  processedRows: number;
  message: string;
  results: Array<{
    rowNumber: number;
    imageUrl: string;
    prompt: string;
    driveLink?: string;
  }>;
}

export interface NanobananaPayload extends WebhookEvent {
  /**
   * The spreadsheet ID containing image URLs and prompts.
   * @canBeFile false
   */
  spreadsheetId: string;
  /**
   * Sheet name to read from. Defaults to "Sheet1".
   * @canBeFile false
   */
  sheetName?: string;
  /**
   * Column header that contains image URLs. Defaults to "image_url".
   * @canBeFile false
   */
  imageUrlColumn?: string;
  /**
   * Column header that contains prompts. Defaults to "prompt".
   * @canBeFile false
   */
  promptColumn?: string;
  /**
   * Column header to store Google Drive links. Defaults to "drive_link".
   * @canBeFile false
   */
  driveLinkColumn?: string;
  /**
   * Optional Google Drive folder ID where generated images will be uploaded.
   * @canBeFile false
   */
  outputFolderId?: string;
  /**
   * Maximum number of rows to process per run (defaults to 25).
   * @canBeFile false
   */
  maxRows?: number;
}

export class NanobananaImagePipeline extends BubbleFlow<'webhook/http'> {
  private columnIndexToLetter(index: number): string {
    let n = index + 1;
    let letters = '';
    while (n > 0) {
      const rem = (n - 1) % 26;
      letters = String.fromCharCode(65 + rem) + letters;
      n = Math.floor((n - 1) / 26);
    }
    return letters;
  }

  private async getSheetValues(spreadsheetId: string, sheetName: string) {
    const sheet = new GoogleSheetsBubble({
      operation: 'getSheetData',
      spreadsheetId,
      sheetName,
    });

    const result = await sheet.action();
    if (!result.success || !result.data?.result?.values) {
      throw new Error(result.error || 'Failed to fetch sheet data');
    }

    return result.data.result.values as Array<Array<any>>;
  }

  private async updateCell(
    spreadsheetId: string,
    sheetName: string,
    columnIndex: number,
    rowIndex: number,
    value: string
  ) {
    const columnLetter = this.columnIndexToLetter(columnIndex);
    const range = `\${sheetName}!\${columnLetter}\${rowIndex}`;
    const update = new GoogleSheetsBubble({
      operation: 'updateCell',
      spreadsheetId,
      range,
      value,
    });

    const result = await update.action();
    if (!result.success) {
      throw new Error(result.error || 'Failed to update sheet cell');
    }
  }

  private async generateImage(imageUrl: string, prompt: string): Promise<string> {
    const agent = new AIAgentBubble({
      model: { model: '${RECOMMENDED_MODELS.IMAGE}' },
      systemPrompt:
        'You are an expert image generator. Enhance or transform the provided image based on the prompt.',
      message: prompt,
      images: [{ type: 'url' as const, url: imageUrl }],
      tools: [],
    });

    const result = await agent.action();
    if (!result.success || !result.data?.response) {
      throw new Error(result.error || 'Image generation failed');
    }

    return result.data.response;
  }

  private extractBase64Image(response: string): string {
    try {
      const parsed = JSON.parse(response);
      if (Array.isArray(parsed)) {
        const imageElement = parsed.find((el: any) => el.type === 'inlineData');
        if (imageElement?.inlineData?.data) {
          return imageElement.inlineData.data;
        }
      }
    } catch {
      // Ignore parsing issues and fall back to raw response
    }

    return response;
  }

  private async uploadToDrive(
    imageBase64: string,
    fileName: string,
    folderId?: string
  ) {
    const drive = new GoogleDriveBubble({
      operation: 'uploadFile',
      fileName,
      content: imageBase64,
      mimeType: 'image/png',
      parents: folderId ? [folderId] : undefined,
    });

    const result = await drive.action();
    if (!result.success || !result.data?.file) {
      throw new Error(result.data?.error || 'Failed to upload to Google Drive');
    }

    return result.data.file;
  }

  async handle(payload: NanobananaPayload): Promise<Output> {
    const {
      spreadsheetId,
      sheetName = 'Sheet1',
      imageUrlColumn = 'image_url',
      promptColumn = 'prompt',
      driveLinkColumn = 'drive_link',
      outputFolderId,
      maxRows = 25,
    } = payload;

    const values = await this.getSheetValues(spreadsheetId, sheetName);
    if (values.length < 2) {
      return { processedRows: 0, message: 'No data rows found.', results: [] };
    }

    const headers = values[0].map((header) => String(header).trim());
    const imageIndex = headers.indexOf(imageUrlColumn);
    const promptIndex = headers.indexOf(promptColumn);
    let driveIndex = headers.indexOf(driveLinkColumn);

    if (imageIndex === -1) {
      throw new Error(`Missing column: \${imageUrlColumn}`);
    }
    if (promptIndex === -1) {
      throw new Error(`Missing column: \${promptColumn}`);
    }

    if (driveIndex === -1) {
      headers.push(driveLinkColumn);
      driveIndex = headers.length - 1;

      const headerUpdate = new GoogleSheetsBubble({
        operation: 'setValues',
        spreadsheetId,
        range: `\${sheetName}!A1`,
        values: [headers],
      });

      const headerResult = await headerUpdate.action();
      if (!headerResult.success) {
        throw new Error(headerResult.error || 'Failed to update headers');
      }
    }

    const results: Output['results'] = [];
    let processedRows = 0;

    for (
      let rowIndex = 1;
      rowIndex < values.length && processedRows < maxRows;
      rowIndex++
    ) {
      const row = values[rowIndex];
      const imageUrl = String(row[imageIndex] || '').trim();
      if (!imageUrl) continue;

      const prompt = String(
        row[promptIndex] ||
          'Enhance this image with a clean, modern aesthetic.'
      ).trim();

      const rawResponse = await this.generateImage(imageUrl, prompt);
      const base64Image = this.extractBase64Image(rawResponse);
      const fileName = `nanobanana-\${rowIndex + 1}-\${Date.now()}.png`;
      const driveFile = await this.uploadToDrive(
        base64Image,
        fileName,
        outputFolderId
      );
      const driveLink =
        driveFile.webViewLink || driveFile.webContentLink || '';

      await this.updateCell(
        spreadsheetId,
        sheetName,
        driveIndex,
        rowIndex + 1,
        driveLink || fileName
      );

      results.push({
        rowNumber: rowIndex + 1,
        imageUrl,
        prompt,
        driveLink,
      });
      processedRows += 1;
    }

    return {
      processedRows,
      message:
        processedRows > 0
          ? 'Images generated and uploaded successfully.'
          : 'No rows processed.',
      results,
    };
  }
}`;

export const metadata = {
  inputsSchema: JSON.stringify({
    type: 'object',
    properties: {
      spreadsheetId: {
        type: 'string',
        description: 'Google Sheets spreadsheet ID containing the image queue.',
      },
      sheetName: {
        type: 'string',
        description: 'Sheet name to read from (default: Sheet1).',
      },
      imageUrlColumn: {
        type: 'string',
        description: 'Column header for image URLs (default: image_url).',
      },
      promptColumn: {
        type: 'string',
        description: 'Column header for prompts (default: prompt).',
      },
      driveLinkColumn: {
        type: 'string',
        description: 'Column header for output drive links (default: drive_link).',
      },
      outputFolderId: {
        type: 'string',
        description: 'Optional Google Drive folder ID for outputs.',
      },
      maxRows: {
        type: 'number',
        description: 'Maximum rows to process per run (default: 25).',
      },
    },
    required: ['spreadsheetId'],
  }),
  requiredCredentials: {
    'google-sheets': ['read', 'write'],
    'google-drive': ['write'],
    'ai-agent': ['generate'],
  },
};
