import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * GoogleSheetsBubble - Google Sheets operations
 */
export class GoogleSheetsBubble extends ServiceBubble<GoogleSheetsParams, GoogleSheetsResult> {
  bubbleName = 'google-sheets';
  type = 'service';
  alias = 'GoogleSheets';
  credentialType = 'google_sheets_api_key';

  params = {
    credentials: z.any(),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { sheets } = await import('@googleapis/sheets');
    const auth = await import('google-auth-library');
    const authClient = auth.JWT.fromJSON(this.params.credentials);
    this.client = sheets({ version: 'v4', auth: authClient });
  }

  async createSpreadsheet(params: { title: string; sheets?: Array<{ title: string }> }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.create({
        requestBody: {
          properties: { title: params.title },
          sheets: params.sheets?.map(s => ({ properties: { title: s.title } }))
        }
      });
      return { success: true, spreadsheet: result.data };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getSheet(params: { spreadsheetId: string; range: string }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.values.get({
        spreadsheetId: params.spreadsheetId,
        range: params.range
      });
      return { success: true, values: result.data.values };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async updateCell(params: { spreadsheetId: string; range: string; values: any[][] }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.values.update({
        spreadsheetId: params.spreadsheetId,
        range: params.range,
        valueInputOption: 'USER_ENTERED',
        requestBody: { values: params.values }
      });
      return { success: true, updatedRows: result.data.updatedRows };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async batchUpdate(params: { spreadsheetId: string; requests: any[] }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.batchUpdate({
        spreadsheetId: params.spreadsheetId,
        requestBody: { requests: params.requests }
      });
      return { success: true, replies: result.data.replies };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async appendRow(params: { spreadsheetId: string; range: string; values: any[] }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.values.append({
        spreadsheetId: params.spreadsheetId,
        range: params.range,
        valueInputOption: 'USER_ENTERED',
        requestBody: { values: [params.values] }
      });
      return { success: true, updates: result.data.updates };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getRow(params: { spreadsheetId: string; range: string }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.values.get({
        spreadsheetId: params.spreadsheetId,
        range: params.range
      });
      return { success: true, row: result.data.values?.[0] || [] };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteRow(params: { spreadsheetId: string; sheetId: number; startIndex: number; endIndex: number }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.batchUpdate({
        spreadsheetId: params.spreadsheetId,
        requestBody: {
          requests: [{
            deleteDimension: {
              range: {
                sheetId: params.sheetId,
                dimension: 'ROWS',
                startIndex: params.startIndex,
                endIndex: params.endIndex
              }
            }
          }]
        }
      });
      return { success: true, deleted: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async addSheet(params: { spreadsheetId: string; title: string }): Promise<GoogleSheetsResult> {
    try {
      const result = await this.client.spreadsheets.batchUpdate({
        spreadsheetId: params.spreadsheetId,
        requestBody: {
          requests: [{
            addSheet: { properties: { title: params.title } }
          }]
        }
      });
      return { success: true, sheet: result.data.replies?.[0]?.addSheet };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface GoogleSheetsParams {
  credentials: any;
  timeout?: number;
}

export interface GoogleSheetsResult {
  success: boolean;
  spreadsheet?: any;
  values?: any[][];
  row?: any[];
  updatedRows?: number;
  replies?: any[];
  updates?: any;
  sheet?: any;
  deleted?: boolean;
  error?: string;
}
