import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * AirtableBubble - Airtable database operations
 */
export class AirtableBubble extends ServiceBubble<AirtableParams, AirtableResult> {
  bubbleName = 'airtable';
  type = 'service';
  alias = 'Airtable';
  credentialType = 'airtable_api_key';

  params = {
    apiKey: z.string().min(1),
    baseId: z.string().min(1),
    timeout: z.number().int().positive().default(30000)
  };

  private base: any = null;

  async connect() {
    const Airtable = await import('airtable');
    this.base = new Airtable.default({ apiKey: this.params.apiKey }).base(this.params.baseId);
  }

  async listRecords(params: { tableName: string; maxRecords?: number; view?: string; filterByFormula?: string }): Promise<AirtableResult> {
    try {
      const query = this.base(params.tableName).select({
        maxRecords: params.maxRecords,
        view: params.view,
        filterByFormula: params.filterByFormula
      });
      const result = await query.all();
      return { success: true, records: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getRecord(params: { tableName: string; recordId: string }): Promise<AirtableResult> {
    try {
      const result = await this.base(params.tableName).find(params.recordId);
      return { success: true, record: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createRecord(params: { tableName: string; fields: any }): Promise<AirtableResult> {
    try {
      const result = await this.base(params.tableName).create(params.fields);
      return { success: true, record: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async updateRecord(params: { tableName: string; recordId: string; fields: any }): Promise<AirtableResult> {
    try {
      const result = await this.base(params.tableName).update(params.recordId, params.fields);
      return { success: true, record: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async deleteRecord(params: { tableName: string; recordId: string }): Promise<AirtableResult> {
    try {
      const result = await this.base(params.tableName).destroy(params.recordId);
      return { success: true, deleted: params.recordId };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async batchCreate(params: { tableName: string; records: any[] }): Promise<AirtableResult> {
    try {
      const result = await this.base(params.tableName).create(params.records);
      return { success: true, records: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async batchUpdate(params: { tableName: string; records: any[] }): Promise<AirtableResult> {
    try {
      const result = await this.base(params.tableName).update(params.records);
      return { success: true, records: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async queryRecords(params: { tableName: string; fields?: string[]; filterByFormula?: string; maxRecords?: number }): Promise<AirtableResult> {
    try {
      const query = this.base(params.tableName).select({
        fields: params.fields,
        filterByFormula: params.filterByFormula,
        maxRecords: params.maxRecords
      });
      const result = await query.all();
      return { success: true, records: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface AirtableParams {
  apiKey: string;
  baseId: string;
  timeout?: number;
}

export interface AirtableResult {
  success: boolean;
  records?: any[];
  record?: any;
  deleted?: string;
  error?: string;
}
