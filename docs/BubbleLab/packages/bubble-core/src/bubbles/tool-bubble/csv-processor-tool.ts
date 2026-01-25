import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * CSVProcessorTool - CSV file processing operations
 */
export class CSVProcessorTool extends ToolBubble<CSVProcessorParams, CSVProcessorResult> {
  bubbleName = 'csv-processor';
  type = 'tool';
  alias = 'csv-processor';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<CSVProcessorResult> {
    try {
      const result = await this.parse(input);
      return { success: true, data: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async parse(params: { csv: string; delimiter?: string }): Promise<CSVProcessorResult> {
    try {
      const delimiter = params.delimiter || ',';
      const lines = params.csv.trim().split('\n');
      const headers = lines[0].split(delimiter);
      const data = lines.slice(1).map(line => {
        const values = line.split(delimiter);
        const row = {};
        headers.forEach((header, i) => {
          row[header.trim()] = values[i]?.trim() || '';
        });
        return row;
      });
      return { success: true, headers, rows: data, count: data.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async transform(params: { data: any[]; transformations: any[] }): Promise<CSVProcessorResult> {
    try {
      let transformed = params.data;
      params.transformations.forEach(t => {
        if (t.type === 'map') {
          transformed = transformed.map(row => ({ ...row, [t.field]: t.value }));
        } else if (t.type === 'filter') {
          transformed = transformed.filter(row => row[t.field] === t.value);
        }
      });
      return { success: true, transformed };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async validate(params: { csv: string; schema?: any }): Promise<CSVProcessorResult> {
    try {
      const lines = params.csv.split('\n');
      const errors = [];
      lines.forEach((line, i) => {
        const values = line.split(',');
        if (values.length < 2) {
          errors.push({ line: i + 1, message: 'Insufficient columns' });
        }
      });
      return { success: true, valid: errors.length === 0, errors };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async merge(params: { csvs: string[] }): Promise<CSVProcessorResult> {
    try {
      const allData = [];
      const headers = new Set();
      params.csvs.forEach(csv => {
        const lines = csv.split('\n');
        lines[0].split(',').forEach(h => headers.add(h.trim()));
        lines.slice(1).forEach(line => {
          const values = line.split(',');
          const row = {};
          Array.from(headers).forEach((header, i) => {
            row[header] = values[i]?.trim() || '';
          });
          allData.push(row);
        });
      });
      return { success: true, merged: allData, headers: Array.from(headers) };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface CSVProcessorParams {
  timeout?: number;
}

export interface CSVProcessorResult {
  success: boolean;
  headers?: string[];
  rows?: any[];
  count?: number;
  transformed?: any[];
  valid?: boolean;
  errors?: any[];
  merged?: any[];
  error?: string;
}
