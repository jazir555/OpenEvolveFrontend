import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * DataTransformerTool - Data transformation operations
 */
export class DataTransformerTool extends ToolBubble<DataTransformerParams, DataTransformerResult> {
  bubbleName = 'data-transformer';
  type = 'tool';
  alias = 'data-transformer';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<DataTransformerResult> {
    try {
      const result = await this.transform(input);
      return { success: true, transformed: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async transform(params: { data: any[]; operations: any[] }): Promise<DataTransformerResult> {
    try {
      let result = params.data;
      params.operations.forEach(op => {
        if (op.type === 'map') {
          result = result.map(item => {
            const newItem = { ...item };
            if (op.field && op.value !== undefined) {
              newItem[op.field] = op.value;
            }
            return newItem;
          });
        } else if (op.type === 'filter') {
          result = result.filter(item => item[op.field] === op.value);
        } else if (op.type === 'rename') {
          result = result.map(item => {
            const newItem = { ...item };
            newItem[op.newField] = newItem[op.oldField];
            delete newItem[op.oldField];
            return newItem;
          });
        }
      });
      return { success: true, transformed: result, count: result.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async map(params: { data: any[]; field: string; transform: (value: any) => any }): Promise<DataTransformerResult> {
    try {
      const result = params.data.map(item => ({
        ...item,
        [params.field]: params.transform(item[params.field])
      }));
      return { success: true, mapped: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async filter(params: { data: any[]; predicate: any }): Promise<DataTransformerResult> {
    try {
      const filtered = params.data.filter(item => {
        for (const [key, value] of Object.entries(params.predicate)) {
          if (item[key] !== value) return false;
        }
        return true;
      });
      return { success: true, filtered, count: filtered.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async aggregate(params: { data: any[]; groupBy: string; aggregations: any[] }): Promise<DataTransformerResult> {
    try {
      const groups = {};
      params.data.forEach(item => {
        const key = item[params.groupBy];
        if (!groups[key]) {
          groups[key] = [];
        }
        groups[key].push(item);
      });

      const result = Object.entries(groups).map(([key, items]) => {
        const aggregated = { [params.groupBy]: key };
        params.aggregations.forEach(agg => {
          if (agg.type === 'count') {
            aggregated[agg.field] = items.length;
          } else if (agg.type === 'sum') {
            aggregated[agg.field] = items.reduce((sum, item) => sum + (item[agg.field] || 0), 0);
          } else if (agg.type === 'avg') {
            const sum = items.reduce((s, item) => s + (item[agg.field] || 0), 0);
            aggregated[agg.field] = sum / items.length;
          }
        });
        return aggregated;
      });

      return { success: true, aggregated: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface DataTransformerParams {
  timeout?: number;
}

export interface DataTransformerResult {
  success: boolean;
  transformed?: any[];
  mapped?: any[];
  filtered?: any[];
  aggregated?: any[];
  count?: number;
  error?: string;
}
