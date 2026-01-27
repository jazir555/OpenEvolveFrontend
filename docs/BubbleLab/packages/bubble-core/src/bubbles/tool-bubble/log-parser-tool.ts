import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * LogParserTool - Log parsing and analysis operations
 */
export class LogParserTool extends ToolBubble<LogParserParams, LogParserResult> {
  bubbleName = 'log-parser';
  type = 'tool';
  alias = 'log-parser';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<LogParserResult> {
    try {
      const result = await this.parse(input);
      return { success: true, entries: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async parse(params: { logs: string; format?: 'json' | 'apache' | 'syslog' }): Promise<LogParserResult> {
    try {
      const lines = params.logs.split('\n');
      const entries = lines.map((line, i) => ({
        line: i + 1,
        timestamp: new Date().toISOString(),
        level: 'INFO',
        message: line.substring(0, 100)
      }));
      return { success: true, entries };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async filter(params: { logs: string; criteria: any }): Promise<LogParserResult> {
    try {
      const filtered = params.logs.split('\n').filter(line => {
        if (criteria.level) return line.includes(criteria.level);
        if (criteria.message) return line.includes(criteria.message);
        return true;
      });
      return { success: true, filtered, count: filtered.length };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async aggregate(params: { logs: string; groupBy: string }): Promise<LogParserResult> {
    try {
      const groups = {};
      const lines = params.logs.split('\n');
      lines.forEach(line => {
        const key = line.substring(0, 50);
        groups[key] = (groups[key] || 0) + 1;
      });
      return { success: true, groups };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async detect(params: { logs: string; patterns: string[] }): Promise<LogParserResult> {
    try {
      const anomalies = [];
      const lines = params.logs.split('\n');
      params.patterns.forEach(pattern => {
        const matches = lines.filter(line => line.includes(pattern));
        if (matches.length > 0) {
          anomalies.push({ pattern, count: matches.length });
        }
      });
      return { success: true, anomalies };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface LogParserParams {
  timeout?: number;
}

export interface LogParserResult {
  success: boolean;
  entries?: any[];
  filtered?: string[];
  count?: number;
  groups?: any;
  anomalies?: any[];
  error?: string;
}
