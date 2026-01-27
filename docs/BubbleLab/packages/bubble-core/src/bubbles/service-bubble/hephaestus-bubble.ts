import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * HephaestusBubble - Code generation and analysis via Hephaestus MCP
 */
export class HephaestusBubble extends ServiceBubble<HephaestusParams, HephaestusResult> {
  bubbleName = 'hephaestus';
  type = 'service';
  alias = 'Hephaestus';
  credentialType = 'hephaestus_api_key';

  params = {
    baseUrl: z.string().url(),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    // Initialize Hephaestus MCP client connection
    const { Client } = await import('@hephaestus/mcp');
    this.client = new Client({ url: this.params.baseUrl });
  }

  async generateCode(params: { description: string; language?: string; context?: string }): Promise<HephaestusResult> {
    try {
      const result = await this.client.generate({
        prompt: params.description,
        language: params.language || 'typescript',
        context: params.context
      });
      return { success: true, code: result.code, explanation: result.explanation };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async explainCode(params: { code: string; language?: string }): Promise<HephaestusResult> {
    try {
      const result = await this.client.explain({
        code: params.code,
        language: params.language || 'typescript'
      });
      return { success: true, explanation: result.explanation, complexity: result.complexity };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async findBugs(params: { code: string; language?: string; severity?: string }): Promise<HephaestusResult> {
    try {
      const result = await this.client.analyze({
        code: params.code,
        language: params.language || 'typescript',
        severity: params.severity || 'medium'
      });
      return { success: true, bugs: result.issues, suggestions: result.fixes };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async suggestOptimizations(params: { code: string; language?: string }): Promise<HephaestusResult> {
    try {
      const result = await this.client.optimize({
        code: params.code,
        language: params.language || 'typescript'
      });
      return { success: true, optimizations: result.suggestions, performance: result.metrics };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async generateDocs(params: { code: string; language?: string; format?: 'markdown' | 'html' | 'javadoc' }): Promise<HephaestusResult> {
    try {
      const result = await this.client.document({
        code: params.code,
        language: params.language || 'typescript',
        format: params.format || 'markdown'
      });
      return { success: true, documentation: result.docs };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createAPI(params: { spec: any; language?: string; framework?: string }): Promise<HephaestusResult> {
    try {
      const result = await this.client.createAPI({
        spec: params.spec,
        language: params.language || 'typescript',
        framework: params.framework || 'express'
      });
      return { success: true, code: result.code, structure: result.structure };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async refactorCode(params: { code: string; language?: string; pattern?: string }): Promise<HephaestusResult> {
    try {
      const result = await this.client.refactor({
        code: params.code,
        language: params.language || 'typescript',
        pattern: params.pattern
      });
      return { success: true, refactoredCode: result.code, changes: result.summary };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface HephaestusParams {
  baseUrl: string;
  timeout?: number;
}

export interface HephaestusResult {
  success: boolean;
  code?: string;
  explanation?: string;
  complexity?: any;
  bugs?: any[];
  suggestions?: any[];
  optimizations?: any[];
  performance?: any;
  documentation?: string;
  structure?: any;
  refactoredCode?: string;
  changes?: any;
  error?: string;
}
