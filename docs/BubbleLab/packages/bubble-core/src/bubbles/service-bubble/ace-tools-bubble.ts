import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * ACEToolsBubble - Code execution and validation via ACE tools
 */
export class ACEToolsBubble extends ServiceBubble<ACEToolsParams, ACEToolsResult> {
  bubbleName = 'ace-tools';
  type = 'service';
  alias = 'ACE Tools';
  credentialType = 'ace_tools_api_key';

  params = {
    apiKey: z.string().min(1),
    baseUrl: z.string().url(),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    const { ACEClient } = await import('@ace/tools');
    this.client = new ACEClient({ apiKey: this.params.apiKey, baseUrl: this.params.baseUrl });
  }

  async executeCode(params: { code: string; language?: string; input?: string }): Promise<ACEToolsResult> {
    try {
      const result = await this.client.execute({
        code: params.code,
        language: params.language || 'python',
        stdin: params.input
      });
      return { success: true, output: result.stdout, error: result.stderr, exitCode: result.exitCode };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async validateCode(params: { code: string; language?: string; rules?: string[] }): Promise<ACEToolsResult> {
    try {
      const result = await this.client.validate({
        code: params.code,
        language: params.language || 'python',
        rules: params.rules || ['syntax', 'style', 'security']
      });
      return { success: true, valid: result.valid, errors: result.errors, warnings: result.warnings };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async formatCode(params: { code: string; language?: string; style?: string }): Promise<ACEToolsResult> {
    try {
      const result = await this.client.format({
        code: params.code,
        language: params.language || 'python',
        style: params.style || 'pep8'
      });
      return { success: true, formattedCode: result.code };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async analyzeCode(params: { code: string; language?: string; metrics?: string[] }): Promise<ACEToolsResult> {
    try {
      const result = await this.client.analyze({
        code: params.code,
        language: params.language || 'python',
        metrics: params.metrics || ['complexity', 'maintainability', 'duplication']
      });
      return { success: true, metrics: result.metrics, score: result.score };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async generateTests(params: { code: string; language?: string; framework?: string }): Promise<ACEToolsResult> {
    try {
      const result = await this.client.generateTests({
        code: params.code,
        language: params.language || 'python',
        framework: params.framework || 'pytest'
      });
      return { success: true, tests: result.testCode, coverage: result.expectedCoverage };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async refactorCode(params: { code: string; language?: string; pattern?: string }): Promise<ACEToolsResult> {
    try {
      const result = await this.client.refactor({
        code: params.code,
        language: params.language || 'python',
        pattern: params.pattern
      });
      return { success: true, refactoredCode: result.code, changes: result.changes };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async documentCode(params: { code: string; language?: string; style?: string }): Promise<ACEToolsResult> {
    try {
      const result = await this.client.document({
        code: params.code,
        language: params.language || 'python',
        style: params.style || 'docstring'
      });
      return { success: true, documentedCode: result.code, docs: result.documentation };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface ACEToolsParams {
  apiKey: string;
  baseUrl: string;
  timeout?: number;
}

export interface ACEToolsResult {
  success: boolean;
  output?: string;
  error?: string;
  exitCode?: number;
  valid?: boolean;
  errors?: any[];
  warnings?: any[];
  formattedCode?: string;
  metrics?: any;
  score?: number;
  tests?: string;
  coverage?: number;
  refactoredCode?: string;
  changes?: any[];
  documentedCode?: string;
  docs?: string;
  code?: string;
}
