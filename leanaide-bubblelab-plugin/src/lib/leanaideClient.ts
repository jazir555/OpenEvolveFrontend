import { ApiHttpError } from './apiTypes';

interface LeanAideConfig {
  serverUrl: string;
  apiKey?: string;
}

export interface LeanAideTaskRequest {
  taskType: string;
  input: string;
  context?: string;
  timeout?: number;
}

export interface LeanAideTaskResponse {
  success: boolean;
  task: string;
  data?: any;
  error?: string;
  logs?: string;
}

export class LeanAideClient {
  private config: LeanAideConfig;
  private baseURL: string;

  constructor(config: LeanAideConfig) {
    this.config = config;
    this.baseURL = config.serverUrl;
  }

  private async request<T>(
    endpoint: string,
    method: 'GET' | 'POST' = 'POST',
    data?: any
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    const options: RequestInit = {
      method,
      headers,
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(url, options);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiHttpError(response.status, errorData);
      }

      return response.json();
    } catch (error) {
      if (error instanceof ApiHttpError) {
        throw error;
      }
      throw new ApiHttpError(500, {
        message: error instanceof Error ? error.message : 'Network error',
      });
    }
  }

  async translateTheorem(
    theoremStatement: string,
    context?: string
  ): Promise<LeanAideTaskResponse> {
    return this.request<LeanAideTaskResponse>('/translate-thm', 'POST', {
      taskType: 'translate_thm',
      input: theoremStatement,
      context,
    });
  }

  async translateDefinition(
    definitionStatement: string,
    context?: string
  ): Promise<LeanAideTaskResponse> {
    return this.request<LeanAideTaskResponse>('/translate-def', 'POST', {
      taskType: 'translate_def',
      input: definitionStatement,
      context,
    });
  }

  async verifySolution(
    problem: string,
    solution: string,
    context?: string
  ): Promise<LeanAideTaskResponse> {
    return this.request<LeanAideTaskResponse>('/verify', 'POST', {
      taskType: 'prove_for_formalization',
      input: problem,
      solution,
      context,
    });
  }

  async elaborateCode(
    leanCode: string,
    context?: string
  ): Promise<LeanAideTaskResponse> {
    return this.request<LeanAideTaskResponse>('/elaborate', 'POST', {
      taskType: 'elaborate',
      input: leanCode,
      context,
    });
  }

  async mathQuery(
    query: string,
    context?: string
  ): Promise<LeanAideTaskResponse> {
    return this.request<LeanAideTaskResponse>('/math-query', 'POST', {
      taskType: 'math_query',
      input: query,
      context,
    });
  }
}