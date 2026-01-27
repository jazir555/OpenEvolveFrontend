import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * AIAgentBubble - AI/LLM completion and analysis operations
 */
export class AIAgentBubble extends ServiceBubble<AIAgentParams, AIAgentResult> {
  bubbleName = 'ai-agent';
  type = 'service';
  alias = 'AI Agent';
  credentialType = 'ai_agent_api_key';

  params = {
    apiKey: z.string().min(1),
    provider: z.enum(['anthropic', 'openai', 'cohere']).default('anthropic'),
    model: z.string().default('claude-3-5-sonnet-20241022'),
    timeout: z.number().int().positive().default(30000)
  };

  private client: any = null;

  async connect() {
    if (this.params.provider === 'anthropic') {
      const Anthropic = await import('@anthropic-ai/sdk');
      this.client = new Anthropic.default({ apiKey: this.params.apiKey });
    } else if (this.params.provider === 'openai') {
      const OpenAI = await import('openai');
      this.client = new OpenAI.default({ apiKey: this.params.apiKey });
    } else if (this.params.provider === 'cohere') {
      const Cohere = await import('cohere-ai');
      this.client = new Cohere.default({ apiKey: this.params.apiKey });
    }
  }

  async generateCompletion(params: { prompt: string; maxTokens?: number; temperature?: number }): Promise<AIAgentResult> {
    try {
      let result;
      if (this.params.provider === 'anthropic') {
        result = await this.client.messages.create({
          model: this.params.model,
          max_tokens: params.maxTokens || 1024,
          messages: [{ role: 'user', content: params.prompt }]
        });
        return { success: true, completion: result.content[0].text, usage: result.usage };
      } else if (this.params.provider === 'openai') {
        result = await this.client.chat.completions.create({
          model: this.params.model,
          messages: [{ role: 'user', content: params.prompt }],
          max_tokens: params.maxTokens || 1024
        });
        return { success: true, completion: result.choices[0].message.content, usage: result.usage };
      } else {
        result = await this.client.generate({
          model: this.params.model,
          prompt: params.prompt,
          maxTokens: params.maxTokens || 1024
        });
        return { success: true, completion: result.generations[0].text };
      }
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async streamCompletion(params: { prompt: string; maxTokens?: number; onChunk?: (chunk: string) => void }): Promise<AIAgentResult> {
    try {
      if (this.params.provider === 'anthropic') {
        const stream = await this.client.messages.create({
          model: this.params.model,
          max_tokens: params.maxTokens || 1024,
          messages: [{ role: 'user', content: params.prompt }],
          stream: true
        });
        let fullText = '';
        for await (const event of stream) {
          if (event.type === 'content_block_delta' && event.delta.text) {
            fullText += event.delta.text;
            params.onChunk?.(event.delta.text);
          }
        }
        return { success: true, completion: fullText };
      } else {
        return { success: false, error: 'Streaming not implemented for this provider' };
      }
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async createChat(params: { messages: Array<{ role: string; content: string }>; maxTokens?: number }): Promise<AIAgentResult> {
    try {
      let result;
      if (this.params.provider === 'anthropic') {
        result = await this.client.messages.create({
          model: this.params.model,
          max_tokens: params.maxTokens || 1024,
          messages: params.messages.map(m => ({ role: m.role, content: m.content }))
        });
        return { success: true, completion: result.content[0].text, usage: result.usage };
      } else if (this.params.provider === 'openai') {
        result = await this.client.chat.completions.create({
          model: this.params.model,
          messages: params.messages,
          max_tokens: params.maxTokens || 1024
        });
        return { success: true, completion: result.choices[0].message.content, usage: result.usage };
      } else {
        return { success: false, error: 'Chat not implemented for this provider' };
      }
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async embedText(params: { text: string }): Promise<AIAgentResult> {
    try {
      if (this.params.provider === 'openai') {
        const result = await this.client.embeddings.create({
          model: 'text-embedding-ada-002',
          input: params.text
        });
        return { success: true, embedding: result.data[0].embedding };
      } else if (this.params.provider === 'cohere') {
        const result = await this.client.embed({
          model: 'embed-english-v3.0',
          texts: [params.text]
        });
        return { success: true, embedding: result.embeddings[0] };
      } else {
        return { success: false, error: 'Embeddings not implemented for this provider' };
      }
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async countTokens(params: { text: string }): Promise<AIAgentResult> {
    try {
      if (this.params.provider === 'anthropic') {
        const result = await this.client.messages.countTokens({
          model: this.params.model,
          messages: [{ role: 'user', content: params.text }]
        });
        return { success: true, count: result.input_tokens };
      } else {
        const roughEstimate = Math.ceil(params.text.length / 4);
        return { success: true, count: roughEstimate };
      }
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async listModels(params?: {}): Promise<AIAgentResult> {
    try {
      if (this.params.provider === 'openai') {
        const result = await this.client.models.list();
        return { success: true, models: result.data };
      } else {
        const models = {
          anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'],
          cohere: ['command', 'command-light', 'command-nightly']
        };
        return { success: true, models: models[this.params.provider] || [] };
      }
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getModelInfo(params?: {}): Promise<AIAgentResult> {
    try {
      const info = {
        provider: this.params.provider,
        model: this.params.model,
        capabilities: {
          streaming: this.params.provider === 'anthropic',
          embeddings: this.params.provider === 'openai' || this.params.provider === 'cohere',
          functionCalling: this.params.provider === 'openai' || this.params.provider === 'anthropic',
          vision: this.params.provider === 'anthropic' || this.params.provider === 'openai'
        }
      };
      return { success: true, info };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface AIAgentParams {
  apiKey: string;
  provider?: string;
  model?: string;
  timeout?: number;
}

export interface AIAgentResult {
  success: boolean;
  completion?: string;
  embedding?: number[];
  count?: number;
  usage?: any;
  models?: string[];
  info?: any;
  error?: string;
}
