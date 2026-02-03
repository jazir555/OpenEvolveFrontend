import { AIAgentBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';
import type { BubbleResult } from '@bubblelab/shared-schemas';
import { CredentialType, AvailableModel } from '@bubblelab/shared-schemas';
import {
  JudgeResponseSchema,
  type JudgeEvaluation,
  type JudgeInput,
  type JudgeResponse,
} from './types.js';

const JsonObjectRegex = /\{[\s\S]*\}/;

export type BaseVisualJudgeConfig = {
  agentName: string;
  provider: string;
  systemPrompt: string;
  model: AvailableModel;
  temperature?: number;
};

const safeParseResponse = (rawResponse: string): JudgeResponse => {
  const trimmed = rawResponse.trim();
  const candidate = trimmed.startsWith('{')
    ? trimmed
    : trimmed.match(JsonObjectRegex)?.[0] || trimmed;

  try {
    const parsed = JSON.parse(candidate);
    return JudgeResponseSchema.parse(parsed);
  } catch (error) {
    return {
      score: 0,
      reasoning:
        'Failed to parse judge response. Returning default score of 0.',
      highlights: [],
      issues: [
        error instanceof Error ? error.message : 'Unknown parsing error',
      ],
      recommendations: [],
    };
  }
};

export class BaseVisualJudge extends AIAgentBubble {
  private readonly agentName: string;
  private readonly provider: string;
  private readonly systemPrompt: string;
  private readonly model: string;
  private readonly temperature: number;

  constructor(config: BaseVisualJudgeConfig) {
    super({
      message: 'Evaluate the provided design.',
      systemPrompt: config.systemPrompt,
      model: {
        model: config.model,
        temperature: config.temperature ?? 0.2,
      },
      images: [],
      expectedOutputSchema: JudgeResponseSchema as z.ZodTypeAny,
    });

    this.agentName = config.agentName;
    this.provider = config.provider;
    this.systemPrompt = config.systemPrompt;
    this.model = config.model;
    this.temperature = config.temperature ?? 0.2;
  }

  get agent(): string {
    return this.agentName;
  }

  get providerName(): string {
    return this.provider;
  }

  private buildMessage(input: JudgeInput): string {
    const lines: string[] = [
      'Evaluate the supplied design screenshot based on the criteria below.',
      'Return JSON that matches the required schema.',
    ];

    if (input.criteria) {
      lines.push(`Criteria: ${input.criteria}`);
    }

    if (input.metadata) {
      lines.push(`Metadata: ${JSON.stringify(input.metadata)}`);
    }

    if (input.html) {
      const snippet = input.html.slice(0, 4000);
      lines.push('HTML snippet:', snippet);
    }

    return lines.join('\n');
  }

  private estimateCost(result: BubbleResult<any>): number {
    void result;
    return 0;
  }

  async evaluate(
    input: JudgeInput,
    credentials: Record<CredentialType, string>
  ): Promise<JudgeEvaluation> {
    const message = this.buildMessage(input);
    this.setParam('message', message);
    this.setParam('systemPrompt', this.systemPrompt);

    // Get current model config and update only specific fields
    const currentModel = this.params.model as any;
    this.setParam('model', {
      model: this.model as any,
      temperature: this.temperature,
      maxTokens: currentModel.maxTokens ?? 64000,
      maxRetries: currentModel.maxRetries ?? 3,
      reasoningEffort: currentModel.reasoningEffort,
      provider: currentModel.provider,
      jsonMode: currentModel.jsonMode ?? false,
      backupModel: currentModel.backupModel,
    });

    this.setParam('images', [input.image as any]);
    this.setParam('credentials', credentials);
    this.setParam('expectedOutputSchema', JudgeResponseSchema as z.ZodTypeAny);

    const result = await this.action();
    if (!result.success || !result.data) {
      throw new Error(result.error || 'Visual judge failed');
    }

    const rawResponse = result.data.response;
    const parsed = safeParseResponse(rawResponse);

    return {
      ...parsed,
      agent: this.agentName,
      provider: this.provider,
      rawResponse,
      costUsd: this.estimateCost(result),
    };
  }
}
