import { z } from 'zod';

export const JudgeResponseSchema = z.object({
  score: z.number().min(0).max(1),
  reasoning: z.string().min(1),
  highlights: z.array(z.string()).default([]),
  issues: z.array(z.string()).default([]),
  recommendations: z.array(z.string()).default([]),
});

export type JudgeResponse = z.infer<typeof JudgeResponseSchema>;

export type JudgeImage =
  | {
      type: 'base64';
      data: string;
      mimeType?: string;
      description?: string;
    }
  | {
      type: 'url';
      url: string;
      description?: string;
    };

export type JudgeInput = {
  image: JudgeImage;
  criteria?: string;
  html?: string;
  metadata?: Record<string, unknown>;
};

export type JudgeEvaluation = JudgeResponse & {
  agent: string;
  provider: string;
  rawResponse: string;
  costUsd: number;
};

export type JudgeAggregateResult = {
  score: number;
  weights: Record<string, number>;
  agents: JudgeEvaluation[];
};
