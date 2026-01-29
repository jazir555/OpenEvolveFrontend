/**
 * MAKER and Adaptive Configuration Schemas for BubbleLab
 */

import { z } from 'zod';

/**
 * MAKER Engine Configuration Schema
 */
export const MAKERConfigSchema = z.object({
  mode: z.enum(['sequential', 'recursive', 'hybrid']).default('recursive'),
  k_ahead: z.number().int().min(1).default(3),
  num_candidates: z.number().int().min(1).default(5),
  max_depth: z.number().int().min(1).default(5),
  enable_red_flagging: z.boolean().default(true),
  max_token_length: z.number().int().default(750),
  timeout_seconds: z.number().int().default(300),
  preset: z.string().optional().describe('Named preset (FAST, BALANCED, ZERO_ERROR, RESEARCH)'),
});

export type MAKERConfig = z.infer<typeof MAKERConfigSchema>;

/**
 * Adaptive MDAP Configuration Schema
 */
export const AdaptiveConfigSchema = z.object({
  enabled: z.boolean().default(true),
  enable_selection: z.boolean().default(true),
  enable_allocation: z.boolean().default(true),
  preset: z.enum(['conservative', 'balanced', 'aggressive']).optional(),
  classifier: z.object({
    feature_weights: z.record(z.number()).optional(),
  }).optional(),
  allocator: z.object({
    thresholds: z.array(z.number()).length(4).optional(),
    enable_learning: z.boolean().optional(),
  }).optional(),
});

export type AdaptiveConfig = z.infer<typeof AdaptiveConfigSchema>;

/**
 * Combined OpenEvolve Parameter Schema
 */
export const OpenEvolveParametersSchema = z.object({
  maker: MAKERConfigSchema.optional(),
  adaptive: AdaptiveConfigSchema.optional(),
  strategy: z.string().optional(),
  max_refinement_cycles: z.number().int().optional(),
});

export type OpenEvolveParameters = z.infer<typeof OpenEvolveParametersSchema>;
