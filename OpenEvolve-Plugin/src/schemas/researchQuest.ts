/**
 * Research Quest Configuration Schema
 *
 * Defines the configuration schema for the Research Quest node
 */

import { z } from 'zod';

// Define the parameters schema separately for better error handling
const parametersSchema = z.object({
  P1_0: z.boolean().optional().default(true),
  P1_1: z.boolean().optional().default(true),
  P1_2: z.boolean().optional().default(true),
  P1_3: z.boolean().optional().default(true),
  P1_4: z.boolean().optional().default(true),
  P1_5: z.boolean().optional().default(true),
  P1_6: z.boolean().optional().default(true),
  P1_7: z.boolean().optional().default(true),
  P1_8: z.boolean().optional().default(true),
  P1_9: z.boolean().optional().default(true),
  P1_10: z.boolean().optional().default(true),
  P1_11: z.boolean().optional().default(true),
  P1_12: z.boolean().optional().default(true),
  P1_13: z.boolean().optional().default(true),
  P1_14: z.boolean().optional().default(true),
  P1_15: z.boolean().optional().default(true),
  P1_16: z.boolean().optional().default(true),
  P1_17: z.boolean().optional().default(true),
  P1_18: z.boolean().optional().default(true),
  P1_19: z.boolean().optional().default(true),
  P1_20: z.boolean().optional().default(true),
  P1_21: z.boolean().optional().default(true),
  P1_22: z.boolean().optional().default(true),
  P1_23: z.boolean().optional().default(true),
  P1_24: z.boolean().optional().default(true),
  P1_25: z.boolean().optional().default(true),
  P1_26: z.boolean().optional().default(true),
  P1_27: z.boolean().optional().default(true),
  P1_28: z.boolean().optional().default(true),
  P1_29: z.boolean().optional().default(true),
}).optional().default({
  P1_0: true,
  P1_1: true,
  P1_2: true,
  P1_3: true,
  P1_4: true,
  P1_5: true,
  P1_6: true,
  P1_7: true,
  P1_8: true,
  P1_9: true,
  P1_10: true,
  P1_11: true,
  P1_12: true,
  P1_13: true,
  P1_14: true,
  P1_15: true,
  P1_16: true,
  P1_17: true,
  P1_18: true,
  P1_19: true,
  P1_20: true,
  P1_21: true,
  P1_22: true,
  P1_23: true,
  P1_24: true,
  P1_25: true,
  P1_26: true,
  P1_27: true,
  P1_28: true,
  P1_29: true,
});

// Main schema with error handling wrapper
const createResearchQuestConfigSchema = () => {
  try {
    return z.object({
      enableMultiLayer: z.boolean().optional().default(true),
      maxHypotheses: z.number().min(3).max(5).optional().default(5),
      enableCausalInference: z.boolean().optional().default(true),
      enableTemporalAnalysis: z.boolean().optional().default(true),
      enableBiasAssessment: z.boolean().optional().default(true),
      enableFalsificationChecks: z.boolean().optional().default(true),
      enableImpactScoring: z.boolean().optional().default(true),
      enableInterdisciplinaryBridges: z.boolean().optional().default(true),
      enableKnowledgeGapDetection: z.boolean().optional().default(true),
      enableProbabilisticConfidence: z.boolean().optional().default(true),
      enableGraphRestructuring: z.boolean().optional().default(true),
      enableTopologyAnalysis: z.boolean().optional().default(true),
      enableInformationTheoryMetrics: z.boolean().optional().default(true),
      enableAttributionTracking: z.boolean().optional().default(true),
      enableStatisticalPowerAnalysis: z.boolean().optional().default(true),
      enableMultiScaleAnalysis: z.boolean().optional().default(true),
      enableCostEstimation: z.boolean().optional().default(true),
      enableSelfAudit: z.boolean().optional().default(true),
      enableEvidenceIntegration: z.boolean().optional().default(true),
      enablePruningMerging: z.boolean().optional().default(true),
      enableSubgraphExtraction: z.boolean().optional().default(true),
      enableReflection: z.boolean().optional().default(true),
      enableComposition: z.boolean().optional().default(true),
      enableHypothesisGeneration: z.boolean().optional().default(true),
      enableTaskDecomposition: z.boolean().optional().default(true),
      enableInitialization: z.boolean().optional().default(true),
      enableBackendExecution: z.boolean().optional().default(true),
      backendUrl: z.string().optional().default('http://localhost:8000'),
      parameters: parametersSchema,
    });
  } catch (error) {
    console.error('Error creating Research Quest config schema:', error);
    // Return a minimal safe schema as fallback
    return z.object({
      enableBackendExecution: z.boolean().optional().default(true),
      backendUrl: z.string().optional().default('http://localhost:8000'),
    });
  }
};

export const researchQuestConfigSchema = createResearchQuestConfigSchema();

export type ResearchQuestConfig = z.infer<typeof researchQuestConfigSchema>;

export default researchQuestConfigSchema;