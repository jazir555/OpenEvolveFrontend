import type { InferInsertModel, InferSelectModel } from 'drizzle-orm';
import {
  evolutionRequests,
  evolutionDesigns,
  evolutionJudgeScores,
  evolutionResults,
  evolutionScreenshots,
} from './schema';

export type EvolutionRequest = InferSelectModel<typeof evolutionRequests>;
export type NewEvolutionRequest = InferInsertModel<typeof evolutionRequests>;

export type EvolutionDesign = InferSelectModel<typeof evolutionDesigns>;
export type NewEvolutionDesign = InferInsertModel<typeof evolutionDesigns>;

export type EvolutionJudgeScore = InferSelectModel<typeof evolutionJudgeScores>;
export type NewEvolutionJudgeScore = InferInsertModel<
  typeof evolutionJudgeScores
>;

export type EvolutionResult = InferSelectModel<typeof evolutionResults>;
export type NewEvolutionResult = InferInsertModel<typeof evolutionResults>;

export type EvolutionScreenshot = InferSelectModel<typeof evolutionScreenshots>;
export type NewEvolutionScreenshot = InferInsertModel<
  typeof evolutionScreenshots
>;
