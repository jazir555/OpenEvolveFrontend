export type EvolutionStartRequest = {
  html: string;
  css?: string;
  iterations?: number;
  populationSize?: number;
  criteria?: string;
};

export type EvolutionResult = {
  html: string;
  css?: string;
  score: number;
  changes: string[];
};

export type EvolutionRunResponse = {
  runId: string;
  best: EvolutionResult;
  history: EvolutionResult[];
};
