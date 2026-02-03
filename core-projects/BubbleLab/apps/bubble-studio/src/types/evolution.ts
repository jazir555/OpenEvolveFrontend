export type EvolutionParameterType =
  | 'text'
  | 'textarea'
  | 'number'
  | 'slider'
  | 'select'
  | 'boolean';

export interface ParameterOption {
  value: string;
  label: string;
}

export interface ParameterSchema {
  name: string;
  type: EvolutionParameterType;
  label: string;
  description?: string;
  required?: boolean;
  defaultValue?: string | number | boolean;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: ParameterOption[];
  multiline?: boolean;
}

export type ParameterValue = string | number | boolean;

export interface EvolutionStartParameters {
  max_iterations: number;
  population_size: number;
  temperature: number;
  top_p: number;
  mutation_rate: number;
  crossover_rate: number;
  branching_mode?: 'root' | 'lineage';
  children_per_parent?: number;
  survival_threshold?: number;
}

export interface EvolutionModelConfig {
  provider: string;
  model: string;
  api_key: string;
  api_base?: string;
}

export interface EvolutionStartPayload {
  content: string;
  mode: 'standard' | 'quality_diversity' | 'island_model';
  parameters: EvolutionStartParameters;
  models: EvolutionModelConfig[];
  constraints?: Record<string, unknown>;
}

export interface EvolutionStartResponse {
  evolution_id: string;
  status: string;
  created_at: string;
  websocket_url: string;
}

export interface EvolutionWebSocketMessage {
  type: string;
  data: Record<string, unknown>;
  room?: string;
  timestamp?: string;
}
