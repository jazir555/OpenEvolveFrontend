/**
 * Flexible Team LLM Assignment Types
 *
 * Supports arbitrary LLM/vLLM assignment to any team
 * with unified credential management
 */

// ==================== LLM Provider Types ====================

export enum LLMProvider {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  GOOGLE = 'google',
  OPENROUTER = 'openrouter',
  TOGETHER = 'together',
  GROQ = 'groq',
  DEEPSEEK = 'deepseek',
  OPENAI_LIKE = 'openai-like', // vLLM, Ollama, etc.
  CUSTOM = 'custom',
}

export enum LLMCapability {
  TEXT = 'text',
  VISION = 'vision', // vLLM capability
  CODE = 'code',
  MATH = 'math',
  REASONING = 'reasoning',
  TOOL_USE = 'tool_use',
  AGENTIC = 'agentic',
}

export enum CredentialSource {
  OPENEVOLVE_CONFIG = 'openevolve_config', // From .env file
  BUBBLELAB_CREDENTIALS = 'bubblelab_credentials', // From BubbleLab API
  USER_PROVIDED = 'user_provided', // Entered by user at runtime
}

export enum TeamRole {
  BLUE_TEAM = 'blue', // Generates solutions
  RED_TEAM = 'red', // Attacks/evaluates
  JUDGE = 'judge', // Evaluates and decides
  OBSERVER = 'observer', // Watches and learns
  ARBITER = 'arbitrator', // Resolves disputes
}

// ==================== Core Types ====================

export interface LLMModel {
  provider: LLMProvider;
  model_id: string;
  name: string;
  capabilities: LLMCapability[];
  max_tokens: number;
  supports_streaming: boolean;
  supports_function_calling: boolean;
  is_vision: boolean; // True for vLLMs
  input_price_per_1k?: number;
  output_price_per_1k?: number;
}

export interface LLMCredential {
  credential_id: string;
  provider: LLMProvider;
  api_key: string; // Will be encrypted in storage
  source: CredentialSource;
  verified: boolean;
  verified_at?: string; // ISO 8601
  last_used?: string; // ISO 8601
  model_permissions: string[];
  api_base?: string; // For OpenAI-compatible APIs
  organization_id?: string;
  project_id?: string;
  region?: string;
}

export interface TeamMemberLLM {
  member_id: string;
  llm: LLMModel;
  credential_id?: string;
  role: TeamRole;
  temperature: number;
  max_tokens: number;
  system_prompt?: string;
  personality?: string;
  total_requests: number;
  successful_requests: number;
  average_latency_ms?: number;
}

export interface Team {
  team_id: string;
  name: string;
  description?: string;
  members: TeamMemberLLM[];
  voting_strategy: 'consensus' | 'majority' | 'weighted' | 'leader_decides';
  quorum_threshold: number; // 0.0 to 1.0
  require_vision_for_design: boolean;
  require_diverse_providers: boolean;
}

export interface TeamAssignmentRequest {
  team_id: string;
  llm_provider: LLMProvider;
  llm_model_id: string;
  role: TeamRole;
  temperature: number;
  max_tokens: number;
  credential_id?: string;
  system_prompt?: string;
}

export interface CredentialVerificationRequest {
  credential_id?: string;
  provider: LLMProvider;
  api_key: string;
  api_base?: string;
  model_to_test?: string;
}

export interface CredentialVerificationResponse {
  verified: boolean;
  credential_id?: string;
  message: string;
  test_model?: string;
  latency_ms?: number;
  available_models?: string[];
}

// ==================== UI Types ====================

export interface LLMSearchFilters {
  provider?: LLMProvider;
  capability?: LLMCapability;
  vision_only?: boolean;
  search_query?: string;
}

export interface LLMGroup {
  category: string;
  llms: LLMModel[];
}

export interface TeamComposition {
  team_id: string;
  name: string;
  members: TeamMemberLLM[];
  has_vision: boolean;
  provider_diversity: number; // Number of unique providers
  total_cost_estimate?: number; // Estimated cost per 1K tokens
}

export interface CredentialFormData {
  provider: LLMProvider;
  api_key: string;
  api_base?: string;
  organization_id?: string;
  project_id?: string;
  region?: string;
}

// ==================== Response Types ====================

export interface LLMSearchResponse {
  llms: LLMModel[];
  grouped: Record<string, LLMModel[]>;
  total: number;
  vision_llms: LLMModel[];
  text_llms: LLMModel[];
}

export interface CredentialsListResponse {
  credentials: Array<{
    credential_id: string;
    provider: LLMProvider;
    source: CredentialSource;
    verified: boolean;
    // API key is masked
    api_key_preview: string;
  }>;
  total: number;
  sources: CredentialSource[];
}

export interface TeamCreateResponse {
  team_id: string;
  name: string;
  members: TeamMemberLLM[];
  created_at: string;
}

export interface TeamTemplate {
  id: string;
  name: string;
  description: string;
  composition: Array<{
    role: TeamRole;
    llm: string; // model_id
    count: number;
  }>;
}

// ==================== Helper Types ====================

export type LLMProviderOption = {
  value: LLMProvider;
  label: string;
  vision_support: boolean;
};

export type TeamRoleOption = {
  value: TeamRole;
  label: string;
  description: string;
  color: string; // For UI
};

export type CapabilityBadge = {
  capability: LLMCapability;
  label: string;
  color: string; // For UI
};

// ==================== Default Values ====================

export const DEFAULT_TEAM_COMPOSITION = {
  voting_strategy: 'consensus' as const,
  quorum_threshold: 0.7,
  require_vision_for_design: true,
  require_diverse_providers: false,
};

export const DEFAULT_MEMBER_CONFIG = {
  temperature: 0.7,
  max_tokens: 4096,
};

// ==================== UI Constants ====================

export const TEAM_ROLE_OPTIONS: TeamRoleOption[] = [
  {
    value: TeamRole.BLUE_TEAM,
    label: 'Blue Team',
    description: 'Generates solutions',
    color: '#3b82f6', // blue
  },
  {
    value: TeamRole.RED_TEAM,
    label: 'Red Team',
    description: 'Attacks and evaluates',
    color: '#ef4444', // red
  },
  {
    value: TeamRole.JUDGE,
    label: 'Judge',
    description: 'Evaluates and decides',
    color: '#8b5cf6', // purple
  },
  {
    value: TeamRole.OBSERVER,
    label: 'Observer',
    description: 'Watches and learns',
    color: '#6b7280', // gray
  },
  {
    value: TeamRole.ARBITER,
    label: 'Arbiter',
    description: 'Resolves disagreements',
    color: '#f59e0b', // amber
  },
];

export const CAPABILITY_BADGES: Record<LLMCapability, CapabilityBadge> = {
  [LLMCapability.TEXT]: {
    capability: LLMCapability.TEXT,
    label: 'Text',
    color: '#6b7280',
  },
  [LLMCapability.VISION]: {
    capability: LLMCapability.VISION,
    label: 'Vision',
    color: '#10b981', // green for vLLM
  },
  [LLMCapability.CODE]: {
    capability: LLMCapability.CODE,
    label: 'Code',
    color: '#3b82f6',
  },
  [LLMCapability.MATH]: {
    capability: LLMCapability.MATH,
    label: 'Math',
    color: '#8b5cf6',
  },
  [LLMCapability.REASONING]: {
    capability: LLMCapability.REASONING,
    label: 'Reasoning',
    color: '#f59e0b',
  },
  [LLMCapability.TOOL_USE]: {
    capability: LLMCapability.TOOL_USE,
    label: 'Tools',
    color: '#ec4899',
  },
  [LLMCapability.AGENTIC]: {
    capability: LLMCapability.AGENTIC,
    label: 'Agentic',
    color: '#14b8a6',
  },
};

// ==================== Helper Functions ====================

export function isVisionLLM(llm: LLMModel): boolean {
  return llm.is_vision || llm.capabilities.includes(LLMCapability.VISION);
}

export function getLLMProviderLabel(provider: LLMProvider): string {
  const labels: Record<LLMProvider, string> = {
    [LLMProvider.OPENAI]: 'OpenAI',
    [LLMProvider.ANTHROPIC]: 'Anthropic',
    [LLMProvider.GOOGLE]: 'Google',
    [LLMProvider.OPENROUTER]: 'OpenRouter',
    [LLMProvider.TOGETHER]: 'Together',
    [LLMProvider.GROQ]: 'Groq',
    [LLMProvider.DEEPSEEK]: 'DeepSeek',
    [LLMProvider.OPENAI_LIKE]: 'OpenAI-Compatible',
    [LLMProvider.CUSTOM]: 'Custom',
  };
  return labels[provider] || provider;
}

export function formatCredentialSource(source: CredentialSource): string {
  const labels: Record<CredentialSource, string> = {
    [CredentialSource.OPENEVOLVE_CONFIG]: 'OpenEvolve Config',
    [CredentialSource.BUBBLELAB_CREDENTIALS]: 'Saved Credentials',
    [CredentialSource.USER_PROVIDED]: 'User Provided',
  };
  return labels[source] || source;
}

export function estimateCost(
  llm: LLMModel,
  inputTokens: number,
  outputTokens: number,
): number {
  const inputCost = (inputTokens / 1000) * (llm.input_price_per_1k || 0);
  const outputCost = (outputTokens / 1000) * (llm.output_price_per_1k || 0);
  return inputCost + outputCost;
}

export function groupLLMsByCapability(llms: LLMModel[]): Record<string, LLMModel[]> {
  const groups: Record<string, LLMModel[]> = {};

  for (const llm of llms) {
    let key: string;

    if (llm.is_vision) {
      key = 'Vision/Multimodal (vLLM)';
    } else if (llm.capabilities.includes(LLMCapability.CODE)) {
      key = 'Code Generation';
    } else if (llm.capabilities.includes(LLMCapability.REASONING)) {
      key = 'Reasoning & Analysis';
    } else {
      key = 'General Purpose';
    }

    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(llm);
  }

  return groups;
}
