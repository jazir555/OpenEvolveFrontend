import { CredentialType } from '@bubblelab/shared-schemas';

/**
 * OpenAI-compatible base URLs per model provider prefix.
 *
 * The model string passed to AIAgentBubble is `<provider>/<modelName>`
 * (e.g. `openrouter/anthropic/claude-sonnet-4.5`). These base URLs are used
 * for providers that expose an OpenAI-compatible `/chat/completions` API and
 * are therefore drivable through LangChain's `ChatOpenAI` client.
 */
export const LLM_PROVIDER_BASE_URLS: Record<string, string> = {
  openrouter: 'https://openrouter.ai/api/v1',
  fireworks: 'https://api.fireworks.ai/inference/v1',
  nvidia: 'https://integrate.api.nvidia.com/v1',
};

/**
 * Maps a model provider prefix to the credential type that supplies its API key.
 */
export const LLM_PROVIDER_CREDENTIALS: Partial<Record<string, CredentialType>> = {
  openai: CredentialType.OPENAI_CRED,
  google: CredentialType.GOOGLE_GEMINI_CRED,
  anthropic: CredentialType.ANTHROPIC_CRED,
  openrouter: CredentialType.OPENROUTER_CRED,
  fireworks: CredentialType.FIREWORKS_CRED,
  nvidia: CredentialType.NVIDIA_NIM_CRED,
};
