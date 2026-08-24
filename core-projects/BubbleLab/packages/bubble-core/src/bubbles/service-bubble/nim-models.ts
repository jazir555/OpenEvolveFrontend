/**
 * Runtime detection of NVIDIA NIM models.
 *
 * NVIDIA NIM catalog ids are unprefixed raw ids (e.g. `deepseek-ai/deepseek-v4-flash-0731`,
 * `meta/llama-3.1-8b-instruct`, `nvidia/llama-3.1-nemotron-51b-instruct`). They are NOT
 * enumerated in code (the UI selector fetches them live from `GET /api/nvidia-nim/models`),
 * so routing to the NVIDIA NIM provider is done by provider-prefix heuristics:
 *
 *   - `nvidia/...`                -> NVIDIA NIM (explicit)
 *   - a known non-NIM provider    -> NOT NIM (openai, google, anthropic, openrouter, fireworks, deepseek)
 *   - any other prefix            -> treated as a NVIDIA NIM catalog id
 *
 * This keeps the model list fully dynamic: no per-model file, no manual regeneration.
 */

const KNOWN_NON_NIM_PROVIDERS = new Set([
  'openai',
  'google',
  'anthropic',
  'openrouter',
  'fireworks',
  'deepseek',
]);

export function isNvidiaNimModel(model: string | undefined | null): boolean {
  if (!model || typeof model !== 'string') return false;
  const provider = model.split('/')[0]?.toLowerCase();
  if (!provider) return false;
  if (provider === 'nvidia') return true;
  if (KNOWN_NON_NIM_PROVIDERS.has(provider)) return false;
  // Unknown provider prefix -> assume it is a NVIDIA NIM catalog id.
  return true;
}
