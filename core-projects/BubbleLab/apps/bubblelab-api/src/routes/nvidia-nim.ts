import { Hono } from 'hono';

/**
 * NVIDIA NIM model catalog proxy.
 *
 * NVIDIA exposes a public (no-auth) OpenAI-compatible models endpoint. This
 * route fetches and caches the chat/instruct model list so the Studio model
 * selector is automatically populated instead of hardcoded.
 *
 *   GET /api/nvidia-nim/models
 */
const app = new Hono();

const NIM_MODELS_URL = 'https://integrate.api.nvidia.com/v1/models';
const CACHE_TTL_MS = 60_000; // 60s
const NIM_CHAT_PATTERN = /instruct|chat|nemotron|reasoning|flash/i;
const NIM_SKIP_PATTERN = /embed|rerank|vision|vlm|image|tts|asr|stt|audio/i;

interface NimModelEntry {
  id: string;
  name: string;
  owned_by: string;
}

let cache:
  | { at: number; models: NimModelEntry[] }
  | undefined;

/** Pretty-print a model id like `meta/llama-3.1-8b-instruct` -> `Llama 3.1 8B Instruct`. */
function toDisplayName(id: string): string {
  const model = id.split('/').at(-1) ?? id;
  return model
    .split('-')
    .map((part) =>
      part ? part.charAt(0).toUpperCase() + part.slice(1) : part
    )
    .join(' ');
}

async function fetchModels(): Promise<NimModelEntry[]> {
  if (cache && Date.now() - cache.at < CACHE_TTL_MS) {
    return cache.models;
  }

  const res = await fetch(NIM_MODELS_URL, {
    headers: { Accept: 'application/json' },
    signal: AbortSignal.timeout(15_000),
  });
  if (!res.ok) {
    throw new Error(
      `NVIDIA NIM models endpoint returned ${res.status} ${res.statusText}`
    );
  }

  const json = (await res.json()) as {
    data?: { id?: string; owned_by?: string }[];
  };

  const ids = (json.data ?? [])
    .map((m) => m.id)
    .filter((id): id is string => Boolean(id))
    .filter(
      (id) =>
        NIM_CHAT_PATTERN.test(id) && !NIM_SKIP_PATTERN.test(id)
    )
    .sort((a, b) => a.localeCompare(b));

  const models: NimModelEntry[] = ids.map((id) => ({
    id,
    name: `${toDisplayName(id)} (NVIDIA NIM)`,
    owned_by: id.split('/')[0] ?? '',
  }));

  cache = { at: Date.now(), models };
  return models;
}

app.get('/models', async (c) => {
  try {
    const models = await fetchModels();
    return c.json({ models });
  } catch (err) {
    return c.json(
      {
        error: err instanceof Error ? err.message : 'Failed to fetch NIM models',
        models: [],
      },
      502
    );
  }
});

export default app;