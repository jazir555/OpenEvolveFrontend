// Centralized environment variable access for the Bubbleflow IDE

// Normalize and provide sensible fallbacks between different env var names
const resolveApiBaseUrl = (): string => {
  const configured =
    import.meta.env.VITE_API_URL || import.meta.env.VITE_API_ENDPOINT || '';

  const base =
    configured && configured.trim().length > 0
      ? configured
      : 'http://localhost:3001';

  // Ensure no trailing slash for consistent concatenation
  return base.replace(/\/$/, '');
};

export const API_BASE_URL: string = resolveApiBaseUrl();

// OpenEvolve API Configuration (new FastAPI service)
export const OPENEVOLVE_API_BASE_URL: string = (
  import.meta.env.VITE_OPENEVOLVE_API_URL || 'http://localhost:8000'
).replace(/\/$/, '');
export const EVOLUTION_API_BASE_URL: string = OPENEVOLVE_API_BASE_URL;

// OneKE Knowledge Extraction API Configuration (FastAPI service)
export const ONEKE_API_BASE_URL: string = (
  import.meta.env.VITE_ONEKE_API_URL || 'http://localhost:8765'
).replace(/\/$/, '');
export const OPENEVOLVE_API_KEY: string | undefined = import.meta.env
  .VITE_OPENEVOLVE_API_KEY;

export const CLERK_PUBLISHABLE_KEY: string | undefined = import.meta.env
  .VITE_CLERK_PUBLISHABLE_KEY;

// LeanAide API Configuration (standalone Python server on port 7654)
export const LEANAIDE_API_URL: string = (
  import.meta.env.VITE_LEANAIDE_API_URL || 'http://localhost:7654'
).replace(/\/$/, '');

// Generic Knowledge Extraction Tool API (standalone FastAPI server on port 8766)
export const GKET_API_URL: string = (
  import.meta.env.VITE_GKET_API_URL || 'http://localhost:8766'
).replace(/\/$/, '');

// BubbleLab API proxy base (Hono proxy that fronts the LeanAide benchmark routes)
export const BUBBLELAB_API_BASE_URL: string = (
  import.meta.env.VITE_BUBBLELAB_API_BASE_URL || 'http://localhost:8787'
).replace(/\/$/, '');
export const SHOW_LEGACY_PARAMS: boolean =
  import.meta.env.VITE_SHOW_LEGACY_PARAMS === 'true';
export const DISABLE_AUTH: boolean =
  import.meta.env.VITE_DISABLE_AUTH === 'true';

// PostHog Analytics Configuration
export const POSTHOG_API_KEY: string | undefined = import.meta.env
  .VITE_POSTHOG_API_KEY;
export const POSTHOG_HOST: string =
  import.meta.env.VITE_POSTHOG_HOST || 'https://us.i.posthog.com';
export const ANALYTICS_ENABLED: boolean =
  import.meta.env.VITE_ANALYTICS_ENABLED !== 'false'; // Enabled by default unless explicitly disabled

// Google Picker API Configuration
export const GOOGLE_API_KEY: string | undefined = import.meta.env
  .VITE_GOOGLE_API_KEY;
export const GOOGLE_OAUTH_CLIENT_ID: string | undefined = import.meta.env
  .VITE_GOOGLE_OAUTH_CLIENT_ID;

// Debug mode - enables extra developer features (copy buttons, etc.)
// Defaults to true in dev mode (import.meta.env.DEV), can be overridden via env var
export const DEBUG_MODE: boolean =
  import.meta.env.VITE_DEBUG_MODE === 'true' || import.meta.env.DEV;

console.log('DISABLE_AUTH', DISABLE_AUTH);
