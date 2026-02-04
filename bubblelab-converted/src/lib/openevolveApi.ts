import type { Team, GauntletDefinition } from "./types";

export interface ApiConfig {
  baseUrl?: string;
  apiKey?: string;
}

const resolveBaseUrl = (override?: string) => {
  if (override) {
    return override;
  }
  const fromWindow = (globalThis as any)?.OPENEVOLVE_API_BASE as string | undefined;
  if (fromWindow) {
    return fromWindow;
  }
  return "";
};

const resolveApiKey = (override?: string) => {
  if (override) {
    return override;
  }
  try {
    const stored = globalThis?.localStorage?.getItem("openevolve_api_key");
    if (stored) {
      return stored;
    }
  } catch (_) {
    // ignore storage errors
  }
  return "";
};

const buildHeaders = (apiKey?: string) => {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  return headers;
};

async function request<T>(
  path: string,
  options: RequestInit = {},
  config: ApiConfig = {},
): Promise<T> {
  const baseUrl = resolveBaseUrl(config.baseUrl);
  const apiKey = resolveApiKey(config.apiKey);
  const url = `${baseUrl}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      ...buildHeaders(apiKey),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const openevolveApi = {
  listTeams: (config?: ApiConfig) => request<Team[]>("/teams", {}, config),
  getTeam: (teamName: string, config?: ApiConfig) =>
    request<Team>(`/teams/${encodeURIComponent(teamName)}`, {}, config),
  createTeam: (team: Team, config?: ApiConfig) =>
    request<Team>("/teams", { method: "POST", body: JSON.stringify(team) }, config),
  updateTeam: (teamName: string, team: Team, config?: ApiConfig) =>
    request<Team>(
      `/teams/${encodeURIComponent(teamName)}`,
      { method: "PUT", body: JSON.stringify(team) },
      config,
    ),
  deleteTeam: (teamName: string, config?: ApiConfig) =>
    request<{ success: boolean }>(
      `/teams/${encodeURIComponent(teamName)}`,
      { method: "DELETE" },
      config,
    ),
  listGauntlets: (config?: ApiConfig) => request<GauntletDefinition[]>("/gauntlets", {}, config),
  getGauntlet: (name: string, config?: ApiConfig) =>
    request<GauntletDefinition>(`/gauntlets/${encodeURIComponent(name)}`, {}, config),
  createGauntlet: (gauntlet: GauntletDefinition, config?: ApiConfig) =>
    request<GauntletDefinition>(
      "/gauntlets",
      { method: "POST", body: JSON.stringify(gauntlet) },
      config,
    ),
  updateGauntlet: (name: string, gauntlet: GauntletDefinition, config?: ApiConfig) =>
    request<GauntletDefinition>(
      `/gauntlets/${encodeURIComponent(name)}`,
      { method: "PUT", body: JSON.stringify(gauntlet) },
      config,
    ),
  deleteGauntlet: (name: string, config?: ApiConfig) =>
    request<{ success: boolean }>(
      `/gauntlets/${encodeURIComponent(name)}`,
      { method: "DELETE" },
      config,
    ),
};
