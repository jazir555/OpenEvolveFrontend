/**
 * Shared OpenEvolve server health helper.
 *
 * Per the integration contract, a bubble's `health_check` must verify the
 * OpenEvolve HTTP backend at `OPENEVOLVE_BASE_URL` (default
 * http://localhost:8000) by issuing a real GET to `/api/v1/health`. Success is
 * derived from the response; an unreachable server yields `success: false`
 * with a clear error (never hardcoded success).
 */

export const DEFAULT_OPENEVOLVE_BASE_URL = 'http://localhost:8000';

export function getOpenEvolveBaseUrl(): string {
  return process.env.OPENEVOLVE_BASE_URL || DEFAULT_OPENEVOLVE_BASE_URL;
}

export interface OpenEvolveHealthResult {
  ok: boolean;
  status: number;
  data: unknown;
  error?: string;
}

export async function checkOpenEvolveHealth(): Promise<OpenEvolveHealthResult> {
  const baseUrl = getOpenEvolveBaseUrl();
  try {
    const res = await fetch(`${baseUrl}/api/v1/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    let data: unknown;
    try {
      data = await res.json();
    } catch {
      data = undefined;
    }
    return {
      ok: res.ok,
      status: res.status,
      data,
      error: res.ok ? undefined : `OpenEvolve health check failed: ${res.status}`,
    };
  } catch (error) {
    return {
      ok: false,
      status: 0,
      data: undefined,
      error: `OpenEvolve server unreachable: ${error instanceof Error ? error.message : 'Unknown error'}`,
    };
  }
}
