/**
 * Backend-launch control plane for BubbleLab.
 *
 * Exposes a small Hono app (mounted under `/api/backends`) that lets the UI
 * START / STOP / CHECK STATUS of the individual backend servers the product
 * depends on:
 *
 *   - LeanAide            (FastAPI-style http.server on :7654)
 *   - OneKE               (FastAPI on :8765)
 *   - GKET                (FastAPI on :8766)
 *   - OpenEvolve API      (uvicorn openevolve_api.main:app on :8000)
 *   - OpenEvolve Engine   (engines/other/api_server.py on :8001)
 *
 * Backends are spawned with `node:child_process.spawn` using `detached: true`
 * so the process survives independently of the Hono server. The tracked PID is
 * kept in a module-level `Map`. Status checks probe the port over TCP so a
 * closed port is treated as "not running".
 *
 * Process spawning uses `node:child_process` (available under Bun) so the
 * module type-checks with `tsc --noEmit` without requiring Bun-specific types.
 */

import { Hono } from 'hono';
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import * as net from 'node:net';

export interface BackendConfig {
  name: string;
  label: string;
  description: string;
  port: number;
  /** argv array to spawn, e.g. ['python', 'core-projects/OneKE/server.py']. */
  cmd: string[];
  /** Absolute working directory to spawn the process in. */
  cwd: string;
  /** HTTP path probed on the backend (best-effort, in addition to TCP). */
  healthPath: string;
  /** Extra environment variables merged into the child process. */
  env?: Record<string, string>;
}

const PYTHON = process.env.PYTHON_PATH && process.env.PYTHON_PATH.trim().length > 0
  ? process.env.PYTHON_PATH.trim()
  : 'python';

/**
 * Resolve the OpenEvolveFrontend repo root. Walk up from this module's
 * directory until we find a directory that contains both `engines` and
 * `core-projects`. Falls back to `BUBBLELAB_REPO_ROOT` or the current working
 * directory.
 */
function resolveRepoRoot(start: string): string {
  if (process.env.BUBBLELAB_REPO_ROOT && existsSync(process.env.BUBBLELAB_REPO_ROOT)) {
    return process.env.BUBBLELAB_REPO_ROOT;
  }

  let dir = start;
  for (let i = 0; i < 12; i++) {
    if (
      existsSync(join(dir, 'engines', 'other', 'api_server.py')) &&
      existsSync(join(dir, 'core-projects', 'BubbleLab'))
    ) {
      return dir;
    }
    const parent = join(dir, '..');
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

const REPO_ROOT = resolveRepoRoot(__dirname);
const SERVICES_DIR = join(REPO_ROOT, 'core-projects', 'BubbleLab', 'services');

export const BACKENDS: BackendConfig[] = [
  {
    name: 'leanaide',
    label: 'LeanAide',
    description:
      'Lean 4 theorem proving server (leanaide_server.py). Provides translate / prove / verify tasks on :7654.',
    port: 7654,
    cmd: ['python', 'core-projects/LeanAide/leanaide_server.py'],
    cwd: REPO_ROOT,
    healthPath: '/',
  },
  {
    name: 'oneke',
    label: 'OneKE',
    description:
      'OneKE knowledge extraction FastAPI server on :8765 (NER / RE / EE / Triple / Base tasks).',
    port: 8765,
    cmd: ['python', 'core-projects/OneKE/server.py'],
    cwd: REPO_ROOT,
    healthPath: '/',
  },
  {
    name: 'gket',
    label: 'GKET',
    description:
      'Generic Knowledge Extraction Tool FastAPI server on :8766 (parse / generate-models / extract / export).',
    port: 8766,
    cmd: ['python', 'core-projects/Generic-Knowledge-Extraction-Tool/server.py'],
    cwd: REPO_ROOT,
    healthPath: '/healthz',
  },
  {
    name: 'openevolve-api',
    label: 'OpenEvolve API',
    description:
      'OpenEvolve FastAPI service (uvicorn openevolve_api.main:app) on :8000 — workflows, teams, gauntlets, executions.',
    port: 8000,
    cmd: ['python', '-m', 'uvicorn', 'openevolve_api.main:app', '--port', '8000'],
    cwd: SERVICES_DIR,
    healthPath: '/health',
  },
  {
    name: 'openevolve-engine',
    label: 'OpenEvolve Engine',
    description:
      'OpenEvolve engine (engines/other/api_server.py) on :8001 — real engine-backed evolution runs.',
    port: 8001,
    cmd: ['python', 'engines/other/api_server.py'],
    cwd: REPO_ROOT,
    healthPath: '/',
    env: {
      OPENEVOLVE_API_KEY: process.env.OPENEVOLVE_API_KEY || '',
    },
  },
];

export function getBackend(name: string): BackendConfig | undefined {
  return BACKENDS.find((b) => b.name === name);
}

/** Tracked running processes: backend name -> { pid, startedAt }. */
const tracked = new Map<string, { pid: number; startedAt: number }>();

function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** Probe a TCP port; returns true if something is listening. */
function checkPort(port: number, timeoutMs = 1500): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    const done = (ok: boolean) => {
      if (settled) return;
      settled = true;
      try {
        socket.destroy();
      } catch {
        /* ignore */
      }
      resolve(ok);
    };

    socket.setTimeout(timeoutMs);
    socket.once('connect', () => done(true));
    socket.once('timeout', () => done(false));
    socket.once('error', () => done(false));
    try {
      socket.connect(port, '127.0.0.1');
    } catch {
      done(false);
    }
  });
}

export interface BackendStatus {
  name: string;
  label: string;
  description: string;
  port: number;
  running: boolean;
  pid?: number;
  startedAt?: number;
  healthPath: string;
  error?: string;
}

/** Compute the effective status for a single backend (port + tracked pid). */
async function statusFor(cfg: BackendConfig): Promise<BackendStatus> {
  const trackedInfo = tracked.get(cfg.name);
  let pid: number | undefined = trackedInfo?.pid;
  let startedAt = trackedInfo?.startedAt;

  // Drop stale tracked PIDs that are no longer alive.
  if (pid !== undefined && !isPidAlive(pid)) {
    tracked.delete(cfg.name);
    pid = undefined;
    startedAt = undefined;
  }

  let running = false;
  let error: string | undefined;
  try {
    running = await checkPort(cfg.port);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  return {
    name: cfg.name,
    label: cfg.label,
    description: cfg.description,
    port: cfg.port,
    running,
    pid,
    startedAt,
    healthPath: cfg.healthPath,
    error,
  };
}

/** Build the spawn argv, substituting a configured Python interpreter. */
function buildArgv(cfg: BackendConfig): string[] {
  const [first, ...rest] = cfg.cmd;
  if (first === 'python' || first === 'python3') {
    return [PYTHON, ...rest];
  }
  return cfg.cmd;
}

export const backendsApp = new Hono();

// GET /api/backends  -> list all backends with current status
backendsApp.get('/', async (c) => {
  const statuses = await Promise.all(BACKENDS.map((b) => statusFor(b)));
  return c.json({ backends: statuses });
});

// GET /api/backends/:name/status -> health-check a single backend
backendsApp.get('/:name/status', async (c) => {
  const name = c.req.param('name');
  const cfg = getBackend(name);
  if (!cfg) {
    return c.json({ error: `Unknown backend: ${name}` }, 404);
  }
  // Re-probe quickly for the status endpoint.
  const status = await statusFor(cfg);
  return c.json(status);
});

// POST /api/backends/:name/start -> spawn the backend process
backendsApp.post('/:name/start', async (c) => {
  const name = c.req.param('name');
  const cfg = getBackend(name);
  if (!cfg) {
    return c.json({ error: `Unknown backend: ${name}` }, 404);
  }

  // Avoid spawning duplicates: if a tracked PID is alive, report it.
  const existing = tracked.get(cfg.name);
  if (existing && isPidAlive(existing.pid)) {
    const status = await statusFor(cfg);
    return c.json({ started: true, alreadyRunning: true, ...status });
  }
  if (existing) {
    tracked.delete(cfg.name);
  }

  try {
    const argv = buildArgv(cfg);
    const child: ChildProcess = spawn(argv[0], argv.slice(1), {
      cwd: cfg.cwd,
      detached: true,
      stdio: 'ignore',
      env: { ...process.env, ...(cfg.env || {}) },
    });

    child.unref();

    const pid = child.pid;
    if (!pid) {
      return c.json({ error: 'Failed to spawn process (no pid)' }, 500);
    }
    tracked.set(cfg.name, { pid, startedAt: Date.now() });

    return c.json({
      started: true,
      alreadyRunning: false,
      name: cfg.name,
      pid,
      port: cfg.port,
      cwd: cfg.cwd,
      cmd: argv,
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return c.json({ error: `Failed to start ${cfg.name}: ${message}` }, 500);
  }
});

// POST /api/backends/:name/stop -> kill the tracked backend process
backendsApp.post('/:name/stop', async (c) => {
  const name = c.req.param('name');
  const cfg = getBackend(name);
  if (!cfg) {
    return c.json({ error: `Unknown backend: ${name}` }, 404);
  }

  const existing = tracked.get(cfg.name);
  if (!existing || !isPidAlive(existing.pid)) {
    tracked.delete(cfg.name);
    return c.json({ stopped: true, wasRunning: false, name: cfg.name });
  }

  try {
    process.kill(existing.pid);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return c.json({ error: `Failed to stop ${cfg.name}: ${message}` }, 500);
  }

  tracked.delete(cfg.name);
  return c.json({ stopped: true, wasRunning: true, name: cfg.name });
});

export default backendsApp;
