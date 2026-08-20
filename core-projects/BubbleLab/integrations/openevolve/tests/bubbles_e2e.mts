/**
 * Bubble-level end-to-end test for the OpenEvolve integration.
 *
 * Unlike `tests/e2e_contract.mjs` (which speaks raw HTTP), this test exercises
 * the REAL bubble classes — `WorkflowOrchestratorBubble` and `KnowledgeEngineBubble` —
 * against a LIVE OpenEvolve server, proving the bubbles function end-to-end and
 * that `start_workflow` -> `get_status`/`get_results` chaining works through the
 * bubble layer (the `workflowId` returned by start is fed back into get_status).
 *
 * Server strategy (mirrors services/openevolve-api/scripts/smoke_boot.py):
 *   - PREFER `services/openevolve-api` (FastAPI): `uvicorn openevolve_api.main:app`.
 *     Because the service dir is named with a hyphen and has no top-level
 *     __init__.py, we generate a thin `openevolve_api` package stub on PYTHONPATH
 *     whose __path__ points at the real service dir, then add the openevolve
 *     python library to PYTHONPATH.
 *   - FALLBACK to `core-projects/openevolve` stdlib server
 *     (`python -m openevolve.server_stdlib`) if the FastAPI boot fails.
 *
 * Usage:  npx tsx tests/bubbles_e2e.mts
 *         npm run test:bubbles
 *
 * Env overrides:
 *   PYTHON            python executable (default "python")
 *   OPENEVOLVE_REPO   path to core-projects/openevolve (python lib)
 *   SERVICE_DIR       path to core-projects/BubbleLab/services/openevolve-api
 *   BOOT_TIMEOUT_MS   health-poll budget for server boot (default 20000)
 *   RUN_TIMEOUT_MS    run-completion budget (default 30000)
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, delimiter } from 'node:path';
import { fileURLToPath } from 'node:url';

import { WorkflowOrchestratorBubble } from '../service-bubbles/workflow-orchestrator-bubble';
import { KnowledgeEngineBubble } from '../service-bubbles/knowledge-engine-bubble';

const HERE = dirname(fileURLToPath(import.meta.url));
const INTEGRATION_DIR = join(HERE, '..');

const PYTHON = process.env.PYTHON || 'python';
const OPENEVOLVE_REPO = process.env.OPENEVOLVE_REPO
  || join(INTEGRATION_DIR, '..', '..', '..', 'openevolve');
const SERVICE_DIR = process.env.SERVICE_DIR
  || join(INTEGRATION_DIR, '..', '..', '..', 'BubbleLab', 'services', 'openevolve-api');

const BOOT_TIMEOUT_MS = Number(process.env.BOOT_TIMEOUT_MS || 20000);
const RUN_TIMEOUT_MS = Number(process.env.RUN_TIMEOUT_MS || 30000);
const POLL_INTERVAL_MS = 400;

const BASE_URL = 'http://127.0.0.1:8000';
const DEAD_URL = 'http://127.0.0.1:9999';

// ---------------------------------------------------------------------------
// Assertion harness
// ---------------------------------------------------------------------------
const results: { name: string; ok: boolean; detail: string }[] = [];
const notes: string[] = [];

function check(name: string, ok: boolean, detail = ''): boolean {
  results.push({ name, ok: Boolean(ok), detail });
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ` -> ${detail}` : ''}`);
  return Boolean(ok);
}
function note(msg: string): void {
  notes.push(msg);
  console.log(`  [NOTE] ${msg}`);
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
function truncate(v: unknown, max = 300): string {
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  if (s === undefined) return 'undefined';
  return s.length > max ? `${s.slice(0, max)}...` : s;
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------
let child: ChildProcess | null = null;
let usedServer: 'openevolve-api' | 'server_stdlib' | null = null;
const serverStdout: string[] = [];
const serverStderr: string[] = [];

function serverLogTail(limit = 40): string {
  const out = serverStdout.join('').split('\n').filter(Boolean).slice(-limit);
  const err = serverStderr.join('').split('\n').filter(Boolean).slice(-limit);
  return [
    out.length ? `--- server stdout (tail) ---\n${out.join('\n')}` : '',
    err.length ? `--- server stderr (tail) ---\n${err.join('\n')}` : '',
  ].filter(Boolean).join('\n');
}

function isPortFree(port: number): boolean {
  // Crude check via a quick fetch; we only use it to pick 8000 vs 8011.
  return true;
}

function makeApiStub(): string {
  // Generate a thin `openevolve_api` package whose __path__ points at SERVICE_DIR,
  // so `uvicorn openevolve_api.main:app` can import the hyphenated service dir.
  const stubDir = mkdtempSync(join(tmpdir(), 'oe_api_stub_'));
  const pkgDir = join(stubDir, 'openevolve_api');
  mkdirSync(pkgDir, { recursive: true });
  writeFileSync(join(pkgDir, '__init__.py'), `__path__ = [${JSON.stringify(SERVICE_DIR)}]\n`);
  return stubDir;
}

function buildEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, PYTHONUNBUFFERED: '1' };
  env.WORKFLOW_DB_PATH = env.WORKFLOW_DB_PATH || 'C:\\Temp\\openevolve_api_bubbles_e2e.db';
  env.OPENEVOLVE_BRIDGE_ENABLED = '1';
  return env;
}

function spawnFastApi(): ChildProcess {
  const stub = makeApiStub();
  const env = buildEnv();
  env.PYTHONPATH = [stub, SERVICE_DIR, OPENEVOLVE_REPO, env.PYTHONPATH || ''].join(delimiter);
  console.log(`  spawning FastAPI: ${PYTHON} -m uvicorn openevolve_api.main:app (cwd=${SERVICE_DIR})`);
  const proc = spawn(
    PYTHON,
    ['-u', '-m', 'uvicorn', 'openevolve_api.main:app', '--host', '127.0.0.1', '--port', '8000'],
    {
      cwd: SERVICE_DIR,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
      windowsHide: true,
    },
  );
  proc.stdout?.on('data', (d) => serverStdout.push(d.toString()));
  proc.stderr?.on('data', (d) => serverStderr.push(d.toString()));
  proc.on('error', (err) => serverStderr.push(`[spawn error] ${err.message}\n`));
  return proc;
}

function spawnStdlib(): ChildProcess {
  const env = buildEnv();
  env.PYTHONPATH = [OPENEVOLVE_REPO, env.PYTHONPATH || ''].join(delimiter);
  console.log(`  spawning stdlib: ${PYTHON} -m openevolve.server_stdlib (cwd=${OPENEVOLVE_REPO})`);
  const proc = spawn(
    PYTHON,
    ['-u', '-m', 'openevolve.server_stdlib'],
    {
      cwd: OPENEVOLVE_REPO,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
      windowsHide: true,
    },
  );
  proc.stdout?.on('data', (d) => serverStdout.push(d.toString()));
  proc.stderr?.on('data', (d) => serverStderr.push(d.toString()));
  proc.on('error', (err) => serverStderr.push(`[spawn error] ${err.message}\n`));
  return proc;
}

async function stopServer(proc: ChildProcess | null): Promise<void> {
  if (!proc || proc.exitCode !== null || proc.signalCode !== null) return;
  const exited = new Promise<void>((resolve) => proc.once('exit', () => resolve()));
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
    } else {
      process.kill(-proc.pid, 'SIGTERM');
    }
  } catch {
    try { proc.kill('SIGTERM'); } catch { /* gone */ }
  }
  const timedOut = await Promise.race([exited.then(() => false), sleep(5000).then(() => true)]);
  if (timedOut) {
    try {
      if (process.platform !== 'win32') process.kill(-proc.pid, 'SIGKILL');
      else proc.kill('SIGKILL');
    } catch { /* gone */ }
    await Promise.race([exited, sleep(2000)]);
  }
}

async function getHealth(): Promise<{ status: number; ok: boolean; body: any }> {
  const res = await fetch(`${BASE_URL}/api/v1/health`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(3000),
  });
  const text = await res.text();
  let body: any;
  try { body = JSON.parse(text); } catch { body = { _raw: text }; }
  return { status: res.status, ok: res.ok, body };
}

async function waitForHealth(deadlineMs: number): Promise<{ status: number; body: any; bootMs: number }> {
  const started = Date.now();
  let lastErr = 'never attempted';
  while (Date.now() - started < deadlineMs) {
    if (child && child.exitCode !== null) {
      throw new Error(`Python server exited early with code ${child.exitCode}.\n${serverLogTail()}`);
    }
    try {
      const res = await getHealth();
      if (res.status === 200) return { ...res, bootMs: Date.now() - started };
      lastErr = `HTTP ${res.status}`;
    } catch (err) {
      lastErr = err instanceof Error ? err.message : String(err);
    }
    await sleep(POLL_INTERVAL_MS);
  }
  throw new Error(`Server did not become healthy within ${deadlineMs}ms (last error: ${lastErr}).\n${serverLogTail()}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  // The bubbles read OPENEVOLVE_BASE_URL/env to derive their base URL. Unset it so
  // the bubbles use the explicitly-passed baseUrl (critical for the negative test).
  delete process.env.OPENEVOLVE_BASE_URL;
  delete process.env.OPENEVOLVE_API_URL;

  console.log('OpenEvolve Bubble-level E2E test');
  console.log(`  python     : ${PYTHON}`);
  console.log(`  base url   : ${BASE_URL}`);
  console.log(`  repo       : ${OPENEVOLVE_REPO}`);
  console.log(`  service    : ${SERVICE_DIR}`);
  console.log('');

  // --- 1. boot the LIVE server (prefer FastAPI, fallback to stdlib) ----------
  console.log('1) server boot + GET /api/v1/health');
  child = spawnFastApi();
  let booted = false;
  try {
    await waitForHealth(BOOT_TIMEOUT_MS);
    usedServer = 'openevolve-api';
    booted = true;
  } catch (err) {
    note(`FastAPI boot failed, falling back to server_stdlib: ${(err as Error).message.split('\n')[0]}`);
    await stopServer(child);
    child = spawnStdlib();
    await waitForHealth(BOOT_TIMEOUT_MS);
    usedServer = 'server_stdlib';
    booted = true;
  }
  const health = await getHealth();
  check('live server health responds HTTP 200', health.status === 200, `status=${health.status}`);
  note(`server backend used: ${usedServer} (port 8000)`);

  // --- 2. start_workflow via the REAL bubble ---------------------------------
  console.log('');
  console.log('2) WorkflowOrchestratorBubble.start_workflow -> POST /api/v1/workflows/orchestrate');
  const startBubble = new WorkflowOrchestratorBubble({
    operation: 'start_workflow',
    system: 'evolutionary',
    problemStatement: 'evolve a function that adds two numbers',
    generations: 2,
    populationSize: 4,
    baseUrl: BASE_URL,
  });
  const startRes = await startBubble.action();
  check('start_workflow success === true', startRes.success === true, `success=${startRes.success} error=${startRes.error || ''}`);
  const workflowId = startRes.workflowId;
  check(
    'start_workflow returns a non-undefined workflowId',
    typeof workflowId === 'string' && workflowId.length > 0,
    `workflowId=${truncate(workflowId, 80)}`,
  );
  if (!workflowId) {
    throw new Error('No workflowId returned from start_workflow; cannot test chaining.');
  }

  // --- 3. chaining: get_status / get_results via the REAL bubble -------------
  console.log('');
  console.log(`3) chaining: get_status + get_results (poll run ${workflowId} until completed)`);
  const pollStart = Date.now();
  let finalStatus = '';
  let resultsBubbleRes: any = null;
  let sawRunning = false;

  while (Date.now() - pollStart < RUN_TIMEOUT_MS) {
    const statusBubble = new WorkflowOrchestratorBubble({
      operation: 'get_status',
      system: 'integrated',
      workflowId,
      baseUrl: BASE_URL,
    });
    const statusRes = await statusBubble.action();
    if (statusRes.status) {
      if (statusRes.status !== finalStatus) {
        finalStatus = statusRes.status;
        console.log(`     status=${statusRes.status} (+${Date.now() - pollStart}ms)`);
      }
      if (statusRes.status === 'running' || statusRes.status === 'pending') sawRunning = true;
    }
    if (statusRes.status === 'completed' || statusRes.status === 'failed') {
      resultsBubbleRes = new WorkflowOrchestratorBubble({
        operation: 'get_results',
        system: 'integrated',
        workflowId,
        baseUrl: BASE_URL,
      }).action();
      resultsBubbleRes = await resultsBubbleRes;
      break;
    }
    await sleep(POLL_INTERVAL_MS);
  }

  const elapsed = Date.now() - pollStart;
  check(
    'get_status reported status (chaining works)',
    typeof finalStatus === 'string' && finalStatus.length > 0,
    `finalStatus=${finalStatus} after ${elapsed}ms`,
  );
  check('run reached status "completed"', finalStatus === 'completed', `status=${finalStatus}`);
  check(
    'get_results returns a non-null result with best_code',
    !!resultsBubbleRes && !!resultsBubbleRes.result &&
      typeof (resultsBubbleRes.result as any)?.best_code === 'string' &&
      ((resultsBubbleRes.result as any).best_code as string).length > 0,
    `best_code(${(resultsBubbleRes?.result?.best_code ?? '').length} chars)=${truncate(resultsBubbleRes?.result?.best_code, 120)}`,
  );
  if (sawRunning) note('observed running/pending -> completed transition through the bubble layer');

  // --- 4. KnowledgeEngineBubble health_check against the LIVE server ----------
  console.log('');
  console.log('4) KnowledgeEngineBubble.health_check (live server)');
  // KnowledgeEngineBubble derives its URL from OPENEVOLVE_BASE_URL (no baseUrl param),
  // so point it at the live server.
  process.env.OPENEVOLVE_BASE_URL = BASE_URL;
  const keLive = new KnowledgeEngineBubble({ operation: 'health_check', backend: 'qdrant' });
  const keLiveRes = await keLive.action();
  check(
    'knowledge health_check success === true (real, server up)',
    keLiveRes.success === true,
    `success=${keLiveRes.success} error=${keLiveRes.error || ''}`,
  );

  // --- 5. NEGATIVE: health_check against a DEAD endpoint must NOT fake success
  console.log('');
  console.log('5) NEGATIVE: KnowledgeEngineBubble.health_check (dead endpoint)');
  process.env.OPENEVOLVE_BASE_URL = DEAD_URL;
  const keDead = new KnowledgeEngineBubble({ operation: 'health_check', backend: 'qdrant' });
  const keDeadRes = await keDead.action();
  check(
    'knowledge health_check success === false on dead endpoint (does NOT fake success)',
    keDeadRes.success === false,
    `success=${keDeadRes.success} error=${keDeadRes.error || ''}`,
  );
  check(
    'negative health_check reports a clear error',
    !!keDeadRes.error && keDeadRes.error.length > 0,
    `error=${truncate(keDeadRes.error, 200)}`,
  );

  // restore so other code (none) is unaffected; server still alive for cleanup.
  delete process.env.OPENEVOLVE_BASE_URL;
}

// ---------------------------------------------------------------------------
let fatal: Error | null = null;
try {
  await main();
} catch (err) {
  fatal = err instanceof Error ? err : new Error(String(err));
  check('test completed without a fatal error', false, fatal.message);
} finally {
  await stopServer(child);
}

const failed = results.filter((r) => !r.ok);
const passed = results.length - failed.length;

console.log('');
console.log('='.repeat(72));
console.log(`BUBBLE E2E SUMMARY: ${passed}/${results.length} checks passed`);
console.log(`server backend   : ${usedServer ?? 'none'}`);
if (notes.length) {
  console.log('');
  console.log('Notes:');
  for (const n of notes) console.log(`  - ${n}`);
}
if (failed.length) {
  console.log('');
  console.log('Failed checks:');
  for (const f of failed) console.log(`  - ${f.name}${f.detail ? ` -> ${f.detail}` : ''}`);
  const tail = serverLogTail();
  if (tail) { console.log(''); console.log(tail); }
}
console.log('');
console.log(failed.length === 0 ? 'RESULT: PASS' : 'RESULT: FAIL');
console.log('='.repeat(72));

process.exitCode = failed.length === 0 ? 0 : 1;
