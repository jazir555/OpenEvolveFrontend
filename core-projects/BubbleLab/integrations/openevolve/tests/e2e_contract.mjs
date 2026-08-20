/**
 * End-to-end HTTP contract test: OpenEvolve Python server <-> BubbleLab bubbles.
 *
 * Proves the integration is not orphaned by exercising the exact endpoints that
 * `service-bubbles/workflow-orchestrator-bubble.ts` and
 * `service-bubbles/knowledge-engine-bubble.ts` call:
 *
 *   GET  /api/v1/health                 (healthCheck)
 *   POST /api/v1/workflows/orchestrate  (startOrchestrateWorkflow)
 *   GET  /api/v1/runs/{id}              (getStatus / getResults)
 *
 * Plain Node ESM (Node 18+ global fetch). No TS compile step required.
 *
 * Usage:  node tests/e2e_contract.mjs
 *         npm run test:e2e
 *
 * Requirements: Python 3.11 on PATH, with the `openevolve` library importable
 * from core-projects/openevolve (the server defaults to the offline mock LLM,
 * so no API keys are needed).
 *
 * Env overrides:
 *   OPENEVOLVE_BASE_URL   base URL to test against (default http://127.0.0.1:8000)
 *   OPENEVOLVE_REPO       path to the python repo root containing openevolve/ package
 *   PYTHON                python executable (default "python")
 *   E2E_BOOT_TIMEOUT_MS   health-poll budget for server boot (default 20000)
 *   E2E_RUN_TIMEOUT_MS    run-completion budget (default 30000)
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const INTEGRATION_DIR = path.resolve(HERE, '..');

// tests/ -> integrations/openevolve -> integrations -> BubbleLab -> core-projects -> core-projects/openevolve
const DEFAULT_REPO = path.resolve(INTEGRATION_DIR, '..', '..', '..', 'openevolve');

const PY_REPO = path.resolve(process.env.OPENEVOLVE_REPO || DEFAULT_REPO);
const PYTHON = process.env.PYTHON || 'python';
const BASE_URL = (process.env.OPENEVOLVE_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

const BOOT_TIMEOUT_MS = Number(process.env.E2E_BOOT_TIMEOUT_MS || 20000);
const RUN_TIMEOUT_MS = Number(process.env.E2E_RUN_TIMEOUT_MS || 30000);
const POLL_INTERVAL_MS = 400;

// ---------------------------------------------------------------------------
// Tiny assertion harness
// ---------------------------------------------------------------------------
const results = [];
const notes = [];

function check(name, ok, detail = '') {
  results.push({ name, ok: Boolean(ok), detail });
  const tag = ok ? 'PASS' : 'FAIL';
  console.log(`  [${tag}] ${name}${detail ? ` -> ${detail}` : ''}`);
  return Boolean(ok);
}

function note(msg) {
  notes.push(msg);
  console.log(`  [NOTE] ${msg}`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function truncate(value, max = 300) {
  const s = typeof value === 'string' ? value : JSON.stringify(value);
  if (s === undefined) return 'undefined';
  return s.length > max ? `${s.slice(0, max)}...` : s;
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------
let child = null;
const serverStdout = [];
const serverStderr = [];

function serverLogTail(limit = 40) {
  const out = serverStdout.join('').split('\n').filter(Boolean).slice(-limit);
  const err = serverStderr.join('').split('\n').filter(Boolean).slice(-limit);
  return [
    out.length ? `--- server stdout (tail) ---\n${out.join('\n')}` : '',
    err.length ? `--- server stderr (tail) ---\n${err.join('\n')}` : '',
  ]
    .filter(Boolean)
    .join('\n');
}

function startServer() {
  const serverModule = path.join(PY_REPO, 'openevolve', 'server_stdlib.py');
  if (!existsSync(serverModule)) {
    throw new Error(
      `Cannot find the OpenEvolve server module at ${serverModule}. ` +
        `Set OPENEVOLVE_REPO to the python repo root that contains openevolve/server_stdlib.py.`,
    );
  }

  console.log(`  spawning: ${PYTHON} -m openevolve.server_stdlib (cwd=${PY_REPO})`);

  const proc = spawn(PYTHON, ['-u', '-m', 'openevolve.server_stdlib'], {
    cwd: PY_REPO, // ensures `import openevolve` resolves to this repo's package
    env: { ...process.env, PYTHONPATH: PY_REPO, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
    // Own process group on POSIX so we can kill the whole tree; on Windows we
    // use taskkill /T instead (detached would pop a new console window).
    detached: process.platform !== 'win32',
    windowsHide: true,
  });

  proc.stdout.on('data', (d) => serverStdout.push(d.toString()));
  proc.stderr.on('data', (d) => serverStderr.push(d.toString()));
  proc.on('error', (err) => serverStderr.push(`[spawn error] ${err.message}\n`));

  return proc;
}

async function stopServer(proc) {
  if (!proc || proc.exitCode !== null || proc.signalCode !== null) return;

  const exited = new Promise((resolve) => proc.once('exit', resolve));

  try {
    if (process.platform === 'win32') {
      // Kill the process tree; python spawns evaluator subprocesses.
      spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], {
        stdio: 'ignore',
        windowsHide: true,
      });
    } else {
      process.kill(-proc.pid, 'SIGTERM');
    }
  } catch {
    try {
      proc.kill('SIGTERM');
    } catch {
      /* already gone */
    }
  }

  const timedOut = await Promise.race([exited.then(() => false), sleep(5000).then(() => true)]);

  if (timedOut) {
    try {
      if (process.platform !== 'win32') process.kill(-proc.pid, 'SIGKILL');
      else proc.kill('SIGKILL');
    } catch {
      /* already gone */
    }
    await Promise.race([exited, sleep(2000)]);
  }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------
async function getJson(url, timeoutMs = 10000) {
  const res = await fetch(url, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { _raw: text };
  }
  return { status: res.status, ok: res.ok, body };
}

async function postJson(url, payload, timeoutMs = 30000) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { _raw: text };
  }
  return { status: res.status, ok: res.ok, body };
}

async function waitForHealth(deadlineMs) {
  const started = Date.now();
  let lastErr = 'never attempted';

  while (Date.now() - started < deadlineMs) {
    if (child && child.exitCode !== null) {
      throw new Error(
        `Python server exited early with code ${child.exitCode}.\n${serverLogTail()}`,
      );
    }
    try {
      const res = await getJson(`${BASE_URL}/api/v1/health`, 3000);
      if (res.status === 200) return { ...res, bootMs: Date.now() - started };
      lastErr = `HTTP ${res.status}`;
    } catch (err) {
      lastErr = err instanceof Error ? err.message : String(err);
    }
    await sleep(POLL_INTERVAL_MS);
  }

  throw new Error(
    `Server did not become healthy within ${deadlineMs}ms (last error: ${lastErr}).\n${serverLogTail()}`,
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log('OpenEvolve <-> BubbleLab E2E HTTP contract test');
  console.log(`  node       : ${process.version} (${os.platform()})`);
  console.log(`  base url   : ${BASE_URL}`);
  console.log(`  python repo: ${PY_REPO}`);
  console.log('');

  // --- 0. reuse an already-running server if present ------------------------
  let reused = false;
  try {
    const pre = await getJson(`${BASE_URL}/api/v1/health`, 1500);
    if (pre.status === 200) {
      reused = true;
      note('an OpenEvolve server is already listening; reusing it instead of spawning');
    }
  } catch {
    /* nothing listening: expected */
  }

  console.log('1) server boot + GET /api/v1/health');
  if (!reused) child = startServer();

  const health = await waitForHealth(BOOT_TIMEOUT_MS);
  check('health responds HTTP 200', health.status === 200, `status=${health.status}`);
  check(
    'health body.status === "healthy"',
    health.body?.status === 'healthy',
    truncate(health.body),
  );
  if (!reused) note(`server became healthy in ${health.bootMs}ms`);

  // --- 2. POST /api/v1/workflows/orchestrate -------------------------------
  console.log('');
  console.log('2) POST /api/v1/workflows/orchestrate');
  const orchestrateBody = {
    system: 'evolutionary',
    problemStatement: 'evolve a function that adds two numbers',
    generations: 2,
    populationSize: 4,
  };
  const orch = await postJson(`${BASE_URL}/api/v1/workflows/orchestrate`, orchestrateBody);
  check(
    'orchestrate returns 2xx',
    orch.ok,
    `status=${orch.status} body=${truncate(orch.body)}`,
  );

  const workflowId = orch.body?.workflowId;
  check(
    'orchestrate returns a workflowId',
    typeof workflowId === 'string' && workflowId.length > 0,
    `workflowId=${truncate(workflowId, 80)}`,
  );

  // Contract nuance worth surfacing: WorkflowOrchestratorBubble.request() reads
  // `data.workflow_id` (snake_case), while this endpoint returns `workflowId`.
  // Non-fatal, but it means the bubble's WorkflowResult.workflowId is undefined
  // for start_workflow unless the caller already passed params.workflowId.
  if (orch.body && orch.body.workflow_id === undefined && workflowId) {
    note(
      'server returns "workflowId" but WorkflowOrchestratorBubble.request() reads "data.workflow_id" ' +
        '-> bubble start_workflow results carry an undefined workflowId (bubble-side mapping gap)',
    );
  }

  if (!workflowId) {
    throw new Error(`No workflowId returned; cannot poll run. body=${truncate(orch.body, 800)}`);
  }

  // --- 3. GET /api/v1/runs/{id} until completed ----------------------------
  console.log('');
  console.log(`3) GET /api/v1/runs/${workflowId} (poll until completed)`);
  const runUrl = `${BASE_URL}/api/v1/runs/${workflowId}`;
  const pollStart = Date.now();
  let run = null;
  let sawRunning = false;
  let statusLine = '';

  while (Date.now() - pollStart < RUN_TIMEOUT_MS) {
    const res = await getJson(runUrl, 5000);
    if (res.status !== 200) {
      throw new Error(`run poll returned HTTP ${res.status}: ${truncate(res.body, 500)}`);
    }
    run = res.body;
    if (run.status !== statusLine) {
      statusLine = run.status;
      console.log(`     status=${run.status} (+${Date.now() - pollStart}ms)`);
    }
    if (run.status === 'running' || run.status === 'pending') sawRunning = true;
    if (run.status === 'completed' || run.status === 'failed') break;
    await sleep(POLL_INTERVAL_MS);
  }

  const elapsed = Date.now() - pollStart;

  check(
    'run is retrievable via GET /api/v1/runs/{id}',
    run !== null && typeof run === 'object',
    `run_id=${truncate(run?.run_id, 80)}`,
  );
  check(
    'run reached status "completed"',
    run?.status === 'completed',
    `status=${truncate(run?.status)} after ${elapsed}ms${run?.error ? ` error=${truncate(run.error, 400)}` : ''}`,
  );
  check(
    'run.result is non-null',
    run?.result !== null && run?.result !== undefined,
    `resultKeys=${run?.result ? truncate(Object.keys(run.result)) : 'null'}`,
  );
  check(
    'run.result contains a non-empty best_code',
    typeof run?.result?.best_code === 'string' && run.result.best_code.length > 0,
    `best_code(${run?.result?.best_code?.length ?? 0} chars)=${truncate(run?.result?.best_code, 120)}`,
  );

  if (run?.result) {
    note(
      `best_score=${truncate(run.result.best_score, 40)} metrics=${truncate(run.result.metrics, 160)}`,
    );
  }
  if (sawRunning) note('observed the async running -> completed transition (202 + poll contract)');
}

// ---------------------------------------------------------------------------
let fatal = null;
try {
  await main();
} catch (err) {
  fatal = err instanceof Error ? err : new Error(String(err));
  check('test completed without a fatal error', false, fatal.message);
} finally {
  await stopServer(child);
}

// --- summary ----------------------------------------------------------------
const failed = results.filter((r) => !r.ok);
const passed = results.length - failed.length;

console.log('');
console.log('='.repeat(72));
console.log(`E2E CONTRACT SUMMARY: ${passed}/${results.length} checks passed`);
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
  if (tail) {
    console.log('');
    console.log(tail);
  }
}
console.log('');
console.log(failed.length === 0 ? 'RESULT: PASS' : 'RESULT: FAIL');
console.log('='.repeat(72));

process.exitCode = failed.length === 0 ? 0 : 1;
