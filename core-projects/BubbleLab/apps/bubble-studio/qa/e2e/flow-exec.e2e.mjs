import { chromium } from 'playwright';
import fs from 'node:fs';

const BASE = 'http://localhost:3000';
const API = 'http://localhost:3001';
const SHOTS = 'qa/screenshots';
fs.mkdirSync(SHOTS, { recursive: true });

const wd = setTimeout(() => {
  console.error('WATCHDOG_TIMEOUT');
  process.exit(2);
}, 300000);

const consoleMsgs = [];
const pageErrors = [];
const failedReqs = [];

const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
});
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
page.on('console', (m) => consoleMsgs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => pageErrors.push(String(e)));
page.on('requestfailed', (r) => failedReqs.push(`${r.url()} :: ${r.failure()?.errorText}`));

const log = (...a) => console.log('[FLOW]', ...a);
const shot = async (n) => {
  try {
    await page.screenshot({ path: `${SHOTS}/${n}.png` });
    log('shot', n);
  } catch (e) {
    log('shot fail', n, String(e).split('\n')[0]);
  }
};
async function clickText(t, timeout = 8000, exact = true) {
  try {
    const el = page.getByText(t, { exact }).first();
    if ((await el.count()) > 0) {
      await el.click({ timeout });
      return true;
    }
  } catch (e) {
    log('clickText failed for', JSON.stringify(t), String(e).split('\n')[0]);
  }
  return false;
}
async function dismissModal() {
  for (let i = 0; i < 4; i++) {
    const overlay = page.locator('div.fixed.inset-0').first();
    if ((await overlay.count()) === 0) return;
    const option = overlay.getByText('Software Engineer', { exact: false }).first();
    if ((await option.count()) > 0) {
      try { await option.click({ timeout: 2000 }); await page.waitForTimeout(300); } catch {}
    }
    const next = overlay.getByText('Next', { exact: true }).first();
    if ((await next.count()) > 0) {
      try { await next.click({ timeout: 3000 }); log('onboarding: selected + Next'); await page.waitForTimeout(1000); continue; } catch {}
    }
    let closed = false;
    for (const label of ['Skip', 'Maybe later', 'Close', 'Not now', 'Get started', 'Continue', 'Dismiss']) {
      const b = overlay.getByText(label, { exact: false }).first();
      if ((await b.count()) > 0) { try { await b.click({ timeout: 3000 }); closed = true; log('clicked modal button:', label); break; } catch {} }
    }
    if (closed) { await page.waitForTimeout(800); continue; }
    await page.keyboard.press('Escape').catch(() => {});
    await page.waitForTimeout(500);
    if ((await overlay.count()) === 0) return;
    try { await overlay.click({ position: { x: 5, y: 5 }, timeout: 2000 }); } catch {}
    await page.waitForTimeout(500);
  }
}
const waitUrl = async (substr, ms = 20000) => {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (page.url().includes(substr)) return page.url();
    await page.waitForTimeout(500);
  }
  return page.url();
};

const HELLO_CODE = [
  "import { BubbleFlow, HelloWorldBubble, type WebhookEvent } from '@bubblelab/bubble-core';",
  '',
  'export interface Output {',
  '  greeting: string;',
  '}',
  '',
  'export interface CustomWebhookPayload extends WebhookEvent {',
  '  /**',
  '   * Name to include in the greeting.',
  '   * @canBeFile false',
  '   */',
  '  name?: string;',
  '}',
  '',
  "export class E2EHelloFlow extends BubbleFlow<'webhook/http'> {",
  '  async handle(payload: CustomWebhookPayload): Promise<Output> {',
  "    const { name = 'E2E Test' } = payload;",
  '',
  '    const helloWorld = new HelloWorldBubble({',
  "      name: name,",
  "      message: 'Hello from Bubble Studio E2E!',",
  '    });',
  '',
  '    const result = await helloWorld.action();',
  '',
  '    if (!result.success) {',
  '      throw new Error(`HelloWorld failed: ${result.error}`);',
  '    }',
  '',
  '    return { greeting: result.data.greeting };',
  '  }',
  '}',
].join('\n');

// Create a self-contained hello-world flow via the product's own API (dev bypass).
async function createHelloFlow() {
  const body = JSON.stringify({
    name: 'E2E HelloWorld',
    description: 'BubbleLab E2E hello-world flow',
    code: HELLO_CODE,
    prompt: '',
    eventType: 'webhook/http',
  });
  const res = await fetch(`${API}/bubble-flow`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });
  const txt = await res.text();
  if (!res.ok) throw new Error(`create failed ${res.status}: ${txt.slice(0, 300)}`);
  const json = JSON.parse(txt);
  return json.id ?? json.data?.id ?? json.flow?.id;
}

try {
  // ---- PHASE A: browser-driven CREATE (proves UI creation works) ----
  log('goto', BASE);
  await page.addInitScript(() => {
    try { localStorage.setItem('onboardingCompleted', 'true'); } catch (e) {}
  });
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(8000);
  await shot('e1-home');
  log('url after load:', page.url());
  await dismissModal();
  await page.waitForTimeout(1000);
  await shot('e1b-after-modal');

  let uiCreatedUrl = '';
  for (const t of ['Start from scratch', 'New Flow', 'Create Flow']) {
    if (await clickText(t, 6000)) { log('clicked create control:', t); break; }
  }
  uiCreatedUrl = await waitUrl('/flow/', 20000);
  log('UI create -> url:', uiCreatedUrl);
  await page.waitForTimeout(3000);
  await shot('e2-ui-created');
  log('UI CREATE: ' + (uiCreatedUrl.includes('/flow/') ? 'SUCCESS' : 'FAILED'));

  // ---- PHASE B: clean hello-world RUN via browser (read execution DOM) ----
  log('creating hello-world flow via API...');
  const helloId = await createHelloFlow();
  log('hello-world flow id:', helloId);
  if (!helloId) throw new Error('no flow id returned from API');

  await page.goto(`${BASE}/flow/${helloId}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(6000);
  await shot('e3-hello-ide');

  const hasRun = (await page.getByText('Run', { exact: true }).count()) > 0;
  const hasEditor = (await page.locator('.monaco-editor').count()) > 0;
  log('IDE check -> Run button:', hasRun, '| Monaco editor:', hasEditor);

  // Click Run (enabled button)
  let ran = false;
  const runEl = page.getByText('Run', { exact: true }).first();
  if ((await runEl.count()) > 0) {
    const disabled = await runEl.getAttribute('disabled').catch(() => null);
    log('Run disabled attr:', disabled);
    if (disabled === null) { await runEl.click({ timeout: 8000 }); ran = true; log('clicked Run'); }
  }
  if (!ran) log('!! could not click enabled Run');

  await page.waitForTimeout(4000);
  await shot('e4-running');

  let execFound = false;
  for (let i = 0; i < 40; i++) {
    const txt = await page.evaluate(() => document.body.innerText);
    if (/greeting|Hello from Bubble Studio E2E|execution_complete|Executing|Result/i.test(txt)) {
      execFound = true; break;
    }
    await page.waitForTimeout(1000);
  }
  log('execFound:', execFound);
  await page.waitForTimeout(3000);
  await shot('e5-execution');

  const execText = await page.evaluate(() => {
    const sel = '[class*="exec" i],[class*="log" i],[class*="result" i],[class*="output" i],[class*="console" i],[class*="stream" i]';
    const els = Array.from(document.querySelectorAll(sel));
    return els.map((e) => e.innerText).join('\n---\n').slice(0, 3000);
  });
  console.log('EXEC_DOM_START');
  console.log(execText);
  console.log('EXEC_DOM_END');

  const body = await page.evaluate(() => document.body.innerText.slice(0, 3500));
  console.log('BODY_AFTER_START');
  console.log(body);
  console.log('BODY_AFTER_END');
} catch (e) {
  console.error('SCRIPT_ERROR', e);
} finally {
  console.log('CONSOLE_ERRORS', consoleMsgs.filter((m) => m.startsWith('[error]')).length);
  console.log('PAGE_ERRORS', pageErrors.length);
  console.log('FAILED_REQS', failedReqs.length);
  console.log('CONSOLE_SAMPLE_START');
  console.log(consoleMsgs.slice(0, 80).join('\n'));
  console.log('CONSOLE_SAMPLE_END');
  if (pageErrors.length) {
    console.log('PAGEERRORS_START');
    console.log(pageErrors.join('\n'));
    console.log('PAGEERRORS_END');
  }
  await browser.close().catch(() => {});
  clearTimeout(wd);
  process.exit(0);
}
