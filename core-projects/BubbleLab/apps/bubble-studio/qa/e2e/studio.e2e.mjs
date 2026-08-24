import { chromium } from 'playwright';
import fs from 'node:fs';

const BASE = 'http://localhost:3000';
const SHOTS = 'qa/screenshots';
fs.mkdirSync(SHOTS, { recursive: true });

// WATCHDOG: never hang
const watchdog = setTimeout(() => { console.error('WATCHDOG_TIMEOUT'); process.exit(2); }, 240000);

const consoleMsgs = [], pageErrors = [], failedReqs = [];
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox','--disable-setuid-sandbox','--disable-gpu','--disable-dev-shm-usage'] });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
page.on('console', m => consoleMsgs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', e => pageErrors.push(String(e)));
page.on('requestfailed', r => failedReqs.push(`${r.url()} :: ${r.failure()?.errorText}`));
const log = (...a) => console.log('[E2E]', ...a);

try {
  log('goto', BASE);
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(9000); // let vite settle
  log('title=', await page.title());
  await page.screenshot({ path: `${SHOTS}/01-home.png` });

  const bodyText = (await page.evaluate(() => document.body.innerText)).slice(0, 1800);
  console.log('BODYTEXT_START'); console.log(bodyText); console.log('BODYTEXT_END');

  const navText = await page.evaluate(() => Array.from(document.querySelectorAll('nav, aside, header, [class*="Sidebar"], [class*="sidebar"]')).map(e => e.innerText).join('\n---\n').slice(0, 1800));
  console.log('NAVTEXT_START'); console.log(navText); console.log('NAVTEXT_END');
  await page.screenshot({ path: `${SHOTS}/02-nav.png` });

  // manifest count directly
  try {
    const bj = await (await fetch('http://localhost:3000/bubbles.json')).json();
    log('manifest bubble count =', bj.bubbles.length, '(expect 282)');
  } catch (e) { log('manifest fetch failed', String(e)); }

  // enumerate clickable labels
  const labels = await page.evaluate(() => Array.from(document.querySelectorAll('a,button')).map(e => (e.innerText||'').trim()).filter(Boolean).slice(0, 120));
  console.log('LINKS_START'); console.log(labels.join(' | ')); console.log('LINKS_END');

  // try to open a Flow IDE / create flow (best effort)
  for (const sel of ['text=New Flow','text=Create Flow','text=New Bubble Flow','text=Bubble Flows','[class*="new-flow"]','[data-testid*="new"]']) {
    try { const el = page.locator(sel).first(); if (await el.count() > 0) { await el.click({ timeout: 5000 }); log('clicked', sel); break; } } catch {}
  }
  await page.waitForTimeout(4000);
  await page.screenshot({ path: `${SHOTS}/03-flow.png` });

  // try to search for hello-world in any search input (best effort)
  try {
    const search = page.locator('input[type="search"], input[placeholder*="Search" i], input[placeholder*="search" i]').first();
    if (await search.count() > 0) { await search.fill('hello-world'); await page.waitForTimeout(1500); await page.screenshot({ path: `${SHOTS}/04-search.png` }); log('searched hello-world'); }
  } catch (e) { log('search step skipped', String(e)); }
} catch (e) {
  console.error('SCRIPT_ERROR', e);
} finally {
  console.log('CONSOLE_ERRORS', consoleMsgs.filter(m => m.startsWith('[error]')).length);
  console.log('PAGE_ERRORS', pageErrors.length);
  console.log('FAILED_REQS', failedReqs.length);
  console.log('CONSOLE_SAMPLE_START'); console.log(consoleMsgs.slice(0, 50).join('\n')); console.log('CONSOLE_SAMPLE_END');
  if (pageErrors.length) { console.log('PAGEERRORS_START'); console.log(pageErrors.join('\n')); console.log('PAGEERRORS_END'); }
  if (failedReqs.length) { console.log('FAILEDREQS_START'); console.log(failedReqs.slice(0,20).join('\n')); console.log('FAILEDREQS_END'); }
  await browser.close().catch(()=>{});
  clearTimeout(watchdog);
  process.exit(0);
}
