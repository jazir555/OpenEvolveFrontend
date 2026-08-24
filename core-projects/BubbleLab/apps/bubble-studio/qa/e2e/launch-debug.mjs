import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';

const log = (m) => {
  const line = `[${new Date().toISOString()}] ${m}`;
  writeFileSync('launch-debug.log', line + '\n', { flag: 'a' });
  console.log(line);
};

const withTimeout = (p, ms, label) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error('TIMEOUT ' + label)), ms)),
  ]);

(async () => {
  log('start');
  try {
    log('launching (timeout 25s)...');
    const b = await withTimeout(chromium.launch({ headless: true }), 25000, 'launch');
    log('launched OK');
    const p = await b.newPage();
    await withTimeout(p.goto('http://localhost:3000', { waitUntil: 'domcontentloaded' }), 20000, 'goto');
    log('goto OK, title=' + (await p.title()));
    await b.close();
    log('closed OK');
  } catch (e) {
    log('ERROR: ' + (e && e.stack ? e.stack : e));
  }
  process.exit(0);
})();
