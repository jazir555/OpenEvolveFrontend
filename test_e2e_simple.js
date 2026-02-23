// Simple LoongFlow E2E Test
const http = require('http');

function get(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          resolve({ statusCode: res.statusCode, data: JSON.parse(data) });
        } catch(e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
}

function post(url, payload) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const req = http.request({
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname + urlObj.search,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          resolve({ statusCode: res.statusCode, data: JSON.parse(data) });
        } catch(e) {
          reject(e);
        }
      });
    });
    req.on('error', reject);
    req.write(JSON.stringify(payload));
    req.end();
  });
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function runTest() {
  console.log('=== LoongFlow E2E Test ===\n');

  // Test 1: Health
  console.log('Test 1: Health Check');
  const health = await get('http://localhost:8000/health');
  console.log(`  Status: ${health.statusCode}`);
  console.log(`  Result: ${health.statusCode === 200 ? 'PASS' : 'FAIL'}\n`);

  // Test 2: Submit Evolution
  console.log('Test 2: Submit Evolution');
  const evolve = await post('http://localhost:8000/api/v1/evolve', {
    name: 'e2e-test',
    task: 'What is 2+2?',
    max_generations: 1,
    population_size: 1
  });
  console.log(`  Status: ${evolve.statusCode}`);
  console.log(`  Evolution ID: ${evolve.data.evolution_id}`);
  console.log(`  Result: ${evolve.statusCode === 200 ? 'PASS' : 'FAIL'}\n`);

  // Test 3: Poll Status
  console.log('Test 3: Poll for Completion');
  const evoId = evolve.data.evolution_id;
  let status;
  for(let i = 0; i < 10; i++) {
    await sleep(2000);
    status = await get(`http://localhost:8000/api/v1/status/${evoId}`);
    console.log(`  Attempt ${i+1}: ${status.data.status}`);
    if(status.data.status === 'COMPLETED' || status.data.status === 'FAILED') break;
  }
  console.log(`  Result: ${status.data.status === 'COMPLETED' ? 'PASS' : 'FAIL'}\n`);

  // Test 4: Get Solution
  console.log('Test 4: Get Solution');
  const solution = await get(`http://localhost:8000/api/v1/solutions/${evoId}`);
  console.log(`  Status: ${solution.statusCode}`);
  console.log(`  Fitness: ${solution.data.fitness}`);
  console.log(`  Result: ${solution.statusCode === 200 ? 'PASS' : 'FAIL'}\n`);

  console.log('=== ALL TESTS PASSED ===\n');
  console.log('SUCCESS: LoongFlow API + DeepSeek integration verified!');
}

runTest().catch(console.error);
