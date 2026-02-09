/**
 * SSV DAO Sybil Logic Demo (JS)
 * Demonstrates DAO fee inflation via dust clusters.
 */

const VICTIM_ASSETS = 10000;
const CLUSTER_COUNT = 50;
const DUST_DEPOSIT = 10;
const DAO_FEE = 0.5;
const WAIT_BLOCKS = 500;

function runDemo() {
    console.log(">>> SSV POC 4: DAO Sybil Inflation (JS Demo)");
    
    let totalDust = CLUSTER_COUNT * DUST_DEPOSIT;
    let pool = VICTIM_ASSETS + totalDust;
    
    console.log(`[INIT] Victim: ${VICTIM_ASSETS}, Pool: ${pool}`);
    
    // Bankruptcy Logic
    let bankruptBlock = Math.floor(DUST_DEPOSIT / DAO_FEE);
    let zombieBlocks = WAIT_BLOCKS - bankruptBlock;
    
    // The Bug: DAO earns unconditionally
    let daoUnbacked = zombieBlocks * DAO_FEE * CLUSTER_COUNT;
    
    console.log(`[DAO] Unbacked Fees Accrued: ${daoUnbacked}`);
    
    // Withdrawal
    let withdrawn = Math.min(daoUnbacked, pool);
    pool -= withdrawn;
    
    console.log(`[FINAL] Pool Remaining: ${pool}`);
    
    if (pool < VICTIM_ASSETS) {
        let loss = VICTIM_ASSETS - pool;
        console.log(`CRITICAL: DAO Sybils stole ${loss} SSV!`);
    }
}

runDemo();
