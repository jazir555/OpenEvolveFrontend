/**
 * SSV Multi-Cluster Insolvency Logic Demo (JS)
 * Demonstrates how multiple bankrupt clusters compound the debt.
 */

const VICTIM_DEPOSIT = 10000;
const SMALL_CLUSTERS = [100, 50, 25];
const BLOCKS = 150;
const OP_FEE = 1;
const DAO_FEE = 0.5;

function runDemo() {
    console.log(">>> SSV POC 2: Multi-Cluster Insolvency (JS Demo)");
    
    let poolAssets = VICTIM_DEPOSIT + SMALL_CLUSTERS.reduce((a, b) => a + b, 0);
    console.log(`[INIT] Pool Assets: ${poolAssets}`);
    
    let totalVirtualDebt = 0;
    
    SMALL_CLUSTERS.forEach((deposit, i) => {
        let burnRate = OP_FEE + DAO_FEE;
        let bankruptBlock = Math.floor(deposit / burnRate);
        
        // Post-Bankruptcy Window
        let unbackedBlocks = Math.max(0, BLOCKS - bankruptBlock);
        
        // Unbacked Earnings
        let opUnbacked = unbackedBlocks * OP_FEE;
        let daoUnbacked = unbackedBlocks * DAO_FEE;
        
        let debt = opUnbacked + daoUnbacked;
        totalVirtualDebt += debt;
        
        console.log(`[CLUSTER ${i+1}] Bankrupt @ ${bankruptBlock}. Unbacked Blocks: ${unbackedBlocks}`);
        console.log(`            Generated Virtual Debt: ${debt}`);
    });
    
    console.log(`[TOTAL] Global Virtual Debt: ${totalVirtualDebt}`);
    
    // Withdrawals (Bank Run)
    poolAssets -= totalVirtualDebt;
    console.log(`[FINAL] Pool Assets Remaining: ${poolAssets}`);
    
    if (poolAssets < VICTIM_DEPOSIT) {
        console.log(`CRITICAL: Victim Lost ${VICTIM_DEPOSIT - poolAssets}! Bank Run Logic Confirmed.`);
    }
}

runDemo();
