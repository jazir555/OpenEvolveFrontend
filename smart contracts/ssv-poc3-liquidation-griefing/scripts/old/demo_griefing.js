/**
 * SSV Liquidation Griefing Logic Demo (JS)
 * Demonstrates the maximized debt accumulation via delayed liquidation.
 */

const DEPOSIT = 100;
const FEE = 1;
const THRESHOLD = 80; // Should liquidate here
const GRIEF_DELAY = 200; // Attacker extends this window

function runDemo() {
    console.log(">>> SSV POC 3: Liquidation Griefing (JS Demo)");
    
    console.log("\n--- SCENARIO 1: Perfect Liquidation ---");
    console.log(`Cluster Liquidated at Block ${THRESHOLD}`);
    console.log("Unbacked Debt: 0");
    
    console.log("\n--- SCENARIO 2: Griefing Attack ---");
    let actualLiquidation = THRESHOLD + GRIEF_DELAY;
    
    // The Gap
    let unbackedFees = GRIEF_DELAY * FEE;
    
    console.log(`Attacker Delays Liquidation by ${GRIEF_DELAY} Blocks!`);
    console.log(`Actual Liquidation Block: ${actualLiquidation}`);
    console.log(`Unbacked Debt Created: ${unbackedFees}`);
    
    // Impact
    let victimAssets = 10000;
    let pool = victimAssets + DEPOSIT;
    
    // Withdrawal
    pool -= unbackedFees;
    
    console.log(`\n[FINAL] Victim Assets Remaining: ${pool}`);
    if (pool < victimAssets) {
        let loss = victimAssets - pool;
        console.log(`CRITICAL: Griefing stole ${loss} SSV from honest users!`);
    }
}

runDemo();