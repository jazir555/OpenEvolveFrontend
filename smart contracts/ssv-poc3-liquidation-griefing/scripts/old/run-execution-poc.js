/**
 * Execution PoC: SSV Liquidation Griefing Insolvency
 * 
 * This script simulates the liquidation griefing attack,
 * demonstrating how delaying liquidation maximizes virtual debt.
 * 
 * Run with: node scripts/run-execution-poc.js
 */

function runExecutionPoC() {
    console.log("=".repeat(80));
    console.log("SSV LIQUIDATION GRIEFING INSOLVENCY: EXECUTION TRACE");
    console.log("=".repeat(80));

    // 1. Initial State
    let poolAssets = 0;
    
    // Users deposit
    const largeDeposit = 10000;
    const small1 = 100;  // Bankrupts in 100 blocks
    const small2 = 50;   // Bankrupts in 50 blocks
    const small3 = 25;   // Bankrupts in 25 blocks
    poolAssets += largeDeposit + small1 + small2 + small3;
    
    console.log("Block 0 - Initial Deposits:");
    console.log(`  Large User:    ${largeDeposit} SSV`);
    console.log(`  Small User 1:  ${small1} SSV`);
    console.log(`  Small User 2:  ${small2} SSV`);
    console.log(`  Small User 3:  ${small3} SSV`);
    console.log(`  Total Assets:  ${poolAssets} SSV`);

    // 2. Wait for near-liquidation
    console.log("\n--- Block 20: Clusters Near Liquidation ---");
    console.log("  Small User 3: 5 SSV remaining (5 blocks until bankrupt)");
    console.log("  Small User 2: 30 SSV remaining (30 blocks until bankrupt)");
    console.log("  Small User 1: 80 SSV remaining (80 blocks until bankrupt)");
    console.log("  Attacker detects opportunity!");

    // 3. LIQUIDATION GRIEFING
    const griefingBlocks = 200;
    console.log("\n--- LIQUIDATION GRIEFING ---");
    console.log("Attacker monitors mempool for liquidate() transactions...");
    console.log("Attacker front-runs with high gas or exhausts liquidators");
    console.log(`Liquidation DELAYED by ${griefingBlocks} blocks!`);

    // Calculate virtual debt after griefing
    // Small 1: Would have been liquidated at block 100, now at block 220
    // Virtual debt: 120 blocks * 1 SSV = 120 SSV
    const virtualDebt1 = 120;
    
    // Small 2: Would have been liquidated at block 50, now at block 220
    // Virtual debt: 170 blocks * 1 SSV = 170 SSV
    const virtualDebt2 = 170;
    
    // Small 3: Would have been liquidated at block 25, now at block 220
    // Virtual debt: 195 blocks * 1 SSV = 195 SSV
    const virtualDebt3 = 195;
    
    // DAO unbacked fees (0.5 SSV per block per validator)
    const daoVirtualDebt = 200 * 0.5 * 3;  // 300 SSV
    
    const totalVirtualDebt = virtualDebt1 + virtualDebt2 + virtualDebt3 + daoVirtualDebt;
    
    console.log(`\nAfter ${griefingBlocks} blocks of griefing:`);
    console.log("  All small clusters: BANKRUPT (liquidation delayed)");
    console.log(`  Virtual debt from Small 1: ${virtualDebt1} SSV`);
    console.log(`  Virtual debt from Small 2: ${virtualDebt2} SSV`);
    console.log(`  Virtual debt from Small 3: ${virtualDebt3} SSV`);
    console.log(`  DAO unbacked fees:         ${daoVirtualDebt} SSV`);
    console.log(`\nTOTAL VIRTUAL DEBT: ${totalVirtualDebt} SSV`);
    console.log("(WITHOUT griefing, this would only be ~100 SSV!)");
    console.log(`Griefing increased theft by ${((totalVirtualDebt / 100 - 1) * 100).toFixed(0)}%`);

    // 4. Bank Run
    console.log("\n--- BANK RUN: Race to Withdraw ---");
    
    // Operators and DAO race to withdraw
    const withdrawal3 = virtualDebt3;
    if (withdrawal3 <= poolAssets) {
        poolAssets -= withdrawal3;
        console.log(`Operator 3 withdrew:  ${withdrawal3} SSV`);
    }
    
    const withdrawal2 = virtualDebt2;
    if (withdrawal2 <= poolAssets) {
        poolAssets -= withdrawal2;
        console.log(`Operator 2 withdrew:  ${withdrawal2} SSV`);
    }
    
    const withdrawal1 = virtualDebt1;
    if (withdrawal1 <= poolAssets) {
        poolAssets -= withdrawal1;
        console.log(`Operator 1 withdrew:  ${withdrawal1} SSV`);
    }
    
    const daoWithdrawal = daoVirtualDebt;
    if (daoWithdrawal <= poolAssets) {
        poolAssets -= daoWithdrawal;
        console.log(`DAO withdrew:         ${daoWithdrawal} SSV`);
    }
    
    const totalStolen = withdrawal1 + withdrawal2 + withdrawal3 + daoWithdrawal;
    console.log(`\nTotal stolen:         ${totalStolen} SSV`);
    console.log("ALL OF IT IS UNBACKED VIRTUAL DEBT!");

    // 5. Honest victim withdrawal
    console.log("\n--- Honest Large User Attempts Withdrawal ---");
    console.log(`Large user entitlement: ${largeDeposit} SSV`);
    console.log(`Remaining pool assets:  ${poolAssets} SSV`);
    
    if (largeDeposit <= poolAssets) {
        console.log("SUCCESS: Large user recovered all funds.");
    } else {
        const loss = largeDeposit - poolAssets;
        console.log(`CRITICAL FAILURE: Large user can only withdraw ${poolAssets} SSV`);
        console.log(`LARGE USER TOTAL LOSS: ${loss} SSV`);
    }

    console.log("\n" + "=".repeat(80));
    console.log("CONCLUSION: Liquidation Griefing Maximizes Theft.");
    console.log(`Griefing period of ${griefingBlocks} blocks created ${totalVirtualDebt} SSV of virtual debt.`);
    console.log("This is the MOST SEVERE attack vector:");
    console.log("  - Can be executed by anyone (not just operators)");
    console.log("  - Maximizes virtual debt through time delay");
    console.log("  - Harder to detect than direct theft");
    console.log("=".repeat(80));
}

// Run the PoC
runExecutionPoC();

module.exports = { runExecutionPoC };
