/**
 * SSV Insolvency Logic Demo (JS)
 * Demonstrates the core "Accounting Mismatch" flaw.
 */

const TOTAL_ASSETS = 1010; // 1000 (Victim) + 10 (Bankrupt User)
const OPERATOR_FEE = 5;
const BLOCKS = 10;
const USER_B_DEPOSIT = 10;

function runDemo() {
    console.log(">>> SSV POC 1: Single Cluster Insolvency (JS Demo)");
    
    let assets = TOTAL_ASSETS;
    let operatorClaim = 0;
    let userBBalance = USER_B_DEPOSIT;
    
    console.log(`[INIT] Assets: ${assets}, User B: ${userBBalance}`);
    
    // Simulate Time
    let feesAccrued = OPERATOR_FEE * BLOCKS;
    
    // 1. Operator Logic (Unchecked)
    operatorClaim += feesAccrued;
    console.log(`[OP]   Fees Accrued: ${feesAccrued} (Unchecked)`);
    
    // 2. Cluster Logic (Capped)
    let actualDeduction = (feesAccrued > userBBalance) ? userBBalance : feesAccrued;
    userBBalance -= actualDeduction;
    console.log(`[USER] Balance Burned: ${actualDeduction} (Capped at 0)`);
    
    // 3. The Insolvency Gap
    let gap = feesAccrued - actualDeduction;
    console.log(`[GAP]  Virtual Debt Created: ${gap}`);
    
    // 4. Withdrawal
    assets -= operatorClaim;
    console.log(`[WITHDRAW] Operator takes ${operatorClaim}`);
    
    // 5. Victim Check
    console.log(`[FINAL] Assets Remaining: ${assets}`);
    if (assets < 1000) {
        console.log(`CRITICAL: Victim lost ${1000 - assets}! Insolvency Confirmed.`);
    }
}

runDemo();
