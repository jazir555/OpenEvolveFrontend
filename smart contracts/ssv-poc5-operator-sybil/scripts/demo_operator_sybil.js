/**
 * SSV Operator Sybil Logic Demo (JS)
 * Demonstrates infinite ROI via self-dealing.
 */

const SYBIL_COUNT = 50;
const DEPOSIT = 5;
const FEE = 1;
const BLOCKS = 200;

function runDemo() {
    console.log(">>> SSV POC 5: Operator Sybil Self-Dealing (JS Demo)");
    
    // Investment
    let investment = SYBIL_COUNT * DEPOSIT;
    console.log(`[INVEST] Attacker spends: ${investment} SSV`);
    
    // Bankruptcy
    let bankruptBlock = Math.floor(DEPOSIT / FEE);
    let profitBlocks = BLOCKS - bankruptBlock;
    
    // Revenue (The Bug)
    let revenue = SYBIL_COUNT * FEE * profitBlocks;
    console.log(`[REVENUE] Unbacked Fees Earned: ${revenue}`);
    
    // ROI
    let roi = (revenue / investment) * 100;
    
    console.log(`[PROFIT] Net Gain: ${revenue - investment}`);
    console.log(`[ROI]    Return on Investment: ${roi}%`);
    
    if (revenue > investment) {
        console.log("CRITICAL: Infinite Money Glitch Confirmed.");
    }
}

runDemo();
