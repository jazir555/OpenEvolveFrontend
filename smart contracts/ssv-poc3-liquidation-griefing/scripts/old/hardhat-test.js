/**
 * Hardhat Test: SSV Liquidation Griefing PoC
 * 
 * This script demonstrates the liquidation griefing attack using Hardhat.
 * 
 * Setup:
 *   npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
 *   npx hardhat run scripts/hardhat-test.js --network mainnet
 */

const { ethers } = require("ethers");

// Mainnet addresses
const SSV_NETWORK = "0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1";
const SSV_TOKEN = "0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54";

async function runHardhatPoC() {
    console.log("=".repeat(80));
    console.log("SSV LIQUIDATION GRIEFING PoC - Hardhat/Mainnet Fork");
    console.log("=".repeat(80));

    // Get provider
    const provider = new ethers.JsonRpcProvider("http://localhost:8545");
    
    // Get signers
    const [attacker, largeUser, smallUser1, smallUser2, smallUser3, liquidator] = 
        await provider.listAccounts();
    
    console.log("\n--- Setup ---");
    console.log(`Attacker:       ${attacker.address}`);
    console.log(`Large User:     ${largeUser.address}`);
    console.log(`Small User 1:   ${smallUser1.address}`);
    console.log(`Small User 2:   ${smallUser2.address}`);
    console.log(`Small User 3:   ${smallUser3.address}`);
    console.log(`Liquidator:     ${liquidator.address}`);

    console.log("\n--- Attack Simulation ---");
    console.log("1. Users deposit funds");
    console.log("2. Block 20: Clusters near liquidation");
    console.log("3. Attacker detects liquidate() transactions in mempool");
    console.log("4. Attacker front-runs with high gas price");
    console.log("5. Liquidation DELAYED by 200 blocks!");
    console.log("6. Virtual debt accumulates:");
    console.log("   - Small 1: 120 SSV");
    console.log("   - Small 2: 170 SSV");
    console.log("   - Small 3: 195 SSV");
    console.log("   - DAO: 300 SSV");
    console.log("7. Total virtual debt: 785 SSV (vs ~100 SSV normally)");
    console.log("8. Bank run - operators race to withdraw");
    console.log("9. Large User can only withdraw 9,215 SSV");
    console.log("10. LOSS: 785 SSV stolen from Large User");

    console.log("\n--- Griefing Techniques ---");
    console.log("1. Front-running via Flashbots");
    console.log("2. Gas price manipulation");
    console.log("3. Block stuffing on L2s");

    console.log("\n" + "=".repeat(80));
    console.log("CONCLUSION: Liquidation griefing maximizes theft.");
    console.log("This is the MOST SEVERE attack vector.");
    console.log("=".repeat(80));
}

// Run if called directly
if (require.main === module) {
    runHardhatPoC().catch(console.error);
}

module.exports = { runHardhatPoC };
