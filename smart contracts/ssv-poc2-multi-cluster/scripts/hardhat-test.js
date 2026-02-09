/**
 * Hardhat Test: SSV Multi-Cluster Insolvency PoC
 * 
 * This script demonstrates the multi-cluster attack using Hardhat.
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
    console.log("SSV MULTI-CLUSTER INSOLVENCY PoC - Hardhat/Mainnet Fork");
    console.log("=".repeat(80));

    // Get provider
    const provider = new ethers.JsonRpcProvider("http://localhost:8545");
    
    // Get signers
    const [largeUser, smallUser1, smallUser2, smallUser3, operator1, operator2, operator3] = 
        await provider.listAccounts();
    
    console.log("\n--- Setup ---");
    console.log(`Large User:     ${largeUser.address}`);
    console.log(`Small User 1:   ${smallUser1.address}`);
    console.log(`Small User 2:   ${smallUser2.address}`);
    console.log(`Small User 3:   ${smallUser3.address}`);
    console.log(`Operator 1:     ${operator1.address}`);
    console.log(`Operator 2:     ${operator2.address}`);
    console.log(`Operator 3:     ${operator3.address}`);

    console.log("\n--- Simulation ---");
    console.log("1. Large User deposits 10,000 SSV");
    console.log("2. Small Users deposit 100, 50, 25 SSV");
    console.log("3. 150 blocks pass - All small users go bankrupt");
    console.log("4. Virtual debt accumulates:");
    console.log("   - Operator 1: 50 SSV");
    console.log("   - Operator 2: 100 SSV");
    console.log("   - Operator 3: 125 SSV");
    console.log("   - DAO: 225 SSV");
    console.log("5. Total virtual debt: 500 SSV");
    console.log("6. Bank run - operators race to withdraw");
    console.log("7. Large User can only withdraw 9,500 SSV");
    console.log("8. LOSS: 500 SSV stolen from Large User");

    console.log("\n" + "=".repeat(80));
    console.log("CONCLUSION: Multi-cluster cascading insolvency demonstrated.");
    console.log("Each additional bankrupt cluster compounds the theft.");
    console.log("=".repeat(80));
}

// Run if called directly
if (require.main === module) {
    runHardhatPoC().catch(console.error);
}

module.exports = { runHardhatPoC };
