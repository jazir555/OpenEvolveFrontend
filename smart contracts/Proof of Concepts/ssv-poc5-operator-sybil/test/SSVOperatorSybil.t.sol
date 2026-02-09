// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {Test} from "forge-std/Test.sol";
import {SSVOperatorSybilPoC} from "../src/SSVOperatorSybilPoC.sol";

contract SSVOperatorSybilTest is Test {
    SSVOperatorSybilPoC public poc;

    function setUp() public {
        vm.createSelectFork(vm.rpcUrl("mainnet"), 19000000);
        
        // Drain existing pool to isolate our test assets
        deal(0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54, 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1, 0);
        
        poc = new SSVOperatorSybilPoC();
    }

    function testOperatorSybilAttack() public {
        poc.initiateAttack();
    }
}
