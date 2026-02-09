// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import {Test} from "forge-std/Test.sol";
import {SSVDaoSybilPoC} from "../src/SSVDaoSybilPoC.sol";

contract SSVDaoSybilTest is Test {
    SSVDaoSybilPoC public poc;

    function setUp() public {
        // We use a specific block for fork consistency
        vm.createSelectFork(vm.rpcUrl("mainnet"), 19000000);
        
        // Drain existing pool to isolate our test assets
        deal(0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54, 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1, 0);
        
        poc = new SSVDaoSybilPoC();
    }

    function testDaoSybilAttack() public {
        poc.initiateAttack();
    }
}
