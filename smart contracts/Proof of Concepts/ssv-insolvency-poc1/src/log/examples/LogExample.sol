pragma solidity ^0.8.0;

import {Log} from "../Log.sol";

contract LogExample is Log {
    function initializeAttack() public modInitializeAttack {
        // Turn off all logs except from those inside the `initializeAttack()` function
        //_setLogType(LogType.INITIALIZE_ATTACK);

        // Turn off all logs except from those inside the `_executeAttack()` function
        //_setLogType(LogType.EXECUTE_ATTACK);

        // Turn off all logs except from those inside the `_completeAttack()` function
        //_setLogType(LogType.COMPLETE_ATTACK);

        // Turn off all logs
        _setLogType(LogType.NONE);

        _log("\n>>> Initialize attack");

        _executeAttack();
    }

    function _executeAttack() internal modExecuteAttack {
        _log("\n>>> Execute attack");

        _completeAttack();
    }

    function _completeAttack() internal modCompleteAttack {
        _log("\n>>> Complete attack");
    }
}
