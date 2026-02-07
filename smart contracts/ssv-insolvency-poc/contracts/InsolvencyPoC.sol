// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SSV Insolvency PoC
 * @dev This contract isolates the EXACT accounting logic from ssv.network:
 * 1. Operator balance increment (OperatorLib.sol:19)
 * 2. Cluster balance decrement capped at zero (ClusterLib.sol:16)
 */
contract MockSSVAccounting {
    struct Snapshot {
        uint64 block;
        uint256 balance;
    }

    struct Operator {
        uint256 fee;
        Snapshot snapshot;
    }

    struct Cluster {
        uint256 balance;
        uint256 index;
    }

    mapping(uint64 => Operator) public operators;
    mapping(bytes32 => Cluster) public clusters;
    uint256 public totalContractAssets;

    // --- Core Protocol Logic (Replicated from Source) ---

    function deposit(bytes32 clusterId, uint256 amount) external {
        clusters[clusterId].balance += amount;
        totalContractAssets += amount;
    }

    /**
     * Replicates OperatorLib.updateSnapshotSt logic
     */
    function updateOperatorSnapshot(uint64 operatorId, uint32 currentBlock) public {
        Operator storage op = operators[operatorId];
        uint256 blockDiff = currentBlock - op.snapshot.block;
        uint256 earnings = blockDiff * op.fee;
        
        op.snapshot.balance += earnings; // Unconditional increment (The Flaw)
        op.snapshot.block = currentBlock;
    }

    /**
     * Replicates ClusterLib.updateBalance logic
     */
    function updateClusterBalance(bytes32 clusterId, uint64 operatorId, uint32 currentBlock) public {
        Cluster storage clus = clusters[clusterId];
        Operator storage op = operators[operatorId];
        
        uint256 blockDiff = currentBlock - op.snapshot.block;
        uint256 usage = blockDiff * op.fee;

        // Capped at zero (The Mismatch)
        clus.balance = usage > clus.balance ? 0 : clus.balance - usage;
    }

    function withdrawOperatorEarnings(uint64 operatorId, uint256 amount) external {
        require(operators[operatorId].snapshot.balance >= amount, "Insufficient virtual balance");
        require(totalContractAssets >= amount, "PROTOCOL INSOLVENT: Contract Empty!");
        
        operators[operatorId].snapshot.balance -= amount;
        totalContractAssets -= amount;
    }

    function setOperator(uint64 id, uint256 fee, uint32 startBlock) external {
        operators[id].fee = fee;
        operators[id].snapshot.block = startBlock;
    }
}
