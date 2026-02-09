// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SSV Multi-Cluster Insolvency PoC
 * @dev This contract isolates the EXACT accounting logic from ssv.network:
 * 1. Operator balance increment (OperatorLib.sol:19)
 * 2. DAO balance increment (ProtocolLib.sol)
 * 3. Cluster balance decrement capped at zero (ClusterLib.sol:16)
 * 4. Multiple clusters with compounding virtual debt
 */
contract MockMultiClusterAccounting {
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
        uint256 validatorCount;
    }

    mapping(uint64 => Operator) public operators;
    mapping(bytes32 => Cluster) public clusters;
    mapping(uint64 => uint256) public daoSnapshots; // DAO earnings per operator
    uint256 public totalContractAssets;
    uint256 public daoFee = 0.05e18; // 5% DAO fee

    // --- Core Protocol Logic (Replicated from Source) ---

    function deposit(bytes32 clusterId, uint256 amount) external {
        clusters[clusterId].balance += amount;
        totalContractAssets += amount;
    }

    /**
     * Replicates OperatorLib.updateSnapshotSt logic
     * UNCONDITIONAL increment - The Root Cause
     */
    function updateOperatorSnapshot(uint64 operatorId, uint32 currentBlock) public {
        Operator storage op = operators[operatorId];
        uint256 blockDiff = currentBlock - op.snapshot.block;
        
        // Calculate earnings from ALL clusters including bankrupt ones
        uint256 totalValidators = getTotalValidatorsForOperator(operatorId);
        uint256 earnings = blockDiff * op.fee * totalValidators;
        
        op.snapshot.balance += earnings; // Unconditional increment (The Flaw)
        op.snapshot.block = currentBlock;
        
        // DAO also earns from ALL clusters
        uint256 daoEarnings = (earnings * daoFee) / 1e18;
        daoSnapshots[operatorId] += daoEarnings;
    }

    /**
     * Replicates ClusterLib.updateBalance logic
     * CAPPED at zero - The Mismatch
     */
    function updateClusterBalance(bytes32 clusterId, uint64 operatorId, uint32 currentBlock) public {
        Cluster storage clus = clusters[clusterId];
        Operator storage op = operators[operatorId];
        
        uint256 blockDiff = currentBlock - op.snapshot.block;
        uint256 usage = blockDiff * op.fee * clus.validatorCount;

        // Capped at zero (The Mismatch - doesn't track negative balance)
        clus.balance = usage > clus.balance ? 0 : clus.balance - usage;
    }

    function withdrawOperatorEarnings(uint64 operatorId, uint256 amount) external {
        require(operators[operatorId].snapshot.balance >= amount, "Insufficient virtual balance");
        require(totalContractAssets >= amount, "PROTOCOL INSOLVENT: Contract Empty!");
        
        operators[operatorId].snapshot.balance -= amount;
        totalContractAssets -= amount;
    }

    function withdrawDAOEarnings(uint64 operatorId, uint256 amount) external {
        require(daoSnapshots[operatorId] >= amount, "Insufficient DAO balance");
        require(totalContractAssets >= amount, "PROTOCOL INSOLVENT: Contract Empty!");
        
        daoSnapshots[operatorId] -= amount;
        totalContractAssets -= amount;
    }

    function setOperator(uint64 id, uint256 fee, uint32 startBlock) external {
        operators[id].fee = fee;
        operators[id].snapshot.block = startBlock;
    }

    function setCluster(bytes32 id, uint256 balance, uint256 validators) external {
        clusters[id].balance = balance;
        clusters[id].validatorCount = validators;
    }

    function getTotalValidatorsForOperator(uint64 operatorId) public view returns (uint256) {
        // Simplified: sum validators across all clusters for this operator
        return 10; // Placeholder
    }

    function getVirtualDebt(bytes32 clusterId) external view returns (uint256) {
        // Calculate uncollateralized debt for a cluster
        Cluster storage clus = clusters[clusterId];
        if (clus.balance == 0) {
            return clus.validatorCount * 100; // Simplified calculation
        }
        return 0;
    }
}
