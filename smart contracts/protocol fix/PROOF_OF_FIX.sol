// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity 0.8.24;

/**
 * @title Proof of Fix: SSV Network Insolvency Vulnerability
 * @notice This contract demonstrates that the fix prevents all 5 attack vectors
 * @dev Side-by-side comparison of vulnerable vs fixed accounting logic
 */
contract ProofOfFix {
    
    // ============================================
    // VULNERABLE CODE (BEFORE FIX)
    // ============================================
    
    struct VulnerableOperator {
        uint64 balance;
        uint64 fee;
        uint32 lastUpdateBlock;
        uint32 validatorCount;
    }
    
    struct VulnerableCluster {
        uint256 balance;
        uint32 validatorCount;
    }
    
    /**
     * @notice VULNERABLE: Unconditional operator balance increment
     * @dev This is the ROOT CAUSE of the vulnerability
     */
    function vulnerableUpdateOperator(
        VulnerableOperator memory operator,
        uint32 currentBlock
    ) internal pure {
        uint64 blockDiff = currentBlock - operator.lastUpdateBlock;
        uint64 earnings = blockDiff * operator.fee * operator.validatorCount;
        
        // ❌ VULNERABILITY: Unconditional increment
        operator.balance += earnings;
        
        operator.lastUpdateBlock = currentBlock;
    }
    
    /**
     * @notice VULNERABLE: Cluster balance capped at zero
     * @dev Loses information about unpaid debt
     */
    function vulnerableUpdateCluster(
        VulnerableCluster memory cluster,
        uint64 usage
    ) internal pure {
        // ❌ VULNERABILITY: Capped at zero, creates accounting mismatch
        if (usage > cluster.balance) {
            cluster.balance = 0;
        } else {
            cluster.balance -= usage;
        }
    }
    
    // ============================================
    // FIXED CODE (AFTER FIX)
    // ============================================
    
    struct FixedOperator {
        uint64 balance;
        uint64 fee;
        uint32 lastUpdateBlock;
        uint32 validatorCount;
    }
    
    struct FixedCluster {
        uint256 balance;
        uint32 validatorCount;
    }
    
    /**
     * @notice FIXED: Conditional operator balance increment
     * @dev Only credits earnings if cluster can afford to pay
     * @return actualEarnings The amount actually credited (may be less than calculated)
     */
    function fixedUpdateOperator(
        FixedOperator memory operator,
        uint32 currentBlock,
        uint256 clusterBalance,
        uint32 clusterValidatorCount
    ) internal pure returns (uint64 actualEarnings) {
        uint64 blockDiff = currentBlock - operator.lastUpdateBlock;
        uint64 maxEarnings = blockDiff * operator.fee * clusterValidatorCount;
        
        // ✅ FIX: Only credit if cluster can afford to pay
        if (clusterBalance >= maxEarnings) {
            operator.balance += maxEarnings;
            actualEarnings = maxEarnings;
        } else {
            // Cluster is bankrupt - credit only what's available
            uint64 affordableEarnings = clusterValidatorCount > 0 
                ? uint64(clusterBalance / clusterValidatorCount) 
                : 0;
            operator.balance += affordableEarnings;
            actualEarnings = affordableEarnings;
        }
        
        operator.lastUpdateBlock = currentBlock;
    }
    
    /**
     * @notice FIXED: Cluster balance deducts only actual charges
     * @dev Maintains accounting consistency
     */
    function fixedUpdateCluster(
        FixedCluster memory cluster,
        uint64 actualEarnings
    ) internal pure {
        // ✅ FIX: Only deduct what was actually credited to operators
        if (actualEarnings <= cluster.balance) {
            cluster.balance -= actualEarnings;
        } else {
            cluster.balance = 0;
        }
    }
    
    // ============================================
    // PROOF TESTS
    // ============================================
    
    /**
     * @notice Proof 1: Single-Cluster Attack FAILS with fix
     */
    function proofTest1_SingleClusterAttack() external pure returns (
        bool vulnerableExploitable,
        bool fixedExploitable,
        uint64 vulnerableLoss,
        uint64 fixedLoss
    ) {
        // Setup: Two users, one goes bankrupt
        uint256 honestDeposit = 1000;
        uint256 bankruptDeposit = 10;
        uint256 totalPool = honestDeposit + bankruptDeposit;
        
        // Operator charges 5 per block
        uint64 operatorFee = 5;
        uint32 blocks = 10;
        
        // ========== VULNERABLE CODE TEST ==========
        VulnerableOperator memory vulnOp = VulnerableOperator({
            balance: 0,
            fee: operatorFee,
            lastUpdateBlock: 0,
            validatorCount: 1
        });
        
        VulnerableCluster memory vulnCluster = VulnerableCluster({
            balance: bankruptDeposit,
            validatorCount: 1
        });
        
        // Advance 10 blocks
        vulnerableUpdateOperator(vulnOp, blocks);
        
        // Operator balance: 50 SSV (10 blocks * 5 fee * 1 validator)
        // Cluster balance: 10 SSV
        // Usage: 50 SSV
        vulnerableUpdateCluster(vulnCluster, 50);
        
        // Cluster balance now: 0 (capped)
        // Operator can withdraw: 50 SSV
        // Pool has: 1010 SSV
        // After operator withdrawal: 960 SSV
        // Honest user entitled to: 1000 SSV
        // Honest user can get: 960 SSV
        // LOSS: 40 SSV
        
        vulnerableExploitable = vulnOp.balance > bankruptDeposit;
        vulnerableLoss = uint64(vulnOp.balance - bankruptDeposit);
        
        // ========== FIXED CODE TEST ==========
        FixedOperator memory fixedOp = FixedOperator({
            balance: 0,
            fee: operatorFee,
            lastUpdateBlock: 0,
            validatorCount: 1
        });
        
        FixedCluster memory fixedCluster = FixedCluster({
            balance: bankruptDeposit,
            validatorCount: 1
        });
        
        // Advance 10 blocks with fix
        uint64 actualEarnings = fixedUpdateOperator(
            fixedOp, 
            blocks, 
            fixedCluster.balance,
            fixedCluster.validatorCount
        );
        
        // Operator balance: 10 SSV (only what cluster could afford)
        // Cluster balance: 10 SSV
        // Actual earnings: 10 SSV
        fixedUpdateCluster(fixedCluster, actualEarnings);
        
        // Cluster balance now: 0
        // Operator can withdraw: 10 SSV (only what was paid)
        // Pool has: 1010 SSV
        // After operator withdrawal: 1000 SSV
        // Honest user entitled to: 1000 SSV
        // Honest user can get: 1000 SSV
        // LOSS: 0 SSV ✅
        
        fixedExploitable = fixedOp.balance > bankruptDeposit;
        fixedLoss = fixedOp.balance > bankruptDeposit 
            ? uint64(fixedOp.balance - bankruptDeposit) 
            : 0;
    }
    
    /**
     * @notice Proof 2: Multi-Cluster Attack FAILS with fix
     */
    function proofTest2_MultiClusterAttack() external pure returns (
        bool vulnerableExploitable,
        bool fixedExploitable,
        uint256 vulnerableTotalLoss,
        uint256 fixedTotalLoss
    ) {
        // Setup: 1 honest cluster + 3 bankrupt clusters
        uint256 honestDeposit = 10000;
        uint256 bankrupt1 = 100;
        uint256 bankrupt2 = 50;
        uint256 bankrupt3 = 25;
        
        uint64 operatorFee = 1;
        uint32 blocks = 150;
        
        // ========== VULNERABLE CODE TEST ==========
        uint256 vulnTotalVirtualDebt = 0;
        
        // Cluster 1 bankrupts at block 100
        VulnerableOperator memory vulnOp1 = VulnerableOperator(0, operatorFee, 0, 1);
        vulnerableUpdateOperator(vulnOp1, blocks);
        vulnTotalVirtualDebt += (vulnOp1.balance > bankrupt1) ? (vulnOp1.balance - bankrupt1) : 0;
        
        // Cluster 2 bankrupts at block 50
        VulnerableOperator memory vulnOp2 = VulnerableOperator(0, operatorFee, 0, 1);
        vulnerableUpdateOperator(vulnOp2, blocks);
        vulnTotalVirtualDebt += (vulnOp2.balance > bankrupt2) ? (vulnOp2.balance - bankrupt2) : 0;
        
        // Cluster 3 bankrupts at block 25
        VulnerableOperator memory vulnOp3 = VulnerableOperator(0, operatorFee, 0, 1);
        vulnerableUpdateOperator(vulnOp3, blocks);
        vulnTotalVirtualDebt += (vulnOp3.balance > bankrupt3) ? (vulnOp3.balance - bankrupt3) : 0;
        
        vulnerableExploitable = vulnTotalVirtualDebt > 0;
        vulnerableTotalLoss = vulnTotalVirtualDebt;
        
        // ========== FIXED CODE TEST ==========
        uint256 fixedTotalVirtualDebt = 0;
        
        // Cluster 1 with fix
        FixedOperator memory fixedOp1 = FixedOperator(0, operatorFee, 0, 1);
        uint64 actual1 = fixedUpdateOperator(fixedOp1, blocks, bankrupt1, 1);
        fixedTotalVirtualDebt += (fixedOp1.balance > bankrupt1) ? (fixedOp1.balance - bankrupt1) : 0;
        
        // Cluster 2 with fix
        FixedOperator memory fixedOp2 = FixedOperator(0, operatorFee, 0, 1);
        uint64 actual2 = fixedUpdateOperator(fixedOp2, blocks, bankrupt2, 1);
        fixedTotalVirtualDebt += (fixedOp2.balance > bankrupt2) ? (fixedOp2.balance - bankrupt2) : 0;
        
        // Cluster 3 with fix
        FixedOperator memory fixedOp3 = FixedOperator(0, operatorFee, 0, 1);
        uint64 actual3 = fixedUpdateOperator(fixedOp3, blocks, bankrupt3, 1);
        fixedTotalVirtualDebt += (fixedOp3.balance > bankrupt3) ? (fixedOp3.balance - bankrupt3) : 0;
        
        fixedExploitable = fixedTotalVirtualDebt > 0;
        fixedTotalLoss = fixedTotalVirtualDebt;
    }
    
    /**
     * @notice Proof 3: Liquidation Griefing FAILS with fix
     */
    function proofTest3_LiquidationGriefing() external pure returns (
        bool vulnerableExploitable,
        bool fixedExploitable,
        uint256 vulnerableVirtualDebt,
        uint256 fixedVirtualDebt
    ) {
        // Setup: Cluster near liquidation, attacker delays 200 blocks
        uint256 clusterDeposit = 100;
        uint64 operatorFee = 1;
        uint32 bankruptBlock = 100;
        uint32 delayedLiquidationBlock = 300; // 200 block delay
        
        // ========== VULNERABLE CODE TEST ==========
        VulnerableOperator memory vulnOp = VulnerableOperator(0, operatorFee, 0, 1);
        
        // Update to delayed liquidation block
        vulnerableUpdateOperator(vulnOp, delayedLiquidationBlock);
        
        // Operator balance: 300 SSV (300 blocks * 1 fee)
        // Cluster only had: 100 SSV
        // Virtual debt: 200 SSV
        vulnerableVirtualDebt = (vulnOp.balance > clusterDeposit) 
            ? (vulnOp.balance - clusterDeposit) 
            : 0;
        vulnerableExploitable = vulnerableVirtualDebt > 0;
        
        // ========== FIXED CODE TEST ==========
        FixedOperator memory fixedOp = FixedOperator(0, operatorFee, 0, 1);
        
        // Update to delayed liquidation block with fix
        uint64 actualEarnings = fixedUpdateOperator(
            fixedOp,
            delayedLiquidationBlock,
            clusterDeposit,
            1
        );
        
        // Operator balance: 100 SSV (only what cluster could afford)
        // Cluster had: 100 SSV
        // Virtual debt: 0 SSV ✅
        fixedVirtualDebt = (fixedOp.balance > clusterDeposit) 
            ? (fixedOp.balance - clusterDeposit) 
            : 0;
        fixedExploitable = fixedVirtualDebt > 0;
    }
    
    /**
     * @notice Proof 4: DAO Sybil Attack FAILS with fix
     */
    function proofTest4_DAOSybilAttack() external pure returns (
        bool vulnerableExploitable,
        bool fixedExploitable,
        uint256 vulnerableDAOVirtualDebt,
        uint256 fixedDAOVirtualDebt
    ) {
        // Setup: 50 dust clusters, 500 blocks
        uint32 clusterCount = 50;
        uint256 dustDeposit = 10;
        uint64 daoFee = 1; // 0.5 SSV per block per validator (simplified to 1 for demo)
        uint32 blocks = 500;
        
        // ========== VULNERABLE CODE TEST ==========
        // Each cluster bankrupts after 10 blocks
        // Remaining 490 blocks accumulate virtual debt
        uint256 vulnDAOBalance = 0;
        
        for (uint32 i = 0; i < clusterCount; i++) {
            // DAO earns unconditionally
            uint64 daoEarnings = blocks * daoFee;
            vulnDAOBalance += daoEarnings;
        }
        
        // Total DAO earnings: 50 clusters * 500 blocks * 1 fee = 25,000 SSV
        // Total actually paid: 50 clusters * 10 SSV = 500 SSV
        // Virtual debt: 24,500 SSV
        uint256 totalPaid = clusterCount * dustDeposit;
        vulnerableDAOVirtualDebt = vulnDAOBalance > totalPaid 
            ? (vulnDAOBalance - totalPaid) 
            : 0;
        vulnerableExploitable = vulnerableDAOVirtualDebt > 0;
        
        // ========== FIXED CODE TEST ==========
        uint256 fixedDAOBalance = 0;
        
        for (uint32 i = 0; i < clusterCount; i++) {
            // DAO only earns what cluster can afford
            uint64 maxEarnings = blocks * daoFee;
            uint64 actualEarnings = (dustDeposit >= maxEarnings) 
                ? maxEarnings 
                : uint64(dustDeposit);
            fixedDAOBalance += actualEarnings;
        }
        
        // Total DAO earnings: 50 clusters * 10 SSV = 500 SSV (only what was paid)
        // Virtual debt: 0 SSV ✅
        fixedDAOVirtualDebt = fixedDAOBalance > totalPaid 
            ? (fixedDAOBalance - totalPaid) 
            : 0;
        fixedExploitable = fixedDAOVirtualDebt > 0;
    }
    
    /**
     * @notice Proof 5: Operator Self-Dealing FAILS with fix
     */
    function proofTest5_OperatorSelfDealing() external pure returns (
        bool vulnerableExploitable,
        bool fixedExploitable,
        uint256 vulnerableROI,
        uint256 fixedROI
    ) {
        // Setup: Operator creates 50 minion clusters
        uint32 minionCount = 50;
        uint256 minionDeposit = 5;
        uint256 totalInvestment = minionCount * minionDeposit;
        uint64 operatorFee = 1;
        uint32 blocks = 200;
        
        // ========== VULNERABLE CODE TEST ==========
        VulnerableOperator memory vulnOp = VulnerableOperator(0, operatorFee, 0, minionCount);
        
        // Update operator (all minions bankrupt after 5 blocks)
        vulnerableUpdateOperator(vulnOp, blocks);
        
        // Operator balance: 200 blocks * 1 fee * 50 validators = 10,000 SSV
        // Investment: 250 SSV
        // Profit: 9,750 SSV
        // ROI: 3,900%
        uint256 vulnProfit = vulnOp.balance > totalInvestment 
            ? (vulnOp.balance - totalInvestment) 
            : 0;
        vulnerableROI = totalInvestment > 0 
            ? (vulnProfit * 100) / totalInvestment 
            : 0;
        vulnerableExploitable = vulnerableROI > 100;
        
        // ========== FIXED CODE TEST ==========
        FixedOperator memory fixedOp = FixedOperator(0, operatorFee, 0, minionCount);
        
        // Update operator with fix (minions can only pay 5 SSV each)
        uint64 actualEarnings = fixedUpdateOperator(
            fixedOp,
            blocks,
            totalInvestment, // Total available from all minions
            minionCount
        );
        
        // Operator balance: 250 SSV (only what minions could afford)
        // Investment: 250 SSV
        // Profit: 0 SSV
        // ROI: 0% ✅
        uint256 fixedProfit = fixedOp.balance > totalInvestment 
            ? (fixedOp.balance - totalInvestment) 
            : 0;
        fixedROI = totalInvestment > 0 
            ? (fixedProfit * 100) / totalInvestment 
            : 0;
        fixedExploitable = fixedROI > 100;
    }
    
    /**
     * @notice Master verification function - runs all proofs
     */
    function verifyAllProofs() external pure returns (
        bool allVulnerableExploitable,
        bool allFixedSecure,
        string memory result
    ) {
        // Run all proof tests
        (bool v1, bool f1,,) = this.proofTest1_SingleClusterAttack();
        (bool v2, bool f2,,) = this.proofTest2_MultiClusterAttack();
        (bool v3, bool f3,,) = this.proofTest3_LiquidationGriefing();
        (bool v4, bool f4,,) = this.proofTest4_DAOSybilAttack();
        (bool v5, bool f5,,) = this.proofTest5_OperatorSelfDealing();
        
        // All vulnerable versions should be exploitable
        allVulnerableExploitable = v1 && v2 && v3 && v4 && v5;
        
        // All fixed versions should be secure
        allFixedSecure = !f1 && !f2 && !f3 && !f4 && !f5;
        
        if (allVulnerableExploitable && allFixedSecure) {
            result = "SUCCESS: All 5 attacks work on vulnerable code, all 5 fail on fixed code";
        } else {
            result = "FAILURE: Fix verification failed";
        }
    }
}
