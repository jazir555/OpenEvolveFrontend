# TVL Update Guide for PoC Submission

## What is TVL?

**TVL = Total Value Locked**

This is the total amount of user funds (in SSV tokens) currently deposited in the SSV Network protocol. For this vulnerability, TVL represents the **total funds at risk** because the accounting flaw affects the entire shared pool.

---

## Current Issue in README.md

Section 4 of the README has **placeholder values** that need real data:

```markdown
## 4. Amount of Funds at Risk
As of current data, the SSV Network Vault contains significant user deposits.
Total Value Locked (TVL) in SSV tokens: **~[Current TVL] SSV**  <-- REPLACE THIS
Average Price of SSV: **~$[Current Price]**  <-- REPLACE THIS
**Estimated Total Funds at Risk: TVL * Price = $[Total Amount] USD**  <-- CALCULATE THIS
```

---

## Step-by-Step: How to Get Real Data

### Step 1: Get Current SSV Vault Balance

**Vault Address:** `0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D`

**Option A - Etherscan:**
1. Go to https://etherscan.io/address/0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D
2. Look for "Token Holdings" → SSV Token
3. Note the balance

**Option B - From bounty instructions:**
- Assets in vault: **60.6k SSV** (as of last update)

### Step 2: Get Current SSV Price

**Sources:**
- CoinGecko: https://www.coingecko.com/en/coins/ssv-network
- CoinMarketCap: https://coinmarketcap.com/currencies/ssv-network/

**Example price:** ~$3.50 - $4.00 USD (check current)

### Step 3: Calculate Funds at Risk

```
Funds at Risk = TVL × Current Price

Example:
  TVL = 60,600 SSV
  Price = $3.55 USD
  Funds at Risk = 60,600 × $3.55 = $215,130 USD
```

---

## Example UPDATE (Replace Section 4)

```markdown
## 4. Amount of Funds at Risk

**Vault Address:** 0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D  
**Data as of:** February 6, 2026

| Metric | Value |
|--------|-------|
| Total Value Locked (TVL) | 60,600 SSV |
| Average Price of SSV | $3.55 USD |
| **Total Funds at Risk** | **$215,130 USD** |

The vulnerability affects the *entire* shared pool of SSV tokens in the SSVNetwork 
contract, as any uncollateralized virtual debt is fulfilled from the total contract 
balance. All user deposits are at risk of partial or total loss due to protocol 
insolvency.
```

---

## Why This Matters for Immunefi

### Bounty Calculation

Per the bounty instructions:

> "For critical Smart Contract bugs, the reward amount is **10% of the funds directly 
> affected** up to a maximum of USD $1,000,000."

| Metric | Value |
|--------|-------|
| Funds at Risk | $215,130 USD |
| 10% of Funds | $21,513 USD |
| **Minimum Bounty** | $50,000 USD |
| **Expected Bounty** | **$50,000 USD** (capped at minimum) |
| Maximum Possible | $1,000,000 USD |

**Note:** The minimum bounty ($50,000) applies because 10% of the funds ($21,513) 
is below the minimum threshold.

### What Immunefi Requires

From `POC_rules.txt`:

> "Additionally, the whitehat should also ideally determine and provide data on 
> the amount of funds at risk, which can be determined by calculating the total 
> amount of tokens multiplied by the average price of the token at the time of 
> the submission."

---

## Quick Reference: Data Sources

| Data | Source | URL |
|------|--------|-----|
| SSV Vault Balance | Etherscan | https://etherscan.io/address/0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D |
| SSV Price | CoinGecko | https://www.coingecko.com/en/coins/ssv-network |
| SSV Price | CoinMarketCap | https://coinmarketcap.com/currencies/ssv-network/ |
| Bounty Info | Immunefi | https://immunefi.com/bug-bounty/ssvnetwork/ |

---

## Action Items

Before submitting to Immunefi:

1. [ ] Check current SSV balance in vault address
2. [ ] Get current SSV price from CoinGecko/CoinMarketCap
3. [ ] Calculate total funds at risk (TVL × Price)
4. [ ] Update README.md Section 4 with real numbers
5. [ ] Include date of data collection
6. [ ] Save screenshots of price/data for verification

---

## Example Calculation (Template)

```
Date: [TODAY'S DATE]
SSV Vault Balance: [AMOUNT] SSV
SSV Price (CoinGecko): $[PRICE] USD
Calculation: [AMOUNT] × $[PRICE] = $[TOTAL] USD

Total Funds at Risk: $[TOTAL] USD
```

---

**Note:** The vulnerability affects ALL user deposits in the shared pool. 
The entire TVL should be considered at risk because the insolvency mechanism 
can drain the entire contract balance over time.
