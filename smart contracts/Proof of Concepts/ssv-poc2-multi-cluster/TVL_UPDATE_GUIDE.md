# TVL Update Guide - SSV Multi-Cluster Insolvency PoC

**Purpose:** Guide for updating TVL and funds-at-risk calculations in the PoC.

**Last Updated:** February 7, 2026  
**Current TVL:** ~60,600 SSV (~$215,130 USD)  
**Current SSV Price:** ~$3.55 USD

---

## Quick Reference

| Metric | Value | Source |
|--------|-------|--------|
| SSV Token Price | ~$3.55 USD | CoinGecko/CoinMarketCap |
| Total TVL (SSV) | ~60,600 SSV | On-chain data |
| Total TVL (USD) | ~$215,130 USD | Calculated |
| 10% of TVL | ~$21,513 USD | Calculated |
| **Minimum Bounty** | **$50,000 USD** | Immunefi tier |

---

## Where TVL Is Referenced

### Files with TVL References

```bash
# Find all TVL references
grep -r "TVL\|215130\|60600" --include="*.md" --include="*.sol" .
```

### Specific Files

| File | Line | Content to Update |
|------|------|-------------------|
| README.md | TVL section | TVL, USD value, 10% calc |
| SUBMISSION_GUIDE.md | Funds at Risk | TVL and USD amount |
| POC_COMPLIANCE_REPORT.md | TVL Calculation | All references |
| FINAL_AUDIT_REPORT.md | Funds at Risk | TVL amount |

---

## How to Update TVL

### Step 1: Get Current TVL

**Method 1: On-Chain (Recommended)**
```javascript
// Get SSV balance of Network contract
const ssvToken = await ethers.getContractAt("IERC20", SSV_TOKEN);
const balance = await ssvToken.balanceOf(SSV_NETWORK);
console.log("TVL:", ethers.formatEther(balance), "SSV");
```

**Method 2: DeFiLlama**
- Visit: https://defillama.com/protocol/ssv-network
- Look for "TVL" metric
- Note: May include staked ETH value too

**Method 3: Dune Analytics**
- Use SSV Network dashboard
- Query: `SELECT SUM(balance) FROM ssv_deposits`

### Step 2: Get Current SSV Price

**Sources:**
- CoinGecko: https://www.coingecko.com/en/coins/ssv-network
- CoinMarketCap: https://coinmarketcap.com/currencies/ssv-network/

**Calculation:**
```
Price = (CoinGecko Price + CoinMarketCap Price) / 2
```

### Step 3: Calculate USD Value

```
TVL_USD = TVL_SSV * SSV_Price
```

### Step 4: Calculate 10%

```
Ten_Percent = TVL_USD * 0.10
```

### Step 5: Update All Files

#### Update README.md

```markdown
## Funds at Risk

| Metric | Value |
|--------|-------|
| **TVL** | ~[NEW_TVL] SSV (~$[NEW_USD] USD) |
| **SSV Price** | ~$[NEW_PRICE] USD |
| **10% of TVL** | ~$[NEW_TEN_PERCENT] USD |
| **Minimum Bounty** | **$50,000 USD** |
```

#### Update Foundry Test

```solidity
console.log("TVL Update: [DATE]");
console.log("SSV Price: $[PRICE] USD");
console.log("TVL: [TVL] SSV (~$[USD] USD)");
console.log("10% of TVL: $[TEN_PERCENT] USD");
```

---

## Update Script

### Python Script

```python
# update_tvl.py
import json
from datetime import datetime

# Configuration
SSV_PRICE = 3.55  # Update this
TVL_SSV = 60600   # Update this

# Calculate
TVL_USD = TVL_SSV * SSV_PRICE
TEN_PERCENT = TVL_USD * 0.10

tvl_data = {
    "date": datetime.now().isoformat(),
    "ssv_price_usd": SSV_PRICE,
    "tvl_ssv": TVL_SSV,
    "tvl_usd": round(TVL_USD, 2),
    "ten_percent_usd": round(TEN_PERCENT, 2),
    "minimum_bounty": 50000
}

with open("TVL_DATA.json", "w") as f:
    json.dump(tvl_data, f, indent=2)

print(f"TVL Updated: {TVL_SSV} SSV = ${TVL_USD:,.2f} USD")
print(f"10% of TVL: ${TEN_PERCENT:,.2f} USD")
print(f"Minimum Bounty: $50,000 USD")
```

---

## Validation

After updating:

1. **Verify calculations:**
   ```bash
   # Check TVL_DATA.json
   cat TVL_DATA.json
   ```

2. **Update all files:**
   ```bash
   # Use sed or manual update
   sed -i 's/60600/[NEW_TVL]/g' README.md
   sed -i 's/215130/[NEW_USD]/g' README.md
   sed -i 's/3.55/[NEW_PRICE]/g' README.md
   ```

3. **Verify consistency:**
   ```bash
   grep -r "[NEW_TVL]\|[NEW_USD]" --include="*.md" .
   ```

---

## Historical TVL Data

| Date | TVL (SSV) | SSV Price | TVL (USD) | Source |
|------|-----------|-----------|-----------|--------|
| Feb 2026 | 60,600 | $3.55 | $215,130 | Multi-cluster PoC |

---

## Important Notes

1. **SSV Price Volatility:**
   - Check price within 24 hours of submission
   - Use average of CoinGecko + CoinMarketCap
   - Document price source

2. **TVL Fluctuations:**
   - TVL changes daily with deposits/withdrawals
   - Update before each submission
   - Use on-chain data when possible

3. **Documentation:**
   - Always include price source
   - Always include date of calculation
   - Round to reasonable precision

---

## Quick Update Checklist

- [ ] Get current SSV price (CoinGecko + CMC)
- [ ] Get current TVL (on-chain preferred)
- [ ] Calculate USD value
- [ ] Calculate 10%
- [ ] Update README.md
- [ ] Update test files
- [ ] Update submission docs
- [ ] Verify all files consistent
- [ ] Document update date

---

*Guide Version: 1.0*  
*Last Updated: February 7, 2026*
