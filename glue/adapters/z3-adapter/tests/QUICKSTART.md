# Z3 Adapter Contract Tests - Quick Reference

## 🚀 Quick Start (30 seconds)

```bash
cd glue/adapters/z3-adapter
npm install
npm test
```

## 📋 Essential Commands

| Command | Description |
|---------|-------------|
| `npm test` | Run all tests |
| `npm run test:contract` | Run contract tests only |
| `npm run test:watch` | Watch mode (development) |
| `npm run test:coverage` | Generate coverage report |
| `make test` | Run tests via Makefile |

## ⚠️ CRITICAL RULES

1. **FAIL FAST**: If tests fail → Adapter does NOT start
2. **NO TRUST**: Mock only - don't require running Z3
3. **UTC TIME**: All timestamps in ISO-8601 with Z suffix
4. **CORRELATE**: Every request/response must be traceable

## 📁 File Structure

```
glue/adapters/z3-adapter/
├── package.json                  # Dependencies and Jest config
├── tsconfig.json                 # TypeScript config
├── Makefile                      # Convenient commands
├── tests/
│   ├── contract.test.ts          # Main contract tests
│   ├── jest.setup.ts             # Test setup
│   ├── README.md                 # Full documentation
│   ├── QUICKSTART.md             # This file
│   └── INTEGRATION_EXAMPLE.md    # Integration examples
└── .husky/
    └── pre-commit                # Git hook
```

## 🎯 Test Categories

### 1. API Contracts (80% of tests)
- Health endpoint responses
- Solve/Optimize/Simplify/Tactic/Fixedpoint endpoints
- Response structure validation
- Schema conformance

### 2. Correlation (5% of tests)
- Request/response correlation
- CanonicalLogEntry transformation

### 3. Database (5% of tests)
- Knowledge queries
- ORM model structures
- UTC timestamps

### 4. Knowledge Extraction (10% of tests)
- Graph structures (nodes/edges)
- Edge cases (empty, disconnected)
- Complex nested data

## 🔧 Common Tasks

### Add New Endpoint Test

```typescript
describe('POST /new-endpoint', () => {
  test('conforms to schema', () => {
    const mockResponse = { /* data */ };
    const result = YourSchema.safeParse(mockResponse);
    expect(result.success).toBe(true);
  });
});
```

### Update Mock Data

Edit `tests/contract.test.ts`:

```typescript
const mockNewResponse = {
  field: 'value',
  // ...
};
```

### Debug Failing Test

```bash
# Run with verbose output
npm run test:verbose

# Run specific test
npm test -- --testNamePattern="should return"
```

## 🔍 Test Results Interpretation

### ✅ PASS
- Contracts validated
- Adapter can start
- Safe to deploy

### ❌ FAIL
- **STOP DO NOT START ADAPTER**
- API contract violated
- Fix required before deployment
- Check Z3 version/release notes

## 📦 Dependencies

```json
{
  "jest": "^29.7.0",
  "ts-jest": "^29.1.1",
  "zod": "^3.22.4",
  "typescript": "^5.3.3"
}
```

## 🔗 Related Files

- **Project Constitution**: `../../../CLAUDE.md`
- **Z3 Schemas**: `../../../BubbleLab/apps/bubblelab-api/src/schemas/z3.ts`
- **Canonical Models**: `../../../BubbleLab/integrations/openevolve/schemas/canonical-models.ts`

## 💡 Tips

1. **Before committing**: Run `npm run test:contract`
2. **API changes**: Update tests FIRST, then adapter code
3. **Schema errors**: Check import paths in tsconfig.json
4. **Timezone issues**: Always use `.toISOString()` and verify `Z` suffix

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Update `tsconfig.json` baseUrl/paths |
| Timeout | Increase `testTimeout` in package.json |
| Schema mismatch | Verify Z3 version, update mocks |
| Type errors | Run `npx tsc --noEmit` |

## 📞 Support

- Check `tests/README.md` for detailed docs
- See `tests/INTEGRATION_EXAMPLE.md` for integration help
- Review `../../../CLAUDE.md` for architecture principles

---

**Remember**: These tests are the GATEKEEPER. If they fail, the system is protecting you from data corruption. Fix the contract, don't bypass the tests.
