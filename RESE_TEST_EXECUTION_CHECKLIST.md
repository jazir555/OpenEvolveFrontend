# RESE Final Integration Test - Execution Checklist

**Purpose:** Step-by-step guide to execute the RESE Framework Final Integration Test

---

## Pre-Execution Checklist

### 1. Environment Configuration

- [ ] Copy `.env.example` to `.env`
  ```bash
  cp .env.example .env
  ```

- [ ] Edit `.env` and set the following variables:

  **Required Variables:**
  ```bash
  # OpenAI API Key
  OPENAI_API_KEY=sk-proj-...

  # RESE Paths
  RESE_DATA_DIR=./data
  RESE_LOGS_DIR=./logs
  RESE_OUTPUT_DIR=./output

  # Phase Models
  PHASE1_MODEL=gpt-4
  PHASE2_MODEL=gpt-4
  PHASE3_MODEL=gpt-4
  PHASE4_MODEL=gpt-4
  ```

  **Optional Variables:**
  ```bash
  ENVIRONMENT=development
  OPENEVOLVE_LOG_LEVEL=INFO
  ```

### 2. Directory Setup

- [ ] Create required directories:
  ```bash
  mkdir -p data logs output
  ```

- [ ] Verify directories exist:
  ```bash
  ls -la data logs output
  ```

### 3. Verify Phase Executors

- [ ] Check Phase I executor exists:
  ```bash
  ls glue/adapters/rese-phase1/src/phase1_executor.py
  ```

- [ ] Check Phase II executor exists:
  ```bash
  ls glue/adapters/rese-phase2/src/phase2_executor.py
  ```

- [ ] Check Phase III executor exists:
  ```bash
  ls glue/adapters/rese-phase3/src/phase3_executor.py
  ```

- [ ] Check Phase IV executor exists:
  ```bash
  ls glue/adapters/rese-phase4/src/phase4_executor.py
  ```

### 4. Verify Test Script

- [ ] Check test script exists:
  ```bash
  ls glue/tests/test_rese_final_integration.py
  ```

- [ ] Verify test script is executable:
  ```bash
  python -m py_compile glue/tests/test_rese_final_integration.py
  ```

### 5. Environment Validation

- [ ] Check Python version (3.8+ required):
  ```bash
  python --version
  ```

- [ ] Verify required packages installed:
  ```bash
  pip list | grep -E "openai|pydantic"
  ```

- [ ] Test environment variable loading:
  ```bash
  python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
  ```

---

## Execution Steps

### Step 1: Navigate to Project Root

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
```

**Verify:**
```bash
pwd
# Should output: /c/Users/mmeadow/Documents/OpenEvolve/Frontend
```

### Step 2: Load Environment Variables

**Option A: Using python-dotenv**
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); print('Environment loaded')"
```

**Option B: Manual export**
```bash
export $(cat .env | xargs)
```

**Verify:**
```bash
python -c "import os; print('OPENAI_API_KEY:', os.getenv('OPENAI_API_KEY')[:10] + '...')"
```

### Step 3: Run the Test

```bash
python glue/tests/test_rese_final_integration.py
```

**Expected Output:**
```
================================================================================
RESE FRAMEWORK FINAL INTEGRATION TEST
================================================================================
Test Start Time: 2026-02-04T...
================================================================================

[Step 1] Validating Environment Variables...
  [OK] OPENAI_API_KEY: sk-proj...
  [OK] RESE_DATA_DIR: ./data
  ...

[DOC] Step 2: Initializing Executors...
[OK] Executors initialized

[DOC] Step 3: Executing Test Pipeline...
```

---

## Post-Execution Checklist

### 1. Verify Report Generation

- [ ] Check markdown report was created:
  ```bash
  ls glue/FINAL_INTEGRATION_TEST_REPORT.md
  ```

- [ ] Check JSON results were created:
  ```bash
  ls glue/FINAL_INTEGRATION_TEST_RESULTS.json
  ```

### 2. Review Test Summary

Check console output for:
- [ ] Phases Passed: 4/4
- [ ] Total Duration: < 300 seconds
- [ ] Peak Memory: < 2048 MB
- [ ] Production Ready: [OK] YES

### 3. Examine Reports

**Markdown Report:**
- [ ] Open `glue/FINAL_INTEGRATION_TEST_REPORT.md`
- [ ] Review Executive Summary
- [ ] Check Phase-by-Phase Results
- [ ] Review Performance Metrics
- [ ] Check CLAUDE.md Compliance
- [ ] Read Recommendations

**JSON Results:**
- [ ] Open `glue/FINAL_INTEGRATION_TEST_RESULTS.json`
- [ ] Verify metadata structure
- [ ] Check metrics accuracy
- [ ] Review phase results

### 4. Validate Results

**Data Flow Validation:**
- [ ] Phase I → Phase II: Valid
- [ ] Phase II → Phase III: Valid
- [ ] Phase III → Phase IV: Valid
- [ ] Final Architecture: Present

**Performance Validation:**
- [ ] Phase I duration < 180s
- [ ] Phase II duration < 180s
- [ ] Phase III duration < 180s
- [ ] Phase IV duration < 180s
- [ ] Total duration < 300s
- [ ] Peak memory < 2048 MB

**Compliance Validation:**
- [ ] Law 1: Air Gap - PASS
- [ ] Law 2: Runtime Truth - PASS
- [ ] Law 3: Untouchable DB - PASS
- [ ] Law 4: Idempotency - NOT TESTED (expected)
- [ ] Law 5: Configuration Explicitness - PASS
- [ ] Law 6: UTC - PASS

---

## Troubleshooting

### Issue: Environment Variables Not Found

**Symptoms:**
```
[X] Missing: OPENAI_API_KEY
[X] Missing: RESE_DATA_DIR
```

**Solution:**
1. Verify `.env` file exists
2. Check variable names in `.env` match exactly
3. Try exporting manually:
   ```bash
   export OPENAI_API_KEY=sk-proj-...
   export RESE_DATA_DIR=./data
   export RESE_LOGS_DIR=./logs
   export RESE_OUTPUT_DIR=./output
   export PHASE1_MODEL=gpt-4
   export PHASE2_MODEL=gpt-4
   export PHASE3_MODEL=gpt-4
   export PHASE4_MODEL=gpt-4
   ```

### Issue: Import Errors

**Symptoms:**
```
ERROR: Cannot import phase executors: ...
```

**Solution:**
1. Verify phase executor files exist
2. Check Python path includes glue layer
3. Try running from different directory:
   ```bash
   cd glue/tests
   python test_rese_final_integration.py
   ```

### Issue: Executor Initialization Failed

**Symptoms:**
```
[X] Failed to create executors: ...
```

**Solution:**
1. Check executor class names match
2. Verify executor modules are valid Python
3. Test individual executors:
   ```bash
   python -c "from adapters.rese-phase1.src.phase1_executor import EpistemicAuditExecutor; print('OK')"
   ```

### Issue: Phase Execution Timeout

**Symptoms:**
```
[X] Phase X failed: timeout
```

**Solution:**
1. Check network connectivity
2. Verify API key is valid
3. Check OpenAI API status
4. Increase timeout in executor config

### Issue: Memory Exceeded

**Symptoms:**
```
Peak Memory: > 2048 MB
[X] Excessive
```

**Solution:**
1. Close other applications
2. Reduce problem complexity
3. Implement data streaming
4. Increase available RAM

---

## Success Criteria

The test is considered **SUCCESSFUL** when:

### Primary Criteria
- ✅ All 4 phases execute successfully
- ✅ No critical errors
- ✅ Data flow validation passes
- ✅ Reports generated correctly

### Secondary Criteria
- ✅ Total execution time < 5 minutes
- ✅ Peak memory < 2GB
- ✅ No circuit breaker trips
- ✅ DLQ is empty

### Compliance Criteria
- ✅ All CLAUDE.md laws pass (except Law 4 which requires multiple runs)
- ✅ Correlation ID consistent
- ✅ All timestamps in UTC

### Output Criteria
- ✅ Markdown report generated
- ✅ JSON results generated
- ✅ Production readiness assessment complete

---

## Test Results Interpretation

### Production Ready: [OK] YES

**Meaning:**
- All phases executed successfully
- No critical failures
- Performance within acceptable bounds
- CLAUDE.md compliant

**Next Steps:**
- Deploy to staging environment
- Run load tests
- Monitor for 24 hours
- Deploy to production

### Production Ready: [X] NO

**Meaning:**
- One or more phases failed
- Critical issues detected
- Performance issues
- Compliance violations

**Next Steps:**
- Review Issues section in report
- Follow Recommendations (prioritized)
- Fix critical issues
- Re-run test

---

## Cleanup

After successful test execution:

### Optional Cleanup

```bash
# Remove test outputs
rm glue/FINAL_INTEGRATION_TEST_REPORT.md
rm glue/FINAL_INTEGRATION_TEST_RESULTS.json

# Clear test data
rm -rf data/*
rm -rf logs/*
rm -rf output/*
```

**Note:** Keep test outputs for debugging and audit purposes

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: RESE Integration Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Configure environment
        run: |
          cp .env.example .env
          echo "OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" >> .env

      - name: Run integration test
        run: |
          python glue/tests/test_rese_final_integration.py

      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: |
            glue/FINAL_INTEGRATION_TEST_REPORT.md
            glue/FINAL_INTEGRATION_TEST_RESULTS.json
```

---

## Appendix: Quick Reference

### Environment Variables Template

```bash
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
RESE_DATA_DIR=./data
RESE_LOGS_DIR=./logs
RESE_OUTPUT_DIR=./output
PHASE1_MODEL=gpt-4
PHASE2_MODEL=gpt-4
PHASE3_MODEL=gpt-4
PHASE4_MODEL=gpt-4
ENVIRONMENT=development
OPENEVOLVE_LOG_LEVEL=INFO
```

### Test Command

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
python glue/tests/test_rese_final_integration.py
```

### Report Locations

- Markdown: `glue/FINAL_INTEGRATION_TEST_REPORT.md`
- JSON: `glue/FINAL_INTEGRATION_TEST_RESULTS.json`

### Expected Test Duration

- Fast: 30-60 seconds (simple problem)
- Normal: 1-3 minutes (moderate problem)
- Slow: 3-5 minutes (complex problem)

### Key Files

- Test: `glue/tests/test_rese_final_integration.py`
- Docs: `glue/FINAL_INTEGRATION_TEST_DELIVERABLE.md`
- Summary: `RESE_FINAL_INTEGRATION_TEST_SUMMARY.md`

---

**Checklist Version:** 1.0
**Last Updated:** 2026-02-04
**Status:** Ready for Use
