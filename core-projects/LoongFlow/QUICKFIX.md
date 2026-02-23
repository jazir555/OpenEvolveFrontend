# LoongFlow API Server - Quick Fix Guide

**Issue:** Import errors prevent api_server.py from starting
**Fix Time:** 5 minutes
**Difficulty:** Easy

---

## Problem

When running `python api_server.py`, you get:
```
ModuleNotFoundError: No module named 'loongflow.agents'
```

## Root Cause

The import statements (lines 29-43) assume LoongFlow has a different structure than it actually has.

**Incorrect (current):**
```python
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loongflow.agents.general_agent.evaluator import GeneralEvaluator
from loongflow.agents.general_agent.general_evolve_agent import GeneralPESAgent
from loongflow.framework.pes.context import EvolveChainConfig
```

**Actual LoongFlow Structure:**
```
LoongFlow/
├── agents/                    # ← HERE (not in src/loongflow/agents)
│   ├── general_agent/
│   │   ├── evaluator.py
│   │   └── general_evolve_agent.py
├── src/loongflow/
│   ├── framework/pes/context.py
│   └── ...
```

---

## Solution

### Option 1: Comment Out Unused Imports (Phase 1 - Recommended)

Since Phase 1 simulates evolution and doesn't use these imports, just comment them out:

```python
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# LoongFlow imports
# NOTE: Phase 1 uses simulated evolution, so these imports aren't needed yet
# TODO: Uncomment and fix paths for Phase 2 integration
# from loongflow.agents.general_agent.evaluator import GeneralEvaluator
# from loongflow.agents.general_agent.general_evolve_agent import GeneralPESAgent
# from loongflow.framework.pes.context import EvolveChainConfig
```

**Result:** Server starts successfully, runs simulated evolutions.

### Option 2: Fix Import Paths (Phase 2 - For Real Integration)

If you want to prepare for Phase 2 integration:

```python
# Add both project root AND src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import from correct locations
from agents.general_agent.evaluator import GeneralEvaluator
from agents.general_agent.general_evolve_agent import GeneralPESAgent
from loongflow.framework.pes.context import EvolveChainConfig
```

**Result:** Server starts, imports available for Phase 2 integration work.

---

## Step-by-Step Fix

1. **Open the file:**
   ```bash
   cd core-projects/LoongFlow
   nano api_server.py  # or use your editor
   ```

2. **Find lines 29-43** (the import section)

3. **Replace with Option 1 or Option 2** (above)

4. **Save and exit**

5. **Test the fix:**
   ```bash
   export LOONGFLOW_LLM_API_KEY="sk-test-key-for-validation"
   python api_server.py
   ```

6. **Expected output:**
   ```
   INFO:     Started server process [12345]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

7. **Test health endpoint** (in another terminal):
   ```bash
   curl http://localhost:8000/health
   ```

   **Expected response:**
   ```json
   {
     "status": "healthy",
     "service": "loongflow-api",
     "version": "1.0.0",
     "timestamp": "2026-02-22T17:30:00.000Z"
   }
   ```

---

## Which Option Should I Choose?

### Choose Option 1 (Comment Out) If:
- ✅ You want to test the API server quickly
- ✅ You're building the adapter/federation layer
- ✅ You don't need real evolution yet
- ✅ You want to avoid potential import issues

**This is the RECOMMENDED approach for Task #39 completion.**

### Choose Option 2 (Fix Paths) If:
- ✅ You're starting Phase 2 integration work
- ✅ You need to access GeneralPESAgent
- ✅ You're implementing real evolution execution
- ✅ You're comfortable debugging import errors

---

## After the Fix

Once the server starts successfully:

1. **Test the API:**
   ```bash
   # Start evolution
   curl -X POST http://localhost:8000/api/v1/evolve \
     -H "Content-Type: application/json" \
     -d '{
       "name": "test-evolution",
       "task": "Optimize sorting algorithm",
       "max_generations": 5
     }'

   # Check status (use the evolution_id from response)
   curl http://localhost:8000/api/v1/status/evo_XXXXX

   # Get solution (when complete)
   curl http://localhost:8000/api/v1/solutions/evo_XXXXX
   ```

2. **Test the adapter:**
   ```bash
   cd glue/adapters/loongflow-adapter
   npm test
   ```

3. **Run probe scripts:**
   ```bash
   cd glue/adapters/loongflow-adapter/probes
   ./check_api.sh
   ```

---

## Common Issues

### Issue: "LOONGFLOW_LLM_API_KEY is required"
**Fix:** Set the environment variable:
```bash
export LOONGFLOW_LLM_API_KEY="sk-test-key-for-validation"
```

### Issue: Port 8000 already in use
**Fix:** Use a different port:
```bash
export LOONGFLOW_API_PORT=8001
python api_server.py
```

### Issue: ModuleNotFoundError for other modules
**Fix:** Make sure LoongFlow is installed:
```bash
cd core-projects/LoongFlow
pip install -e .
```

---

## Summary

**Option 1 (Comment Out):** 2 minutes, safe, works immediately ✅
**Option 2 (Fix Paths):** 5 minutes, prepares for Phase 2, may have follow-on issues ⚠️

**Recommendation:** Use Option 1 for now, revisit imports when starting Phase 2 integration.

---

*Last Updated: 2026-02-22*
*Related: STATUS.md, API.md*
