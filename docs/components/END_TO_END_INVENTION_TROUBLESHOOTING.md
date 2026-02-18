# End-to-End Invention Planner - Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Issues](#configuration-issues)
3. [Runtime Issues](#runtime-issues)
4. [Performance Issues](#performance-issues)
5. [Integration Issues](#integration-issues)
6. [Quality Issues](#quality-issues)
7. [Debug Mode](#debug-mode)
8. [FAQ](#faq)

---

## Installation Issues

### Issue: ModuleNotFoundError

**Symptom:**
```
ModuleNotFoundError: No module named 'end_to_end_invention_planner'
```

**Causes:**
1. Not in the correct directory
2. Module file missing
3. Python path not set correctly

**Solutions:**

```bash
# 1. Verify you're in the correct directory
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
pwd  # Should show: C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# 2. Check if the file exists
ls end_to_end_invention_planner.py
# Should show: end_to_end_invention_planner.py

# 3. Install dependencies
pip install -r requirements.txt

# 4. Try importing in Python
python -c "from end_to_end_invention_planner import plan_invention; print('OK')"
```

**If file is missing:**
```bash
# Check if it exists elsewhere
find . -name "end_to_end_invention_planner.py"

# Or recreate it (contact administrator)
```

---

### Issue: Dependency Installation Fails

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**

```bash
# 1. Update pip
python -m pip install --upgrade pip

# 2. Install specific packages individually
pip install pydantic
pip install openai
pip install anthropic
pip install python-dotenv

# 3. Use specific versions if needed
pip install pydantic==2.0.0
pip install openai==1.0.0

# 4. Try with conda (if using conda)
conda install pydantic openai anthropic
```

---

### Issue: Python Version Incompatible

**Symptom:**
```
SyntaxError or import errors due to Python version
```

**Solution:**

```bash
# Check Python version
python --version  # Should be 3.9 or higher

# If too old, install newer Python
# Windows: Download from python.org
# Mac: brew install python@3.11
# Linux: sudo apt-get install python3.11

# Verify again
python3.11 --version
```

---

## Configuration Issues

### Issue: API Key Not Found

**Symptom:**
```
openai.error.AuthenticationError: No API key provided
```

**Solutions:**

```bash
# 1. Check if .env file exists
ls .env

# 2. If not, create it
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
EOF

# 3. Verify keys are set
cat .env

# 4. Or set environment variables directly
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# 5. Test with Python
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

---

### Issue: Invalid API Key

**Symptom:**
```
openai.error.InvalidAPIKey: Provided API key is invalid
```

**Solutions:**

```bash
# 1. Verify API key is correct
# Check for typos, extra spaces, etc.

# 2. Generate new API key if needed
# OpenAI: https://platform.openai.com/api-keys
# Anthropic: https://console.anthropic.com/

# 3. Update .env file with new key
# Edit .env:
# OPENAI_API_KEY=sk-new-key-here

# 4. Reload environment
source .env  # Linux/Mac
# Or restart your terminal/shell
```

---

### Issue: API Quota Exceeded

**Symptom:**
```
openai.error.RateLimitError: You exceeded your current quota
```

**Solutions:**

```python
# 1. Check API usage
# Visit OpenAI/Anthropic dashboard to check quota

# 2. Add credit/purchase more quota
# OpenAI: https://platform.openai.com/account/usage
# Anthropic: https://console.anthropic.com/

# 3. Reduce API calls with lower configuration
from generic_maker_integration import MAKERConfig

config = MAKERConfig(
    voting_threshold=3,  # Reduce from 5
    max_generations=20,  # Reduce from 50
    population_size=15   # Reduce from 30
)

# 4. Use cached results when possible
```

---

## Runtime Issues

### Issue: TimeoutError

**Symptom:**
```
asyncio.exceptions.TimeoutError: Operation timed out
```

**Causes:**
1. Invention too complex for current timeout
2. API responses slow
3. Network issues

**Solutions:**

```python
# 1. Increase timeout
from generic_maker_integration import MAKERConfig

config = MAKERConfig(
    timeout_seconds=600  # Increase from 300
)

planner = EndToEndInventionPlanner(config=config)

# 2. Reduce complexity
config = MAKERConfig(
    voting_threshold=3,  # Lower threshold
    max_generations=20,  # Fewer iterations
    population_size=15   # Smaller population
)

# 3. Simplify prompt
# Instead of complex multi-constraint prompt,
# break into simpler sub-problems

# 4. Retry with exponential backoff
import asyncio

async def plan_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await plan_invention(prompt, domain="physics")
        except TimeoutError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                print(f"Timeout, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
```

---

### Issue: Memory Error

**Symptom:**
```
MemoryError: Unable to allocate memory
```

**Solutions:**

```python
# 1. Reduce memory usage
from generic_maker_integration import MAKERConfig

config = MAKERConfig(
    voting_threshold=3,
    max_generations=20,
    population_size=15,
    enable_decomposition=False  # Disable if not critical
)

# 2. Process in smaller chunks
# Instead of one large invention, break into parts

# 3. Clear cache between runs
import gc

planner = EndToEndInventionPlanner()
plan = await planner.plan_invention(...)
# Save and clear
del plan
gc.collect()

# 4. Use 64-bit Python if using 32-bit
python --version  # Should show 64-bit
```

---

### Issue: JSON Parsing Error

**Symptom:**
```
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Causes:**
1. LLM returned invalid JSON
2. Response parsing failed
3. Unexpected output format

**Solutions:**

```python
# 1. Add error handling
import json

async def safe_plan_invention(prompt):
    try:
        plan = await plan_invention(prompt)
        return plan
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Retrying with different configuration...")
        # Retry with higher voting threshold
        config = MAKERConfig(voting_threshold=7)
        planner = EndToEndInventionPlanner(config)
        return await planner.plan_invention(prompt)

# 2. Use fallback parsing
# The system already has fallback parsing built-in
# It will extract JSON from LLM text output

# 3. Report issue if persistent
# If JSON errors persist, it may indicate a bug
```

---

## Performance Issues

### Issue: Planning Takes Too Long

**Symptom:**
Planning takes 30+ minutes for simple inventions

**Solutions:**

```python
# 1. Reduce configuration
config = MAKERConfig(
    voting_threshold=3,     # Lower threshold
    max_generations=20,     # Fewer generations
    population_size=15,     # Smaller population
    timeout_seconds=120     # Lower timeout
)

# 2. Disable non-essential features
config = MAKERConfig(
    enable_voting=True,
    voting_threshold=3,
    enable_decomposition=False  # Skip decomposition
)

# 3. Use simpler domain
plan = await plan_invention(
    prompt="your prompt",
    domain="general"  # Faster than specific domains
)

# 4. Optimize prompt
# Be specific and concise
# Avoid vague requirements

# 5. Check network speed
# Slow API calls increase time
# Consider using faster API endpoint
```

---

### Issue: High API Costs

**Symptom:**
Large API bills from extensive use

**Solutions:**

```python
# 1. Reduce API calls
config = MAKERConfig(
    voting_threshold=3,     # Fewer voting rounds
    max_generations=20,     # Fewer generations
    population_size=15      # Smaller population
)

# 2. Use cheaper models
# Configure in environment variables or LLM provider settings

# 3. Cache results
import pickle

# Save results
plan = await plan_invention(prompt)
with open("cached_plan.pkl", "wb") as f:
    pickle.dump(plan, f)

# Load results later
with open("cached_plan.pkl", "rb") as f:
    plan = pickle.load(f)

# 4. Batch process efficiently
# Plan multiple related inventions together
# to share knowledge retrieval

# 5. Monitor usage
stats = planner.get_statistics()
print(f"Total planning time: {stats['total_planning_time']:.1f}s")
print(f"Inventions planned: {stats['inventions_planned']}")
```

---

## Integration Issues

### Issue: LeanAide Not Available

**Symptom:**
```
Warning: LeanAide not available - math formalization will be simulated
```

**Explanation:**
This is a warning, not an error. The system will function but with simulated rather than formal math verification.

**Solutions:**

```python
# Option 1: Ignore and use simulation
# System will work, just without formal proofs

# Option 2: Install LeanAide
cd LeanAide
pip install -r requirements.txt
python leanaide_server.py

# Option 3: Check installation
try:
    from leanaide_client import LeanAideClient
    print("LeanAide installed correctly")
except ImportError as e:
    print(f"LeanAide not available: {e}")
    print("Install from: https://github.com/leanaide/leanaide")

# Option 4: Use without LeanAide
# Set environment variable
import os
os.environ['LEANAIDE_ENABLED'] = 'false'
```

---

### Issue: Knowledge Engine Not Connected

**Symptom:**
```
Warning: Knowledge Engine not available - using generic knowledge retrieval
```

**Solutions:**

```python
# Option 1: Use generic retrieval (default)
# System works with built-in knowledge

# Option 2: Configure Bedrock KB
# AWS Bedrock Knowledge Base
import boto3

bedrock_client = boto3.client('bedrock', region_name='us-east-1')
# Configure in .env:
# BEDROCK_ENABLED=true
# BEDROCK_REGION=us-east-1

# Option 3: Configure Elasticsearch
# Start Elasticsearch
docker run -d -p 9200:9200 elasticsearch:8.0.0

# Configure in .env:
# ELASTICSEARCH_URL=http://localhost:9200
# ELASTICSEARCH_ENABLED=true

# Option 4: Use without knowledge engine
# Set environment
import os
os.environ['KNOWLEDGE_ENGINE_ENABLED'] = 'false'
```

---

### Issue: CrewAI Connection Failed

**Symptom:**
```
ConnectionError: Unable to connect to CrewAI server
```

**Solutions:**

```python
# Option 1: Disable CrewAI (default)
# System works without it

# Option 2: Start CrewAI server
cd crewai
python crewai_server.py

# Option 3: Check connection
import aiohttp

async def check_crewai(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url + "/health") as resp:
                return resp.status == 200
    except:
        return False

# Option 4: Configure in .env
# CREWAI_ENABLED=false  # Disable
# CREWAI_URL=http://localhost:9000  # Set URL
```

---

## Quality Issues

### Issue: Low Confidence Score

**Symptom:**
Plan confidence < 80%

**Solutions:**

```python
# 1. Check validation summary
plan = await plan_invention(prompt)
print(f"Confidence: {plan.validation_summary['confidence']:.1%}")

# 2. Identify failing validations
for aspect, passed in plan.physics_validation.items():
    if not passed:
        print(f"✗ {aspect} failed")

# 3. Review critical error sources
critical = [e for e in plan.error_sources if e.impact == "critical"]
print(f"Critical errors: {len(critical)}")

# 4. Increase voting threshold
config = MAKERConfig(voting_threshold=7)
planner = EndToEndInventionPlanner(config)
plan = await planner.plan_invention(prompt, domain=domain)

# 5. Add more specific constraints
plan = await plan_invention(
    prompt,
    domain=domain,
    constraints=["specific constraint 1", "specific constraint 2"]
)

# 6. Consider plan not ready
if not plan.validation_summary['ready_for_execution']:
    print("Plan not ready - review before use")
```

---

### Issue: Vague or Generic Plan

**Symptom:**
Plan lacks specificity, has generic steps

**Solutions:**

```python
# 1. Improve prompt specificity
# Instead of:
plan = await plan_invention("invent something")

# Use:
plan = await plan_invention(
    prompt="Create a plan to invent iron oxide magnetic nanoparticles with size 10-15 nm for biomedical MRI contrast agents",
    domain="chemistry",
    constraints=[
        "Biocompatible coating required",
        "Hydrodynamic diameter < 50 nm",
        "Magnetization > 60 emu/g"
    ]
)

# 2. Add domain specification
plan = await plan_invention(prompt, domain="chemistry")

# 3. Include equipment constraints
plan = await plan_invention(
    prompt,
    domain=domain,
    available_equipment=["Furnace up to 1000°C", "Centrifuge 10000 rpm"]
)

# 4. Use higher voting threshold
config = MAKERConfig(voting_threshold=7)
```

---

### Issue: Missing Error Analysis

**Symptom:**
Few error sources identified (< 20)

**Solutions:**

```python
# 1. Increase thoroughness
config = MAKERConfig(
    voting_threshold=7,
    max_generations=100,
    population_size=50
)

# 2. Manually request error analysis
# The system should automatically do this,
# but verify it's running

# 3. Check for parsing errors
# If error sources weren't parsed correctly

# 4. Report issue if persistent
# Should typically find 50+ error sources
```

---

## Debug Mode

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or for specific module
logger = logging.getLogger('end_to_end_invention_planner')
logger.setLevel(logging.DEBUG)
```

### Trace Execution

```python
import asyncio

async def debug_plan_invention(prompt):
    """Plan with detailed tracing"""
    print(f"Starting planning for: {prompt[:100]}")

    planner = EndToEndInventionPlanner()

    # Track each stage
    import time
    stage_times = {}

    # The planner logs each stage, monitor logs
    plan = await planner.plan_invention(prompt, domain="physics")

    # Check results
    print(f"\nPlanning complete!")
    print(f"Knowledge sources: {len(plan.knowledge_base)}")
    print(f"Decomposition steps: {len(plan.decomposition.get('steps', []))}")
    print(f"Math formalized: {len(plan.formalized_math)}")
    print(f"Error sources: {len(plan.error_sources)}")
    print(f"Red team findings: {len(plan.red_team_findings)}")
    print(f"Blue team fixes: {len(plan.blue_team_fixes)}")
    print(f"Success criteria: {len(plan.success_criteria)}")
    print(f"Confidence: {plan.validation_summary['confidence']:.1%}")

    return plan
```

### Export Debug Information

```python
def export_debug_info(plan, filename="debug_info.json"):
    """Export plan details for debugging"""
    import json

    debug_info = {
        "invention_goal": {
            "target": plan.invention_goal.target,
            "domain": plan.invention_goal.domain,
            "complexity": plan.invention_goal.complexity_score
        },
        "knowledge_base_count": len(plan.knowledge_base),
        "decomposition_steps": len(plan.decomposition.get('steps', [])),
        "formalized_math_count": len(plan.formalized_math),
        "error_sources_count": len(plan.error_sources),
        "red_team_findings_count": len(plan.red_team_findings),
        "blue_team_fixes_count": len(plan.blue_team_fixes),
        "success_criteria_count": len(plan.success_criteria),
        "validation_summary": plan.validation_summary,
        "physics_validation": plan.physics_validation
    }

    with open(filename, 'w') as f:
        json.dump(debug_info, f, indent=2)

    print(f"Debug info exported to {filename}")

# Usage
plan = await plan_invention("your prompt")
export_debug_info(plan)
```

---

## FAQ

### Q: How do I know if the system is working correctly?

**A:**
```python
# Run this test
import asyncio
from end_to_end_invention_planner import plan_invention

async def test_system():
    try:
        plan = await plan_invention(
            prompt="Create a plan to invent iron oxide nanoparticles",
            domain="chemistry"
        )

        # Check basics
        assert plan.invention_goal is not None
        assert len(plan.knowledge_base) > 0
        assert len(plan.decomposition.get('steps', [])) > 0
        assert len(plan.error_sources) > 0
        assert len(plan.success_criteria) > 0

        print("✓ System working correctly!")
        print(f"  Confidence: {plan.validation_summary['confidence']:.1%}")

    except Exception as e:
        print(f"✗ System error: {e}")

asyncio.run(test_system())
```

### Q: What should I do if planning fails?

**A:**
1. Check error message
2. Verify API keys are valid
3. Check internet connection
4. Try simpler prompt
5. Reduce configuration complexity
6. Check system logs

### Q: Can I use the system without API keys?

**A:**
No, the system requires LLM API keys (OpenAI or Anthropic) for core functionality. The system uses these APIs for:
- Prompt analysis
- Knowledge retrieval
- Decomposition
- Error analysis
- Red/blue team testing
- SOP generation

### Q: How do I improve plan quality?

**A:**
1. Use specific, detailed prompts
2. Choose appropriate domain
3. Add explicit constraints
4. Increase voting threshold (7-10)
5. Enable all optional integrations
6. Review and iterate on results

### Q: Is the output guaranteed to be correct?

**A:**
No system is perfect. The output has:
- Confidence score (check validation_summary)
- Multiple validation layers
- Red/blue team testing
- But human review is still recommended for critical applications

### Q: How do I report bugs or issues?

**A:**
1. Gather debug information (see Debug Mode above)
2. Document the error (full error message, stack trace)
3. Note your configuration (Python version, OS, etc.)
4. Check if issue is known
5. Report with complete information

### Q: Can I use the system commercially?

**A:**
Yes, but:
- Review licensing terms
- Ensure API usage compliance
- Validate results for your use case
- Consider liability for critical applications

---

## Getting Help

### Resources

1. **Documentation**:
   - [END_TO_END_INVENTION_GUIDE.md](END_TO_END_INVENTION_GUIDE.md)
   - [END_TO_END_INVENTION_QUICKSTART.md](END_TO_END_INVENTION_QUICKSTART.md)
   - [END_TO_END_INVENTION_API_REFERENCE.md](END_TO_END_INVENTION_API_REFERENCE.md)
   - [END_TO_END_INVENTION_INTEGRATIONS.md](END_TO_END_INVENTION_INTEGRATIONS.md)

2. **System Logs**:
   - Enable debug logging (see above)
   - Check console output
   - Review error messages

3. **Validation**:
   - Check validation_summary
   - Review physics_validation
   - Inspect error_sources

### Contact

If issues persist:
1. Gather all debug information
2. Document steps to reproduce
3. Include system configuration
4. Contact support with complete details

---

**Version**: 1.0.0
**Last Updated**: 2025-12-30
**Paper**: arXiv:2511.09030
