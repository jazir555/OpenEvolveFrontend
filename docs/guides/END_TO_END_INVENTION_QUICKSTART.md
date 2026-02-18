# End-to-End Invention Planner - Quick Start Guide

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [First Invention Example](#first-invention-example)
4. [Common Pitfalls](#common-pitfalls)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Internet connection (for LLM API calls)

### Step 1: Clone or Navigate to Repository

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Dependencies:**
- `asyncio` - Async/await support
- `pydantic` - Data validation
- `openai` - LLM API access
- `anthropic` - Claude API access
- `python-dotenv` - Environment configuration

### Step 3: Configure Environment Variables

Create a `.env` file in the Frontend directory:

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: LeanAide Configuration (if using math formalization)
LEANAIDE_SERVER_URL=http://localhost:8080
LEANAIDE_ENABLED=true

# Optional: Knowledge Engine Configuration
KNOWLEDGE_ENGINE_ENABLED=true
BEDROCK_KB_ENABLED=false
ELASTICSEARCH_URL=http://localhost:9200

# Optional: CrewAI Delegation
CREWAI_ENABLED=false
CREWAI_URL=http://localhost:9000
```

### Step 4: Verify Installation

```bash
python -c "from end_to_end_invention_planner import EndToEndInventionPlanner; print('Installation successful!')"
```

---

## Configuration

### Basic Configuration

The system uses sensible defaults, but you can customize behavior:

```python
from end_to_end_invention_planner import EndToEndInventionPlanner
from generic_maker_integration import MAKERConfig

# Create custom configuration
config = MAKERConfig(
    enable_voting=True,           # Use multi-agent voting for reliability
    voting_threshold=5,           # First-to-5-ahead voting
    enable_decomposition=True,    # Use MDAP decomposition
    max_generations=50,          # Evolutionary optimization generations
    population_size=30,          # Population size for evolution
    timeout_seconds=300          # Timeout per operation
)

# Initialize planner with custom config
planner = EndToEndInventionPlanner(config=config)
```

### Domain-Specific Configuration

Different technical domains may require different configurations:

```python
# Physics/Materials Science (high precision)
physics_config = MAKERConfig(
    enable_voting=True,
    voting_threshold=7,  # Higher threshold for physics
    max_generations=100,  # More iterations
    population_size=50
)

# Biology/Chemistry (moderate precision)
bio_config = MAKERConfig(
    enable_voting=True,
    voting_threshold=5,
    max_generations=50,
    population_size=30
)

# Engineering (practical focus)
eng_config = MAKERConfig(
    enable_voting=True,
    voting_threshold=3,  # Lower threshold, faster results
    max_generations=30,
    population_size=20
)
```

### LeanAide Integration (Optional)

If you have LeanAide server running for mathematical formalization:

```python
from leanaide_client import LeanAideConfig

lean_config = LeanAideConfig(
    server_url="http://localhost:8080",
    timeout=30,
    max_retries=3
)

# The planner will auto-detect LeanAide availability
# No additional configuration needed if server is running
```

---

## First Invention Example

### Example 1: Simple Chemistry Invention

Create a file `first_invention.py`:

```python
import asyncio
from end_to_end_invention_planner import plan_invention

async def main():
    """Plan your first invention"""
    print("Creating invention plan...")

    # Define your invention goal
    plan = await plan_invention(
        prompt="Create a plan to invent iron oxide magnetic nanoparticles for biomedical applications",
        domain="chemistry",
        constraints=[
            "Must be biocompatible",
            "Particle size 10-15 nm",
            "Must be dispersible in water"
        ],
        available_equipment=[
            "Standard chemistry lab",
            "Furnace",
            "Centrifuge"
        ]
    )

    # Save the complete plan
    document = plan.to_executable_document()

    with open("magnetic_nanoparticles_plan.md", "w") as f:
        f.write(document)

    # Display results
    print("\n" + "="*80)
    print("INVENTION PLAN COMPLETE!")
    print("="*80)
    print(f"\nInvention: {plan.invention_goal.target}")
    print(f"Domain: {plan.invention_goal.domain}")
    print(f"Complexity: {plan.invention_goal.complexity_score:.2f}")
    print(f"\nValidation Summary:")
    print(f"  Confidence: {plan.validation_summary['confidence']:.1%}")
    print(f"  Ready for Execution: {plan.validation_summary['ready_for_execution']}")
    print(f"\nPlan Components:")
    print(f"  Knowledge Sources: {len(plan.knowledge_base)}")
    print(f"  Decomposition Steps: {len(plan.decomposition.get('steps', []))}")
    print(f"  Formalized Math: {len(plan.formalized_math)}")
    print(f"  Error Sources Analyzed: {len(plan.error_sources)}")
    print(f"  Red Team Findings: {len(plan.red_team_findings)}")
    print(f"  Blue Team Fixes: {len(plan.blue_team_fixes)}")
    print(f"  Success Criteria: {len(plan.success_criteria)}")
    print(f"\nOutput saved to: magnetic_nanoparticles_plan.md")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python first_invention.py
```

### Expected Output

The system will:
1. Analyze your prompt (Stage 1)
2. Retrieve relevant knowledge (Stage 2)
3. Decompose into atomic steps (Stage 3)
4. Formalize mathematics in Lean (Stage 4)
5. Validate physics/logic (Stage 5)
6. Analyze all error sources (Stage 6)
7. Run red/blue team testing (Stage 7)
8. Generate bulletproof SOP (Stage 8)
9. Define binary success criteria (Stage 9)

Output file (`magnetic_nanoparticles_plan.md`) contains:
- Complete invention goal analysis
- Scientific knowledge base
- Detailed step-by-step decomposition
- Formalized mathematical relationships (in Lean 4)
- Physics/logic validation results
- Comprehensive error source analysis
- Red team vulnerability findings
- Blue team fix implementations
- Complete turnkey-ready SOP
- Binary success criteria (pass/fail)

### Example 2: Complex Physics Invention

```python
import asyncio
from end_to_end_invention_planner import plan_invention

async def main():
    """Plan a complex physics invention"""
    plan = await plan_invention(
        prompt="""
        Create a plan to invent a high-temperature superconducting wire with:
        - Critical temperature: 77 K or higher
        - Current density: 10^6 A/cm² or higher
        - Wire length: 10 meters
        - Diameter: 1 mm
        - Must be manufacturable with standard lab equipment
        """,
        domain="physics"
    )

    # Check formalized math
    print(f"\nMathematical Relationships Formalized: {len(plan.formalized_math)}")
    for math in plan.formalized_math:
        print(f"\n{math.description}")
        print(f"  Theorem: {math.lean_theorem}")
        print(f"  Confidence: {math.confidence:.1%}")

    # Check critical error sources
    critical_errors = [e for e in plan.error_sources if e.impact == "critical"]
    print(f"\nCritical Error Sources: {len(critical_errors)}")
    for error in critical_errors:
        print(f"\n[CRITICAL] {error.description}")
        print(f"  Probability: {error.probability:.1%}")
        print(f"  Mitigation: {error.mitigation_strategy}")

    # Save plan
    with open("superconductor_plan.md", "w") as f:
        f.write(plan.to_executable_document())

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Biology/Medicine Invention

```python
import asyncio
from end_to_end_invention_planner import plan_invention

async def main():
    """Plan a biotechnology invention"""
    plan = await plan_invention(
        prompt="""
        Create a plan to invent a CRISPR-based gene therapy for:
        - Genetic disease: Duchenne muscular dystrophy
        - Delivery method: Intravenous injection
        - Target tissue: Muscle cells
        - Must be safe for human clinical trials
        """,
        domain="biology"
    )

    # Check safety validation
    print("\nPhysics/Biology Validation:")
    for aspect, validated in plan.physics_validation.items():
        status = "✓ PASS" if validated else "✗ FAIL"
        print(f"  {status}: {aspect}")

    # Check binary success criteria
    print(f"\nBinary Success Criteria: {len(plan.success_criteria)}")
    for i, criterion in enumerate(plan.success_criteria, 1):
        print(f"\n{i}. {criterion.criterion}")
        print(f"   Measurement: {criterion.measurement_method}")
        print(f"   Pass if: {criterion.pass_threshold} {criterion.units}")
        print(f"   Binary: PASS or FAIL (no ambiguity)")

    # Save plan
    with open("crispr_therapy_plan.md", "w") as f:
        f.write(plan.to_executable_document())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Common Pitfalls

### Pitfall 1: Vague Prompts

**Problem:**
```python
# Too vague - will produce low-quality plan
plan = await plan_invention(
    prompt="invent something cool",
    domain="general"
)
```

**Solution:**
```python
# Be specific about goals, requirements, and constraints
plan = await plan_invention(
    prompt="Create a plan to invent a lightweight aluminum alloy with strength-to-weight ratio exceeding titanium-6Al-4V",
    domain="materials_science",
    constraints=[
        "Must use aluminum as base metal",
        "Must exceed titanium specific strength",
        "Manufacturable with standard metallurgy equipment",
        "Cost competitive with titanium alloys"
    ]
)
```

### Pitfall 2: Wrong Domain Selection

**Problem:**
```python
# Wrong domain - will retrieve irrelevant knowledge
plan = await plan_invention(
    prompt="Create a gene therapy for cancer",
    domain="engineering"  # Should be "biology"
)
```

**Solution:**
```python
# Use correct domain for best results
plan = await plan_invention(
    prompt="Create a gene therapy for cancer",
    domain="biology"  # Correct domain
)
```

**Supported Domains:**
- `"physics"` - Physics inventions, materials, devices
- `"chemistry"` - Chemical synthesis, reactions, materials
- `"biology"` - Biotechnology, genetics, medicine
- `"materials_science"` - Alloys, polymers, composites
- `"engineering"` - Mechanical, electrical, software
- `"general"` - Multi-domain or unspecified

### Pitfall 3: Ignoring Error Analysis

**Problem:**
```python
# Not reviewing error sources before execution
plan = await plan_invention(...)
# Just executing the SOP without checking error sources
```

**Solution:**
```python
plan = await plan_invention(...)

# Always review critical error sources first
critical_errors = [e for e in plan.error_sources if e.impact == "critical"]
if critical_errors:
    print(f"WARNING: {len(critical_errors)} critical error sources identified!")
    for error in critical_errors:
        print(f"  - {error.description}")
        print(f"    Probability: {error.probability:.1%}")
        print(f"    Mitigation: {error.mitigation_strategy}")
```

### Pitfall 4: Missing Dependencies

**Problem:**
```python
# LeanAide not available, but expecting full formalization
# (Will fall back to simulated formalization)
```

**Solution:**
```python
# Check capabilities first
from end_to_end_invention_planner import get_invention_planner_capabilities

capabilities = get_invention_planner_capabilities()
if not capabilities["math_formalization"]:
    print("Warning: LeanAide not available - math will be simulated")
    print("Install LeanAide for full formalization support")
```

### Pitfall 5: Not Using Validation Results

**Problem:**
```python
plan = await plan_invention(...)
# Proceeding even if validation failed
```

**Solution:**
```python
plan = await plan_invention(...)

# Always check validation before proceeding
if not plan.validation_summary['ready_for_execution']:
    print("WARNING: Plan not ready for execution!")
    print(f"Confidence: {plan.validation_summary['confidence']:.1%}")
    print(f"\nValidation Issues:")
    for aspect, passed in plan.physics_validation.items():
        if not passed:
            print(f"  ✗ {aspect}")

    # Don't use the plan if validation fails
    print("\nPlease review and address validation issues before execution.")
else:
    print("Plan validated successfully - ready for execution!")
```

---

## Troubleshooting

### Issue 1: Import Errors

**Error:**
```
ImportError: No module named 'end_to_end_invention_planner'
```

**Solution:**
```bash
# Ensure you're in the correct directory
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Check if file exists
ls end_to_end_invention_planner.py

# Install dependencies
pip install -r requirements.txt
```

### Issue 2: LLM API Errors

**Error:**
```
openai.error.AuthenticationError: No API key provided
```

**Solution:**
```bash
# Check .env file exists
cat .env

# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# If not set, add to .env file:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### Issue 3: Timeout Errors

**Error:**
```
asyncio.exceptions.TimeoutError
```

**Solution:**
```python
# Increase timeout in configuration
config = MAKERConfig(
    timeout_seconds=600,  # Increase from default 300
    max_generations=30,   # Reduce generations to speed up
    population_size=20    # Reduce population size
)

planner = EndToEndInventionPlanner(config=config)
```

### Issue 4: LeanAide Connection Errors

**Error:**
```
ConnectionError: LeanAide server not available
```

**Solution:**
```python
# Option 1: Disable LeanAide (math will be simulated)
import os
os.environ['LEANAIDE_ENABLED'] = 'false'

# Option 2: Start LeanAide server
# In another terminal:
cd LeanAide
python leanaide_server.py

# Option 3: Check LeanAide is available before using
try:
    from leanaide_client import LeanAideClient
    print("LeanAide available - full math formalization enabled")
except ImportError:
    print("LeanAide not available - math will be simulated")
```

### Issue 5: Memory Errors

**Error:**
```
MemoryError: Unable to allocate memory
```

**Solution:**
```python
# Reduce computational requirements
config = MAKERConfig(
    enable_voting=True,
    voting_threshold=3,  # Lower threshold
    max_generations=20,  # Fewer generations
    population_size=15,  # Smaller population
    enable_decomposition=False  # Disable decomposition if not critical
)

planner = EndToEndInventionPlanner(config=config)
```

### Issue 6: Slow Performance

**Problem:** Planning takes too long

**Solution:**
```python
# Speed optimization strategies
config = MAKERConfig(
    # Reduce voting threshold
    voting_threshold=3,

    # Reduce evolutionary iterations
    max_generations=20,
    population_size=15,

    # Faster timeout
    timeout_seconds=120,

    # Optional: Disable decomposition for simple inventions
    enable_decomposition=False
)

# Use simpler domain if possible
plan = await plan_invention(
    prompt="your prompt",
    domain="general"  # Faster than specific domains
)
```

---

## FAQ

### Q1: What types of inventions can I plan?

**A:** You can plan inventions in these domains:
- **Physics**: Superconductors, quantum devices, optical systems
- **Chemistry**: Nanoparticles, catalysts, synthesis methods
- **Biology**: Gene therapies, diagnostics, bioassays
- **Materials Science**: Alloys, polymers, composites
- **Engineering**: Mechanical devices, electrical systems, software

### Q2: How long does planning take?

**A:** Typical times:
- Simple invention: 5-15 minutes
- Moderate complexity: 15-30 minutes
- Complex invention: 30-60 minutes
- Very complex (with LeanAide): 60+ minutes

Factors affecting time:
- Domain complexity
- Number of decomposition steps
- Mathematical formalization requirements
- Voting threshold
- Evolutionary optimization generations

### Q3: What's the output format?

**A:** The output is a complete Markdown document containing:
- Invention goal analysis
- Scientific knowledge base
- Detailed decomposition into atomic steps
- Formalized mathematics (Lean 4 theorems and proofs)
- Physics/logic validation results
- Comprehensive error analysis (50+ error sources typical)
- Red team vulnerability findings
- Blue team fix implementations
- Complete turnkey-ready SOP
- Binary success criteria (clear pass/fail)

### Q4: Do I need LeanAide installed?

**A:** No, LeanAide is optional. The system will:
- With LeanAide: Full formal mathematical verification
- Without LeanAide: Simulated formalization (still functional)

For production use, LeanAide is recommended for mathematical inventions.

### Q5: Can I execute the generated SOP directly?

**A:** Yes, if:
1. Validation summary shows `ready_for_execution: true`
2. Confidence score is ≥80%
3. All physics_validation checks show PASS
4. You have access to required equipment and materials

The SOP is designed to be **turnkey-ready** - any qualified lab/engineer can execute it without understanding the underlying science.

### Q6: How accurate are the plans?

**A:** Accuracy depends on:
- **Prompt quality**: Specific, detailed prompts produce better plans
- **Domain knowledge**: Well-established scientific domains are more accurate
- **Validation confidence**: Check the confidence score in validation summary
- **Error coverage**: More error sources analyzed = more robust plan

Typical confidence scores:
- Simple inventions: 85-95%
- Moderate complexity: 75-85%
- Complex/ novel inventions: 60-75%

### Q7: What if the plan fails during execution?

**A:** The system includes:
- **Comprehensive error analysis**: 50+ error sources identified
- **Mitigation strategies**: For each error source
- **Contingency procedures**: Fallback options in the SOP
- **Binary success criteria**: Clear pass/fail to detect failure early

If execution fails:
1. Check which error source caused failure
2. Review mitigation strategy in plan
3. Apply blue team fix for that vulnerability
4. Re-validate with updated parameters

### Q8: Can I customize the planning process?

**A:** Yes, through:
1. **Configuration**: Adjust MAKERConfig parameters
2. **Domain selection**: Choose appropriate technical domain
3. **Constraints**: Add specific requirements
4. **Equipment**: Specify available equipment
5. **Custom evaluators**: Create specialized evaluators for your needs

### Q9: Is this suitable for production use?

**A:** The system is production-ready for:
- Research and development planning
- Prototyping and experimentation
- Process optimization
- Educational purposes

For critical applications (medical, safety-critical):
- Review plans with domain experts
- Validate all mathematical formalizations
- Test procedures in controlled environment first
- Use highest voting thresholds (7-10)
- Enable all validation layers

### Q10: How do I get help or report issues?

**A:**
- Check this guide and the main documentation
- Review troubleshooting section above
- Check error logs for detailed messages
- Verify all dependencies are installed correctly
- Ensure API keys are valid and have sufficient credits

---

## Next Steps

1. **Read the Full Guide**: See [END_TO_END_INVENTION_GUIDE.md](END_TO_END_INVENTION_GUIDE.md)
2. **API Reference**: See [END_TO_END_INVENTION_API_REFERENCE.md](END_TO_END_INVENTION_API_REFERENCE.md)
3. **Integration Guide**: See [END_TO_END_INVENTION_INTEGRATIONS.md](END_TO_END_INVENTION_INTEGRATIONS.md)
4. **Advanced Usage**: Explore custom evaluators, domain-specific configurations

---

## Quick Reference

### Basic Usage
```python
from end_to_end_invention_planner import plan_invention

plan = await plan_invention(
    prompt="your invention description",
    domain="physics|chemistry|biology|materials_science|engineering|general"
)

document = plan.to_executable_document()
```

### Check Validation
```python
if plan.validation_summary['ready_for_execution']:
    print("Ready to execute!")
    print(f"Confidence: {plan.validation_summary['confidence']:.1%}")
```

### Review Critical Errors
```python
critical = [e for e in plan.error_sources if e.impact == "critical"]
for error in critical:
    print(f"{error.description} - {error.mitigation_strategy}")
```

---

**Version**: 1.0.0
**Last Updated**: 2025-12-30
**Paper**: arXiv:2511.09030
