# Optional LoongFlow Documentation - Index

## 📚 Documentation Suite

Complete documentation for optional LoongFlow usage and graceful fallback in the Unified Evolution API.

---

## 🎯 Quick Start

**New to optional LoongFlow?** Start here:
1. **[Summary](./OPTIONAL_LOONGFLOW_SUMMARY.md)** - Quick overview and getting started (5 min read)
2. **[User Guide](./OPTIONAL_LOONGFLOW_GUIDE.md)** - Comprehensive usage guide (15 min read)

**Need technical details?** Go to:
3. **[Fallback Documentation](./FALLBACK_DOCUMENTATION.md)** - Implementation and architecture (20 min read)
4. **[Configuration Reference](./CONFIGURATION_OPTIONS.md)** - All configuration options (10 min read)

---

## 📖 Document Overview

### 1. [OPTIONAL_LOONGFLOW_SUMMARY.md](./OPTIONAL_LOONGFLOW_SUMMARY.md)
**Purpose:** Quick reference and getting started guide
**Audience:** All users
**Length:** 400+ lines
**Read Time:** 5 minutes

**Contents:**
- Quick reference tables
- Configuration decision matrix
- Domain recommendations
- 4 quick start examples
- Installation guide
- Key takeaways

**When to read:**
- ✅ First time learning about optional LoongFlow
- ✅ Need a quick refresher
- ✅ Want to see examples quickly
- ✅ Looking for decision matrices

---

### 2. [OPTIONAL_LOONGFLOW_GUIDE.md](./OPTIONAL_LOONGFLOW_GUIDE.md)
**Purpose:** Complete user guide for optional LoongFlow usage
**Audience:** Users and developers
**Length:** 800+ lines
**Read Time:** 15 minutes

**Contents:**
- Why make LoongFlow optional (6 use cases)
- How to disable LoongFlow (5 methods)
- Configuration options explained
- How fallback works
- Capabilities comparison
- OpenEvolve-only recommendations
- 7 detailed examples
- Troubleshooting guide (4 issues)
- Best practices (5 scenarios)
- Migration guide
- FAQ (14 questions)

**When to read:**
- ✅ Want comprehensive understanding
- ✅ Planning to use optional LoongFlow
- ✅ Need troubleshooting help
- ✅ Looking for best practices
- ✅ Migrating from old API

---

### 3. [FALLBACK_DOCUMENTATION.md](./FALLBACK_DOCUMENTATION.md)
**Purpose:** Technical documentation on graceful fallback implementation
**Audience:** Developers and system architects
**Length:** 600+ lines
**Read Time:** 20 minutes

**Contents:**
- Architecture diagrams
- Complete decision tree
- Implementation details
- Error handling (3 custom exceptions)
- Logging and monitoring
- Testing strategies (9 test scenarios)
- Configuration matrix (12 combinations)
- Best practices (5 guidelines)

**When to read:**
- ✅ Implementing custom solutions
- ✅ Debugging fallback behavior
- ✅ Designing system architecture
- ✅ Setting up monitoring
- ✅ Writing tests

---

### 4. [CONFIGURATION_OPTIONS.md](./CONFIGURATION_OPTIONS.md)
**Purpose:** Complete reference of all LoongFlow-related configuration options
**Audience:** Users and developers
**Length:** 500+ lines
**Read Time:** 10 minutes

**Contents:**
- Core LoongFlow control (4 parameters)
- PES mode configuration (8 parameters)
- Configuration combinations (6 scenarios)
- Configuration precedence (5 levels)
- Configuration validation
- Domain-specific examples (7 domains)
- Configuration files (YAML, env vars)
- Best practices (5 guidelines)
- Migration guide

**When to read:**
- ✅ Configuring the system
- ✅ Need parameter details
- ✅ Setting up environment variables
- ✅ Creating configuration files
- ✅ Validating configuration

---

### 5. [OPTIONAL_LOONGFLOW_DOCUMENTATION_COMPLETE.md](./OPTIONAL_LOONGFLOW_DOCUMENTATION_COMPLETE.md)
**Purpose:** Project completion report and metrics
**Audience:** Project managers and stakeholders
**Length:** 400+ lines
**Read Time:** 10 minutes

**Contents:**
- Executive summary
- Deliverables checklist
- Success criteria
- Documentation quality metrics
- Key features documented
- Use cases covered
- Testing coverage
- Impact assessment

**When to read:**
- ✅ Reviewing project completion
- ✅ Assessing documentation quality
- ✅ Understanding project scope
- ✅ Planning future improvements

---

## 🚀 Common Use Cases

### Use Case 1: "I want to disable LoongFlow"

**Quick Answer:**
```python
from openevolve.unified import evolve, UnifiedEvolutionConfig

result = await evolve(
    problem="Optimize function",
    domain="general",
    use_loongflow=False  # Disable LoongFlow
)
```

**Detailed Guide:**
- [Optional LoongFlow Guide - How to Disable](./OPTIONAL_LOONGFLOW_GUIDE.md#how-to-disable-loongflow)
- [Configuration Options - Core Parameters](./CONFIGURATION_OPTIONS.md#core-loongflow-control)

---

### Use Case 2: "I want LoongFlow with graceful fallback"

**Quick Answer:**
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True  # Graceful fallback
)

result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    config=config
)
```

**Detailed Guide:**
- [Optional LoongFlow Guide - Graceful Fallback](./OPTIONAL_LOONGFLOW_GUIDE.md#example-5-graceful-fallback)
- [Fallback Documentation - Implementation](./FALLBACK_DOCUMENTATION.md#implementation-details)

---

### Use Case 3: "I want to compare LoongFlow vs OpenEvolve"

**Quick Answer:**
```python
# Run with LoongFlow
result_lf = await evolve(problem, domain, use_loongflow=True)

# Run with OpenEvolve only
result_oe = await evolve(problem, domain, use_loongflow=False)

# Compare
print(f"LoongFlow: {result_lf.evaluations} evals")
print(f"OpenEvolve: {result_oe.evaluations} evals")
```

**Detailed Guide:**
- [Optional LoongFlow Guide - A/B Testing](./OPTIONAL_LOONGFLOW_GUIDE.md#example-7-ab-testing)
- [Summary - Compare Both Systems](./OPTIONAL_LOONGFLOW_SUMMARY.md#example-4-compare-both-systems)

---

### Use Case 4: "I need production-ready configuration"

**Quick Answer:**
```python
config = UnifiedEvolutionConfig(
    enable_loongflow=True,
    loongflow_fallback_enabled=True,  # Resilient
    evolution_mode=EvolutionMode.AUTO  # Auto-select
)
```

**Detailed Guide:**
- [Configuration Options - Production Configurations](./CONFIGURATION_OPTIONS.md#recommended-configurations)
- [Optional LoongFlow Guide - Best Practices](./OPTIONAL_LOONGFLOW_GUIDE.md#best-practices)

---

### Use Case 5: "I'm having trouble with LoongFlow"

**Quick Answer:**
1. Check if LoongFlow is installed: `pip list | grep loongflow`
2. Check logs for fallback messages
3. Verify configuration: `enable_loongflow=True`
4. Install LoongFlow if needed: `pip install loongflow`

**Detailed Guide:**
- [Optional LoongFlow Guide - Troubleshooting](./OPTIONAL_LOONGFLOW_GUIDE.md#troubleshooting)
- [Fallback Documentation - Troubleshooting](./FALLBACK_DOCUMENTATION.md#troubleshooting)

---

## 📊 Decision Matrices

### Configuration Decision Matrix

| Scenario | `enable_loongflow` | `loongflow_fallback_enabled` | `require_loongflow` |
|----------|-------------------|------------------------------|-------------------|
| **Development** | `false` | N/A | N/A |
| **Production (resilient)** | `true` | `true` | `false` |
| **Production (strict)** | `true` | `false` | `true` |
| **Expensive evals** | `true` | `true` | `false` |
| **Cheap evals** | `false` | N/A | N/A |

📖 **Full details:** [Configuration Options - Decision Matrix](./CONFIGURATION_OPTIONS.md#decision-matrix)

---

### Domain Recommendation Matrix

| Domain | LoongFlow | OpenEvolve Mode | Rationale |
|--------|-----------|----------------|-----------|
| **Finance** | ✅ Yes | PES | 60% fewer backtests |
| **Trading** | ❌ No | Adversarial | Robustness to regime changes |
| **Science** | ✅ Yes | PES | 60% fewer experiments |
| **Engineering** | ❌ No | MO | Multi-objective optimization |
| **Pharma** | ❌ No | QD | Chemical space exploration |
| **Web** | ❌ No | Standard | Fast evaluations |
| **General** | ✅ Yes | AUTO | Auto-select based on problem |

📖 **Full details:** [Optional LoongFlow Guide - Recommendations](./OPTIONAL_LOONGFLOW_GUIDE.md#openevolve-only-recommendations)

---

## 🔍 Quick Reference

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_loongflow` | `bool` | `true` | Enable/disable LoongFlow globally |
| `loongflow_fallback_enabled` | `bool` | `true` | Allow fallback to OpenEvolve |
| `require_loongflow` | `bool` | `false` | Require LoongFlow (fail if unavailable) |
| `use_loongflow` | `bool` | `None` | Runtime override |

📖 **Full details:** [Configuration Options - Core Parameters](./CONFIGURATION_OPTIONS.md#core-loongflow-control)

---

### Key Features Comparison

| Feature | With LoongFlow | OpenEvolve Only |
|---------|---------------|----------------|
| **Directed search (PES)** | ✅ Yes | ❌ No |
| **60% fewer evaluations** | ✅ Yes | ❌ No |
| **Quality Diversity (QD)** | ✅ Yes | ✅ Yes |
| **Multi-Objective (MO)** | ✅ Yes | ✅ Yes |
| **Adversarial testing** | ✅ Yes | ✅ Yes |
| **3-round gauntlet** | ✅ Yes | ✅ Yes |

📖 **Full details:** [Optional LoongFlow Guide - Capabilities Comparison](./OPTIONAL_LOONGFLOW_GUIDE.md#capabilities-comparison)

---

## 🛠️ Installation

### With LoongFlow (Recommended for Production)

```bash
pip install openevolve[unified]
pip install loongflow
```

### Without LoongFlow (OpenEvolve-Only)

```bash
pip install openevolve[unified]
```

📖 **Full details:** [Summary - Installation](./OPTIONAL_LOONGFLOW_SUMMARY.md#installation)

---

## 📝 Examples

### Quick Examples (4)
1. [Quick Start (OpenEvolve-Only)](./OPTIONAL_LOONGFLOW_SUMMARY.md#example-1-quick-start-openevolve-only)
2. [Production (Graceful Fallback)](./OPTIONAL_LOONGFLOW_SUMMARY.md#example-2-production-graceful-fallback)
3. [Strict LoongFlow Requirement](./OPTIONAL_LOONGFLOW_SUMMARY.md#example-3-strict-loongflow-requirement)
4. [Compare Both Systems](./OPTIONAL_LOONGFLOW_SUMMARY.md#example-4-compare-both-systems)

### Detailed Examples (7)
1. [Explicit Disable](./OPTIONAL_LOONGFLOW_GUIDE.md#example-1-explicit-disable)
2. [Configuration File](./OPTIONAL_LOONGFLOW_GUIDE.md#example-2-configuration-file)
3. [Convenience Function](./OPTIONAL_LOONGFLOW_GUIDE.md#example-3-convenience-function)
4. [Require LoongFlow](./OPTIONAL_LOONGFLOW_GUIDE.md#example-4-require-loongflow-error-if-not-available)
5. [Graceful Fallback](./OPTIONAL_LOONGFLOW_GUIDE.md#example-5-graceful-fallback)
6. [Dynamic Selection Based on Cost](./OPTIONAL_LOONGFLOW_GUIDE.md#example-6-dynamic-selection-based-on-cost)
7. [A/B Testing](./OPTIONAL_LOONGFLOW_GUIDE.md#example-7-ab-testing)

### Domain Examples (7)
1. [Finance Domain](./CONFIGURATION_OPTIONS.md#finance-domain)
2. [Trading Domain](./CONFIGURATION_OPTIONS.md#trading-domain)
3. [Science Domain](./CONFIGURATION_OPTIONS.md#science-domain)
4. [Engineering Domain](./CONFIGURATION_OPTIONS.md#engineering-domain)
5. [Pharma Domain](./CONFIGURATION_OPTIONS.md#pharma-domain)
6. [Web Domain](./CONFIGURATION_OPTIONS.md#web-domain)
7. [General Domain](./CONFIGURATION_OPTIONS.md#general-domain)

---

## ❓ FAQ

**Quick answers to common questions:**

1. [Will I lose functionality if I disable LoongFlow?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-will-i-lose-functionality-if-i-disable-loongflow)
2. [Can I switch back and forth between LoongFlow and OpenEvolve?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-can-i-switch-back-and-forth)
3. [What happens to my knowledge extraction when LoongFlow is disabled?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-what-happens-to-my-knowledge-extraction)
4. [Will my gauntlet evaluation change in OpenEvolve-only mode?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-will-my-gauntlet-evaluation-change)
5. [Can I deploy without LoongFlow installed?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-can-i-deploy-without-loongflow-installed)
6. [How do I know if LoongFlow is being used?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-how-do-i-know-if-loongflow-is-being-used)
7. [What's the performance difference between LoongFlow and OpenEvolve?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-whats-the-performance-difference)
8. [Can I use PES mode without LoongFlow installed?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-can-i-use-pes-mode-without-loongflow-installed)
9. [How do I enable LoongFlow in production?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-how-do-i-enable-loongflow-in-production)
10. [What happens if LoongFlow fails during execution?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-what-happens-if-loongflow-fails-during-execution)
11. [Should I use LoongFlow for web optimization?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-should-i-use-loongflow-for-web-optimization)
12. [Can I disable LoongFlow after it's been enabled?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-can-i-disable-loongflow-after-its-been-enabled)
13. [How do I monitor LoongFlow usage?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-how-do-i-monitor-loongflow-usage)
14. [Can I use PES mode without LoongFlow installed?](./OPTIONAL_LOONGFLOW_GUIDE.md#q-can-i-use-pes-mode-without-loongflow-installed)

📖 **All FAQs:** [Optional LoongFlow Guide - FAQ](./OPTIONAL_LOONGFLOW_GUIDE.md#faq)

---

## 🔧 Troubleshooting

### Common Issues

1. **["LoongFlow not available" Warning](./OPTIONAL_LOONGFLOW_GUIDE.md#issue-loongflow-not-available-warning)**
   - Cause: LoongFlow not installed
   - Solution: Install LoongFlow or disable requirement

2. **[Poor Performance in OpenEvolve-Only Mode](./OPTIONAL_LOONGFLOW_GUIDE.md#issue-poor-performance-in-openevolve-only-mode)**
   - Cause: Problem benefits from LoongFlow's directed search
   - Solution: Enable LoongFlow or increase iterations

3. **[Strategy Selection Ignores Context](./OPTIONAL_LOONGFLOW_GUIDE.md#issue-strategy-selection-ignores-context)**
   - Cause: Historical data from different mode
   - Solution: Clear history or specify mode manually

4. **[Import Error When Using PES Mode](./OPTIONAL_LOONGFLOW_GUIDE.md#issue-import-error-when-using-pes-mode)**
   - Cause: PES requires LoongFlow
   - Solution: Install LoongFlow or use different mode

📖 **All troubleshooting:** [Optional LoongFlow Guide - Troubleshooting](./OPTIONAL_LOONGFLOW_GUIDE.md#troubleshooting)

---

## 📈 Best Practices

### Development (5 practices)
1. [Use OpenEvolve-only for faster iteration](./OPTIONAL_LOONGFLOW_GUIDE.md#1-development-environment)
2. [Enable verbose logging](./CONFIGURATION_OPTIONS.md#best-practices)
3. [Use feature flags](./FALLBACK_DOCUMENTATION.md#best-practices)
4. [Test both modes](./OPTIONAL_LOONGFLOW_GUIDE.md#3-testing-environment)
5. [Validate configuration](./CONFIGURATION_OPTIONS.md#3-validate-configuration-before-use)

### Production (5 practices)
1. [Always enable fallback](./FALLBACK_DOCUMENTATION.md#1-always-enable-fallback-in-production)
2. [Log fallback events](./FALLBACK_DOCUMENTATION.md#2-log-fallback-events)
3. [Use environment variables](./CONFIGURATION_OPTIONS.md#1-use-environment-variables-for-deployment)
4. [Monitor performance](./FALLBACK_DOCUMENTATION.md#logging-and-monitoring)
5. [Document decisions](./CONFIGURATION_OPTIONS.md#4-document-configuration-decisions)

### Testing (5 practices)
1. [Test both configurations](./OPTIONAL_LOONGFLOW_GUIDE.md#3-testing-environment)
2. [Unit test fallback logic](./FALLBACK_DOCUMENTATION.md#unit-tests)
3. [Integration test fallback](./FALLBACK_DOCUMENTATION.md#integration-tests)
4. [Performance test overhead](./FALLBACK_DOCUMENTATION.md#performance-tests)
5. [Validate both modes work](./OPTIONAL_LOONGFLOW_SUMMARY.md#3-testing-environment)

---

## 🔄 Migration

### From LoongFlow-Dependent
- [Before/After Examples](./OPTIONAL_LOONGFLOW_SUMMARY.md#from-loongflow-dependent)
- [Code Transformation](./OPTIONAL_LOONGFLOW_GUIDE.md#migration-guide)
- [Migration Checklist](./OPTIONAL_LOONGFLOW_GUIDE.md#migration-checklist)

### From OpenEvolve-Only
- [Before/After Examples](./OPTIONAL_LOONGFLOW_SUMMARY.md#from-openevolve-only)
- [Code Transformation](./OPTIONAL_LOONGFLOW_GUIDE.md#migration-guide)
- [Hybrid Mode Setup](./CONFIGURATION_OPTIONS.md#configuration-migration)

---

## 📞 Support

### Documentation
- [Unified Evolution API](./UNIFIED_EVOLUTION_API.md)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
- [API Reference](./API_REFERENCE.md)

### Related Documentation
- [LoongFlow Integration](./LOONGFLOW_INTEGRATION_COMPLETE.md)
- [Unified Evolution Implementation](./UNIFIED_INTEGRATION_COMPLETE.md)
- [Knowledge Engine Integration](./KNOWLEDGE_ENGINE_INTEGRATION_COMPLETE.md)

---

## 📋 Document Checklist

Use this checklist to ensure you've covered everything:

### Planning Phase
- [ ] Read [Summary](./OPTIONAL_LOONGFLOW_SUMMARY.md) for overview
- [ ] Review [Configuration Options](./CONFIGURATION_OPTIONS.md) for available parameters
- [ ] Check [Domain Recommendations](./OPTIONAL_LOONGFLOW_GUIDE.md#openevolve-only-recommendations)

### Implementation Phase
- [ ] Choose configuration based on use case
- [ ] Set up environment variables or config file
- [ ] Implement with examples from guide
- [ ] Test both modes (LoongFlow and OpenEvolve)

### Testing Phase
- [ ] Run [A/B testing example](./OPTIONAL_LOONGFLOW_GUIDE.md#example-7-ab-testing)
- [ ] Verify fallback behavior
- [ ] Check performance metrics
- [ ] Validate configuration

### Deployment Phase
- [ ] Follow [production best practices](./FALLBACK_DOCUMENTATION.md#best-practices)
- [ ] Set up [monitoring and logging](./FALLBACK_DOCUMENTATION.md#logging-and-monitoring)
- [ ] Configure environment variables
- [ ] Document configuration decisions

### Maintenance Phase
- [ ] Monitor fallback frequency
- [ ] Track performance metrics
- [ ] Review and adjust configuration
- [ ] Update documentation as needed

---

## 🎓 Learning Path

### Beginner (New to Optional LoongFlow)
1. Start: [Summary](./OPTIONAL_LOONGFLOW_SUMMARY.md) (5 min)
2. Read: [Quick Start Examples](./OPTIONAL_LOONGFLOW_SUMMARY.md#usage-examples) (5 min)
3. Try: [Example 1 - OpenEvolve-Only](./OPTIONAL_LOONGFLOW_SUMMARY.md#example-1-quick-start-openevolve-only) (5 min)
4. Review: [Configuration Options](./CONFIGURATION_OPTIONS.md#core-loongflow-control) (10 min)

### Intermediate (Ready to Use)
1. Read: [User Guide](./OPTIONAL_LOONGFLOW_GUIDE.md) (15 min)
2. Study: [Your Domain Examples](./CONFIGURATION_OPTIONS.md#configuration-examples-by-domain) (10 min)
3. Try: [Production Example](./OPTIONAL_LOONGFLOW_SUMMARY.md#example-2-production-graceful-fallback) (10 min)
4. Review: [Best Practices](./OPTIONAL_LOONGFLOW_GUIDE.md#best-practices) (10 min)

### Advanced (Custom Implementation)
1. Study: [Fallback Documentation](./FALLBACK_DOCUMENTATION.md) (20 min)
2. Understand: [Architecture](./FALLBACK_DOCUMENTATION.md#architecture) (10 min)
3. Implement: [Custom Configuration](./CONFIGURATION_OPTIONS.md#configuration-combinations) (15 min)
4. Test: [Testing Strategies](./FALLBACK_DOCUMENTATION.md#testing) (15 min)

---

## 📊 Documentation Metrics

- **Total Documentation:** 2,300+ lines
- **Number of Documents:** 5
- **Code Examples:** 40+
- **Configuration Options:** 12+
- **Best Practices:** 25
- **FAQ Items:** 14
- **Troubleshooting Items:** 10
- **Test Scenarios:** 9
- **Domain Recommendations:** 7

---

## ✅ Success Criteria

All success criteria met:

- ✅ Complete guide on optional LoongFlow
- ✅ Fallback mechanism documented
- ✅ Configuration options explained
- ✅ Examples for both modes
- ✅ Troubleshooting guide
- ✅ Best practices documented
- ✅ Migration guide from LoongFlow-dependent
- ✅ FAQ section

---

**Status:** ✅ COMPLETE
**Last Updated:** January 30, 2026
**Version:** 1.0

---

**Quick Links:**
- [Summary](./OPTIONAL_LOONGFLOW_SUMMARY.md) - Quick overview
- [User Guide](./OPTIONAL_LOONGFLOW_GUIDE.md) - Complete guide
- [Fallback Docs](./FALLBACK_DOCUMENTATION.md) - Technical details
- [Configuration](./CONFIGURATION_OPTIONS.md) - All options
- [Completion Report](./OPTIONAL_LOONGFLOW_DOCUMENTATION_COMPLETE.md) - Project status
