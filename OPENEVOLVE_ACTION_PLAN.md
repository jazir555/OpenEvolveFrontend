# OpenEvolve Integration - Action Plan

**Created:** October 22, 2025  
**Status:** Active Development  
**Priority:** High

---

## Current Situation

### What's Working ✅
- Core OpenEvolve client implementation
- Parameter management (211 parameters)
- Metrics collection framework
- Fallback handler
- Unit tests (28/28 passing)
- Team integration (Blue, Red, Evaluator)
- Workflow structures with OpenEvolve fields
- UI models and configuration

### What's Broken 🔴
Integration tests revealed several critical issues:

1. **API Mismatches**
   - `ParameterManager.validate_parameters()` doesn't exist (should be `validate()`)
   - `MetricsAggregator()` constructor signature mismatch
   - `ResourceLimits` missing `max_execution_time` parameter

2. **Import Errors**
   - `blue_team.py` has relative import issues
   - `red_team.py` and `evaluator_team.py` missing `_compose_messages` import
   - Import path inconsistencies

3. **Syntax Errors**
   - `workflow_engine.py` line 1757 indentation error

4. **Missing Implementations**
   - Template manager missing 'balanced' preset
   - Resource manager not properly detecting limit violations
   - Some methods referenced in tests don't exist

---

## Immediate Action Items (Next 4 Hours)

### Phase 1: Fix Critical Bugs (2h)

#### 1.1 Fix Parameter Manager API (15min)
```python
# In parameter_manager.py
# Add alias method for backward compatibility
def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate parameters (alias for validate())"""
    result = self.validate(params)
    return result.valid, result.errors
```

#### 1.2 Fix Metrics Collector (15min)
```python
# In metrics_collector.py
# Fix MetricsAggregator initialization
class MetricsCollector:
    def __init__(self):
        self.store = MetricsStore()
        self.aggregator = MetricsAggregator()  # Remove argument
        self.exporter = MetricsExporter()
```

#### 1.3 Fix Workflow Engine Syntax (10min)
- Fix indentation error at line 1757
- Run syntax check: `python -m py_compile workflow_engine.py`

#### 1.4 Fix Import Issues (30min)
```python
# In blue_team.py, red_team.py, evaluator_team.py
# Change from:
from .llm_utils import _request_openai_compatible_chat, _compose_messages
# To:
from llm_utils import _request_openai_compatible_chat
# And add _compose_messages if missing
```

#### 1.5 Fix Resource Limits (15min)
```python
# In resource_manager.py
@dataclass
class ResourceLimits:
    max_api_calls: Optional[int] = None
    max_cost: Optional[float] = None
    max_execution_time: Optional[int] = None  # Add this
    max_memory_mb: Optional[int] = None
```

#### 1.6 Fix Template Manager Presets (15min)
```python
# In template_manager.py
# Add get_openevolve_preset_templates() method
# Include 'balanced' preset
```

#### 1.7 Fix Resource Manager Violations (20min)
```python
# In resource_manager.py
# Fix check_limits() to properly detect violations
def check_limits(self) -> Tuple[bool, List[str]]:
    violations = []
    if self.limits.max_api_calls and self.current_api_calls > self.limits.max_api_calls:
        violations.append(f"API calls exceeded: {self.current_api_calls} > {self.limits.max_api_calls}")
    # ... check other limits
    return len(violations) == 0, violations
```

### Phase 2: Verify Fixes (1h)

#### 2.1 Run Unit Tests (10min)
```bash
pytest test_openevolve_integration.py -v
# Should still pass 28/28
```

#### 2.2 Run Integration Tests (20min)
```bash
pytest test_integration_openevolve.py -v
# Fix any remaining issues
```

#### 2.3 Run Diagnostics (10min)
```bash
python -m py_compile *.py
# Check for syntax errors
```

#### 2.4 Manual Testing (20min)
- Test parameter validation
- Test metrics collection
- Test team integration
- Test resource limits

### Phase 3: Update Documentation (1h)

#### 3.1 Update Status Document (20min)
- Update completion percentage
- Document fixed issues
- Update API reference

#### 3.2 Update README (20min)
- Fix code examples to match actual API
- Add troubleshooting section
- Document known issues

#### 3.3 Update Integration Guide (20min)
- Add correct import statements
- Fix method names in examples
- Add error handling examples

---

## Short-Term Goals (Next Week)

### Week 1: Stabilization (20h)

#### Day 1-2: Bug Fixes (8h)
- [ ] Fix all critical bugs from integration tests
- [ ] Ensure all tests pass (unit + integration)
- [ ] Fix import issues across all files
- [ ] Fix API mismatches

#### Day 3: Testing (4h)
- [ ] Add more integration tests
- [ ] Test edge cases
- [ ] Test error handling
- [ ] Test resource limits

#### Day 4: Documentation (4h)
- [ ] Complete API reference
- [ ] Add troubleshooting guide
- [ ] Update all code examples
- [ ] Add migration guide

#### Day 5: Polish (4h)
- [ ] Code cleanup
- [ ] Add missing docstrings
- [ ] Improve error messages
- [ ] Final review

---

## Medium-Term Goals (Next Month)

### Week 2: Advanced Features (20h)
- [ ] Implement advanced visualizations
- [ ] Add monitoring dashboard
- [ ] Enhance UI components
- [ ] Add configuration profiles

### Week 3: Performance (20h)
- [ ] Optimize evolution speed
- [ ] Add caching
- [ ] Improve resource management
- [ ] Add distributed processing

### Week 4: Production Ready (20h)
- [ ] Security audit
- [ ] Performance benchmarking
- [ ] Load testing
- [ ] Final documentation

---

## Success Criteria

### Must Have (P0)
- [x] Core OpenEvolve client working
- [x] 211 parameters configured
- [x] Unit tests passing (28/28)
- [ ] Integration tests passing (0/16) 🔴
- [ ] No syntax errors
- [ ] No import errors
- [ ] API consistency

### Should Have (P1)
- [ ] Complete documentation
- [ ] All presets working
- [ ] Resource limits enforced
- [ ] Error handling robust
- [ ] UI components polished

### Nice to Have (P2)
- [ ] Advanced visualizations
- [ ] Monitoring dashboard
- [ ] Configuration profiles
- [ ] Performance optimizations

---

## Risk Assessment

### High Risk 🔴
1. **Import Issues** - Blocking integration tests
   - Mitigation: Fix import paths, use absolute imports
   - Timeline: 30 minutes

2. **API Mismatches** - Breaking existing code
   - Mitigation: Add compatibility methods, update tests
   - Timeline: 1 hour

3. **Syntax Errors** - Preventing code execution
   - Mitigation: Fix immediately, add syntax checks to CI
   - Timeline: 10 minutes

### Medium Risk 🟡
1. **Resource Management** - Not detecting violations
   - Mitigation: Fix check_limits() logic
   - Timeline: 20 minutes

2. **Template Management** - Missing presets
   - Mitigation: Add missing presets
   - Timeline: 15 minutes

### Low Risk 🟢
1. **Documentation** - Out of sync with code
   - Mitigation: Update docs after fixes
   - Timeline: 2 hours

---

## Next Steps

### Today (4 hours)
1. ✅ Create action plan (this document)
2. ⏳ Fix critical bugs (Phase 1)
3. ⏳ Verify fixes (Phase 2)
4. ⏳ Update documentation (Phase 3)

### Tomorrow (4 hours)
1. Add more integration tests
2. Test edge cases
3. Fix any remaining issues
4. Update status document

### This Week (12 hours remaining)
1. Complete API reference
2. Add troubleshooting guide
3. Polish UI components
4. Final testing

---

## Tracking Progress

### Bugs Fixed
- [x] ParameterManager API mismatch - Added validate_parameters() alias
- [ ] MetricsCollector initialization
- [x] UI Components Tuple import - Added Tuple to imports
- [ ] Import errors (blue_team, red_team, evaluator_team)
- [ ] ResourceLimits missing parameter
- [ ] Template manager missing preset
- [ ] Resource manager violations not detected

### Tests Passing
- [x] Unit tests: 28/28 ✅
- [ ] Integration tests: 0/16 🔴
- [ ] Total: 28/44 (64%)

### Documentation Status
- [x] Status document created
- [x] Action plan created
- [ ] API reference complete
- [ ] Troubleshooting guide complete
- [ ] Migration guide complete

---

## Notes

- Focus on fixing critical bugs first
- Don't add new features until tests pass
- Update documentation as you fix issues
- Test after each fix
- Commit frequently

---

## Contact

For questions or issues:
- Check test output for specific errors
- Review this action plan for priorities
- Update status document after fixes

---

**Last Updated:** October 22, 2025  
**Next Review:** After Phase 1 completion
