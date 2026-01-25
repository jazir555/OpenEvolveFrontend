# Bug Report Template

Use this template when reporting bugs in RESE components.

**Version:** 1.0.0
**Created:** 2025-12-31

---

## Bug Description

<!-- Clear, concise description of the bug -->

**Component:** [Phase I / Phase II / Phase III / Phase IV / Core]
**Severity:** [Critical / High / Medium / Low]

### Expected Behavior

<!-- What should happen -->

### Actual Behavior

<!-- What actually happens -->

### Steps to Reproduce

1.
2.
3.

### Minimal Reproducible Example

```python
# Paste code that reproduces the bug
def test_reproducing_bug():
    ...
```

---

## Environment

- **Python Version:** 3.10.x
- **OS:** [Ubuntu 22.04 / macOS / Windows]
- **RESE Version:** x.x.x
- **Dependencies:**
  ```
  # Paste output of pip list
  ```

---

## Test Case

### Test That Fails

```python
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_bug_reproduction():
    """
    Test that reproduces the bug.

    Issue: [Link to issue]
    """
    # Test code
    ...
```

### Test Output

```
# Paste test output
```

---

## Logs & Error Messages

```
# Paste relevant logs or error messages
```

---

## Screenshots (if applicable)

<!-- Add screenshots to help explain the problem -->

---

## Additional Context

<!-- Add any other context about the problem here -->

### Related Issues

- Issue #xxx
- PR #yyy

### Possible Fix

<!-- If you have a fix in mind, describe it here -->

### Priority Assessment

**Why is this bug important?**
- [ ] Breaks critical functionality
- [ ] Affects many users
- [ ] Has workaround
- [ ] Blocks release
- [ ] Other: ______

---

## Checklist

- [ ] I have searched for similar issues
- [ ] I have provided a minimal reproducible example
- [ ] I have included test output/error messages
- [ ] I have specified the component and severity
- [ ] I have included environment information
- [ ] I have suggested a possible fix (optional)

---

## Severity Guidelines

| Severity | Description | Examples |
|----------|-------------|----------|
| **Critical** | System crash, data loss, security issue | Application crashes, data corruption, security vulnerabilities |
| **High** | Major feature broken, no workaround | Core functionality fails, critical path blocked |
| **Medium** | Feature broken, has workaround | Non-critical bugs, workarounds available |
| **Low** | Minor issue, cosmetic problem | Typos, UI issues, minor annoyances |

---

## Response Time Targets

| Severity | Target Response | Target Resolution |
|----------|----------------|-------------------|
| Critical | 4 hours | 1 day |
| High | 1 day | 3 days |
| Medium | 3 days | 1 week |
| Low | 1 week | 2 weeks |

---

## Contact

**Reporter:** [Your Name]
**Date:** [YYYY-MM-DD]
**Assigned To:** [To be filled]

---

**Template Version:** 1.0.0
**Last Updated:** 2025-12-31
