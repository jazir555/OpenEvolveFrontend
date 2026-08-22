# WORKFLOW TEST SUMMARY - STATISTICS & ANALYSIS

Generated: 2026-01-19T02:03:30.285350

## OVERALL STATISTICS

Total Bubbles Audited: 90

### WORKFLOW BUBBLES

- Total: 17
- With Tests: 0 (0.0%)
- Total Issues: 288
- Avg Complexity: 95.2

### SERVICE BUBBLES

- Total: 40
- With Tests: 20 (50.0%)
- Total Issues: 647
- Avg Complexity: 110.7

### TOOL BUBBLES

- Total: 33
- With Tests: 14 (42.4%)
- Total Issues: 338
- Avg Complexity: 128.8

## SECURITY ISSUES SUMMARY

Total Security Issues: 115

### By Severity

- High: 38
- Medium: 33
- Low: 44

### By Category

- logging: 44
- rate_limiting: 32
- timeout: 31
- code_injection: 6
- env_validation: 1
- error_handling: 1

## CODE QUALITY ISSUES SUMMARY

Total Quality Issues: 1158

- code_quality: 892
- error_handling: 222
- resource_management: 44

## TOP PROBLEMATIC FILES

| File | Type | Security Issues | Quality Issues | Total |
|------|------|-----------------|----------------|-------|
| file-processor-tool | tool | 2 | 80 | 82 |
| apify-bubble | service | 1 | 40 | 41 |
| backup-restore.workflow | workflow | 3 | 34 | 37 |
| google-sheets-bubble | service | 0 | 37 | 37 |
| pdf-form-operations.workflow | workflow | 0 | 36 | 36 |
| parse-document.workflow | workflow | 1 | 33 | 34 |
| http-fix-validation | service | 3 | 28 | 31 |
| gmail-bubble | service | 3 | 26 | 29 |
| generate-document.workflow | workflow | 0 | 27 | 27 |
| pdf-ocr.workflow | workflow | 0 | 27 | 27 |

## RECOMMENDATIONS

### HIGH PRIORITY: 38 High Issues

- Missing input validation
- Missing timeout handling
- Missing rate limiting

### TESTING: 56 Bubbles Without Tests

Create comprehensive test suites for all bubbles:
- Environment validation tests
- Authentication tests
- Rate limiting tests
- Input validation tests
- Error handling tests
- Integration tests

