# Technical Debt Analysis Report

Generated: 2026-01-18 01:44:57

## Summary

- **Files Analyzed:** 110
- **Total Lines:** 73,478
- **Total Issues:** 2,081

### Severity Breakdown

- **HIGH:** 38
- **MEDIUM:** 389
- **LOW:** 1,654

### Top Issue Categories

- **Magic Number:** 1,128
- **Console Log:** 480
- **Any Type:** 210
- **Long Method:** 163
- **Poor Naming:** 46
- **Hardcoded Url:** 26
- **Technical Debt Marker:** 21
- **Complex Conditional:** 7

## Top Files Requiring Attention


### 1. tool-bubble\chart-js-tool.ts

- **Issues:** 102
- **Lines:** 772
- **Functions:** 18

**Issue Breakdown:**
- Magic Numbers: 91
- Console Logs: 8
- Long Methods: 2
- Poor Naming: 1

### 2. service-bubble\ai-agent.ts

- **Issues:** 86
- **Lines:** 1,890
- **Functions:** 78

**Issue Breakdown:**
- Magic Numbers: 50
- Console Logs: 24
- Long Methods: 5
- Any Types: 4
- Hardcoded Urls: 3

### 3. tool-bubble\reddit-scrape-tool.ts

- **Issues:** 77
- **Lines:** 516
- **Functions:** 24

**Issue Breakdown:**
- Magic Numbers: 68
- Console Logs: 5
- Long Methods: 2
- Poor Naming: 1
- Any Types: 1

### 4. workflow-bubble\generate-document.workflow.ts

- **Issues:** 55
- **Lines:** 820
- **Functions:** 23

**Issue Breakdown:**
- Magic Numbers: 30
- Console Logs: 23
- Long Methods: 2

### 5. tool-bubble\pdf-generator-tool.ts

- **Issues:** 50
- **Lines:** 892
- **Functions:** 54

**Issue Breakdown:**
- Magic Numbers: 34
- Any Types: 12
- Console Logs: 3
- Long Methods: 1

### 6. service-bubble\github.ts

- **Issues:** 49
- **Lines:** 1,321
- **Functions:** 26

**Issue Breakdown:**
- Magic Numbers: 39
- Poor Naming: 9
- Long Methods: 1

### 7. service-bubble\ace-tools-bubble.ts

- **Issues:** 47
- **Lines:** 748
- **Functions:** 22

**Issue Breakdown:**
- Magic Numbers: 23
- Console Logs: 13
- Any Types: 7
- Long Methods: 4

### 8. service-bubble\stripe-bubble.ts

- **Issues:** 47
- **Lines:** 1,293
- **Functions:** 23

**Issue Breakdown:**
- Magic Numbers: 40
- Long Methods: 4
- Any Types: 3

### 9. workflow-bubble\parse-document.workflow.ts

- **Issues:** 46
- **Lines:** 822
- **Functions:** 12

**Issue Breakdown:**
- Console Logs: 24
- Magic Numbers: 20
- Long Methods: 2

### 10. workflow-bubble\pdf-ocr.workflow.ts

- **Issues:** 44
- **Lines:** 994
- **Functions:** 24

**Issue Breakdown:**
- Magic Numbers: 22
- Console Logs: 20
- Long Methods: 2

### 11. service-bubble\hephaestus-bubble.ts

- **Issues:** 42
- **Lines:** 1,106
- **Functions:** 15

**Issue Breakdown:**
- Magic Numbers: 21
- Any Types: 14
- Console Logs: 3
- Long Methods: 3
- Hardcoded Urls: 1

### 12. service-bubble\airtable.ts

- **Issues:** 40
- **Lines:** 1,552
- **Functions:** 43

**Issue Breakdown:**
- Debt Markers: 20
- Magic Numbers: 16
- Long Methods: 2
- Console Logs: 1
- Poor Naming: 1

### 13. tool-bubble\research-agent-tool.ts

- **Issues:** 40
- **Lines:** 755
- **Functions:** 26

**Issue Breakdown:**
- Magic Numbers: 25
- Console Logs: 11
- Long Methods: 4

### 14. tool-bubble\file-processor-tool.ts

- **Issues:** 37
- **Lines:** 1,416
- **Functions:** 119

**Issue Breakdown:**
- Console Logs: 27
- Magic Numbers: 7
- Long Methods: 2
- Complex Conditionals: 1

### 15. service-bubble\notion\notion.ts

- **Issues:** 36
- **Lines:** 1,927
- **Functions:** 28

**Issue Breakdown:**
- Magic Numbers: 30
- Long Methods: 5
- Poor Naming: 1

### 16. tool-bubble\metrics-collector-tool.ts

- **Issues:** 35
- **Lines:** 1,428
- **Functions:** 60

**Issue Breakdown:**
- Magic Numbers: 25
- Any Types: 4
- Long Methods: 3
- Poor Naming: 2
- Console Logs: 1

### 17. workflow-bubble\backup-restore.workflow.ts

- **Issues:** 35
- **Lines:** 891
- **Functions:** 31

**Issue Breakdown:**
- Console Logs: 28
- Long Methods: 3
- Magic Numbers: 2
- Poor Naming: 2

### 18. workflow-bubble\pdf-form-operations.workflow.ts

- **Issues:** 35
- **Lines:** 1,212
- **Functions:** 34

**Issue Breakdown:**
- Console Logs: 23
- Magic Numbers: 8
- Long Methods: 4

### 19. service-bubble\github-bubble.ts

- **Issues:** 34
- **Lines:** 721
- **Functions:** 25

**Issue Breakdown:**
- Console Logs: 12
- Any Types: 11
- Magic Numbers: 10
- Long Methods: 1

### 20. service-bubble\notion-bubble.ts

- **Issues:** 33
- **Lines:** 1,066
- **Functions:** 29

**Issue Breakdown:**
- Magic Numbers: 15
- Any Types: 11
- Long Methods: 7

## Refactoring Recommendations

### High Priority

1. **Extract Long Methods:** Break down functions over 100 lines
2. **Reduce Deep Nesting:** Apply Guard Clause pattern
3. **Remove Code Duplication:** Extract common patterns to utilities

### Medium Priority

1. **Replace Magic Numbers:** Use named constants
2. **Extract Complex Conditionals:** Create descriptive variable names
3. **Remove Hardcoded URLs:** Move to configuration

### Low Priority

1. **Improve Naming:** Use descriptive variable names
2. **Remove Console.log:** Use proper logging
3. **Replace 'any' Types:** Use specific types
