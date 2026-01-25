# Security Quick Reference: Expression Evaluation

## Overview
The CSV and Data Transformer tools now use **safe expression evaluation** to prevent code injection attacks.

---

## What Changed?

### Before (INSECURE):
```typescript
// ❌ Allows arbitrary code execution
const result = eval(userInput);
```

### After (SECURE):
```typescript
// ✅ Only allows mathematical operations
import { evaluate } from 'mathjs';
const result = evaluate(userInput, scope);
```

---

## Supported Expressions

### Mathematical Operations
```javascript
// Basic arithmetic
"price * 1.1"
"{quantity} + {bonus}"
"({total} - {discount}) * 0.9"

// Complex expressions
"({price} * {quantity}) + ({tax} * 0.1)"
"Math.sqrt({x}^2 + {y}^2)"  // Not supported - see below
```

### Field References
```javascript
// Use {fieldName} syntax
"{price} + {tax}"
"({quantity} * {price}) / 100"
```

### Operators
- `+` Addition
- `-` Subtraction
- `*` Multiplication
- `/` Division
- `(` `)` Parentheses
- `%` Modulo
- `.` Decimal point

---

## What's NOT Supported

### ❌ JavaScript Functions
```javascript
// NOT supported - will throw error
"Math.round({price})"
"Math.max({a}, {b})"
"parseInt({value})"
```

**Alternative:** Use mathematical equivalents or pre-process data

### ❌ Property Access
```javascript
// NOT supported
"{object.property}"
"{array[0]}"
```

**Alternative:** Use field references directly

### ❌ String Operations
```javascript
// NOT supported
"'Hello ' + {name}"
"{text}.toUpperCase()"
```

**Alternative:** Use the 'format' or 'replace' operations instead

### ❌ Code Execution
```javascript
// BLOCKED - security violation
"eval('...')"
"fetch('...')"
"require('fs')"
```

---

## Usage Examples

### CSV Processor Tool
```typescript
const result = await csvTool.performAction({
  operation: CSVOperationType.TRANSFORM,
  csvData: "price,quantity\n10,5\n20,3",
  transformRules: [
    {
      column: "total",
      operation: "calculate",
      expression: "{price} * {quantity}"  // ✅ Safe
    }
  ]
});
```

### Data Transformer Tool
```typescript
const result = await transformerTool.performAction({
  operation: "map",
  inputData: [
    { price: 10, quantity: 5 },
    { price: 20, quantity: 3 }
  ],
  mapOperations: [
    {
      targetField: "total",
      transform: "calculate",
      expression: "{price} * {quantity}"  // ✅ Safe
    }
  ]
});
```

---

## Security Features

### 1. Input Validation
- ✅ Empty expression check
- ✅ Maximum length: 1000 characters
- ✅ Character whitelist enforcement

### 2. Safe Evaluation
- ✅ Uses `mathjs` library (sandboxed)
- ✅ No code execution possible
- ✅ Only mathematical operations allowed

### 3. Error Handling
- ✅ Detailed error messages
- ✅ Graceful failure
- ✅ Audit logging

---

## Error Messages

### Expression Too Long
```
Error: Expression too long (max 1000 characters)
```
**Solution:** Break into multiple smaller expressions

### Invalid Characters
```
Error: Invalid expression: "...". Only mathematical operations are allowed.
```
**Solution:** Remove non-mathematical characters

### Empty Expression
```
Error: Expression cannot be empty
```
**Solution:** Provide a valid expression

### Invalid Result
```
Error: Expression result is not a valid number: NaN
```
**Solution:** Check expression logic and field values

---

## Custom Transformations

### ⚠️ SECURITY WARNING
Custom transformations are **DISABLED BY DEFAULT** for security reasons.

### To Enable (Development Only)
```bash
export ALLOW_CUSTOM_TRANSFORMATIONS=true
```

### Blocked Patterns
Custom scripts cannot contain:
- `eval(` - Dynamic code execution
- `Function(` - Dynamic function creation
- `require(` - Module loading
- `import ` - ES6 imports
- `fetch(` - Network requests
- `XMLHttpRequest` - Network requests
- `process.` - Node.js APIs
- `child_process` - Process spawning
- `fs.` - File system access
- `__dirname`, `__filename` - Path access
- `../` - Path traversal
- `document.`, `window.` - Browser APIs

### Safe Custom Script Example
```javascript
// ✅ Safe - only uses array methods
(data) => data
  .filter(row => row.price > 10)
  .map(row => ({
    ...row,
    total: row.price * row.quantity
  }))
  .sort((a, b) => a.total - b.total)
```

### Dangerous Script Example (BLOCKED)
```javascript
// ❌ Blocked - contains dangerous pattern
(data) => {
  eval('process.exit()');  // BLOCKED
  return data;
}
```

---

## Best Practices

### DO ✅
1. Use mathematical expressions for calculations
2. Validate user input before passing to tools
3. Keep expressions simple and readable
4. Test expressions thoroughly
5. Monitor logs for errors

### DON'T ❌
1. Never try to inject JavaScript code
2. Never use `eval()` or `Function()` in expressions
3. Never bypass security warnings
4. Never enable custom transformations in production without review
5. Never trust untrusted input sources

---

## Troubleshooting

### Expression Not Working
1. Check expression syntax
2. Verify field names exist
3. Ensure values are numbers
4. Check length < 1000 characters
5. Review error message

### Custom Transformation Blocked
1. Check if `ALLOW_CUSTOM_TRANSFORMATIONS=true`
2. Review script for dangerous patterns
3. Ensure script returns array
4. Check script length < 10000 characters
5. Review error message

---

## Migration Guide

### Old Code (If You Used eval)
```typescript
// ❌ OLD - Don't do this
const result = eval(expression);
```

### New Code
```typescript
// ✅ NEW - Use this instead
import { evaluate } from 'mathjs';
const scope = { price: 10, quantity: 5 };
const result = evaluate("{price} * {quantity}", scope);
```

---

## Testing

### Test Safe Expression
```javascript
const safe = "{price} * {quantity}";
// Should work ✅
```

### Test Blocked Pattern
```javascript
const blocked = "eval('process.exit()')";
// Should throw error ❌
```

### Test Length Limit
```javascript
const tooLong = "1".repeat(1001);
// Should throw error ❌
```

---

## Support

### Documentation
- CSV Tool: `./csv-processor-tool.ts`
- Data Transformer: `./data-transformer-tool.ts`
- Full Report: `./SECURITY_FIX_REPORT.md`

### Getting Help
1. Check error messages
2. Review this quick reference
3. See full security report
4. Check mathjs documentation: https://mathjs.org/

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
