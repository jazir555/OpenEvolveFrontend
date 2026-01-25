# ALL WORKFLOW BUBBLES - COMPREHENSIVE BUG REPORT

Generated: 2026-01-19T02:03:30.280880

## SECURITY ISSUES

Total Security Issues: 115

### High Severity (38 issues)

#### code_injection

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:441`

**Issue:** Use of eval() is dangerous

**Impact:** Code injection or XSS vulnerability

**Recommendation:** Remove or sanitize this pattern

**Code:**
```
* const result = await context.eval(code, { timeout });
```

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\agi-inc.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\eleven-labs.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### env_validation

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\followupboss.ts:1071`

**Issue:** Environment variables used without validation

**Impact:** Application may crash or behave unexpectedly with missing/invalid env vars

**Recommendation:** Add environment variable validation at startup

**Code:**
```
'X-System': process.env.FUB_SYSTEM_NAME || 'Bubble-Lab',
```

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\followupboss.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-calendar.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\bubbleflow-validation-tool.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### code_injection

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:349`

**Issue:** Use of exec() is dangerous

**Impact:** Code injection or XSS vulnerability

**Recommendation:** Remove or sanitize this pattern

**Code:**
```
//   exec(`${this.scannerCommand} "${filePath}"`, { timeout: this.timeout }, (error, stdout, stderr) => {
```

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### code_injection

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:163`

**Issue:** Use of exec() is dangerous

**Impact:** Code injection or XSS vulnerability

**Recommendation:** Remove or sanitize this pattern

**Code:**
```
while ((match = kvPattern.exec(line)) !== null) {
```

---

#### code_injection

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:660`

**Issue:** Use of exec() is dangerous

**Impact:** Code injection or XSS vulnerability

**Recommendation:** Remove or sanitize this pattern

**Code:**
```
while ((match = styleRegex.exec(css)) !== null) {
```

---

#### code_injection

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:668`

**Issue:** Use of exec() is dangerous

**Impact:** Code injection or XSS vulnerability

**Recommendation:** Remove or sanitize this pattern

**Code:**
```
while ((propMatch = propRegex.exec(properties)) !== null) {
```

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\text-analyzer-tool.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### code_injection

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:377`

**Issue:** Use of exec() is dangerous

**Impact:** Code injection or XSS vulnerability

**Recommendation:** Remove or sanitize this pattern

**Code:**
```
while ((match = tagRegex.exec(xml)) !== null) {
```

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\etl-pipeline.workflow.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

#### timeout

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-data-assistant.workflow.ts:1`

**Issue:** No timeout configured for network requests

**Impact:** Application may hang indefinitely

**Recommendation:** Add timeouts to all network requests

---

### Medium Severity (33 issues)

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\eleven-labs.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-calendar.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### error_handling

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hello-world.ts:1`

**Issue:** No error handling detected

**Impact:** Errors may expose sensitive information

**Recommendation:** Add try-catch blocks and proper error handling

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\insforge-db.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\bubbleflow-validation-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\tool-template.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\url-validator-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-scrape-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\api-aggregator.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\etl-pipeline.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\scheduled-task.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

#### rate_limiting

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:1`

**Issue:** No rate limiting detected for API calls

**Impact:** API abuse and potential quota exhaustion

**Recommendation:** Implement rate limiting for all API calls

---

### Low Severity (44 issues)

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\insforge-db.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\code-formatter-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\text-analyzer-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\url-validator-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-crawl-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-scrape-tool.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\api-aggregator.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\etl-pipeline.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\scheduled-task.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-notifier.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

#### logging

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:1`

**Issue:** Using console.log instead of structured logging

**Impact:** Poor observability and potential information leakage

**Recommendation:** Use structured logging with correlation IDs

---

## CODE QUALITY ISSUES

Total Quality Issues: 1158

### code_quality (892 issues)

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:213`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:273`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:289`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:445`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:551`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:557`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:558`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:577`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:598`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:619`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:636`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:659`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:675`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:689`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:712`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:977`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:978`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1050`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1051`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1052`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1262`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1732`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:296`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:313`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:419`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:446`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:448`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:450`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:452`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:454`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:456`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:458`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:460`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:462`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:464`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:832`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:332`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:406`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:411`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:416`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:429`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:566`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:584`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:608`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:610`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:612`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:614`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:616`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:618`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:620`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:622`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:624`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:626`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:628`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:630`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:107`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:111`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:182`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:186`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:207`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:211`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:246`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:250`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:291`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:295`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:332`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:349`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:384`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:388`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:410`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:414`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:441`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:445`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:449`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:752`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:1324`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:455`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:591`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:618`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:620`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:622`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:624`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:626`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:628`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:630`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:632`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:634`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:636`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:638`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:640`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:642`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:900`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:918`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:963`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:981`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1035`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1053`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1183`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1230`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1277`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1380`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1439`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1457`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1460`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:203`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:251`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:268`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:328`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:344`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:359`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:377`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:396`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:429`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:434`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:451`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:472`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:485`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:496`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:507`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:532`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\eleven-labs.ts:230`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\firecrawl.ts:1194`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:235`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:260`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:273`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:331`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:346`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:393`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:435`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:471`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:474`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:479`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:480`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:504`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:537`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:569`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:572`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:606`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:624`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:637`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:700`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github.ts:690`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:211`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:236`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:249`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:307`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:315`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:321`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:346`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:382`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:387`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:420`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:433`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:437`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:438`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:460`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:463`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:484`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:488`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:502`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:518`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:535`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:561`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:578`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:598`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:832`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:875`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:917`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:944`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:959`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:985`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1007`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1046`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1252`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1321`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1456`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-calendar.ts:402`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:315`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:362`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:456`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:462`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:653`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:660`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:688`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:798`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:803`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:904`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive.ts:529`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive.ts:1163`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive.ts:1194`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:508`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:527`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:624`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:721`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:749`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:751`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:753`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:755`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:757`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:759`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:761`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:763`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:765`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:767`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:769`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:771`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:773`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:775`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:778`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:780`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:782`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:784`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:786`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:788`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:880`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:1301`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:1494`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:1498`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:1558`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:307`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:398`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:399`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:490`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:508`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.ts:299`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.ts:347`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.ts:609`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.ts:631`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:24`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:32`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:34`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:35`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:42`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:44`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:45`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:52`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:54`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:55`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:62`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:64`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:65`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:73`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:75`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:76`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:84`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:86`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:87`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:94`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:96`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:97`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:104`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:106`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:107`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:111`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:112`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-fix-validation.ts:113`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http.ts:129`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http.ts:199`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http.ts:208`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\insforge-db.ts:93`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:64`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:484`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:512`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:526`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:619`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:645`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:648`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:651`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:654`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:657`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:660`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:663`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:666`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:669`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:672`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:675`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:678`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:681`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:684`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:687`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:690`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:693`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:807`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:1113`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:209`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:259`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:262`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:279`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:343`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:359`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:363`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:374`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:393`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:424`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:455`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:476`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:513`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:544`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:567`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:589`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:614`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:248`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:268`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:284`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:394`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:406`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:463`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:541`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:551`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:635`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:691`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:216`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:250`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:265`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:325`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:341`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:350`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:365`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:378`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:397`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:414`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:417`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:420`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:437`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:453`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:470`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:485`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:490`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:493`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:510`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:185`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:215`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:218`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:232`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:289`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:316`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:319`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:336`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:352`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:366`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:379`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:393`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:412`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:431`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:445`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:461`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:470`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:485`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:503`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:201`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:223`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:272`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:308`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:323`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:327`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:350`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:398`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:426`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:455`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:486`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:520`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:206`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:231`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:244`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:302`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:316`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:355`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:379`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:404`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:424`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:444`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:465`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:484`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:511`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:514`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:534`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:589`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1345`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:405`

**Issue:** Todo/Fixme comment

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:454`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:468`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:479`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:486`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:500`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:578`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:598`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:602`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:634`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:667`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:669`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:671`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:673`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:675`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:677`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:679`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:681`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:683`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:685`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:687`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:689`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:691`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:693`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:695`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:1212`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:179`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:204`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:218`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:267`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:295`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:323`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:349`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:368`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:391`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:408`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:433`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:454`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:463`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:465`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:726`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:728`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:730`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:732`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:734`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:736`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:738`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:740`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:742`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:744`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:746`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:748`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:750`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:752`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:1389`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:1434`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:1754`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:1827`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:261`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:319`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:352`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:383`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:409`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:430`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:456`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:480`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:501`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:518`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:543`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:578`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:605`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\chart-js-tool.ts:741`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\code-edit-tool.ts:266`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\code-formatter-tool.ts:251`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\code-formatter-tool.ts:271`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\code-formatter-tool.ts:296`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\code-formatter-tool.ts:456`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:300`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:337`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:562`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:763`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:772`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:284`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:359`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:380`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:442`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:1002`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:265`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:291`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:310`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:499`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\email-validator-tool.ts:512`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:167`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:180`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:182`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:196`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:198`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:210`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:215`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:335`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:340`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:358`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:362`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:402`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:415`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:417`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:431`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:433`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:445`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:450`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:751`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:816`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:893`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:956`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:986`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:987`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1003`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1024`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1097`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1104`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1111`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1118`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1202`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1246`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1278`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1288`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1343`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1353`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1359`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1384`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1389`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1391`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1401`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1447`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1449`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1474`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1477`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1512`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1706`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:161`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:192`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:500`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:561`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:644`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:753`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:878`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\google-maps-tool.ts:238`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:290`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:393`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:417`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:432`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:458`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:477`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:490`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:519`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:532`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:274`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:293`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:345`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:353`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:361`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:367`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:396`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:509`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:624`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:700`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\linkedin-tool.ts:591`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:558`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:743`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:776`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:977`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:989`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:1011`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:695`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:698`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:731`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:784`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:810`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:825`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:854`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:857`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:1424`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:261`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:319`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:322`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:340`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:355`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:374`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:477`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:492`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:695`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:813`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:849`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:862`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:871`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:887`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:102`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:190`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:196`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:200`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:223`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:287`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:289`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:318`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:416`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:438`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:445`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:474`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:161`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:165`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:169`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:196`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:204`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:205`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:206`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:255`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:256`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:257`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:268`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:465`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:632`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:116`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:158`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:168`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:169`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:170`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:181`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:202`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.ts:233`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\text-analyzer-tool.ts:299`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\text-analyzer-tool.ts:346`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\text-analyzer-tool.ts:353`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\text-analyzer-tool.ts:406`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\text-analyzer-tool.ts:498`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\url-validator-tool.ts:242`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\url-validator-tool.ts:268`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\url-validator-tool.ts:287`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\url-validator-tool.ts:365`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:200`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:203`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:204`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:205`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:220`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:223`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:242`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:309`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:357`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:364`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:393`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:406`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-crawl-tool.ts:151`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-crawl-tool.ts:189`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-crawl-tool.ts:192`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-crawl-tool.ts:205`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-crawl-tool.ts:258`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:123`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:124`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:128`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:199`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:200`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:210`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-scrape-tool.ts:182`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-scrape-tool.ts:190`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-scrape-tool.ts:223`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-search-tool.ts:125`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-search-tool.ts:198`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:550`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:587`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:620`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:652`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:660`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:704`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:736`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:770`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:804`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\api-aggregator.workflow.ts:53`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:248`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:270`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:283`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:290`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:305`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:311`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:316`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:335`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:350`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:363`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:378`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:383`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:385`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:390`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:402`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:408`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:414`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:419`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:425`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:432`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:446`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:459`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:483`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:519`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:645`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:653`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:661`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:669`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:748`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:757`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:830`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:858`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:296`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:297`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:307`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:324`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:341`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:359`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:380`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:381`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:382`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:401`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:434`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:462`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:477`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:509`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:528`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:553`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:624`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\database-analyzer.workflow.ts:151`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\database-analyzer.workflow.ts:188`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\database-analyzer.workflow.ts:244`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\database-analyzer.workflow.ts:289`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\database-analyzer.workflow.ts:314`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\etl-pipeline.workflow.ts:55`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:99`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:106`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:125`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:133`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:151`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:162`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:177`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:211`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:220`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:230`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:256`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\event-handler.workflow.ts:260`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:296`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:299`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:304`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:308`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:315`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:327`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:349`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:356`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:371`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:375`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:376`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:403`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:442`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:445`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:467`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:484`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:492`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:526`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:530`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:538`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:567`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:580`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:582`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:599`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:618`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:195`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:199`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:204`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:220`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:224`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:229`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:234`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:251`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:306`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:327`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:334`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:363`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:413`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\monitoring-alert.workflow.ts:417`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:83`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:123`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:147`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:164`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:223`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:344`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:389`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:412`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:431`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:440`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:454`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\multi-step-approval.workflow.ts:471`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:368`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:371`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:375`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:387`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:402`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:420`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:448`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:460`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:471`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:483`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:493`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:537`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:554`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:568`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:572`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:583`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:591`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:599`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:602`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:608`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:644`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:669`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:683`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:700`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:717`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:722`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:766`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:769`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:789`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:475`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:523`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:604`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:617`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:627`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:648`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:649`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:690`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:705`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:724`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:736`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:750`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:770`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:805`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:821`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:844`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:845`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:858`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:875`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:876`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:880`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:920`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:935`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:962`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:978`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:989`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1010`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1027`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1052`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1134`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1138`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1160`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1176`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1204`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:479`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:480`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:487`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:504`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:509`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:529`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:534`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:573`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:618`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:621`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:650`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:678`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:728`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:731`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:740`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:775`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:787`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:806`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:815`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:823`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:846`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:850`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:857`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:926`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\scheduled-task.workflow.ts:54`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\scheduled-task.workflow.ts:76`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\scheduled-task.workflow.ts:98`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\scheduled-task.workflow.ts:107`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-data-assistant.workflow.ts:430`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:630`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:636`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:642`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:657`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:658`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:663`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:843`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:849`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:857`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:869`

**Issue:** Use of 'any' type

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-notifier.workflow.ts:211`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-notifier.workflow.ts:216`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-notifier.workflow.ts:230`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-notifier.workflow.ts:250`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-notifier.workflow.ts:264`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-notifier.workflow.ts:285`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:295`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:296`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:321`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:335`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:347`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:352`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:386`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:417`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:425`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:442`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:450`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\webhook-repeater.workflow.ts:458`

**Issue:** Console logging in production

**Recommendation:** Review and improve code quality

---

### error_handling (222 issues)

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:257`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ace-tools-bubble.ts:294`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\agi-inc.ts:596`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\agi-inc.ts:952`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:741`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:754`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:770`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:787`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:808`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:812`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:943`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1602`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1610`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1618`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1627`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1638`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1742`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.ts:1748`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:290`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:307`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:324`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:340`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-bubble.ts:466`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:360`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:369`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.ts:632`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:836`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:1293`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.ts:1529`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:449`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:466`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:644`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1170`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1216`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1247`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1269`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1310`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1361`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1415`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1501`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\elasticsearch-bubble.ts:312`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\eleven-labs.ts:230`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\followupboss.ts:1108`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\followupboss.ts:1275`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:317`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:366`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:293`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:340`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:858`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1096`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1622`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1665`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-calendar.ts:423`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-calendar.ts:454`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:285`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:426`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:481`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:547`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive.ts:629`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:502`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:521`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:540`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:558`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:582`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:790`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:1562`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:265`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:272`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:325`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:343`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:347`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:384`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:443`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.ts:355`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\insforge-db.ts:235`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\insforge-db.ts:263`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\insforge-db.ts:297`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:493`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\notion-bubble.ts:696`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql-bubble.ts:329`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:578`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:610`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.ts:627`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\qdrant-bubble.ts:309`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\redis-bubble.ts:276`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:218`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:245`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:272`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:299`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:347`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:436`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:482`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:259`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:395`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:423`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:452`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:483`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:517`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:288`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:331`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:586`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1243`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1289`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1301`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1828`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1847`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1871`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:376`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:387`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:435`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.ts:697`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\telegram.ts:895`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\telegram.ts:956`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\twilio-bubble.ts:254`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.ts:754`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:305`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:367`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:446`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:450`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:470`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:474`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:493`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:515`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:564`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\workflow-orchestrator-bubble.ts:593`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\chart-js-tool.ts:264`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\chart-js-tool.ts:772`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:324`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:504`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:341`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:513`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:528`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:531`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:937`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:973`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:1012`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:803`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:837`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:845`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:863`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:871`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:880`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:924`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:931`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:941`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:948`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:970`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1059`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1073`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1123`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1139`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1147`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1153`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1168`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1226`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1270`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1312`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1320`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1325`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1368`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1374`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1394`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1409`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1413`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1429`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1437`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1453`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1468`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\file-processor-tool.ts:1500`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:104`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:491`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.ts:539`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:316`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:322`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:362`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:556`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\instagram-tool.ts:255`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:695`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\linkedin-tool.ts:433`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:579`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:723`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:777`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\metrics-collector-tool.ts:1058`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:367`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\reddit-scrape-tool.ts:475`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:200`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:223`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:227`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:240`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\twitter-tool.ts:298`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:272`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:280`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:349`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:419`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:437`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:138`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.ts:173`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-scrape-tool.ts:161`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-search-tool.ts:166`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:265`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:279`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:574`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\youtube-tool.ts:239`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:429`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:670`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:439`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:394`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:406`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:442`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:511`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-form-operations.workflow.ts:1117`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:501`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:526`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\pdf-ocr.workflow.ts:615`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-data-assistant.workflow.ts:363`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-data-assistant.workflow.ts:416`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-data-assistant.workflow.ts:497`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-data-assistant.workflow.ts:548`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:379`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:389`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:615`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:783`

**Issue:** Error message may expose sensitive information

**Recommendation:** Use sanitized error messages

---

### resource_management (44 issues)

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\eleven-labs.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\firecrawl.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\gmail.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hephaestus-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\insforge-db.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\sendgrid-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack-bubble.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\service-bubble\telegram.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\chart-js-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\code-edit-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\csv-processor-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\data-transformer-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\image-processor-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\instagram-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\json-validator-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\linkedin-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\log-parser-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\pdf-generator-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\tiktok-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\tool-template.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\twitter-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\url-validator-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\vector-search-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\xml-parser-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\youtube-tool.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\backup-restore.workflow.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\data-enrichment.workflow.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\database-analyzer.workflow.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\etl-pipeline.workflow.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\generate-document.workflow.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\parse-document.workflow.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-data-assistant.workflow.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

**File:** `BubbleLab\packages\bubble-core\src\bubbles\workflow-bubble\slack-formatter-agent.ts:1`

**Issue:** No resource cleanup detected

**Recommendation:** Ensure proper resource cleanup in finally blocks

---

