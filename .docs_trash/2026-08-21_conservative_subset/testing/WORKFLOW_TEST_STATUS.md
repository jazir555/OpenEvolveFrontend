# WORKFLOW TEST STATUS REPORT

Generated: 2026-01-19T02:03:30.285350

## WORKFLOW BUBBLES

Total: 17
With Tests: 0 (0.0%)
Without Tests: 17 (100.0%)

| Bubble | Has Test | Test File | Coverage | Missing Tests |
|--------|----------|-----------|----------|---------------|
| api-aggregator.workflow | ❌ | N/A | N/A | - |
| backup-restore.workflow | ❌ | N/A | N/A | - |
| data-enrichment.workflow | ❌ | N/A | N/A | - |
| database-analyzer.workflow | ❌ | N/A | N/A | - |
| etl-pipeline.workflow | ❌ | N/A | N/A | - |
| event-handler.workflow | ❌ | N/A | N/A | - |
| generate-document.workflow | ❌ | N/A | N/A | - |
| monitoring-alert.workflow | ❌ | N/A | N/A | - |
| multi-step-approval.workflow | ❌ | N/A | N/A | - |
| parse-document.workflow | ❌ | N/A | N/A | - |
| pdf-form-operations.workflow | ❌ | N/A | N/A | - |
| pdf-ocr.workflow | ❌ | N/A | N/A | - |
| scheduled-task.workflow | ❌ | N/A | N/A | - |
| slack-data-assistant.workflow | ❌ | N/A | N/A | - |
| slack-formatter-agent | ❌ | N/A | N/A | - |
| slack-notifier.workflow | ❌ | N/A | N/A | - |
| webhook-repeater.workflow | ❌ | N/A | N/A | - |

## SERVICE BUBBLES

Total: 40
With Tests: 20 (50.0%)
Without Tests: 20 (50.0%)

| Bubble | Has Test | Test File | Coverage | Missing Tests |
|--------|----------|-----------|----------|---------------|
| ace-tools-bubble | ❌ | N/A | N/A | - |
| agi-inc | ❌ | N/A | N/A | - |
| ai-agent | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\ai-agent.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| airtable | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable.test.ts | -15.0% | Environment Validation |
| airtable-bubble | ❌ | N/A | N/A | - |
| airtable-wrapper | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\airtable-wrapper.test.ts | -35.0% | Environment Validation, Authentication |
| apify-bubble | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\apify-bubble.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| elasticsearch-bubble | ❌ | N/A | N/A | - |
| eleven-labs | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\eleven-labs.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| firecrawl | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\firecrawl.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| followupboss | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\followupboss.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| github | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\github.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| github-bubble | ❌ | N/A | N/A | - |
| gmail | ❌ | N/A | N/A | - |
| gmail-bubble | ❌ | N/A | N/A | - |
| google-calendar | ❌ | N/A | N/A | - |
| google-drive | ❌ | N/A | N/A | - |
| google-drive-bubble | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-drive-bubble.test.ts | -35.0% | Environment Validation, Input Validation |
| google-sheets-bubble | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\google-sheets-bubble.test.ts | -15.0% | Environment Validation |
| hello-world | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\hello-world.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| crewai-bubble | ❌ | N/A | N/A | - |
| http | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| http-bubble | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\http-bubble.test.ts | -55.0% | Environment Validation, Rate Limiting, Input Validation |
| http-fix-validation | ❌ | N/A | N/A | - |
| insforge-db | ❌ | N/A | N/A | - |
| notion-bubble | ❌ | N/A | N/A | - |
| postgresql | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\postgresql.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| postgresql-bubble | ❌ | N/A | N/A | - |
| qdrant-bubble | ❌ | N/A | N/A | - |
| redis-bubble | ❌ | N/A | N/A | - |
| resend | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\resend.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| sendgrid-bubble | ❌ | N/A | N/A | - |
| slack | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\slack.test.ts | -15.0% | Environment Validation |
| slack-bubble | ❌ | N/A | N/A | - |
| storage | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\storage.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| stripe-bubble | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\stripe-bubble.test.ts | -35.0% | Environment Validation, Input Validation |
| telegram | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\telegram.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| twilio-bubble | ❌ | N/A | N/A | - |
| webhook-bubble | ✅ | BubbleLab\packages\bubble-core\src\bubbles\service-bubble\webhook-bubble.test.ts | -55.0% | Environment Validation, Authentication, Input Validation |
| workflow-orchestrator-bubble | ❌ | N/A | N/A | - |

## TOOL BUBBLES

Total: 33
With Tests: 14 (42.4%)
Without Tests: 19 (57.6%)

| Bubble | Has Test | Test File | Coverage | Missing Tests |
|--------|----------|-----------|----------|---------------|
| bubbleflow-validation-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\bubbleflow-validation-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| chart-js-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\chart-js-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| code-edit-tool | ❌ | N/A | N/A | - |
| code-formatter-tool | ❌ | N/A | N/A | - |
| csv-processor-tool | ❌ | N/A | N/A | - |
| data-transformer-tool | ❌ | N/A | N/A | - |
| email-validator-tool | ❌ | N/A | N/A | - |
| file-processor-tool | ❌ | N/A | N/A | - |
| get-bubble-details-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\get-bubble-details-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| google-maps-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\google-maps-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| image-processor-tool | ❌ | N/A | N/A | - |
| instagram-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\instagram-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| json-validator-tool | ❌ | N/A | N/A | - |
| linkedin-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\linkedin-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| list-bubbles-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\list-bubbles-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| log-parser-tool | ❌ | N/A | N/A | - |
| metrics-collector-tool | ❌ | N/A | N/A | - |
| pdf-generator-tool | ❌ | N/A | N/A | - |
| reddit-scrape-tool | ❌ | N/A | N/A | - |
| research-agent-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\research-agent-tool.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| sql-query-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\sql-query-tool.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| text-analyzer-tool | ❌ | N/A | N/A | - |
| tiktok-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\tiktok-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| tool-template | ❌ | N/A | N/A | - |
| twitter-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\twitter-tool.test.ts | -95.0% | Environment Validation, Authentication, Rate Limiting, Input Validation, Error Handling |
| url-validator-tool | ❌ | N/A | N/A | - |
| vector-search-tool | ❌ | N/A | N/A | - |
| web-crawl-tool | ❌ | N/A | N/A | - |
| web-extract-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-extract-tool.test.ts | -75.0% | Environment Validation, Authentication, Rate Limiting, Input Validation |
| web-scrape-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-scrape-tool.test.ts | -55.0% | Environment Validation, Authentication, Error Handling |
| web-search-tool | ✅ | BubbleLab\packages\bubble-core\src\bubbles\tool-bubble\web-search-tool.test.ts | -55.0% | Environment Validation, Authentication, Input Validation |
| xml-parser-tool | ❌ | N/A | N/A | - |
| youtube-tool | ❌ | N/A | N/A | - |

