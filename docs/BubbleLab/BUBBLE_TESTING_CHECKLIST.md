# ✅ BUBBLE TESTING CHECKLIST
## Systematic Testing Progress Tracker

**Instructions**: Check off bubbles as tests are completed
**Last Updated**: 2026-01-19

---

## 📊 PROGRESS TRACKER

```
Overall Progress: ████░░░░░░ 26.6% (37/139)

Service Bubbles:  ██████░░░░ 34.3% (23/67)
Tool Bubbles:     ███████░░░ 40.0% (14/35)
Workflow Bubbles: ░░░░░░░░░░   0.0% ( 0/17)
Templates:        ░░░░░░░░░░   0.0% ( 0/20)
```

---

## 🔴 CRITICAL PRIORITY (25+ operations, untested)
*These are the most complex and urgent bubbles to test*

### Service Bubbles
- [ ] **NotionBubble** (71 ops) - `packages/bubble-core/src/bubbles/service-bubble/notion-bubble.ts`
- [ ] **AirtableBubble** (47 ops) - `packages/bubble-core/src/bubbles/service-bubble/airtable-bubble.ts`
- [ ] **GmailBubble** (39 ops) - `packages/bubble-core/src/bubbles/service-bubble/gmail.ts`
- [ ] **AceToolsBubble** (36 ops) - `packages/bubble-core/src/bubbles/service-bubble/ace-tools-bubble.ts`
- [ ] **GithubBubble** (34 ops) - `packages/bubble-core/src/bubbles/service-bubble/github-bubble.ts`
- [ ] **GmailBubble** (34 ops) - `packages/bubble-core/src/bubbles/service-bubble/gmail-bubble.ts`
- [ ] **RedisBubble** (34 ops) - `packages/bubble-core/src/bubbles/service-bubble/redis-bubble.ts`
- [ ] **SlackBubble** (34 ops) - `packages/bubble-core/src/bubbles/service-bubble/slack-bubble.ts`
- [ ] **WorkflowOrchestratorBubble** (34 ops) - `packages/bubble-core/src/bubbles/service-bubble/workflow-orchestrator-bubble.ts`
- [ ] **ElasticsearchBubble** (33 ops) - `packages/bubble-core/src/bubbles/service-bubble/elasticsearch-bubble.ts`
- [ ] **HephaestusBubble** (33 ops) - `packages/bubble-core/src/bubbles/service-bubble/hephaestus-bubble.ts`
- [ ] **PostgresqlBubble** (33 ops) - `packages/bubble-core/src/bubbles/service-bubble/postgresql-bubble.ts`
- [ ] **QdrantBubble** (33 ops) - `packages/bubble-core/src/bubbles/service-bubble/qdrant-bubble.ts`
- [ ] **AGIIncBubble** (31 ops) - `packages/bubble-core/src/bubbles/service-bubble/agi-inc.ts`
- [ ] **SendGridBubble** (27 ops) - `packages/bubble-core/src/bubbles/service-bubble/sendgrid-bubble.ts`
- [ ] **TwilioBubble** (27 ops) - `packages/bubble-core/src/bubbles/service-bubble/twilio-bubble.ts`

### Workflow Bubbles
- [ ] **BackupRestoreWorkflow** (22 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/backup-restore.workflow.ts`
- [ ] **PDFFormOperationsWorkflow** (22 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/pdf-form-operations.workflow.ts`

### Tool Bubbles
- [ ] **FileProcessorTool** (14 ops) - `packages/bubble-core/src/bubbles/tool-bubble/file-processor-tool.ts`
- [ ] **MetricsCollectorTool** (13 ops) - `packages/bubble-core/src/bubbles/tool-bubble/metrics-collector-tool.ts`

---

## 🟠 HIGH PRIORITY (15-24 operations, untested)

### Service Bubbles
- [ ] **GoogleCalendarBubble** (20 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-calendar.ts`
- [ ] **GoogleDriveBubble** (12 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-drive.ts`
- [ ] **HephaestusBubble** (15 ops) - `integrations/openevolve/service-bubbles/hephaestus-bubble.ts`
- [ ] **SendGridBubble** (15 ops) - `integrations/openevolve/service-bubbles/sendgrid-bubble.ts`
- [ ] **TwilioBubble** (15 ops) - `integrations/openevolve/service-bubbles/twilio-bubble.ts`
- [ ] **WorkflowOrchestratorBubble** (15 ops) - `integrations/openevolve/service-bubbles/workflow-orchestrator-bubble.ts`
- [ ] **ACEToolsBubble** (14 ops) - `integrations/openevolve/service-bubbles/ace-tools-bubble.ts`

### Tool Bubbles
- [ ] **XMLParserTool** (11 ops) - `packages/bubble-core/src/bubbles/tool-bubble/xml-parser-tool.ts`

### Templates
- [ ] **ContentCreationTrendsFlow** (11 ops) - `apps/bubble-studio/src/components/templates/template_codes/contentCreationTrends.ts`
- [ ] **TelegramBotFlow** (10 ops) - `apps/bubble-studio/src/components/templates/template_codes/telegrambot.ts`
- [ ] **VideoScriptGeneratorFlow** (9 ops) - `apps/bubble-studio/src/components/templates/template_codes/videoScriptGenerator.ts`
- [ ] **RedditFlow** (8 ops) - `apps/bubble-studio/src/components/templates/template_codes/redditLeadGeneration.ts`

---

## 🟡 MEDIUM PRIORITY (5-14 operations, untested)

### Service Bubbles
- [ ] **GitHubBubble** (10 ops) - `integrations/openevolve/service-bubbles/github-bubble.ts`
- [ ] **QdrantBubble** (10 ops) - `integrations/openevolve/service-bubbles/qdrant-bubble.ts`
- [ ] **GmailBubble** (9 ops) - `integrations/openevolve/service-bubbles/gmail-bubble.ts`
- [ ] **PostgreSQLBubbleExtended** (9 ops) - `integrations/openevolve/service-bubbles/postgresql-bubble.ts`
- [ ] **SlackBubble** (9 ops) - `integrations/openevolve/service-bubbles/slack-bubble.ts`
- [ ] **GoogleSheetsStressTest** (8 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-sheets/google-sheets.integration.flow.ts`
- [ ] **KnowledgeEngineBubble** (8 ops) - `integrations/openevolve/service-bubbles/knowledge-engine-bubble.ts`
- [ ] **RedisBubble** (8 ops) - `integrations/openevolve/service-bubbles/redis-bubble.ts`
- [ ] **ElasticsearchBubble** (13 ops) - `integrations/openevolve/service-bubbles/elasticsearch-bubble.ts`

### Tool Bubbles
- [ ] **PDFGeneratorTool** (8 ops) - `packages/bubble-core/src/bubbles/tool-bubble/pdf-generator-tool.ts`
- [ ] **CSVProcessorTool** (8 ops) - `packages/bubble-core/src/bubbles/tool-bubble/csv-processor-tool.ts`
- [ ] **YouTubeTool** (5 ops) - `packages/bubble-core/src/bubbles/tool-bubble/youtube-tool.ts`
- [ ] **LogParserTool** (5 ops) - `packages/bubble-core/src/bubbles/tool-bubble/log-parser-tool.ts`
- [ ] **MetricsCollectorTool** (7 ops) - `integrations/openevolve/tool-bubbles/metrics-collector-tool.ts`
- [ ] **LogParserTool** (5 ops) - `integrations/openevolve/tool-bubbles/log-parser-tool.ts`
- [ ] **EmailValidatorTool** (4 ops) - `packages/bubble-core/src/bubbles/tool-bubble/email-validator-tool.ts`
- [ ] **RedditScrapeTool** (4 ops) - `packages/bubble-core/src/bubbles/tool-bubble/reddit-scrape-tool.ts`

### Workflow Bubbles
- [ ] **MultiStepApprovalWorkflow** (8 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/multi-step-approval.workflow.ts`
- [ ] **DataEnrichmentWorkflow** (6 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/data-enrichment.workflow.ts`
- [ ] **SlackFormatterAgentBubble** (6 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/slack-formatter-agent.ts`
- [ ] **ETLPipelineWorkflow** (5 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/etl-pipeline.workflow.ts`
- [ ] **SlackNotifierWorkflowBubble** (5 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/slack-notifier.workflow.ts`
- [ ] **EventHandlerWorkflow** (4 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/event-handler.workflow.ts`

### Templates
- [ ] **NanobananaImagePipeline** (7 ops) - `apps/bubble-studio/src/components/templates/template_codes/nanobananaImagePipeline.ts`
- [ ] **NotionApprovalMonitor** (7 ops) - `apps/bubble-studio/src/components/templates/template_codes/notionApprovalMonitor.ts`
- [ ] **DailyNewsDigestFlow** (6 ops) - `apps/bubble-studio/src/components/templates/template_codes/dailyNewsDigest.ts`
- [ ] **LinkedinLeadGen** (6 ops) - `apps/bubble-studio/src/components/templates/template_codes/linkedinLeadGen.ts`
- [ ] **SlackDigestAndEmailWorkflow** (6 ops) - `apps/bubble-studio/src/components/templates/template_codes/projectManagementAssistant.ts`
- [ ] **CalendarReportFlow** (6 ops) - `apps/bubble-studio/src/components/templates/template_codes/personalAssistant.ts`
- [ ] **WebsiteLeadGenerationFlow** (6 ops) - `apps/bubble-studio/src/components/templates/template_codes/websiteLeadGeneration.ts`
- [ ] **GithubPRCommenter** (6 ops) - `apps/bubble-studio/src/components/templates/template_codes/githubPRCommenter.ts`
- [ ] **ChatWithYourDatabaseFlow** (5 ops) - `apps/bubble-studio/src/components/templates/template_codes/chatWithYourDatabase.ts`
- [ ] **StockAnalysisFlow** (5 ops) - `apps/bubble-studio/src/components/templates/template_codes/financialAdvisor.ts`
- [ ] **ProductImageTransformer** (5 ops) - `apps/bubble-studio/src/components/templates/template_codes/productImageTransformer.ts`
- [ ] **TechweekSchedulerFlow** (5 ops) - `apps/bubble-studio/src/components/templates/template_codes/techweekScheduler.ts`

---

## 🟢 LOW PRIORITY (<5 operations, untested)

### Service Bubbles
- [ ] **HttpBubble** (3 ops) - `integrations/openevolve/service-bubbles/http-bubble.ts`
- [ ] **InsForgeDbBubble** (4 ops) - `packages/bubble-core/src/bubbles/service-bubble/insforge-db.ts`
- [ ] **APIFY_ACTOR_SCHEMAS** (1 op) - `packages/bubble-core/src/bubbles/service-bubble/apify/apify-scraper.schema.ts`
- [ ] **DataSourcePropertySchema** (1 op) - `packages/bubble-core/src/bubbles/service-bubble/notion/property-schemas.ts`
- [ ] **ValueRangeSchema** (11 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-sheets/google-sheets.schema.ts`
- [ ] **apify-bubble** (0 ops) - `integrations/openevolve/service-bubbles/apify-bubble.ts`
- [ ] **google-sheets.utils** (0 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-sheets/google-sheets.utils.ts`
- [ ] **http-fix-validation** (0 ops) - `packages/bubble-core/src/bubbles/service-bubble/http-fix-validation.ts`
- [ ] **index files** (0 ops) - Multiple index.ts files
- [ ] **types** (0 ops) - `packages/bubble-core/src/bubbles/service-bubble/apify/types.ts`

### Tool Bubbles
- [ ] **URLValidatorTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/url-validator-tool.ts`
- [ ] **VectorSearchTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/vector-search-tool.ts`
- [ ] **WebCrawlTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/web-crawl-tool.ts`
- [ ] **MyCustomTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/tool-template.ts`
- [ ] **CodeFormatterTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/code-formatter-tool.ts`
- [ ] **EditBubbleFlowTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/code-edit-tool.ts`
- [ ] **DataTransformerTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/data-transformer-tool.ts`
- [ ] **ImageProcessorTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/image-processor-tool.ts`
- [ ] **JSONValidatorTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/json-validator-tool.ts`
- [ ] **TextAnalyzerTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/text-analyzer-tool.ts`

### Workflow Bubbles
- [ ] **APIAggregatorWorkflow** (3 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/api-aggregator.workflow.ts`
- [ ] **MonitoringAlertWorkflow** (3 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/monitoring-alert.workflow.ts`
- [ ] **ScheduledTaskWorkflow** (3 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/scheduled-task.workflow.ts`
- [ ] **DatabaseAnalyzerWorkflowBubble** (2 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/database-analyzer.workflow.ts`
- [ ] **GenerateDocumentWorkflow** (2 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/generate-document.workflow.ts`
- [ ] **ParseDocumentWorkflow** (2 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/parse-document.workflow.ts`
- [ ] **PDFOcrWorkflow** (2 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/pdf-ocr.workflow.ts`
- [ ] **SlackDataAssistantWorkflow** (2 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/slack-data-assistant.workflow.ts`
- [ ] **WebhookRepeaterWorkflow** (2 ops) - `packages/bubble-core/src/bubbles/workflow-bubble/webhook-repeater.workflow.ts`

### Templates
- [ ] **ChatWithYourDatabaseFlow** (2 ops) - `apps/bubble-studio/src/components/templates/template_codes/databaseMetricsAssistant.ts`
- [ ] **GmailLabelingFlow** (2 ops) - `apps/bubble-studio/src/components/templates/template_codes/gmailLabeling.ts`
- [ ] **GmailReplyAssistantFlow** (2 ops) - `apps/bubble-studio/src/components/templates/template_codes/gmailReplyAssistant.ts`
- [ ] **GithubContributorScraperFlow** (3 ops) - `apps/bubble-studio/src/components/templates/template_codes/githubScraper.ts`

---

## ✅ ALREADY TESTED (37 bubbles)

### Service Bubbles (23)
- [x] **AirtableBubble** (15 ops) - `packages/bubble-core/src/bubbles/service-bubble/airtable.ts`
- [x] **AirtableWrapperBubble** (61 ops) - `packages/bubble-core/src/bubbles/service-bubble/airtable-wrapper.ts`
- [x] **AIAgentBubble** (8 ops) - `packages/bubble-core/src/bubbles/service-bubble/ai-agent.ts`
- [x] **ApifyBubble** (59 ops) - `packages/bubble-core/src/bubbles/service-bubble/apify-bubble.ts`
- [x] **ApifyBubble** (8 ops) - `packages/bubble-core/src/bubbles/service-bubble/apify/apify.ts`
- [x] **ElevenLabsBubble** (18 ops) - `packages/bubble-core/src/bubbles/service-bubble/eleven-labs.ts`
- [x] **FirecrawlBubble** (12 ops) - `packages/bubble-core/src/bubbles/service-bubble/firecrawl.ts`
- [x] **FollowUpBossBubble** (89 ops) - `packages/bubble-core/src/bubbles/service-bubble/followupboss.ts`
- [x] **GithubBubble** (21 ops) - `packages/bubble-core/src/bubbles/service-bubble/github.ts`
- [x] **GoogleDriveBubble** (43 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-drive-bubble.ts`
- [x] **GoogleSheetsBubble** (87 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-sheets-bubble.ts`
- [x] **GoogleSheetsBubble** (14 ops) - `packages/bubble-core/src/bubbles/service-bubble/google-sheets/google-sheets.ts`
- [x] **HelloWorldBubble** (3 ops) - `packages/bubble-core/src/bubbles/service-bubble/hello-world.ts`
- [x] **HttpBubble** (6 ops) - `packages/bubble-core/src/bubbles/service-bubble/http-bubble.ts`
- [x] **HttpBubble** (3 ops) - `packages/bubble-core/src/bubbles/service-bubble/http.ts`
- [x] **NotionBubble** (29 ops) - `packages/bubble-core/src/bubbles/service-bubble/notion/notion.ts`
- [x] **PostgreSQLBubble** (6 ops) - `packages/bubble-core/src/bubbles/service-bubble/postgresql.ts`
- [x] **ResendBubble** (9 ops) - `packages/bubble-core/src/bubbles/service-bubble/resend.ts`
- [x] **SlackBubble** (18 ops) - `packages/bubble-core/src/bubbles/service-bubble/slack.ts`
- [x] **StorageBubble** (13 ops) - `packages/bubble-core/src/bubbles/service-bubble/storage.ts`
- [x] **StripeBubble** (67 ops) - `packages/bubble-core/src/bubbles/service-bubble/stripe-bubble.ts`
- [x] **TelegramBubble** (23 ops) - `packages/bubble-core/src/bubbles/service-bubble/telegram.ts`
- [x] **WebhookBubble** (61 ops) - `packages/bubble-core/src/bubbles/service-bubble/webhook-bubble.ts`

### Tool Bubbles (14)
- [x] **BubbleFlowValidationTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/bubbleflow-validation-tool.ts`
- [x] **ChartJSTool** (4 ops) - `packages/bubble-core/src/bubbles/tool-bubble/chart-js-tool.ts`
- [x] **GetBubbleDetailsTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/get-bubble-details-tool.ts`
- [x] **GoogleMapsTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/google-maps-tool.ts`
- [x] **InstagramTool** (6 ops) - `packages/bubble-core/src/bubbles/tool-bubble/instagram-tool.ts`
- [x] **LinkedInTool** (5 ops) - `packages/bubble-core/src/bubbles/tool-bubble/linkedin-tool.ts`
- [x] **ListBubblesTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/list-bubbles-tool.ts`
- [x] **ResearchAgentTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/research-agent-tool.ts`
- [x] **SQLQueryTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/sql-query-tool.ts`
- [x] **TikTokTool** (3 ops) - `packages/bubble-core/src/bubbles/tool-bubble/tiktok-tool.ts`
- [x] **TwitterTool** (8 ops) - `packages/bubble-core/src/bubbles/tool-bubble/twitter-tool.ts`
- [x] **WebExtractTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/web-extract-tool.ts`
- [x] **WebScrapeTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/web-scrape-tool.ts`
- [x] **WebSearchTool** (2 ops) - `packages/bubble-core/src/bubbles/tool-bubble/web-search-tool.ts`

### Workflow Bubbles (0)
- *None tested yet*

### Templates (0)
- *None tested yet*

---

## 📈 TESTING STANDARDS CHECKLIST

For each bubble test, ensure:

### Unit Tests
- [ ] All operations covered
- [ ] Edge cases tested
- [ ] Error handling verified
- [ ] Input validation tested
- [ ] Output schemas validated

### Integration Tests
- [ ] External API calls mocked
- [ ] Authentication flows tested
- [ ] Rate limiting handled
- [ ] Retry logic verified
- [ ] Circuit breaker tested

### Performance Tests
- [ ] Response time measured
- [ ] Memory usage checked
- [ ] Concurrent requests tested
- [ ] Large data sets handled

### Security Tests
- [ ] Input sanitization verified
- [ ] Credential handling secure
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] CSRF protection

---

## 🎯 MILESTONES

- [ ] **Milestone 1**: Complete Critical Priority (20 bubbles)
- [ ] **Milestone 2**: Complete High Priority (11 bubbles)
- [ ] **Milestone 3**: Complete Medium Priority (57 bubbles)
- [ ] **Milestone 4**: Complete Low Priority (14 bubbles)
- [ ] **Milestone 5**: 100% Test Coverage (139 bubbles)

---

**Progress**: 37/139 tested (26.6%)
**Remaining**: 102 bubbles
**Target Date**: [Set your target]

---

## 📝 NOTES

- Update checkboxes as tests are completed
- Add test coverage percentage for each bubble
- Document any issues or special cases
- Track time spent on each bubble
- Update this file weekly

---

**Last Updated**: 2026-01-19
**Next Review**: [Set schedule]
