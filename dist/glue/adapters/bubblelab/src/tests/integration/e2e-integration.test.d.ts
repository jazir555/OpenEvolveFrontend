/**
 * End-to-End Integration Test
 *
 * Tests the complete OpenEvolve-BubbleLab integration including:
 * - Plugin registration and initialization
 * - Workflow execution across plugins
 * - Event bus integration
 * - Monitoring and telemetry
 */
export {};
/**
 * Manual Integration Test Checklist
 *
 * Run these steps manually to verify the integration:
 *
 * 1. Plugin Registration
 *    [ ] RAGBits plugin loads successfully
 *    [ ] Datapizza plugin loads successfully
 *    [ ] OpenEvolve API adapter loads
 *    [ ] All plugins appear in registry
 *
 * 2. Plugin Initialization
 *    [ ] Plugins initialize without errors
 *    [ ] Health checks pass for available plugins
 *    [ ] Plugin capabilities are detected
 *
 * 3. Workflow Execution
 *    [ ] Can select workflow template
 *    [ ] Can input workflow parameters
 *    [ ] Workflow executes end-to-end
 *    [ ] Step results are displayed
 *    [ ] Final output is shown
 *
 * 4. Event Bus
 *    [ ] Events are published on workflow start
 *    [ ] Events are published on workflow completion
 *    [ ] Cross-plugin handlers work
 *
 * 5. Monitoring
 *    [ ] Workflow metrics are recorded
 *    [ ] Step metrics are recorded
 *    [ ] Aggregate statistics are available
 *    [ ] Can export metrics
 *
 * 6. UI Integration
 *    [ ] Workflow tab appears in navigation
 *    [ ] Workflow templates display correctly
 *    [ ] Execution shows real-time updates
 *    [ ] History is tracked and displayed
 *
 * 7. Error Handling
 *    [ ] Invalid workflows show validation errors
 *    [ ] Plugin failures don't crash the app
 *    [ ] Retry logic works when configured
 *    [ ] Circuit breakers prevent cascading failures
 */
//# sourceMappingURL=e2e-integration.test.d.ts.map