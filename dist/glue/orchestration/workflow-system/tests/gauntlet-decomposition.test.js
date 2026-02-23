"use strict";
/**
 * Integration Tests for Gauntlet and Decomposition Workflows
 *
 * Tests the complete integration of gauntlet execution and decomposition workflows
 * following CLAUDE.md principles (Law of Runtime Truth).
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const workflow_orchestrator_1 = require("../workflow-orchestrator");
const plugin_registry_1 = require("../plugin-registry");
const workflow_templates_1 = require("../workflow-templates");
(0, globals_1.describe)('Gauntlet and Decomposition Workflows Integration', () => {
    let orchestrator;
    let registry;
    (0, globals_1.beforeAll)(async () => {
        // Initialize the orchestrator and registry
        registry = (0, plugin_registry_1.getPluginRegistry)();
        orchestrator = (0, workflow_orchestrator_1.getWorkflowOrchestrator)();
        // Wait for initialization
        await new Promise(resolve => setTimeout(resolve, 1000));
    });
    (0, globals_1.afterAll)(async () => {
        // Cleanup
        if (orchestrator) {
            await orchestrator.destroy();
        }
    });
    (0, globals_1.describe)('Workflow Templates', () => {
        (0, globals_1.it)('should have all 8 workflow templates registered', () => {
            const templates = (0, workflow_templates_1.getAllWorkflowTemplates)();
            (0, globals_1.expect)(templates).toHaveLength(8);
            (0, globals_1.expect)(templates.map(t => t.id)).toContain('gauntlet-execution');
            (0, globals_1.expect)(templates.map(t => t.id)).toContain('decomposition-execution');
            (0, globals_1.expect)(templates.map(t => t.id)).toContain('gauntlet-decomposition-integrated');
        });
        (0, globals_1.it)('should retrieve gauntlet execution workflow template', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-execution');
            (0, globals_1.expect)(template).toBeDefined();
            (0, globals_1.expect)(template?.id).toBe('gauntlet-execution');
            (0, globals_1.expect)(template?.name).toBe('Gauntlet Execution');
            (0, globals_1.expect)(template?.steps).toHaveLength(7);
        });
        (0, globals_1.it)('should retrieve decomposition execution workflow template', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('decomposition-execution');
            (0, globals_1.expect)(template).toBeDefined();
            (0, globals_1.expect)(template?.id).toBe('decomposition-execution');
            (0, globals_1.expect)(template?.name).toBe('Decomposition Execution');
            (0, globals_1.expect)(template?.steps).toHaveLength(8);
        });
        (0, globals_1.it)('should retrieve integrated gauntlet-decomposition workflow template', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-decomposition-integrated');
            (0, globals_1.expect)(template).toBeDefined();
            (0, globals_1.expect)(template?.id).toBe('gauntlet-decomposition-integrated');
            (0, globals_1.expect)(template?.name).toBe('Gauntlet + Decomposition Integration');
            (0, globals_1.expect)(template?.steps).toHaveLength(8);
        });
    });
    (0, globals_1.describe)('Gauntlet Execution Workflow', () => {
        (0, globals_1.it)('should have correct step dependencies', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-execution');
            (0, globals_1.expect)(template?.steps).toBeDefined();
            // Check that execute-rounds depends on initialize-gauntlet and prepare-content
            const executeRounds = template?.steps.find(s => s.id === 'execute-rounds');
            (0, globals_1.expect)(executeRounds?.dependsOn).toContain('initialize-gauntlet');
            (0, globals_1.expect)(executeRounds?.dependsOn).toContain('prepare-content');
            // Check that formal-verification depends on execute-rounds
            const formalVerification = template?.steps.find(s => s.id === 'formal-verification');
            (0, globals_1.expect)(formalVerification?.dependsOn).toContain('execute-rounds');
        });
        (0, globals_1.it)('should have conditional verification steps', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-execution');
            // Check that formal-verification has a condition
            const formalVerification = template?.steps.find(s => s.id === 'formal-verification');
            (0, globals_1.expect)(formalVerification?.condition).toBeDefined();
            // Check that lean-verification has a condition
            const leanVerification = template?.steps.find(s => s.id === 'lean-verification');
            (0, globals_1.expect)(leanVerification?.condition).toBeDefined();
        });
        (0, globals_1.it)('should use correct plugins for each step', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-execution');
            const initStep = template?.steps.find(s => s.id === 'initialize-gauntlet');
            (0, globals_1.expect)(initStep?.plugin).toBe('openevolve');
            const executeStep = template?.steps.find(s => s.id === 'execute-rounds');
            (0, globals_1.expect)(executeStep?.plugin).toBe('openevolve');
            const storeStep = template?.steps.find(s => s.id === 'store-results');
            (0, globals_1.expect)(storeStep?.plugin).toBe('ragbits');
        });
    });
    (0, globals_1.describe)('Decomposition Execution Workflow', () => {
        (0, globals_1.it)('should have correct step dependencies', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('decomposition-execution');
            (0, globals_1.expect)(template?.steps).toBeDefined();
            // Check workflow execution step dependencies
            const executeSubProblems = template?.steps.find(s => s.id === 'execute-sub-problems');
            (0, globals_1.expect)(executeSubProblems?.dependsOn).toContain('get-dependency-graph');
            // Check reassembly dependencies
            const reassemble = template?.steps.find(s => s.id === 'reassemble-solution');
            (0, globals_1.expect)(reassemble?.dependsOn).toContain('execute-sub-problems');
            (0, globals_1.expect)(reassemble?.dependsOn).toContain('search-knowledge');
        });
        (0, globals_1.it)('should use ROMA analysis for problem decomposition', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('decomposition-execution');
            const analyzeStep = template?.steps.find(s => s.id === 'analyze-problem');
            (0, globals_1.expect)(analyzeStep?.action).toBe('bubblelabsRomaAnalyze');
            (0, globals_1.expect)(analyzeStep?.plugin).toBe('openevolve');
        });
        (0, globals_1.it)('should validate final solution through gauntlets', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('decomposition-execution');
            const validateStep = template?.steps.find(s => s.id === 'validate-solution');
            (0, globals_1.expect)(validateStep?.action).toBe('startEvolutionRun');
            (0, globals_1.expect)(validateStep?.plugin).toBe('openevolve');
        });
    });
    (0, globals_1.describe)('Integrated Gauntlet-Decomposition Workflow', () => {
        (0, globals_1.it)('should have complete end-to-end flow', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-decomposition-integrated');
            (0, globals_1.expect)(template?.steps).toHaveLength(8);
            // Verify the flow: analyze -> create -> get plan -> execute -> get results -> validate -> verify -> track
            const stepIds = template?.steps.map(s => s.id);
            (0, globals_1.expect)(stepIds).toEqual([
                'analyze-and-decompose',
                'create-workflow',
                'get-workflow-plan',
                'execute-sub-problem-gauntlets',
                'get-workflow-results',
                'final-validation',
                'formal-verification',
                'store-and-track'
            ]);
        });
        (0, globals_1.it)('should have conditional formal verification', () => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-decomposition-integrated');
            const formalVerification = template?.steps.find(s => s.id === 'formal-verification');
            (0, globals_1.expect)(formalVerification?.condition).toBeDefined();
        });
    });
    (0, globals_1.describe)('Workflow Template Categories', () => {
        (0, globals_1.it)('should categorize gauntlet workflows correctly', () => {
            const { getWorkflowTemplatesByCategory } = require('../workflow-templates');
            const gauntletTemplates = getWorkflowTemplatesByCategory('gauntlet');
            (0, globals_1.expect)(gauntletTemplates).toHaveLength(1);
            (0, globals_1.expect)(gauntletTemplates[0].id).toBe('gauntlet-execution');
        });
        (0, globals_1.it)('should categorize decomposition workflows correctly', () => {
            const { getWorkflowTemplatesByCategory } = require('../workflow-templates');
            const decompositionTemplates = getWorkflowTemplatesByCategory('decomposition');
            (0, globals_1.expect)(decompositionTemplates).toHaveLength(1);
            (0, globals_1.expect)(decompositionTemplates[0].id).toBe('decomposition-execution');
        });
        (0, globals_1.it)('should categorize integrated workflows correctly', () => {
            const { getWorkflowTemplatesByCategory } = require('../workflow-templates');
            const integratedTemplates = getWorkflowTemplatesByCategory('integrated');
            (0, globals_1.expect)(integratedTemplates).toHaveLength(1);
            (0, globals_1.expect)(integratedTemplates[0].id).toBe('gauntlet-decomposition-integrated');
        });
    });
    (0, globals_1.describe)('Error Handling', () => {
        (0, globals_1.it)('should have appropriate error handling strategies', () => {
            const gauntletTemplate = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-execution');
            const decompositionTemplate = (0, workflow_templates_1.getWorkflowTemplate)('decomposition-execution');
            const integratedTemplate = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-decomposition-integrated');
            (0, globals_1.expect)(gauntletTemplate?.onError).toBe('continue');
            (0, globals_1.expect)(decompositionTemplate?.onError).toBe('continue');
            (0, globals_1.expect)(integratedTemplate?.onError).toBe('continue');
        });
        (0, globals_1.it)('should have reasonable retry limits', () => {
            const gauntletTemplate = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-execution');
            const decompositionTemplate = (0, workflow_templates_1.getWorkflowTemplate)('decomposition-execution');
            const integratedTemplate = (0, workflow_templates_1.getWorkflowTemplate)('gauntlet-decomposition-integrated');
            (0, globals_1.expect)(gauntletTemplate?.maxRetries).toBeGreaterThan(0);
            (0, globals_1.expect)(decompositionTemplate?.maxRetries).toBeGreaterThan(0);
            (0, globals_1.expect)(integratedTemplate?.maxRetries).toBeGreaterThan(0);
        });
    });
    (0, globals_1.describe)('Plugin Registry Integration', () => {
        (0, globals_1.it)('should have OpenEvolve plugin registered', () => {
            const openevolvePlugin = registry.getPlugin('openevolve');
            (0, globals_1.expect)(openevolvePlugin).toBeDefined();
        });
        (0, globals_1.it)('should have RAGBits plugin registered', () => {
            const ragbitsPlugin = registry.getPlugin('ragbits');
            (0, globals_1.expect)(ragbitsPlugin).toBeDefined();
        });
        (0, globals_1.it)('should have Datapizza plugin registered', () => {
            const datapizzaPlugin = registry.getPlugin('datapizza');
            (0, globals_1.expect)(datapizzaPlugin).toBeDefined();
        });
        (0, globals_1.it)('should have all required capabilities', () => {
            const openevolvePlugin = registry.getPlugin('openevolve');
            const capabilities = openevolvePlugin?.capabilities;
            (0, globals_1.expect)(capabilities?.verification).toBe(true);
            (0, globals_1.expect)(capabilities?.analysis).toBe(true);
        });
    });
});
//# sourceMappingURL=gauntlet-decomposition.test.js.map