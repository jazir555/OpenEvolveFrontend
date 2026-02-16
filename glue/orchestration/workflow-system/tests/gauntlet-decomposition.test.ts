/**
 * Integration Tests for Gauntlet and Decomposition Workflows
 *
 * Tests the complete integration of gauntlet execution and decomposition workflows
 * following CLAUDE.md principles (Law of Runtime Truth).
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { getWorkflowOrchestrator } from '../workflow-orchestrator';
import { getPluginRegistry } from '../plugin-registry';
import {
  GAUNTLET_EXECUTION_WORKFLOW,
  DECOMPOSITION_EXECUTION_WORKFLOW,
  GAUNTLET_DECOMPOSITION_WORKFLOW,
  getAllWorkflowTemplates,
  getWorkflowTemplate
} from '../workflow-templates';
import type { WorkflowDefinition } from '../workflow-orchestrator';

describe('Gauntlet and Decomposition Workflows Integration', () => {
  let orchestrator: ReturnType<typeof getWorkflowOrchestrator>;
  let registry: ReturnType<typeof getPluginRegistry>;

  beforeAll(async () => {
    // Initialize the orchestrator and registry
    registry = getPluginRegistry();
    orchestrator = getWorkflowOrchestrator();

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  afterAll(async () => {
    // Cleanup
    if (orchestrator) {
      await orchestrator.destroy();
    }
  });

  describe('Workflow Templates', () => {
    it('should have all 8 workflow templates registered', () => {
      const templates = getAllWorkflowTemplates();
      expect(templates).toHaveLength(8);
      expect(templates.map(t => t.id)).toContain('gauntlet-execution');
      expect(templates.map(t => t.id)).toContain('decomposition-execution');
      expect(templates.map(t => t.id)).toContain('gauntlet-decomposition-integrated');
    });

    it('should retrieve gauntlet execution workflow template', () => {
      const template = getWorkflowTemplate('gauntlet-execution');
      expect(template).toBeDefined();
      expect(template?.id).toBe('gauntlet-execution');
      expect(template?.name).toBe('Gauntlet Execution');
      expect(template?.steps).toHaveLength(7);
    });

    it('should retrieve decomposition execution workflow template', () => {
      const template = getWorkflowTemplate('decomposition-execution');
      expect(template).toBeDefined();
      expect(template?.id).toBe('decomposition-execution');
      expect(template?.name).toBe('Decomposition Execution');
      expect(template?.steps).toHaveLength(8);
    });

    it('should retrieve integrated gauntlet-decomposition workflow template', () => {
      const template = getWorkflowTemplate('gauntlet-decomposition-integrated');
      expect(template).toBeDefined();
      expect(template?.id).toBe('gauntlet-decomposition-integrated');
      expect(template?.name).toBe('Gauntlet + Decomposition Integration');
      expect(template?.steps).toHaveLength(8);
    });
  });

  describe('Gauntlet Execution Workflow', () => {
    it('should have correct step dependencies', () => {
      const template = getWorkflowTemplate('gauntlet-execution') as WorkflowDefinition;
      expect(template?.steps).toBeDefined();

      // Check that execute-rounds depends on initialize-gauntlet and prepare-content
      const executeRounds = template?.steps.find(s => s.id === 'execute-rounds');
      expect(executeRounds?.dependsOn).toContain('initialize-gauntlet');
      expect(executeRounds?.dependsOn).toContain('prepare-content');

      // Check that formal-verification depends on execute-rounds
      const formalVerification = template?.steps.find(s => s.id === 'formal-verification');
      expect(formalVerification?.dependsOn).toContain('execute-rounds');
    });

    it('should have conditional verification steps', () => {
      const template = getWorkflowTemplate('gauntlet-execution') as WorkflowDefinition;

      // Check that formal-verification has a condition
      const formalVerification = template?.steps.find(s => s.id === 'formal-verification');
      expect(formalVerification?.condition).toBeDefined();

      // Check that lean-verification has a condition
      const leanVerification = template?.steps.find(s => s.id === 'lean-verification');
      expect(leanVerification?.condition).toBeDefined();
    });

    it('should use correct plugins for each step', () => {
      const template = getWorkflowTemplate('gauntlet-execution') as WorkflowDefinition;

      const initStep = template?.steps.find(s => s.id === 'initialize-gauntlet');
      expect(initStep?.plugin).toBe('openevolve');

      const executeStep = template?.steps.find(s => s.id === 'execute-rounds');
      expect(executeStep?.plugin).toBe('openevolve');

      const storeStep = template?.steps.find(s => s.id === 'store-results');
      expect(storeStep?.plugin).toBe('ragbits');
    });
  });

  describe('Decomposition Execution Workflow', () => {
    it('should have correct step dependencies', () => {
      const template = getWorkflowTemplate('decomposition-execution') as WorkflowDefinition;
      expect(template?.steps).toBeDefined();

      // Check workflow execution step dependencies
      const executeSubProblems = template?.steps.find(s => s.id === 'execute-sub-problems');
      expect(executeSubProblems?.dependsOn).toContain('get-dependency-graph');

      // Check reassembly dependencies
      const reassemble = template?.steps.find(s => s.id === 'reassemble-solution');
      expect(reassemble?.dependsOn).toContain('execute-sub-problems');
      expect(reassemble?.dependsOn).toContain('search-knowledge');
    });

    it('should use ROMA analysis for problem decomposition', () => {
      const template = getWorkflowTemplate('decomposition-execution') as WorkflowDefinition;

      const analyzeStep = template?.steps.find(s => s.id === 'analyze-problem');
      expect(analyzeStep?.action).toBe('bubblelabsRomaAnalyze');
      expect(analyzeStep?.plugin).toBe('openevolve');
    });

    it('should validate final solution through gauntlets', () => {
      const template = getWorkflowTemplate('decomposition-execution') as WorkflowDefinition;

      const validateStep = template?.steps.find(s => s.id === 'validate-solution');
      expect(validateStep?.action).toBe('startEvolutionRun');
      expect(validateStep?.plugin).toBe('openevolve');
    });
  });

  describe('Integrated Gauntlet-Decomposition Workflow', () => {
    it('should have complete end-to-end flow', () => {
      const template = getWorkflowTemplate('gauntlet-decomposition-integrated') as WorkflowDefinition;
      expect(template?.steps).toHaveLength(8);

      // Verify the flow: analyze -> create -> get plan -> execute -> get results -> validate -> verify -> track
      const stepIds = template?.steps.map(s => s.id);
      expect(stepIds).toEqual([
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

    it('should have conditional formal verification', () => {
      const template = getWorkflowTemplate('gauntlet-decomposition-integrated') as WorkflowDefinition;

      const formalVerification = template?.steps.find(s => s.id === 'formal-verification');
      expect(formalVerification?.condition).toBeDefined();
    });
  });

  describe('Workflow Template Categories', () => {
    it('should categorize gauntlet workflows correctly', () => {
      const { getWorkflowTemplatesByCategory } = require('../workflow-templates');
      const gauntletTemplates = getWorkflowTemplatesByCategory('gauntlet');
      expect(gauntletTemplates).toHaveLength(1);
      expect(gauntletTemplates[0].id).toBe('gauntlet-execution');
    });

    it('should categorize decomposition workflows correctly', () => {
      const { getWorkflowTemplatesByCategory } = require('../workflow-templates');
      const decompositionTemplates = getWorkflowTemplatesByCategory('decomposition');
      expect(decompositionTemplates).toHaveLength(1);
      expect(decompositionTemplates[0].id).toBe('decomposition-execution');
    });

    it('should categorize integrated workflows correctly', () => {
      const { getWorkflowTemplatesByCategory } = require('../workflow-templates');
      const integratedTemplates = getWorkflowTemplatesByCategory('integrated');
      expect(integratedTemplates).toHaveLength(1);
      expect(integratedTemplates[0].id).toBe('gauntlet-decomposition-integrated');
    });
  });

  describe('Error Handling', () => {
    it('should have appropriate error handling strategies', () => {
      const gauntletTemplate = getWorkflowTemplate('gauntlet-execution');
      const decompositionTemplate = getWorkflowTemplate('decomposition-execution');
      const integratedTemplate = getWorkflowTemplate('gauntlet-decomposition-integrated');

      expect(gauntletTemplate?.onError).toBe('continue');
      expect(decompositionTemplate?.onError).toBe('continue');
      expect(integratedTemplate?.onError).toBe('continue');
    });

    it('should have reasonable retry limits', () => {
      const gauntletTemplate = getWorkflowTemplate('gauntlet-execution');
      const decompositionTemplate = getWorkflowTemplate('decomposition-execution');
      const integratedTemplate = getWorkflowTemplate('gauntlet-decomposition-integrated');

      expect(gauntletTemplate?.maxRetries).toBeGreaterThan(0);
      expect(decompositionTemplate?.maxRetries).toBeGreaterThan(0);
      expect(integratedTemplate?.maxRetries).toBeGreaterThan(0);
    });
  });

  describe('Plugin Registry Integration', () => {
    it('should have OpenEvolve plugin registered', () => {
      const openevolvePlugin = registry.getPlugin('openevolve');
      expect(openevolvePlugin).toBeDefined();
    });

    it('should have RAGBits plugin registered', () => {
      const ragbitsPlugin = registry.getPlugin('ragbits');
      expect(ragbitsPlugin).toBeDefined();
    });

    it('should have Datapizza plugin registered', () => {
      const datapizzaPlugin = registry.getPlugin('datapizza');
      expect(datapizzaPlugin).toBeDefined();
    });

    it('should have all required capabilities', () => {
      const openevolvePlugin = registry.getPlugin('openevolve');
      const capabilities = openevolvePlugin?.capabilities;

      expect(capabilities?.verification).toBe(true);
      expect(capabilities?.analysis).toBe(true);
    });
  });
});
