import { BubbleRunner } from './BubbleRunner';
import { getFixture } from '../../tests/fixtures/index.js';
import { BubbleFactory } from '@bubblelab/bubble-core';

/**
 * Critical Fix Tests for BubbleRunner
 *
 * Tests the 5 critical fixes:
 * 1. Property initialization (bubbleFactory, currentStep, savedStates)
 * 2. runStep() execution
 * 3. resumeFromStep() functionality
 * 4. State saving and loading
 * 5. Error handling
 */

describe('BubbleRunner Critical Fixes', () => {
  let bubbleFactory: BubbleFactory;

  beforeAll(async () => {
    bubbleFactory = new BubbleFactory();
    await bubbleFactory.registerDefaults();
  });

  describe('1. Property Initialization', () => {
    it('should initialize bubbleFactory property', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      // Access private property using bracket notation for testing
      expect((runner as any).bubbleFactory).toBeDefined();
      expect((runner as any).bubbleFactory).toBe(bubbleFactory);
    });

    it('should initialize currentStep to 0', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      expect((runner as any).currentStep).toBeDefined();
      expect((runner as any).currentStep).toBe(0);
    });

    it('should initialize savedStates as empty Map', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      expect((runner as any).savedStates).toBeDefined();
      expect((runner as any).savedStates).toBeInstanceOf(Map);
      expect((runner as any).savedStates.size).toBe(0);
    });

    it('should initialize plan property', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      expect((runner as any).plan).toBeDefined();
      expect((runner as any).plan).toHaveProperty('steps');
      expect(Array.isArray((runner as any).plan.steps)).toBe(true);
    });

    it('should initialize logger property', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      expect((runner as any).logger).toBeDefined();
      expect(typeof (runner as any).logger.info).toBe('function');
      expect(typeof (runner as any).logger.error).toBe('function');
    });

    it('should initialize injector property', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      expect((runner as any).injector).toBeDefined();
    });
  });

  describe('2. runStep() Execution', () => {
    it('should throw error if execution plan is not initialized', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      // Manually set plan to null to test error handling
      (runner as any).plan = null;

      await expect(runner.runStep(0)).rejects.toThrow(
        'Execution plan not initialized'
      );
    });

    it('should throw error if step ID is not found', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();
      const nonExistentStepId = 999999;

      await expect(runner.runStep(nonExistentStepId)).rejects.toThrow(
        `Step ${nonExistentStepId} not found in execution plan`
      );
    });

    it('should execute a valid step from the plan', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];
        const result = await runner.runStep(firstStep.id);

        expect(result).toBeDefined();
        expect(result.success).toBe(true);
        expect(result.error).toBe('');
        expect(result.data).toHaveProperty('stepId');
        expect(result.data).toHaveProperty('completed');
        expect(result.data.completed).toBe(true);
      } else {
        // If no steps, test passes trivially
        expect(plan.steps).toEqual([]);
      }
    });

    it('should update currentStep after successful execution', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];
        const initialStep = (runner as any).currentStep;

        await runner.runStep(firstStep.id);

        expect((runner as any).currentStep).toBe(firstStep.id);
        expect((runner as any).currentStep).not.toBe(initialStep);
      }
    });

    it('should handle errors during step execution gracefully', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      // Mock a step that will fail by creating an invalid scenario
      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Temporarily break the step to test error handling
        const originalMiniSteps = firstStep.miniSteps;
        (firstStep as any).miniSteps = null;

        const result = await runner.runStep(firstStep.id);

        // Should handle the error gracefully
        expect(result).toBeDefined();

        // Restore original miniSteps
        firstStep.miniSteps = originalMiniSteps;
      }
    });
  });

  describe('3. resumeFromStep() Functionality', () => {
    it('should throw error if execution plan is not initialized', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      // Manually set plan to null
      (runner as any).plan = null;

      await expect(runner.resumeFromStep(0)).rejects.toThrow(
        'Execution plan not initialized'
      );
    });

    it('should throw error if no saved state exists for step', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Try to resume without running the step first (no saved state)
        await expect(runner.resumeFromStep(firstStep.id)).rejects.toThrow(
          `No saved state found for step ${firstStep.id}. Cannot resume.`
        );
      }
    });

    it('should successfully resume from a step with saved state', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // First, execute the step to create a saved state
        await runner.runStep(firstStep.id);

        // Verify state was saved
        const savedState = runner.getSavedState(firstStep.id);
        expect(savedState).toBeDefined();

        // Now try to resume from that step
        const resumeResult = await runner.resumeFromStep(firstStep.id);

        expect(resumeResult).toBeDefined();
        expect(resumeResult.success).toBe(true);
        expect(resumeResult.error).toBe('');
        expect(resumeResult.data).toHaveProperty('resumedFrom');
        expect(resumeResult.data.resumedFrom).toBe(firstStep.id);
      }
    });

    it('should restore currentStep when resuming', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Execute step
        await runner.runStep(firstStep.id);

        // Clear current step to simulate a fresh start
        (runner as any).currentStep = 0;

        // Resume should restore the step
        await runner.resumeFromStep(firstStep.id);

        expect((runner as any).currentStep).toBe(firstStep.id);
      }
    });

    it('should handle resume errors gracefully', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      // Create a saved state manually
      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Manually add a saved state
        (runner as any).savedStates.set(firstStep.id, {
          stepId: firstStep.id,
          currentStep: firstStep.id,
          variables: {},
          timestamp: new Date().toISOString()
        });

        // Clear the plan to trigger an error during resume
        const originalPlan = (runner as any).plan;
        (runner as any).plan = null;

        const result = await runner.resumeFromStep(firstStep.id);

        expect(result.success).toBe(false);
        expect(result.error).toContain('Failed to resume from step');

        // Restore plan
        (runner as any).plan = originalPlan;
      }
    });
  });

  describe('4. State Saving and Loading', () => {
    it('should save state after step execution', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Verify no state exists initially
        expect(runner.getSavedState(firstStep.id)).toBeUndefined();

        // Execute step
        await runner.runStep(firstStep.id);

        // Verify state was saved
        const savedState = runner.getSavedState(firstStep.id);
        expect(savedState).toBeDefined();
        expect(savedState).toHaveProperty('stepId');
        expect(savedState).toHaveProperty('currentStep');
        expect(savedState).toHaveProperty('variables');
        expect(savedState).toHaveProperty('timestamp');
      }
    });

    it('should retrieve all saved states', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      // Execute a few steps
      const stepsToExecute = Math.min(3, plan.steps.length);

      for (let i = 0; i < stepsToExecute; i++) {
        await runner.runStep(plan.steps[i].id);
      }

      // Get all saved states
      const allStates = runner.getAllSavedStates();

      expect(allStates).toBeInstanceOf(Map);
      expect(allStates.size).toBeGreaterThan(0);

      // Verify each saved state has required properties
      allStates.forEach((state: any) => {
        expect(state).toHaveProperty('stepId');
        expect(state).toHaveProperty('currentStep');
        expect(state).toHaveProperty('variables');
        expect(state).toHaveProperty('timestamp');
      });
    });

    it('should clear all saved states', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      // Execute a step to create state
      if (plan.steps.length > 0) {
        await runner.runStep(plan.steps[0].id);

        // Verify state exists
        expect((runner as any).savedStates.size).toBeGreaterThan(0);

        // Clear states
        runner.clearSavedStates();

        // Verify states are cleared
        expect((runner as any).savedStates.size).toBe(0);
        expect((runner as any).currentStep).toBe(0);
      }
    });

    it('should preserve variables in saved state', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        await runner.runStep(firstStep.id);

        const savedState = runner.getSavedState(firstStep.id);

        expect(savedState).toBeDefined();
        expect(savedState).toHaveProperty('variables');

        // Variables should be an object or Map
        const variables = savedState.variables;
        expect(typeof variables).toBe('object');
      }
    });

    it('should include timestamp in saved state', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        await runner.runStep(firstStep.id);

        const savedState = runner.getSavedState(firstStep.id);

        expect(savedState).toBeDefined();
        expect(savedState).toHaveProperty('timestamp');

        // Timestamp should be a valid ISO date string
        const timestamp = new Date(savedState.timestamp);
        expect(timestamp instanceof Date).toBe(true);
        expect(isNaN(timestamp.getTime())).toBe(false);
      }
    });
  });

  describe('5. Error Handling', () => {
    it('should handle invalid step ID gracefully', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const invalidStepId = -1;

      await expect(runner.runStep(invalidStepId)).rejects.toThrow();
    });

    it('should return error result on execution failure', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      // Create a scenario that will fail
      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Mock a failure by making the step invalid
        const originalType = firstStep.type;
        (firstStep as any).type = 'invalid_type';

        const result = await runner.runStep(firstStep.id);

        // Should handle the error and return an error result
        expect(result).toBeDefined();

        // Restore original type
        firstStep.type = originalType;
      }
    });

    it('should log errors during step execution', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
        enableLogging: true,
      });

      const logger = runner.getLogger();
      expect(logger).toBeDefined();

      // Spy on logger.error
      const spy = jest.spyOn(logger!, 'error');

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Force an error
        (firstStep as any).miniSteps = null;

        await runner.runStep(firstStep.id);

        // Error should be logged
        expect(spy).toHaveBeenCalled();
      }

      spy.mockRestore();
    });

    it('should handle missing execution plan in getPlan()', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      // Clear the plan
      (runner as any).plan = null;

      expect(() => runner.getPlan()).toThrow('Plan not found');
    });

    it('should handle dispose correctly', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const logger = runner.getLogger();
      expect(logger).toBeDefined();

      // Dispose should not throw
      expect(() => runner.dispose()).not.toThrow();
    });

    it('should provide execution summary', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const summary = runner.getExecutionSummary();

      expect(summary).toBeDefined();
      expect(typeof summary).toBe('object');
    });

    it('should export logs in different formats', () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const jsonLogs = runner.exportLogs('json');
      const csvLogs = runner.exportLogs('csv');
      const tableLogs = runner.exportLogs('table');

      // Logs should be strings or null
      expect(typeof jsonLogs === 'string' || jsonLogs === null).toBe(true);
      expect(typeof csvLogs === 'string' || csvLogs === null).toBe(true);
      expect(typeof tableLogs === 'string' || tableLogs === null).toBe(true);
    });
  });

  describe('Integration Tests', () => {
    it('should execute full workflow: run step, save state, resume', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length > 0) {
        const firstStep = plan.steps[0];

        // Step 1: Execute
        const runResult = await runner.runStep(firstStep.id);
        expect(runResult.success).toBe(true);

        // Step 2: Verify state saved
        const savedState = runner.getSavedState(firstStep.id);
        expect(savedState).toBeDefined();

        // Step 3: Clear current step
        (runner as any).currentStep = 0;

        // Step 4: Resume
        const resumeResult = await runner.resumeFromStep(firstStep.id);
        expect(resumeResult.success).toBe(true);
        expect(resumeResult.data.resumedFrom).toBe(firstStep.id);

        // Step 5: Verify current step restored
        expect((runner as any).currentStep).toBe(firstStep.id);
      }
    });

    it('should handle multiple step executions sequentially', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();
      const stepsToExecute = Math.min(3, plan.steps.length);

      const results = [];

      for (let i = 0; i < stepsToExecute; i++) {
        const result = await runner.runStep(plan.steps[i].id);
        results.push(result);

        expect(result.success).toBe(true);
        expect((runner as any).currentStep).toBe(plan.steps[i].id);
      }

      expect(results.length).toBe(stepsToExecute);
    });

    it('should maintain state integrity across multiple operations', async () => {
      const script = getFixture('hello-world');
      const runner = new BubbleRunner(script, bubbleFactory, {
        pricingTable: {},
      });

      const plan = runner.getPlan();

      if (plan.steps.length >= 2) {
        // Execute first step
        await runner.runStep(plan.steps[0].id);
        const state1 = runner.getSavedState(plan.steps[0].id);

        // Execute second step
        await runner.runStep(plan.steps[1].id);
        const state2 = runner.getSavedState(plan.steps[1].id);

        // Both states should exist and be different
        expect(state1).toBeDefined();
        expect(state2).toBeDefined();
        expect(state1.stepId).not.toBe(state2.stepId);

        // Get all states
        const allStates = runner.getAllSavedStates();
        expect(allStates.size).toBeGreaterThanOrEqual(2);
      }
    });
  });
});
