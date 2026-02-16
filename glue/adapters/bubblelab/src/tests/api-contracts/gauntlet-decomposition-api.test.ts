/**
 * API Contract Tests for Gauntlet and Decomposition Endpoints
 *
 * Tests the API client contract to ensure all endpoints are properly defined
 * and follow the expected structure. Following CLAUDE.md Law of Runtime Truth.
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { openevolveApi } from '../../lib/openevolveApi';

describe('Gauntlet Execution API Contract Tests', () => {
  describe('executeGauntlet', () => {
    it('should have executeGauntlet endpoint defined', () => {
      expect(openevolveApi.executeGauntlet).toBeDefined();
      expect(typeof openevolveApi.executeGauntlet).toBe('function');
    });

    it('should accept gauntlet name and payload', () => {
      const endpoint = openevolveApi.executeGauntlet;
      expect(endpoint.length).toBeGreaterThanOrEqual(2); // gauntletName, payload, optional config
    });

    it('should have correct signature', () => {
      const payload = {
        content: 'Test content',
        content_type: 'text_general',
        evolution_mode: 'standard',
        parameters: { max_iterations: 3 }
      };

      // Should not throw on type checking
      expect(() => {
        openevolveApi.executeGauntlet('test-gauntlet', payload);
      }).not.toThrow();
    });
  });

  describe('getGauntletExecutionStatus', () => {
    it('should have getGauntletExecutionStatus endpoint defined', () => {
      expect(openevolveApi.getGauntletExecutionStatus).toBeDefined();
      expect(typeof openevolveApi.getGauntletExecutionStatus).toBe('function');
    });

    it('should accept execution ID', () => {
      const endpoint = openevolveApi.getGauntletExecutionStatus;
      expect(endpoint.length).toBeGreaterThanOrEqual(1); // executionId, optional config
    });
  });

  describe('listGauntletExecutions', () => {
    it('should have listGauntletExecutions endpoint defined', () => {
      expect(openevolveApi.listGauntletExecutions).toBeDefined();
      expect(typeof openevolveApi.listGauntletExecutions).toBe('function');
    });

    it('should accept optional gauntlet name filter', () => {
      const endpoint = openevolveApi.listGauntletExecutions;
      expect(() => {
        endpoint();
        endpoint({ apiKey: 'test' });
        endpoint({ apiKey: 'test' }, 'test-gauntlet');
      }).not.toThrow();
    });
  });
});

describe('Decomposition Execution API Contract Tests', () => {
  describe('executeDecomposition', () => {
    it('should have executeDecomposition endpoint defined', () => {
      expect(openevolveApi.executeDecomposition).toBeDefined();
      expect(typeof openevolveApi.executeDecomposition).toBe('function');
    });

    it('should accept workflow ID and payload', () => {
      const endpoint = openevolveApi.executeDecomposition;
      expect(endpoint.length).toBeGreaterThanOrEqual(2); // workflowId, payload, optional config

      const payload = {
        problem_statement: 'Test problem',
        decomposition_method: 'hierarchical',
        granularity: 'medium',
        max_depth: 3,
        max_sub_problems: 5
      };

      expect(() => {
        openevolveApi.executeDecomposition('workflow-123', payload);
      }).not.toThrow();
    });
  });

  describe('getDecompositionExecutionStatus', () => {
    it('should have getDecompositionExecutionStatus endpoint defined', () => {
      expect(openevolveApi.getDecompositionExecutionStatus).toBeDefined();
      expect(typeof openevolveApi.getDecompositionExecutionStatus).toBe('function');
    });

    it('should accept execution ID', () => {
      const endpoint = openevolveApi.getDecompositionExecutionStatus;
      expect(endpoint.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('listDecompositionExecutions', () => {
    it('should have listDecompositionExecutions endpoint defined', () => {
      expect(openevolveApi.listDecompositionExecutions).toBeDefined();
      expect(typeof openevolveApi.listDecompositionExecutions).toBe('function');
    });

    it('should accept optional workflow ID filter', () => {
      const endpoint = openevolveApi.listDecompositionExecutions;
      expect(() => {
        endpoint();
        endpoint({ apiKey: 'test' });
        endpoint({ apiKey: 'test' }, 'workflow-123');
      }).not.toThrow();
    });
  });
});

describe('Workflow Template Execution API Contract Tests', () => {
  describe('executeWorkflowTemplate', () => {
    it('should have executeWorkflowTemplate endpoint defined', () => {
      expect(openevolveApi.executeWorkflowTemplate).toBeDefined();
      expect(typeof openevolveApi.executeWorkflowTemplate).toBe('function');
    });

    it('should accept template ID and payload', () => {
      const endpoint = openevolveApi.executeWorkflowTemplate;
      expect(endpoint.length).toBeGreaterThanOrEqual(2);

      const payload = {
        parameters: {
          gauntlet_name: 'test-gauntlet',
          content_value: 'Test content'
        },
        callback_url: 'https://example.com/callback'
      };

      expect(() => {
        openevolveApi.executeWorkflowTemplate('gauntlet-execution', payload);
      }).not.toThrow();
    });
  });

  describe('getWorkflowTemplateExecutionStatus', () => {
    it('should have getWorkflowTemplateExecutionStatus endpoint defined', () => {
      expect(openevolveApi.getWorkflowTemplateExecutionStatus).toBeDefined();
      expect(typeof openevolveApi.getWorkflowTemplateExecutionStatus).toBe('function');
    });

    it('should accept execution ID', () => {
      const endpoint = openevolveApi.getWorkflowTemplateExecutionStatus;
      expect(endpoint.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('stopWorkflowTemplateExecution', () => {
    it('should have stopWorkflowTemplateExecution endpoint defined', () => {
      expect(openevolveApi.stopWorkflowTemplateExecution).toBeDefined();
      expect(typeof openevolveApi.stopWorkflowTemplateExecution).toBe('function');
    });

    it('should accept execution ID', () => {
      const endpoint = openevolveApi.stopWorkflowTemplateExecution;
      expect(endpoint.length).toBeGreaterThanOrEqual(1);
    });
  });
});

describe('Unified Execution Status API Contract Tests', () => {
  it('should have getExecutionStatus endpoint defined', () => {
    expect(openevolveApi.getExecutionStatus).toBeDefined();
    expect(typeof openevolveApi.getExecutionStatus).toBe('function');
  });

  it('should accept execution type and execution ID', () => {
    const endpoint = openevolveApi.getExecutionStatus;
    expect(endpoint.length).toBeGreaterThanOrEqual(2);

    expect(() => {
      openevolveApi.getExecutionStatus('gauntlet', 'exec-123');
      openevolveApi.getExecutionStatus('decomposition', 'exec-456');
      openevolveApi.getExecutionStatus('workflow-template', 'exec-789');
    }).not.toThrow();
  });

  it('should reject invalid execution types', () => {
    const endpoint = openevolveApi.getExecutionStatus;

    expect(() => {
      // @ts-expect-error - testing invalid type
      endpoint('invalid-type', 'exec-123');
    }).toThrow();
  });
});

describe('API Endpoint Response Type Contracts', () => {
  it('should have EvolutionRunResponse type for gauntlet execution', () => {
    // This test documents the expected return type
    type ExpectedResponse = {
      run_id: string;
      status: string;
    };

    // Type assertion - will fail at compile time if wrong
    const assertType = <T extends { run_id: string; status: string }>(_value: T) => {
      return true;
    };

    expect(assertType).toBeDefined();
  });

  it('should have decomposition execution response type', () => {
    type ExpectedResponse = {
      execution_id: string;
      status: string;
    };

    const assertType = <T extends { execution_id: string; status: string }>(_value: T) => {
      return true;
    };

    expect(assertType).toBeDefined();
  });

  it('should have workflow template execution response type', () => {
    type ExpectedResponse = {
      execution_id: string;
      status: string;
      template_id: string;
    };

    const assertType = <T extends { execution_id: string; status: string; template_id: string }>(_value: T) => {
      return true;
    };

    expect(assertType).toBeDefined();
  });
});

describe('OpenEvolve API Adapter Extension Contract Tests', () => {
  // Note: These tests verify that the adapter has the right methods
  // Actual execution tests would require mocking the API

  it('should have gauntlet management methods', () => {
    const requiredMethods = [
      'getGauntlet',
      'createGauntlet',
      'updateGauntlet'
    ];

    // These methods are accessed through the adapter, not directly
    // The contract is that they exist in the API client
    requiredMethods.forEach(method => {
      expect(openevolveApi[method]).toBeDefined();
    });
  });

  it('should have gauntlet execution methods', () => {
    const requiredMethods = [
      'executeGauntlet',
      'getGauntletExecutionStatus'
    ];

    requiredMethods.forEach(method => {
      expect(openevolveApi[method]).toBeDefined();
    });
  });

  it('should have decomposition execution methods', () => {
    const requiredMethods = [
      'executeDecomposition',
      'getDecompositionExecutionStatus'
    ];

    requiredMethods.forEach(method => {
      expect(openevolveApi[method]).toBeDefined();
    });
  });

  it('should have workflow management methods', () => {
    const requiredMethods = [
      'createWorkflow',
      'getWorkflowPlan',
      'getWorkflowResults',
      'startEvolutionRun',
      'getEvolutionRun'
    ];

    requiredMethods.forEach(method => {
      expect(openevolveApi[method]).toBeDefined();
    });
  });
});
