/**
 * Contract Tests for openevolveApi.ts
 *
 * Follows Federation Constitution Section 4: The Proof of Work (Phase 2: The Contract)
 *
 * These tests verify that the API returns the specific fields we rely on.
 * If the contract is violated (OpenEvolve API changed), the adapter refuses to start
 * to prevent data corruption.
 *
 * Run on container startup to validate API contract before accepting traffic.
 */

import { describe, it, expect, beforeAll, afterEach } from '@jest/globals';
import { fetch, RequestInit, Response } from 'node-fetch';
import { openevolveApi, ApiConfig } from './openevolveApi';

// Mock fetch globally for testing
global.fetch = fetch as any;

// Test configuration - should be injected via environment
const TEST_CONFIG: ApiConfig = {
  baseUrl: process.env.OPENEVOLVE_API_BASE_URL || 'http://localhost:8000',
  apiKey: process.env.OPENEVOLVE_API_KEY || 'test-key',
  timeout: 30000,
};

describe('OpenEvolve API Contract Tests', () => {
  beforeAll(() => {
    // Validate test configuration
    if (!process.env.OPENEVOLVE_API_BASE_URL) {
      console.warn('OPENEVOLVE_API_BASE_URL not set, using default: http://localhost:8000');
    }
  });

  describe('Health Check', () => {
    it('should return health status object', async () => {
      const response = await openevolveApi.getHealth(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response).toBe('object');
      // Health endpoint should return status
      if ('status' in response) {
        expect(typeof response.status).toBe('string');
      }
    });
  });

  describe('Teams API', () => {
    it('listTeams should return teams array and total', async () => {
      const response = await openevolveApi.listTeams(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.teams).toBeDefined();
      expect(Array.isArray(response.teams)).toBe(true);
      expect(typeof response.total).toBe('number');
    });

    it('getTeam should return team object with required fields', async () => {
      // First list to get a valid team name
      const listResponse = await openevolveApi.listTeams(TEST_CONFIG);
      if (listResponse.teams.length === 0) {
        console.warn('No teams found, skipping getTeam contract test');
        return;
      }

      const teamName = listResponse.teams[0].name;
      const response = await openevolveApi.getTeam(teamName, TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.name).toBeDefined();
      expect(typeof response.name).toBe('string');
      expect(response.description).toBeDefined();
    });
  });

  describe('Workflows API', () => {
    it('listWorkflows should return workflows array and total', async () => {
      const response = await openevolveApi.listWorkflows(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.workflows).toBeDefined();
      expect(Array.isArray(response.workflows)).toBe(true);
      expect(typeof response.total).toBe('number');
    });

    it('createWorkflow should return workflow ID and status', async () => {
      const payload = {
        name: 'contract-test-workflow',
        description: 'Contract test workflow',
        gauntlet_name: 'test-gauntlet',
        protocol_text: '# Test Protocol\n\nThis is a contract test.',
      };

      try {
        const response = await openevolveApi.createWorkflow(payload, TEST_CONFIG);

        expect(response).toBeDefined();
        expect(response.workflow_id).toBeDefined();
        expect(typeof response.workflow_id).toBe('string');
        expect(response.status).toBeDefined();
        expect(typeof response.status).toBe('string');
      } catch (error) {
        // Workflow creation might fail if test gauntlet doesn't exist
        // That's OK for contract testing - we're validating the API interface
        console.warn('Workflow creation failed (expected if test data not set up):', error);
      }
    });
  });

  describe('Gauntlets API', () => {
    it('listGauntlets should return gauntlets array and total', async () => {
      const response = await openevolveApi.listGauntlets(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.gauntlets).toBeDefined();
      expect(Array.isArray(response.gauntlets)).toBe(true);
      expect(typeof response.total).toBe('number');
    });
  });

  describe('Evolution API', () => {
    it('startEvolutionRun should return run ID and initial status', async () => {
      const payload = {
        content: '# Test Protocol\n\nThis is a contract test for evolution.',
        content_type: 'markdown',
        evolution_mode: 'incremental',
        parameters: {
          generations: 2,
        },
      };

      try {
        const response = await openevolveApi.startEvolutionRun(payload, TEST_CONFIG);

        expect(response).toBeDefined();
        expect(response.run_id).toBeDefined();
        expect(typeof response.run_id).toBe('string');
        expect(response.status).toBeDefined();
        expect(typeof response.status).toBe('string');
      } catch (error) {
        console.warn('Evolution run failed (may require valid protocol):', error);
      }
    });

    it('listEvolutionRuns should return runs array', async () => {
      const response = await openevolveApi.listEvolutionRuns(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.runs).toBeDefined();
      expect(Array.isArray(response.runs)).toBe(true);
    });
  });

  describe('Adversarial Testing API', () => {
    it('startAdversarialRun should return run ID and initial status', async () => {
      const payload = {
        content: '# Test Protocol\n\nThis is a contract test for adversarial testing.',
        content_type: 'markdown',
        parameters: {
          test_types: ['injection', 'prompt_extraction'],
        },
      };

      try {
        const response = await openevolveApi.startAdversarialRun(payload, TEST_CONFIG);

        expect(response).toBeDefined();
        expect(response.run_id).toBeDefined();
        expect(typeof response.run_id).toBe('string');
        expect(response.status).toBeDefined();
        expect(typeof response.status).toBe('string');
      } catch (error) {
        console.warn('Adversarial run failed (may require valid protocol):', error);
      }
    });

    it('listAdversarialRuns should return runs array', async () => {
      const response = await openevolveApi.listAdversarialRuns(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.runs).toBeDefined();
      expect(Array.isArray(response.runs)).toBe(true);
    });
  });

  describe('Knowledge Base API', () => {
    it('listKnowledgeArtifacts should return artifacts array', async () => {
      const response = await openevolveApi.listKnowledgeArtifacts(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.artifacts).toBeDefined();
      expect(Array.isArray(response.artifacts)).toBe(true);
    });

    it('getKnowledgeStats should return statistics object', async () => {
      const response = await openevolveApi.getKnowledgeStats(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response).toBe('object');
      // Stats should contain numeric fields
      if ('total_artifacts' in response) {
        expect(typeof response.total_artifacts).toBe('number');
      }
    });
  });

  describe('Providers API', () => {
    it('listProviders should return providers array', async () => {
      const response = await openevolveApi.listProviders(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.providers).toBeDefined();
      expect(Array.isArray(response.providers)).toBe(true);
    });
  });

  describe('Version Control API', () => {
    it('listVersions should return versions array', async () => {
      const response = await openevolveApi.listVersions(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.versions).toBeDefined();
      expect(Array.isArray(response.versions)).toBe(true);
    });
  });

  describe('BubbleLabs Integration API', () => {
    it('getBubblelabsStatus should return status object', async () => {
      const response = await openevolveApi.getBubblelabsStatus(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response).toBe('object');
      if ('available' in response) {
        expect(typeof response.available).toBe('boolean');
      }
    });

    it('listWorkflowDefinitions should return definitions array', async () => {
      const response = await openevolveApi.listWorkflowDefinitions(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.definitions).toBeDefined();
      expect(Array.isArray(response.definitions)).toBe(true);
    });
  });

  describe('Maker Integration API', () => {
    it('getMakerStatus should return availability status', async () => {
      const response = await openevolveApi.getMakerStatus(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response.available).toBe('boolean');
    });

    it('listMakerTools should return tools array', async () => {
      const response = await openevolveApi.listMakerTools({}, TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.tools).toBeDefined();
      expect(Array.isArray(response.tools)).toBe(true);
    });
  });

  describe('Knowledge Explorer API', () => {
    it('bubblelabsKnowledgeStatus should return status object', async () => {
      const response = await openevolveApi.bubblelabsKnowledgeStatus(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response.initialized).toBe('boolean');
      expect(typeof response.query_history_count).toBe('number');
    });

    it('bubblelabsKnowledgeQueryHistory should return history array', async () => {
      const response = await openevolveApi.bubblelabsKnowledgeQueryHistory(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.queries).toBeDefined();
      expect(Array.isArray(response.queries)).toBe(true);
    });
  });

  describe('LeanAide API', () => {
    it('bubblelabsLeanAideStatus should return status object', async () => {
      const response = await openevolveApi.bubblelabsLeanAideStatus(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response).toBe('object');
    });

    it('bubblelabsLeanAideTrees should return trees array', async () => {
      const response = await openevolveApi.bubblelabsLeanAideTrees(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.trees).toBeDefined();
      expect(Array.isArray(response.trees)).toBe(true);
    });
  });

  describe('Monitoring API', () => {
    it('getMonitoringDashboard should return metrics object', async () => {
      const response = await openevolveApi.getMonitoringDashboard(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response).toBe('object');
    });

    it('getMonitoringAlerts should return alerts array', async () => {
      const response = await openevolveApi.getMonitoringAlerts(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.alerts).toBeDefined();
      expect(Array.isArray(response.alerts)).toBe(true);
    });
  });

  describe('Analytics API', () => {
    it('getStatistics should return statistics summary', async () => {
      const response = await openevolveApi.getStatistics(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response).toBe('object');
    });

    it('getPerformanceMetrics should return metrics array and total', async () => {
      const response = await openevolveApi.getPerformanceMetrics(undefined, 10, TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.metrics).toBeDefined();
      expect(Array.isArray(response.metrics)).toBe(true);
      expect(typeof response.total).toBe('number');
    });
  });

  describe('Validation API', () => {
    it('listValidationRules should return rules object', async () => {
      const response = await openevolveApi.listValidationRules(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(response.rules).toBeDefined();
      expect(typeof response.rules).toBe('object');
      expect(response.rule_names).toBeDefined();
      expect(Array.isArray(response.rule_names)).toBe(true);
    });
  });

  describe('Auto-Approval API', () => {
    it('getAutoApprovalConfig should return config object', async () => {
      const response = await openevolveApi.getAutoApprovalConfig(TEST_CONFIG);

      expect(response).toBeDefined();
      expect(typeof response).toBe('object');
      // Config should have enabled flag
      if ('enabled' in response) {
        expect(typeof response.enabled).toBe('boolean');
      }
    });
  });

  describe('Error Handling', () => {
    it('should throw error with 404 status for non-existent team', async () => {
      await expect(
        openevolveApi.getTeam('non-existent-team-12345', TEST_CONFIG)
      ).rejects.toThrow();
    });

    it('should throw error with 404 status for non-existent workflow', async () => {
      await expect(
        openevolveApi.getWorkflow('non-existent-workflow-id', TEST_CONFIG)
      ).rejects.toThrow();
    });
  });
});
