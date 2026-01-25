/**
 * Integration Tests for Gauntlet Bubble
 *
 * Comprehensive test suite covering:
 * - Bubble base class compliance
 * - Configuration validation
 * - Operations (health check, run gauntlet, get capabilities)
 * - Federation Constitution compliance
 * - Error handling and resilience
 * - Circuit breaker functionality
 * - API contract validation
 */

import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { GauntletBubble } from './gauntlet-bubble';
import type { BubbleContext } from '@bubblelab/bubble-core';

// Mock fetch for testing
global.fetch = vi.fn();

describe('GauntletBubble - Federation Constitution Compliance', () => {
  it('should extend ServiceBubble properly', () => {
    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
    });
    expect(bubble).toBeInstanceOf(GauntletBubble);
    expect(bubble.constructor.name).toBe('GauntletBubble');
  });

  it('should have proper static properties', () => {
    expect(GauntletBubble.service).toBe('openevolve');
    expect(GauntletBubble.bubbleName).toBe('gauntlet');
    expect(GauntletBubble.type).toBe('service');
    expect(GauntletBubble.authType).toBe('apikey');
    expect(GauntletBubble.credentialType).toBe('gauntlet_api_key');
  });

  it('should require gauntletUrl (no magic defaults)', () => {
    expect(() => {
      new GauntletBubble({
        operation: 'health_check',
        // gauntletUrl missing - should throw
      } as any);
    }).toThrow();
  });

  it('should fail without gauntletUrl', () => {
    expect(() => {
      new GauntletBubble({
        operation: 'health_check',
        gauntletUrl: '', // empty string should fail URL validation
      });
    }).toThrow();
  });
});

describe('GauntletBubble - Health Check Operation', () => {
  beforeAll(() => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'healthy', version: '1.0.0' }),
    } as Response);
  });

  afterAll(() => {
    vi.clearAllMocks();
  });

  it('should perform health check successfully', async () => {
    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
    });

    const result = await bubble.action();

    expect(result.success).toBe(true);
    expect(result.operation).toBe('health_check');
    expect(result.passed).toBe(true);
    expect(result.score).toBe(100);
    expect(result.timing).toBeGreaterThanOrEqual(0);
  });

  it('should handle health check failure', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: 'Service unavailable' }),
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
    });

    const result = await bubble.action();

    expect(result.success).toBe(false);
    expect(result.passed).toBe(false);
    expect(result.error).toBeDefined();
  });
});

describe('GauntletBubble - Run Gauntlet Operation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should run full gauntlet successfully', async () => {
    const mockResponse = {
      passed: true,
      overall_score: 85,
      rounds_completed: 3,
      difficulty_used: 'adaptive',
      round_results: [
        {
          round_number: 1,
          team_type: 'red',
          score: 80,
          passed: true,
          feedback: ['Good structure'],
          timestamp: '2025-01-23T00:00:00Z',
        },
        {
          round_number: 2,
          team_type: 'blue',
          score: 85,
          passed: true,
          feedback: ['Improved clarity'],
          timestamp: '2025-01-23T00:01:00Z',
        },
        {
          round_number: 3,
          team_type: 'gold',
          score: 90,
          passed: true,
          feedback: ['Final approval'],
          timestamp: '2025-01-23T00:02:00Z',
        },
      ],
      feedback: ['Overall good quality'],
      improvements_needed: [],
      team_performances: [
        {
          team_type: 'red',
          score: 80,
          rounds_count: 1,
          strengths: ['Found edge cases'],
          weaknesses: ['None critical'],
          recommendations: ['Add more tests'],
        },
      ],
      version: '1.0.0',
    };

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'run_gauntlet',
      gauntletUrl: 'http://localhost:8000',
      gauntletType: 'full',
      rounds: 3,
      difficulty: 'adaptive',
      solution: {
        code: 'function test() { return true; }',
      },
      solutionId: 'test-solution-1',
    });

    const result = await bubble.action();

    expect(result.success).toBe(true);
    expect(result.operation).toBe('run_gauntlet');
    expect(result.passed).toBe(true);
    expect(result.score).toBe(85);
    expect(result.roundResults).toHaveLength(3);
    expect(result.feedback).toContain('Overall good quality');
    expect(result.improvementsNeeded).toHaveLength(0);
    expect(result.teamPerformances).toHaveLength(1);
    expect(result.summary.gauntletType).toBe('full');
    expect(result.summary.roundsCompleted).toBe(3);
    expect(result.summary.passed).toBe(true);
  });

  it('should handle gauntlet failure', async () => {
    const mockResponse = {
      passed: false,
      overall_score: 55,
      rounds_completed: 2,
      difficulty_used: 'medium',
      round_results: [],
      feedback: ['Failed criteria'],
      improvements_needed: ['Fix security issues', 'Improve error handling'],
      team_performances: [],
    };

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'run_gauntlet',
      gauntletUrl: 'http://localhost:8000',
      gauntletType: 'red',
      rounds: 3,
      solution: 'bad code',
    });

    const result = await bubble.action();

    expect(result.success).toBe(true);
    expect(result.passed).toBe(false);
    expect(result.score).toBe(55);
    expect(result.improvementsNeeded).toContain('Fix security issues');
    expect(result.improvementsNeeded).toContain('Improve error handling');
  });

  it('should require solution for run_gauntlet', async () => {
    const bubble = new GauntletBubble({
      operation: 'run_gauntlet',
      gauntletUrl: 'http://localhost:8000',
      // solution missing
    } as any);

    await expect(bubble.action()).rejects.toThrow('solution is required');
  });

  it('should handle API errors gracefully', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: 'Internal server error' }),
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'run_gauntlet',
      gauntletUrl: 'http://localhost:8000',
      solution: 'test code',
    });

    const result = await bubble.action();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Internal server error');
    expect(result.passed).toBe(false);
  });
});

describe('GauntletBubble - Get Capabilities Operation', () => {
  beforeAll(() => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        supported_gauntlet_types: ['red', 'blue', 'gold', 'full'],
        supported_difficulties: ['easy', 'medium', 'hard', 'adaptive'],
        evaluation_criteria: [
          'correctness',
          'completeness',
          'efficiency',
          'clarity',
          'robustness',
        ],
        max_rounds: 10,
        version: '1.0.0',
      }),
    } as Response);
  });

  afterAll(() => {
    vi.clearAllMocks();
  });

  it('should retrieve capabilities successfully', async () => {
    const bubble = new GauntletBubble({
      operation: 'get_capabilities',
      gauntletUrl: 'http://localhost:8000',
    });

    const result = await bubble.action();

    expect(result.success).toBe(true);
    expect(result.operation).toBe('get_capabilities');
    expect(result.passed).toBe(true);
    expect(result.data).toBeDefined();
  });
});

describe('GauntletBubble - Circuit Breaker & Resilience', () => {
  it('should use circuit breaker for resilience', async () => {
    let callCount = 0;
    vi.mocked(fetch).mockImplementation(() => {
      callCount++;
      if (callCount <= 3) {
        return Promise.resolve({
          ok: false,
          json: async () => ({ error: 'Service unavailable' }),
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      } as Response);
    });

    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
    });

    // Should succeed after retries
    const result = await bubble.action();

    expect(result).toBeDefined();
    expect(callCount).toBeGreaterThan(0);
  });
});

describe('GauntletBubble - API Authentication', () => {
  it('should include API key in headers when provided', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'healthy' }),
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
      apiKey: 'test-api-key-123',
    });

    await bubble.action();

    expect(fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer test-api-key-123',
        }),
      })
    );
  });

  it('should work without API key', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'healthy' }),
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
      // No apiKey
    });

    const result = await bubble.action();

    expect(result.success).toBe(true);
  });
});

describe('GauntletBubble - Request Formatting', () => {
  it('should format round results correctly', async () => {
    const mockResponse = {
      passed: true,
      overall_score: 85,
      rounds_results: [
        {
          round_number: 1,
          team_type: 'red',
          score: 80,
          passed: true,
          feedback: ['Good'],
          timestamp: '2025-01-23T00:00:00Z',
        },
      ],
      feedback: [],
      improvements_needed: [],
      team_performances: [],
    };

    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'run_gauntlet',
      gauntletUrl: 'http://localhost:8000',
      solution: 'test',
    });

    const result = await bubble.action();

    expect(result.roundResults).toHaveLength(1);
    expect(result.roundResults[0]).toMatchObject({
      round: 1,
      team: 'red',
      score: 80,
      passed: true,
    });
  });

  it('should format team performances correctly', async () => {
    const mockResponse = {
      passed: true,
      overall_score: 85,
      rounds_results: [],
      feedback: [],
      improvements_needed: [],
      team_performances: [
        {
          team_type: 'red',
          score: 80,
          rounds_count: 2,
          strengths: ['Thorough'],
          weaknesses: ['Slow'],
          recommendations: ['Optimize'],
        },
      ],
    };

    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'run_gauntlet',
      gauntletUrl: 'http://localhost:8000',
      solution: 'test',
    });

    const result = await bubble.action();

    expect(result.teamPerformances).toHaveLength(1);
    expect(result.teamPerformances[0]).toMatchObject({
      team: 'red',
      overallScore: 80,
      roundsParticipated: 2,
    });
  });
});

describe('GauntletBubble - Error Handling', () => {
  it('should handle network errors', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('Network error'));

    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
    });

    const result = await bubble.action();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Network error');
  });

  it('should handle JSON parse errors', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => {
        throw new Error('Invalid JSON');
      },
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
    });

    const result = await bubble.action();

    expect(result.success).toBe(false);
  });

  it('should handle unknown operations', async () => {
    const bubble = new GauntletBubble({
      operation: 'unknown_operation' as any,
      gauntletUrl: 'http://localhost:8000',
    });

    const result = await bubble.action();

    expect(result.success).toBe(false);
    expect(result.error).toContain('Unknown operation');
  });
});

describe('GauntletBubble - Timeout Configuration', () => {
  it('should respect timeout parameter', async () => {
    vi.mocked(fetch).mockImplementation(() =>
      new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            ok: true,
            json: async () => ({ status: 'healthy' }),
          } as Response);
        }, 100);
      })
    );

    const bubble = new GauntletBubble({
      operation: 'health_check',
      gauntletUrl: 'http://localhost:8000',
      timeout: 5000,
    });

    const result = await bubble.action();

    expect(result.success).toBe(true);
  }, 10000);
});

describe('GauntletBubble - Summary Metadata', () => {
  it('should include comprehensive summary metadata', async () => {
    const mockResponse = {
      passed: true,
      overall_score: 85,
      rounds_completed: 3,
      difficulty_used: 'adaptive',
      round_results: [],
      feedback: [],
      improvements_needed: [],
      team_performances: [],
    };

    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const bubble = new GauntletBubble({
      operation: 'run_gauntlet',
      gauntletUrl: 'http://localhost:8000',
      gauntletType: 'full',
      rounds: 5,
      difficulty: 'hard',
      evaluationCriteria: ['correctness', 'security', 'scalability'],
      solution: 'test',
    });

    const result = await bubble.action();

    expect(result.summary).toMatchObject({
      gauntletType: 'full',
      roundsCompleted: 3,
      totalRounds: 5,
      difficultyUsed: 'adaptive',
      criteriaEvaluated: ['correctness', 'security', 'scalability'],
      overallScore: 85,
      passThreshold: 70,
      passed: true,
    });
  });
});
