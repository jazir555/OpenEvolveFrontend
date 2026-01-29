// @ts-nocheck
/**
 * Adversarial Node
 *
 * Red team/blue team testing node for robust content validation.
 * Simulates adversarial attacks to identify and patch vulnerabilities.
 *
 * @module nodes
 */

import {
  OpenEvolveBaseNode,
  NodeInputs,
  NodeResult,
  ExecutionContext,
  ValidationError,
  ParameterSchema
} from './OpenEvolveBaseNode';
import { adversarialApi, knowledgeApi } from '@/services/api/endpoints';

/**
 * Attack modes for adversarial testing
 */
export type AttackMode =
  | 'prompt_injection'
  | 'jailbreak'
  | 'adversarial_examples'
  | 'data_poisoning'
  | 'model_extraction'
  | 'semantic_attacks';

/**
 * Adversarial test status
 */
export type TestStatus = 'pending' | 'running' | 'completed' | 'failed' | 'stopped';

/**
 * Adversarial node configuration
 */
export interface AdversarialNodeConfig {
  attackModes?: AttackMode[];
  numRounds?: number;
  redTeamModels?: Array<{ provider: string; model: string }>;
  blueTeamModels?: Array<{ provider: string; model: string }>;
  timeoutMs?: number;
  enableAutoPatch?: boolean;
}

/**
 * Attack result from a single round
 */
export interface AttackResult {
  round: number;
  attackMode: AttackMode;
  success: boolean;
  promptUsed: string;
  responseReceived: string;
  vulnerability: string;
  confidence: number;
}

/**
 * Patch proposal
 */
export interface PatchProposal {
  round: number;
  attackMode: string;
  vulnerability: string;
  proposedFix: string;
  approved: boolean;
  feedback?: string;
}

/**
 * Adversarial test result
 */
export interface AdversarialTestResult {
  testId: string;
  status: TestStatus;
  content: string;
  attackModes: AttackMode[];
  rounds: number;
  attacks: AttackResult[];
  patches: PatchProposal[];
  summary: {
    totalAttacks: number;
    successfulAttacks: number;
    vulnerabilitiesFound: number;
    patchesProposed: number;
    patchesApplied: number;
    overallRobustness: number;
  };
  recommendations: string[];
  metadata: {
    startedAt: Date;
    completedAt?: Date;
    executionTime: number;
    redTeamModels: Array<{ provider: string; model: string }>;
    blueTeamModels: Array<{ provider: string; model: string }>;
  };
}

/**
 * Adversarial Node
 *
 * Executes red team/blue team adversarial testing to identify vulnerabilities.
 * Provides automated patching and comprehensive security analysis.
 */
export class AdversarialNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Adversarial Testing';
  static readonly DESCRIPTION = 'Red team/blue team testing for identifying vulnerabilities and improving robustness';
  static readonly ICON = 'adversarial';
  static readonly CATEGORY = 'testing';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: AdversarialNodeConfig = {}) {
    super(id, {
      attackModes: ['prompt_injection', 'jailbreak'],
      numRounds: 5,
      redTeamModels: [],
      blueTeamModels: [],
      timeoutMs: 300000, // 5 minutes
      enableAutoPatch: false,
      ...config
    });
  }

  /**
   * Execute adversarial testing
   *
   * @param inputs - Must contain 'content' to test
   * @param context - Execution context
   * @returns Promise resolving to adversarial test result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const content = inputs.content as string;
      const attackModes = (inputs.attackModes as AttackMode[]) || this.config.attackModes as AttackMode[];
      const numRounds = (inputs.numRounds as number) || this.config.numRounds as number;
      const redTeamModels = inputs.redTeamModels as Array<{ provider: string; model: string }> | undefined;
      const blueTeamModels = inputs.blueTeamModels as Array<{ provider: string; model: string }> | undefined;

      // Validate required inputs
      if (!content || content.trim().length === 0) {
        return this.createErrorResult('Content is required and cannot be empty');
      }

      if (!attackModes || attackModes.length === 0) {
        return this.createErrorResult('At least one attack mode must be specified');
      }

      context.updateProgress(10, 'Preparing adversarial test configuration');

      // Fetch relevant security knowledge
      let securityContext = '';
      try {
        const redTeamData = await knowledgeApi.getRedTeamAttacks();
        const blueTeamData = await knowledgeApi.getBlueTeamDefenses();
        if (redTeamData || blueTeamData) {
          securityContext = `Recent Attack Patterns: ${JSON.stringify(redTeamData)}\nRecommended Defenses: ${JSON.stringify(blueTeamData)}`;
        }
      } catch (e) {
        console.warn('Failed to fetch security context', e);
      }

      // Prepare test parameters
      const parameters = {
        num_rounds: numRounds,
        red_team_models: redTeamModels || this.config.redTeamModels as Array<{ provider: string; model: string }>,
        blue_team_models: blueTeamModels || this.config.blueTeamModels as Array<{ provider: string; model: string }>,
        context: securityContext // Pass context to backend if supported
      };

      context.updateProgress(20, 'Starting adversarial test');

      // Start adversarial test via API
      const response = await adversarialApi.start({
        content,
        attack_modes: attackModes,
        parameters
      });

      const testId = response.test_id;

      context.updateProgress(30, 'Adversarial test started, monitoring progress');

      // Monitor test progress
      const result = await this.monitorTest(testId, attackModes, context);

      const executionTime = Date.now() - startTime;

      // Generate recommendations
      const recommendations = this.generateRecommendations(result);

      const testResult: AdversarialTestResult = {
        testId,
        status: result.status as TestStatus,
        content,
        attackModes,
        rounds: result.rounds || 0,
        attacks: result.attacks || [],
        patches: result.patches || [],
        summary: {
          totalAttacks: result.attacks?.length || 0,
          successfulAttacks: result.attacks?.filter(a => a.success).length || 0,
          vulnerabilitiesFound: result.vulnerabilities_found || 0,
          patchesProposed: result.patches?.length || 0,
          patchesApplied: result.patches?.filter(p => p.approved).length || 0,
          overallRobustness: this.calculateRobustness(result)
        },
        recommendations,
        metadata: {
          startedAt: new Date(result.started_at || startTime),
          completedAt: result.completed_at ? new Date(result.completed_at) : undefined,
          executionTime,
          redTeamModels: parameters.red_team_models,
          blueTeamModels: parameters.blue_team_models
        }
      };

      context.updateProgress(100, `Adversarial testing complete: ${testResult.summary.vulnerabilitiesFound} vulnerabilities found`);

      return this.createSuccessResult(testResult);

    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error during adversarial testing'
      );
    }
  }

  /**
   * Monitor adversarial test progress until completion
   *
   * @param testId - Test ID to monitor
   * @param attackModes - Attack modes being tested
   * @param context - Execution context
   * @returns Promise resolving to test status
   */
  private async monitorTest(
    testId: string,
    attackModes: AttackMode[],
    context: ExecutionContext
  ): Promise<any> {
    const maxAttempts = 60; // 5 minutes with 5 second intervals
    let attempts = 0;
    const timeoutMs = this.config.timeoutMs as number;
    const startTime = Date.now();

    while (attempts < maxAttempts) {
      // Check timeout
      if (Date.now() - startTime > timeoutMs) {
        throw new Error('Adversarial test monitoring timeout exceeded');
      }

      try {
        const status = await adversarialApi.getStatus(testId);

        // Update progress based on rounds completed
        const progress = status.current_round && status.total_rounds
          ? (status.current_round / status.total_rounds) * 80 + 20
          : 30 + (attempts / maxAttempts) * 70;

        context.updateProgress(
          Math.min(progress, 95),
          `Round ${status.current_round || 0}/${status.total_rounds || '?'} - Testing ${attackModes.join(', ')}`
        );

        // Check if test is complete
        if (status.status === 'completed' || status.status === 'failed' || status.status === 'stopped') {
          return status;
        }

        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;

      } catch (error) {
        // If polling fails, wait and retry
        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
      }
    }

    throw new Error('Adversarial test did not complete within the expected time');
  }

  /**
   * Calculate overall robustness score
   *
   * @param result - Test result
   * @returns Robustness score (0-1)
   */
  private calculateRobustness(result: any): number {
    const attacks = result.attacks || [];
    if (attacks.length === 0) return 1.0;

    const successfulAttacks = attacks.filter((a: AttackResult) => a.success).length;
    const successRate = successfulAttacks / attacks.length;
    return Math.max(0, 1 - successRate);
  }

  /**
   * Generate recommendations based on test results
   *
   * @param result - Test result
   * @returns Array of recommendations
   */
  private generateRecommendations(result: any): string[] {
    const recommendations: string[] = [];
    const attacks = result.attacks || [];

    // Analyze successful attacks
    const successfulAttacks = attacks.filter((a: AttackResult) => a.success);
    const vulnerabilities = successfulAttacks.map((a: AttackResult) => a.vulnerability);

    if (vulnerabilities.length > 0) {
      recommendations.push(
        `Address ${vulnerabilities.length} identified vulnerability/vulnerabilities`,
        'Implement input validation and sanitization',
        'Add rate limiting and request throttling',
        'Review and strengthen prompt engineering',
        'Consider implementing adversarial training'
      );
    }

    // Analyze attack patterns
    const attackModes = new Set(attacks.map((a: AttackResult) => a.attackMode));
    if (attackModes.has('prompt_injection')) {
      recommendations.push('Strengthen defenses against prompt injection attacks');
    }
    if (attackModes.has('jailbreak')) {
      recommendations.push('Implement jailbreak detection and prevention');
    }
    if (attackModes.has('data_poisoning')) {
      recommendations.push('Validate and sanitize all input data');
    }

    return recommendations;
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.content) {
      errors.push({
        field: 'content',
        message: 'Content is required',
        severity: 'error'
      });
    }

    if (inputs.content && typeof inputs.content !== 'string') {
      errors.push({
        field: 'content',
        message: 'Content must be a string',
        severity: 'error'
      });
    }

    if (inputs.content && inputs.content.length < 50) {
      errors.push({
        field: 'content',
        message: 'Content is too short for meaningful adversarial testing (minimum 50 characters)',
        severity: 'warning'
      });
    }

    // Validate attack modes
    if (inputs.attackModes && !Array.isArray(inputs.attackModes)) {
      errors.push({
        field: 'attackModes',
        message: 'Attack modes must be an array',
        severity: 'error'
      });
    }

    if (inputs.attackModes && Array.isArray(inputs.attackModes)) {
      const validModes = ['prompt_injection', 'jailbreak', 'adversarial_examples', 'data_poisoning', 'model_extraction', 'semantic_attacks'];
      const invalidModes = inputs.attackModes.filter(m => !validModes.includes(m));
      if (invalidModes.length > 0) {
        errors.push({
          field: 'attackModes',
          message: `Invalid attack modes: ${invalidModes.join(', ')}`,
          severity: 'error'
        });
      }
    }

    // Validate number of rounds
    if (inputs.numRounds && typeof inputs.numRounds === 'number') {
      if (inputs.numRounds < 1 || inputs.numRounds > 50) {
        errors.push({
          field: 'numRounds',
          message: 'Number of rounds must be between 1 and 50',
          severity: 'error'
        });
      }
    }

    return errors;
  }

  /**
   * Get JSON Schema for configuration parameters
   *
   * @returns Parameter schema
   */
  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        attackModes: {
          type: 'array',
          description: 'Attack modes to test against',
          items: {
            type: 'string',
            enum: ['prompt_injection', 'jailbreak', 'adversarial_examples', 'data_poisoning', 'model_extraction', 'semantic_attacks']
          },
          default: ['prompt_injection', 'jailbreak']
        },
        numRounds: {
          type: 'number',
          description: 'Number of testing rounds',
          minimum: 1,
          maximum: 50,
          default: 5
        },
        redTeamModels: {
          type: 'array',
          description: 'Red team (attacker) models to use',
          items: {
            type: 'object',
            properties: {
              provider: { type: 'string' },
              model: { type: 'string' }
            }
          },
          default: []
        },
        blueTeamModels: {
          type: 'array',
          description: 'Blue team (defender) models to use',
          items: {
            type: 'object',
            properties: {
              provider: { type: 'string' },
              model: { type: 'string' }
            }
          },
          default: []
        },
        timeoutMs: {
          type: 'number',
          description: 'Timeout for adversarial test in milliseconds',
          minimum: 10000,
          maximum: 600000,
          default: 300000
        },
        enableAutoPatch: {
          type: 'boolean',
          description: 'Enable automatic patch application',
          default: false
        }
      },
      required: []
    };
  }

  /**
   * Approve or reject a patch proposal
   *
   * @param testId - Test ID
   * @param round - Round number
   * @param approved - Whether to approve the patch
   * @param feedback - Optional feedback
   * @returns Promise resolving to approval result
   */
  async approvePatch(
    testId: string,
    round: number,
    approved: boolean,
    feedback?: string
  ): Promise<NodeResult> {
    try {
      const result = await adversarialApi.approvePatch(testId, { round, approved, feedback });
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to approve patch'
      );
    }
  }

  /**
   * Stop running adversarial test
   *
   * @param testId - Test ID to stop
   * @returns Promise resolving to stop result
   */
  async stopTest(testId: string): Promise<NodeResult> {
    try {
      const result = await adversarialApi.stop(testId);
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to stop adversarial test'
      );
    }
  }

  /**
   * List all adversarial tests
   *
   * @param params - Optional query parameters
   * @returns Promise resolving to list of tests
   */
  async listTests(params?: { status?: string; limit?: number; offset?: number }): Promise<NodeResult> {
    try {
      const result = await adversarialApi.list(params);
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Failed to list adversarial tests'
      );
    }
  }
}

export default AdversarialNode;
