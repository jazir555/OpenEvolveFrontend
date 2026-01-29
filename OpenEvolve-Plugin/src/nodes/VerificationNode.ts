// @ts-nocheck
/**
 * Verification Node
 *
 * Verifies solutions against requirements and quality standards.
 * Provides comprehensive verification reports with scoring.
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

/**
 * Verification check types
 */
export type VerificationCheck =
  | 'requirements'
  | 'quality'
  | 'completeness'
  | 'correctness'
  | 'consistency'
  | 'feasibility'
  | 'all';

/**
 * Verification result for a single check
 */
export interface CheckResult {
  check: VerificationCheck;
  passed: boolean;
  score: number;
  details: string[];
  severity: 'critical' | 'major' | 'minor' | 'info';
  suggestions: string[];
}

/**
 * Verification report interface
 */
export interface VerificationReport {
  solutionId: string;
  overallScore: number;
  passed: boolean;
  checks: CheckResult[];
  requirements: {
    specified: string[];
    met: string[];
    partiallyMet: string[];
    notMet: string[];
    coverage: number;
  };
  qualityMetrics: {
    completeness: number;
    correctness: number;
    clarity: number;
    consistency: number;
    feasibility: number;
  };
  issues: {
    critical: string[];
    major: string[];
    minor: string[];
  };
  suggestions: string[];
  metadata: {
    verifiedAt: Date;
    verificationTime: number;
    threshold: number;
    verifierVersion: string;
  };
}

/**
 * Verification node configuration
 */
export interface VerificationNodeConfig {
  threshold?: number;
  checks?: VerificationCheck[];
  strictMode?: boolean;
  generateSuggestions?: boolean;
  includeDetails?: boolean;
}

/**
 * Solution Verification Node
 *
 * Validates solutions against requirements and quality standards.
 * Generates detailed verification reports with actionable feedback.
 */
export class VerificationNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'Solution Verification';
  static readonly DESCRIPTION = 'Verify solutions against requirements and quality standards with comprehensive reporting';
  static readonly ICON = 'verification';
  static readonly CATEGORY = 'verification';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config: VerificationNodeConfig = {}) {
    super(id, {
      threshold: 0.7,
      checks: ['all'],
      strictMode: false,
      generateSuggestions: true,
      includeDetails: true,
      ...config
    });
  }

  /**
   * Execute solution verification
   *
   * @param inputs - Must contain 'solution' and 'requirements'
   * @param context - Execution context
   * @returns Promise resolving to verification result
   */
  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      const startTime = Date.now();

      // Extract inputs
      const solution = inputs.solution as string;
      const solutionId = inputs.solutionId as string | undefined;
      const requirements = inputs.requirements as string[] | undefined;
      const qualityStandards = inputs.qualityStandards as Record<string, any> | undefined;
      const originalProblem = inputs.problem as string | undefined;

      // Validate required inputs
      if (!solution || solution.trim().length === 0) {
        return this.createErrorResult('Solution is required and cannot be empty');
      }

      if (!requirements || requirements.length === 0) {
        return this.createErrorResult('Requirements array is required and cannot be empty');
      }

      // Step 1: Perform verification checks
      const checks = await this.performAllChecks(
        solution,
        requirements,
        originalProblem,
        qualityStandards
      );

      // Step 2: Analyze requirements coverage
      const requirementsAnalysis = this.analyzeRequirementsCoverage(
        solution,
        requirements
      );

      // Step 3: Calculate quality metrics
      const qualityMetrics = this.calculateQualityMetrics(
        solution,
        checks,
        requirementsAnalysis
      );

      // Step 4: Identify issues
      const issues = this.identifyIssues(checks);

      // Step 5: Generate suggestions
      const suggestions = this.config.generateSuggestions
        ? this.generateSuggestions(checks, requirementsAnalysis, qualityMetrics)
        : [];

      // Step 6: Calculate overall score
      const overallScore = this.calculateOverallScore(checks, qualityMetrics);

      // Step 7: Determine if passed
      const threshold = this.config.threshold as number;
      const passed = overallScore >= threshold && issues.critical.length === 0;

      const verificationTime = Date.now() - startTime;

      const report: VerificationReport = {
        solutionId: solutionId || 'unknown',
        overallScore,
        passed,
        checks,
        requirements: requirementsAnalysis,
        qualityMetrics,
        issues,
        suggestions,
        metadata: {
          verifiedAt: new Date(),
          verificationTime,
          threshold,
          verifierVersion: VerificationNode.VERSION
        }
      };

      return this.createSuccessResult(report);
    } catch (error) {
      return this.createErrorResult(
        error instanceof Error ? error : 'Unknown error during verification'
      );
    }
  }

  /**
   * Validate input data
   *
   * @param inputs - Input data to validate
   * @returns Array of validation errors
   */
  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.solution) {
      errors.push({
        field: 'solution',
        message: 'Solution is required',
        severity: 'error'
      });
    }

    if (inputs.solution && typeof inputs.solution !== 'string') {
      errors.push({
        field: 'solution',
        message: 'Solution must be a string',
        severity: 'error'
      });
    }

    if (!inputs.requirements) {
      errors.push({
        field: 'requirements',
        message: 'Requirements are required',
        severity: 'error'
      });
    }

    if (inputs.requirements && !Array.isArray(inputs.requirements)) {
      errors.push({
        field: 'requirements',
        message: 'Requirements must be an array',
        severity: 'error'
      });
    }

    if (inputs.solution && inputs.solution.length < 50) {
      errors.push({
        field: 'solution',
        message: 'Solution is too short for meaningful verification (minimum 50 characters)',
        severity: 'warning'
      });
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
        threshold: {
          type: 'number',
          description: 'Minimum score (0-1) for solution to pass verification',
          minimum: 0,
          maximum: 1,
          default: 0.7
        },
        checks: {
          type: 'array',
          description: 'Verification checks to perform',
          items: {
            type: 'string',
            enum: ['requirements', 'quality', 'completeness', 'correctness', 'consistency', 'feasibility', 'all']
          },
          default: ['all']
        },
        strictMode: {
          type: 'boolean',
          description: 'Enable strict verification mode',
          default: false
        },
        generateSuggestions: {
          type: 'boolean',
          description: 'Generate improvement suggestions',
          default: true
        },
        includeDetails: {
          type: 'boolean',
          description: 'Include detailed verification information',
          default: true
        }
      },
      required: []
    };
  }

  /**
   * Perform all verification checks
   *
   * @param solution - Solution to verify
   * @param requirements - Requirements to verify against
   * @param originalProblem - Original problem statement
   * @param qualityStandards - Quality standards
   * @returns Array of check results
   */
  private async performAllChecks(
    solution: string,
    requirements: string[],
    originalProblem?: string,
    qualityStandards?: Record<string, any>
  ): Promise<CheckResult[]> {
    const checksToRun = this.config.checks as VerificationCheck[];
    const allChecks: VerificationCheck[] = checksToRun.includes('all')
      ? ['requirements', 'quality', 'completeness', 'correctness', 'consistency', 'feasibility']
      : checksToRun;

    const results: CheckResult[] = [];

    for (const check of allChecks) {
      const result = await this.performCheck(
        check,
        solution,
        requirements,
        originalProblem,
        qualityStandards
      );
      results.push(result);
    }

    return results;
  }

  /**
   * Perform a single verification check
   *
   * @param check - Check to perform
   * @param solution - Solution to verify
   * @param requirements - Requirements
   * @param originalProblem - Original problem
   * @param qualityStandards - Quality standards
   * @returns Check result
   */
  private async performCheck(
    check: VerificationCheck,
    solution: string,
    requirements: string[],
    originalProblem?: string,
    qualityStandards?: Record<string, any>
  ): Promise<CheckResult> {
    switch (check) {
      case 'requirements':
        return this.checkRequirements(solution, requirements);
      case 'quality':
        return this.checkQuality(solution, qualityStandards);
      case 'completeness':
        return this.checkCompleteness(solution, requirements, originalProblem);
      case 'correctness':
        return this.checkCorrectness(solution, originalProblem);
      case 'consistency':
        return this.checkConsistency(solution);
      case 'feasibility':
        return this.checkFeasibility(solution);
      default:
        return {
          check,
          passed: false,
          score: 0,
          details: ['Unknown check type'],
          severity: 'info',
          suggestions: []
        };
    }
  }

  /**
   * Check requirements coverage
   *
   * @param solution - Solution to check
   * @param requirements - Requirements to verify
   * @returns Check result
   */
  private checkRequirements(solution: string, requirements: string[]): CheckResult {
    const details: string[] = [];
    const suggestions: string[] = [];
    const strictMode = this.config.strictMode as boolean;

    let metCount = 0;
    const requirementResults: Array<{ requirement: string; status: string }> = [];

    requirements.forEach(requirement => {
      const requirementLower = requirement.toLowerCase();
      const solutionLower = solution.toLowerCase();

      // Check if requirement keywords appear in solution
      const keywords = requirementLower.split(/\s+/).filter(w => w.length > 4);
      const keywordMatches = keywords.filter(k => solutionLower.includes(k)).length;
      const matchRatio = keywords.length > 0 ? keywordMatches / keywords.length : 0;

      let status: string;
      if (matchRatio > 0.6) {
        status = 'met';
        metCount++;
      } else if (matchRatio > 0.3) {
        status = 'partially met';
      } else {
        status = 'not met';
      }

      requirementResults.push({ requirement, status });

      if (strictMode || status !== 'met') {
        details.push(`Requirement: "${requirement}" - ${status}`);
      }
    });

    const coverage = requirements.length > 0 ? metCount / requirements.length : 0;
    const score = coverage;
    const passed = strictMode ? coverage >= 0.9 : coverage >= 0.7;

    if (!passed) {
      suggestions.push(
        'Ensure all requirements are explicitly addressed in the solution',
        'Use requirement terminology and language in the solution',
        'Provide clear mappings between requirements and solution components'
      );
    }

    return {
      check: 'requirements',
      passed,
      score,
      details: this.config.includeDetails ? details : [`Coverage: ${ (coverage * 100).toFixed(1) }%`],
      severity: passed ? 'info' : coverage < 0.5 ? 'critical' : 'major',
      suggestions
    };
  }

  /**
   * Check quality standards
   *
   * @param solution - Solution to check
   * @param qualityStandards - Quality standards
   * @returns Check result
   */
  private checkQuality(solution: string, qualityStandards?: Record<string, any>): CheckResult {
    const details: string[] = [];
    const suggestions: string[] = [];

    // Check structure
    const hasStructure = solution.includes('##') || solution.includes('#');
    const hasLists = solution.includes('-') || solution.includes('*');
    const hasCodeBlocks = solution.includes('```') || solution.includes('`');

    // Check length
    const length = solution.length;
    const adequateLength = length > 200;

    // Check clarity indicators
    const sentences = solution.split(/[.!?]+/);
    const avgSentenceLength = sentences.reduce((sum, s) => sum + s.length, 0) / sentences.length;
    const clearSentences = avgSentenceLength < 150;

    let score = 0;
    if (hasStructure) score += 0.25;
    if (hasLists) score += 0.15;
    if (hasCodeBlocks) score += 0.1;
    if (adequateLength) score += 0.25;
    if (clearSentences) score += 0.25;

    details.push(
      `Structure: ${hasStructure ? 'Present' : 'Missing'}`,
      `Lists: ${hasLists ? 'Present' : 'Missing'}`,
      `Length: ${length} characters ${adequateLength ? '(adequate)' : '(too short)'}`,
      `Clarity: ${clearSentences ? 'Good' : 'Needs improvement'}`
    );

    if (score < 0.7) {
      suggestions.push(
        'Add clear structure with headers and sections',
        'Use bullet points for better readability',
        'Include code examples where appropriate',
        'Break up long sentences for better clarity'
      );
    }

    const passed = score >= 0.6;

    return {
      check: 'quality',
      passed,
      score,
      details,
      severity: passed ? 'info' : score < 0.4 ? 'major' : 'minor',
      suggestions
    };
  }

  /**
   * Check completeness
   *
   * @param solution - Solution to check
   * @param requirements - Requirements
   * @param originalProblem - Original problem
   * @returns Check result
   */
  private checkCompleteness(
    solution: string,
    requirements: string[],
    originalProblem?: string
  ): CheckResult {
    const details: string[] = [];
    const suggestions: string[] = [];

    let score = 0.5; // Base score

    // Check for introduction/problem statement
    const hasIntro = originalProblem
      ? solution.toLowerCase().includes(originalProblem.substring(0, 20).toLowerCase())
      : true;
    if (hasIntro) score += 0.1;

    // Check for solution/approach section
    const hasApproach = /solution|approach|method|strategy/i.test(solution);
    if (hasApproach) score += 0.15;

    // Check for implementation details
    const hasImplementation = /implement|execute|step|phase/i.test(solution);
    if (hasImplementation) score += 0.15;

    // Check for validation/testing
    const hasValidation = /test|valid|verif|check|review/i.test(solution);
    if (hasValidation) score += 0.1;

    // Check for conclusion/summary
    const hasConclusion = /conclusion|summary|complete|finish/i.test(solution);
    if (hasConclusion) score += 0.1;

    details.push(
      `Introduction: ${hasIntro ? 'Present' : 'Missing or unclear'}`,
      `Approach: ${hasApproach ? 'Present' : 'Missing'}`,
      `Implementation: ${hasImplementation ? 'Present' : 'Missing'}`,
      `Validation: ${hasValidation ? 'Present' : 'Missing'}`,
      `Conclusion: ${hasConclusion ? 'Present' : 'Missing'}`
    );

    if (score < 0.7) {
      suggestions.push(
        'Add a clear introduction restating the problem',
        'Include a detailed approach or methodology section',
        'Provide implementation steps or phases',
        'Add validation or testing considerations',
        'Include a summary or conclusion'
      );
    }

    const passed = score >= 0.6;

    return {
      check: 'completeness',
      passed,
      score,
      details,
      severity: passed ? 'info' : score < 0.4 ? 'major' : 'minor',
      suggestions
    };
  }

  /**
   * Check correctness
   *
   * @param solution - Solution to check
   * @param originalProblem - Original problem
   * @returns Check result
   */
  private checkCorrectness(solution: string, originalProblem?: string): CheckResult {
    const details: string[] = [];
    const suggestions: string[] = [];

    let score = 0.6; // Base score assuming solution is mostly correct

    // Check if solution addresses the problem
    if (originalProblem) {
      const problemWords = new Set(originalProblem.toLowerCase().split(/\s+/));
      const solutionWords = solution.toLowerCase().split(/\s+/);
      const overlap = solutionWords.filter(w => problemWords.has(w)).length;
      const relevance = overlap / problemWords.size;
      score += relevance * 0.2;
    }

    // Check for logical flow indicators
    const hasLogic = /therefore|thus|consequently|because|since|so/i.test(solution);
    if (hasLogic) score += 0.1;

    // Check for problem-solving patterns
    const hasProblemSolving = /solve|address|resolve|fix|improve/i.test(solution);
    if (hasProblemSolving) score += 0.1;

    details.push(
      `Problem relevance: ${((score - 0.6) / 0.2 * 100).toFixed(0)}%`,
      `Logical flow: ${hasLogic ? 'Present' : 'Could be improved'}`,
      `Problem-solving focus: ${hasProblemSolving ? 'Present' : 'Weak'}`
    );

    if (score < 0.7) {
      suggestions.push(
        'Ensure the solution directly addresses the original problem',
        'Use clear logical connections between ideas',
        'Maintain focus on problem-solving throughout'
      );
    }

    const passed = score >= 0.6;

    return {
      check: 'correctness',
      passed,
      score: Math.min(score, 1.0),
      details,
      severity: passed ? 'info' : score < 0.4 ? 'major' : 'minor',
      suggestions
    };
  }

  /**
   * Check consistency
   *
   * @param solution - Solution to check
   * @returns Check result
   */
  private checkConsistency(solution: string): CheckResult {
    const details: string[] = [];
    const suggestions: string[] = [];

    let score = 0.7; // Base score

    // Check for consistent terminology
    const words = solution.toLowerCase().split(/\s+/);
    const wordFreq = new Map<string, number>();
    words.forEach(w => {
      if (w.length > 5) {
        wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
      }
    });

    const consistentTerminology = wordFreq.size < words.length / 3;
    if (consistentTerminology) score += 0.1;

    // Check for consistent formatting
    const hasConsistentHeaders = /^#{1,3}\s/gm.test(solution);
    if (hasConsistentHeaders) score += 0.1;

    // Check for consistent structure
    const lines = solution.split('\n');
    const bulletLines = lines.filter(l => l.trim().startsWith('-')).length;
    const hasConsistentStructure = bulletLines > 2;
    if (hasConsistentStructure) score += 0.1;

    details.push(
      `Terminology consistency: ${consistentTerminology ? 'Good' : 'Could be improved'}`,
      `Header consistency: ${hasConsistentHeaders ? 'Present' : 'Mixed'}`,
      `Structure consistency: ${hasConsistentStructure ? 'Present' : 'Weak'}`
    );

    if (score < 0.8) {
      suggestions.push(
        'Use consistent terminology throughout the solution',
        'Apply consistent formatting for headers and sections',
        'Maintain consistent structure for similar elements'
      );
    }

    const passed = score >= 0.7;

    return {
      check: 'consistency',
      passed,
      score,
      details,
      severity: passed ? 'info' : 'minor',
      suggestions
    };
  }

  /**
   * Check feasibility
   *
   * @param solution - Solution to check
   * @returns Check result
   */
  private checkFeasibility(solution: string): CheckResult {
    const details: string[] = [];
    const suggestions: string[] = [];

    let score = 0.7; // Base score

    // Check for realistic timeframes
    const hasTimeframes = /quickly|soon|immediate|instant|rapid/i.test(solution);
    if (!hasTimeframes) score += 0.1;

    // Check for practical steps
    const hasSteps = /step|phase|stage|1\.|2\.|3\./i.test(solution);
    if (hasSteps) score += 0.1;

    // Check for resource considerations
    const hasResources = /resource|tool|technology|framework/i.test(solution);
    if (hasResources) score += 0.1;

    details.push(
      `Timeframe realism: ${!hasTimeframes ? 'Good' : 'May be optimistic'}`,
      `Practical steps: ${hasSteps ? 'Present' : 'Could be more detailed'}`,
      `Resource awareness: ${hasResources ? 'Present' : 'Could be improved'}`
    );

    if (score < 0.8) {
      suggestions.push(
        'Provide realistic timeframes for implementation',
        'Break down into concrete, actionable steps',
        'Consider resource requirements and constraints'
      );
    }

    const passed = score >= 0.7;

    return {
      check: 'feasibility',
      passed,
      score,
      details,
      severity: passed ? 'info' : 'minor',
      suggestions
    };
  }

  /**
   * Analyze requirements coverage
   *
   * @param solution - Solution to analyze
   * @param requirements - Requirements
   * @returns Requirements analysis
   */
  private analyzeRequirementsCoverage(
    solution: string,
    requirements: string[]
  ): VerificationReport['requirements'] {
    const specified = requirements;
    const met: string[] = [];
    const partiallyMet: string[] = [];
    const notMet: string[] = [];

    const solutionLower = solution.toLowerCase();

    requirements.forEach(requirement => {
      const requirementLower = requirement.toLowerCase();
      const keywords = requirementLower.split(/\s+/).filter(w => w.length > 4);
      const keywordMatches = keywords.filter(k => solutionLower.includes(k)).length;
      const matchRatio = keywords.length > 0 ? keywordMatches / keywords.length : 0;

      if (matchRatio > 0.6) {
        met.push(requirement);
      } else if (matchRatio > 0.3) {
        partiallyMet.push(requirement);
      } else {
        notMet.push(requirement);
      }
    });

    const coverage = specified.length > 0 ? met.length / specified.length : 0;

    return {
      specified,
      met,
      partiallyMet,
      notMet,
      coverage
    };
  }

  /**
   * Calculate quality metrics
   *
   * @param solution - Solution
   * @param checks - Check results
   * @param requirementsAnalysis - Requirements analysis
   * @returns Quality metrics
   */
  private calculateQualityMetrics(
    solution: string,
    checks: CheckResult[],
    requirementsAnalysis: VerificationReport['requirements']
  ): VerificationReport['qualityMetrics'] {
    const requirementsCheck = checks.find(c => c.check === 'requirements');
    const qualityCheck = checks.find(c => c.check === 'quality');
    const completenessCheck = checks.find(c => c.check === 'completeness');
    const correctnessCheck = checks.find(c => c.check === 'correctness');
    const consistencyCheck = checks.find(c => c.check === 'consistency');
    const feasibilityCheck = checks.find(c => c.check === 'feasibility');

    return {
      completeness: completenessCheck?.score || 0.7,
      correctness: correctnessCheck?.score || 0.7,
      clarity: qualityCheck?.score || 0.7,
      consistency: consistencyCheck?.score || 0.7,
      feasibility: feasibilityCheck?.score || 0.7
    };
  }

  /**
   * Identify issues from check results
   *
   * @param checks - Check results
   * @returns Issues organized by severity
   */
  private identifyIssues(checks: CheckResult[]): VerificationReport['issues'] {
    const critical: string[] = [];
    const major: string[] = [];
    const minor: string[] = [];

    checks.forEach(check => {
      if (!check.passed) {
        const issue = `${check.check}: ${check.details.join(', ')}`;
        if (check.severity === 'critical') {
          critical.push(issue);
        } else if (check.severity === 'major') {
          major.push(issue);
        } else {
          minor.push(issue);
        }
      }
    });

    return { critical, major, minor };
  }

  /**
   * Generate improvement suggestions
   *
   * @param checks - Check results
   * @param requirementsAnalysis - Requirements analysis
   * @param qualityMetrics - Quality metrics
   * @returns Array of suggestions
   */
  private generateSuggestions(
    checks: CheckResult[],
    requirementsAnalysis: VerificationReport['requirements'],
    qualityMetrics: VerificationReport['qualityMetrics']
  ): string[] {
    const suggestions: string[] = [];

    // Collect suggestions from all checks
    checks.forEach(check => {
      suggestions.push(...check.suggestions);
    });

    // Add requirements-specific suggestions
    if (requirementsAnalysis.notMet.length > 0) {
      suggestions.push(
        `Explicitly address the ${requirementsAnalysis.notMet.length} unmet requirement(s)`
      );
    }

    if (requirementsAnalysis.partiallyMet.length > 0) {
      suggestions.push(
        `Strengthen coverage of partially met requirements: ${requirementsAnalysis.partiallyMet.join(', ')}`
      );
    }

    // Add quality-specific suggestions
    const lowMetrics = Object.entries(qualityMetrics)
      .filter(([_, score]) => score < 0.7)
      .map(([metric, _]) => metric);

    if (lowMetrics.length > 0) {
      suggestions.push(
        `Focus on improving: ${lowMetrics.join(', ')}`
      );
    }

    // Deduplicate
    return [...new Set(suggestions)];
  }

  /**
   * Calculate overall verification score
   *
   * @param checks - Check results
   * @param qualityMetrics - Quality metrics
   * @returns Overall score (0-1)
   */
  private calculateOverallScore(
    checks: CheckResult[],
    qualityMetrics: VerificationReport['qualityMetrics']
  ): number {
    // Weighted average of check scores
    const checkScores = checks.map(c => c.score);
    const avgCheckScore = checkScores.reduce((sum, s) => sum + s, 0) / checkScores.length;

    // Weighted average of quality metrics
    const qualityScores = Object.values(qualityMetrics);
    const avgQualityScore = qualityScores.reduce((sum, s) => sum + s, 0) / qualityScores.length;

    // Combine with 70% weight on checks, 30% on quality metrics
    return avgCheckScore * 0.7 + avgQualityScore * 0.3;
  }
}

export default VerificationNode;
