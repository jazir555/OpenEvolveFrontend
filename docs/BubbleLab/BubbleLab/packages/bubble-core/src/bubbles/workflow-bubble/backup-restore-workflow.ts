import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * Constants for BackupRestoreWorkflow
 */
const DEFAULT_TIMEOUT_MS = 300000;
const BACKUP_STEP = 1;
const RESTORE_STEP = 2;
const VALIDATE_STEP = 3;

/**
 * Step execution status enumeration
 */
enum StepStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

/**
 * Workflow step result interface
 */
interface WorkflowStep {
  step: number;
  name: string;
  status: StepStatus;
  result?: StepExecutionResult;
}

/**
 * Step execution result interface
 */
interface StepExecutionResult {
  success: boolean;
  result?: unknown;
  error?: string;
}

/**
 * Parameters for backup operation
 */
interface BackupParams {
  sourcePath?: string;
  destinationPath?: string;
  includeMetadata?: boolean;
}

/**
 * Parameters for restore operation
 */
interface RestoreParams {
  backupPath: string;
  targetPath?: string;
  overwrite?: boolean;
}

/**
 * Parameters for validate operation
 */
interface ValidateParams {
  backupPath: string;
  checksum?: boolean;
  verifyIntegrity?: boolean;
}

/**
 * Input parameters for BackupRestoreWorkflow
 */
export interface BackupRestoreParams {
  timeout?: number;
  backup?: BackupParams;
  restore?: RestoreParams;
  validate?: ValidateParams;
}

/**
 * Result of BackupRestoreWorkflow execution
 */
export interface BackupRestoreResult {
  success: boolean;
  steps?: WorkflowStep[];
  error?: string;
}

/**
 * BackupRestoreWorkflow - Orchestrates backup, restore, and validation operations
 *
 * This workflow executes three sequential steps:
 * 1. Backup: Creates a backup of the specified data
 * 2. Restore: Restores data from the backup
 * 3. Validate: Validates the integrity of the restored data
 *
 * Each step is executed with proper error handling and status tracking.
 */
export class BackupRestoreWorkflow extends WorkflowBubble<BackupRestoreParams, BackupRestoreResult> {
  bubbleName = 'backuprestore';
  type = 'workflow';
  alias = 'backuprestore';

  params = {
    timeout: z.number().int().positive().default(DEFAULT_TIMEOUT_MS)
  };

  /**
   * Executes the complete backup-restore-validate workflow
   * @param input - Workflow parameters including backup, restore, and validate configs
   * @returns Promise<BackupRestoreResult> - Result with success status and step details
   */
  async execute(input: BackupRestoreParams): Promise<BackupRestoreResult> {
    const steps: WorkflowStep[] = [
      { step: BACKUP_STEP, name: 'backup', status: StepStatus.PENDING },
      { step: RESTORE_STEP, name: 'restore', status: StepStatus.PENDING },
      { step: VALIDATE_STEP, name: 'validate', status: StepStatus.PENDING }
    ];

    try {
      // Step 1: Backup
      const step1Result = await this.executeStep(BACKUP_STEP, steps, () => this.backup(input.backup));
      if (!step1Result.success) {
        return { success: false, error: `Backup failed: ${step1Result.error}`, steps };
      }

      // Step 2: Restore
      const step2Result = await this.executeStep(RESTORE_STEP, steps, () => this.restore(input.restore));
      if (!step2Result.success) {
        return { success: false, error: `Restore failed: ${step2Result.error}`, steps };
      }

      // Step 3: Validate
      const step3Result = await this.executeStep(VALIDATE_STEP, steps, () => this.validate(input.validate));
      if (!step3Result.success) {
        return { success: false, error: `Validation failed: ${step3Result.error}`, steps };
      }

      return { success: true, steps };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      return { success: false, error: errorMessage, steps };
    }
  }

  /**
   * Executes a single workflow step with error handling and status tracking
   * @param stepNumber - The step number to execute
   * @param steps - Array of workflow steps to update
   * @param stepFunction - The step function to execute
   * @returns Promise<StepExecutionResult> - Result of the step execution
   */
  private async executeStep(
    stepNumber: number,
    steps: WorkflowStep[],
    stepFunction: () => Promise<StepExecutionResult>
  ): Promise<StepExecutionResult> {
    const stepIndex = stepNumber - 1;
    steps[stepIndex].status = StepStatus.IN_PROGRESS;

    try {
      const result = await stepFunction();
      steps[stepIndex].status = result.success ? StepStatus.COMPLETED : StepStatus.FAILED;
      steps[stepIndex].result = result;
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const errorResult: StepExecutionResult = { success: false, error: errorMessage };
      steps[stepIndex].status = StepStatus.FAILED;
      steps[stepIndex].result = errorResult;
      return errorResult;
    }
  }

  /**
   * Performs backup operation
   * @param params - Backup parameters
   * @returns Promise<StepExecutionResult> - Backup operation result
   */
  async backup(params?: BackupParams): Promise<StepExecutionResult> {
    try {
      const result = await this.client.backup(params);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Backup operation failed';
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Performs restore operation
   * @param params - Restore parameters
   * @returns Promise<StepExecutionResult> - Restore operation result
   */
  async restore(params?: RestoreParams): Promise<StepExecutionResult> {
    try {
      const result = await this.client.restore(params);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Restore operation failed';
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Performs validation operation
   * @param params - Validation parameters
   * @returns Promise<StepExecutionResult> - Validation operation result
   */
  async validate(params?: ValidateParams): Promise<StepExecutionResult> {
    try {
      const result = await this.client.validate(params);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Validation operation failed';
      return { success: false, error: errorMessage };
    }
  }
}
