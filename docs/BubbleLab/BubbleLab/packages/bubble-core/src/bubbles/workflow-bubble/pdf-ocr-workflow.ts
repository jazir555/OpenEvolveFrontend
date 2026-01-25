import { WorkflowBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * Constants for PDFOCRWorkflow
 */
const DEFAULT_TIMEOUT_MS = 300000;
const IDENTIFY_STEP = 1;
const AUTOFILL_STEP = 2;
const EXTRACT_STEP = 3;

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
 * Parameters for PDF identification operation
 */
interface IdentifyParams {
  filePath: string;
  detectType?: boolean;
  analyzeStructure?: boolean;
}

/**
 * Parameters for PDF autofill operation
 */
interface AutofillParams {
  filePath: string;
  formData?: Record<string, unknown>;
  preserveFormatting?: boolean;
}

/**
 * Parameters for PDF text extraction operation
 */
interface ExtractParams {
  filePath: string;
  includeImages?: boolean;
  ocrLanguage?: string;
  preserveLayout?: boolean;
}

/**
 * Input parameters for PDFOCRWorkflow
 */
export interface PDFOCRParams {
  timeout?: number;
  identify?: IdentifyParams;
  autofill?: AutofillParams;
  extract?: ExtractParams;
}

/**
 * Result of PDFOCRWorkflow execution
 */
export interface PDFOCRResult {
  success: boolean;
  steps?: WorkflowStep[];
  error?: string;
}

/**
 * PDFOCRWorkflow - Orchestrates PDF identification, autofill, and extraction operations
 *
 * This workflow executes three sequential steps:
 * 1. Identify: Identifies PDF type and structure
 * 2. Autofill: Automatically fills form fields in the PDF
 * 3. Extract: Extracts text and data from the PDF using OCR
 *
 * Each step is executed with proper error handling and status tracking.
 */
export class PDFOCRWorkflow extends WorkflowBubble<PDFOCRParams, PDFOCRResult> {
  bubbleName = 'pdfocr';
  type = 'workflow';
  alias = 'pdfocr';

  params = {
    timeout: z.number().int().positive().default(DEFAULT_TIMEOUT_MS)
  };

  /**
   * Executes the complete PDF OCR workflow
   * @param input - Workflow parameters including identify, autofill, and extract configs
   * @returns Promise<PDFOCRResult> - Result with success status and step details
   */
  async execute(input: PDFOCRParams): Promise<PDFOCRResult> {
    const steps: WorkflowStep[] = [
      { step: IDENTIFY_STEP, name: 'identify', status: StepStatus.PENDING },
      { step: AUTOFILL_STEP, name: 'autofill', status: StepStatus.PENDING },
      { step: EXTRACT_STEP, name: 'extract', status: StepStatus.PENDING }
    ];

    try {
      // Step 1: Identify PDF type and structure
      const step1Result = await this.executeStep(IDENTIFY_STEP, steps, () => this.identify(input.identify));
      if (!step1Result.success) {
        return { success: false, error: `Identification failed: ${step1Result.error}`, steps };
      }

      // Step 2: Autofill form fields
      const step2Result = await this.executeStep(AUTOFILL_STEP, steps, () => this.autofill(input.autofill));
      if (!step2Result.success) {
        return { success: false, error: `Autofill failed: ${step2Result.error}`, steps };
      }

      // Step 3: Extract text and data
      const step3Result = await this.executeStep(EXTRACT_STEP, steps, () => this.extract(input.extract));
      if (!step3Result.success) {
        return { success: false, error: `Extraction failed: ${step3Result.error}`, steps };
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
   * Identifies PDF type and structure
   * @param params - Identification parameters
   * @returns Promise<StepExecutionResult> - Identification operation result
   */
  async identify(params?: IdentifyParams): Promise<StepExecutionResult> {
    try {
      const result = await this.client.identify(params);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Identification operation failed';
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Automatically fills form fields in the PDF
   * @param params - Autofill parameters
   * @returns Promise<StepExecutionResult> - Autofill operation result
   */
  async autofill(params?: AutofillParams): Promise<StepExecutionResult> {
    try {
      const result = await this.client.autofill(params);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Autofill operation failed';
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Extracts text and data from the PDF using OCR
   * @param params - Extraction parameters
   * @returns Promise<StepExecutionResult> - Extraction operation result
   */
  async extract(params?: ExtractParams): Promise<StepExecutionResult> {
    try {
      const result = await this.client.extract(params);
      return { success: true, result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Extraction operation failed';
      return { success: false, error: errorMessage };
    }
  }
}
