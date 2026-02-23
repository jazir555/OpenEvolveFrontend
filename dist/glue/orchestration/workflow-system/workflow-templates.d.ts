/**
 * Pre-defined Workflow Templates
 *
 * Common workflow patterns for OpenEvolve users.
 * These templates can be customized and executed.
 */
import type { WorkflowDefinition } from './workflow-orchestrator';
/**
 * Research Assistant Workflow
 *
 * Searches knowledge base, processes results, and generates insights
 */
export declare const RESEARCH_ASSISTANT_WORKFLOW: WorkflowDefinition;
/**
 * Data Analysis Pipeline Workflow
 *
 * Ingests data, processes it, and generates analytics
 */
export declare const DATA_ANALYSIS_PIPELINE: WorkflowDefinition;
/**
 * Proof Verification Workflow
 *
 * Verifies mathematical proofs using multiple provers
 */
export declare const PROOF_VERIFICATION_WORKFLOW: WorkflowDefinition;
/**
 * Knowledge Extraction Workflow
 *
 * Extracts knowledge from documents and indexes it
 */
export declare const KNOWLEDGE_EXTRACTION_WORKFLOW: WorkflowDefinition;
/**
 * Problem Solving Workflow
 *
 * Analyzes problems using ROMA and generates solutions
 */
export declare const PROBLEM_SOLVING_WORKFLOW: WorkflowDefinition;
/**
 * Gauntlet Execution Workflow
 *
 * Executes a gauntlet with multiple rounds, team validation, and quorum logic
 */
export declare const GAUNTLET_EXECUTION_WORKFLOW: WorkflowDefinition;
/**
 * Decomposition Execution Workflow
 *
 * Decomposes complex problems and executes sub-problems with proper dependency management
 */
export declare const DECOMPOSITION_EXECUTION_WORKFLOW: WorkflowDefinition;
/**
 * Integrated Gauntlet + Decomposition Workflow
 *
 * Combines decomposition planning with gauntlet validation for complex problem solving
 */
export declare const GAUNTLET_DECOMPOSITION_WORKFLOW: WorkflowDefinition;
/**
 * All workflow templates
 */
export declare const WORKFLOW_TEMPLATES: Record<string, WorkflowDefinition>;
/**
 * Get workflow template by ID
 */
export declare function getWorkflowTemplate(id: string): WorkflowDefinition | undefined;
/**
 * Get all workflow templates
 */
export declare function getAllWorkflowTemplates(): WorkflowDefinition[];
/**
 * Get workflow templates by category
 */
export declare function getWorkflowTemplatesByCategory(category: string): WorkflowDefinition[];
//# sourceMappingURL=workflow-templates.d.ts.map