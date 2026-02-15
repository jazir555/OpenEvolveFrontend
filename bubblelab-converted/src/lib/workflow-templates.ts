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
export const RESEARCH_ASSISTANT_WORKFLOW: WorkflowDefinition = {
  id: 'research-assistant',
  name: 'Research Assistant',
  description: 'Search knowledge base, analyze results, and extract insights',
  version: '1.0.0',
  steps: [
    {
      id: 'search',
      name: 'Search Knowledge Base',
      description: 'Search for relevant documents using RAGBits',
      plugin: 'ragbits',
      action: 'search',
      input: {
        query: '$query',
        topK: 10,
        searchType: 'semantic'
      },
      outputMapping: {
        results: 'search_results'
      }
    },
    {
      id: 'analyze',
      name: 'Analyze Results',
      description: 'Process and categorize search results',
      plugin: 'datapizza',
      action: 'processData',
      input: {
        data: '$search_results',
        processingType: 'classification'
      },
      dependsOn: ['search'],
      outputMapping: {
        categories: 'categories'
      }
    },
    {
      id: 'summarize',
      name: 'Generate Summary',
      description: 'Create a summary of findings',
      plugin: 'datapizza',
      action: 'queryData',
      input: {
        query: 'Summarize the key findings and organize by category',
        dataSource: '$categories'
      },
      dependsOn: ['analyze'],
      outputMapping: {
        summary: 'final_summary'
      }
    }
  ],
  onError: 'continue',
  maxRetries: 2
};

/**
 * Data Analysis Pipeline Workflow
 *
 * Ingests data, processes it, and generates analytics
 */
export const DATA_ANALYSIS_PIPELINE: WorkflowDefinition = {
  id: 'data-analysis-pipeline',
  name: 'Data Analysis Pipeline',
  description: 'Process raw data through analysis pipeline and generate insights',
  version: '1.0.0',
  steps: [
    {
      id: 'ingest',
      name: 'Ingest Data',
      description: 'Load and validate raw data',
      plugin: 'datapizza',
      action: 'runPipeline',
      input: {
        dataSource: '$data_source',
        pipelineType: 'etl'
      },
      outputMapping: {
        processedData: 'processed_data'
      }
    },
    {
      id: 'index',
      name: 'Index Processed Data',
      description: 'Index processed data for search',
      plugin: 'ragbits',
      action: 'ingest',
      input: {
        content: '$processed_data',
        metadata: {
          documentType: 'analytics',
          source: 'datapizza'
        }
      },
      dependsOn: ['ingest'],
      retryOnFailure: true
    },
    {
      id: 'analyze',
      name: 'Run Analytics',
      description: 'Generate statistical analysis',
      plugin: 'datapizza',
      action: 'queryData',
      input: {
        query: '$analysis_query',
        dataSource: '$processed_data'
      },
      dependsOn: ['ingest'],
      outputMapping: {
        insights: 'analytics_insights'
      }
    },
    {
      id: 'track',
      name: 'Track Metrics',
      description: 'Store analytics metrics',
      plugin: 'openevolve',
      action: 'bubblelabsAnalyticsTrack',
      input: {
        workflow_id: '$workflow_id',
        metrics: '$analytics_insights'
      },
      dependsOn: ['analyze']
    }
  ],
  onError: 'stop',
  maxRetries: 3
};

/**
 * Proof Verification Workflow
 *
 * Verifies mathematical proofs using multiple provers
 */
export const PROOF_VERIFICATION_WORKFLOW: WorkflowDefinition = {
  id: 'proof-verification',
  name: 'Proof Verification',
  description: 'Verify mathematical theorems using Z3 and LeanAide',
  version: '1.0.0',
  steps: [
    {
      id: 'z3-verify',
      name: 'Z3 Verification',
      description: 'Verify using Z3 SMT solver',
      plugin: 'openevolve',
      action: 'bubblelabsZ3Prove',
      input: {
        theorem: '$theorem'
      },
      outputMapping: {
        z3_result: 'z3_verification_result'
      }
    },
    {
      id: 'lean-verify',
      name: 'LeanAide Verification',
      description: 'Verify using LeanAide',
      plugin: 'openevolve',
      action: 'bubblelabsLeanAideProve',
      input: {
        theorem: '$theorem'
      },
      outputMapping: {
        lean_result: 'lean_verification_result'
      }
    },
    {
      id: 'cross-validate',
      name: 'Cross-Validate Results',
      description: 'Compare results from both provers',
      plugin: 'datapizza',
      action: 'processData',
      input: {
        data: {
          z3: '$z3_verification_result',
          lean: '$lean_verification_result'
        },
        processingType: 'comparison'
      },
      dependsOn: ['z3-verify', 'lean-verify'],
      outputMapping: {
        validation: 'cross_validation_result'
      }
    },
    {
      id: 'store',
      name: 'Store Proof',
      description: 'Store verified proof in knowledge base',
      plugin: 'ragbits',
      action: 'ingest',
      input: {
        content: '$theorem',
        metadata: {
          documentType: 'proof',
          verified: true,
          z3_result: '$z3_verification_result',
          lean_result: '$lean_verification_result'
        }
      },
      dependsOn: ['cross-validate'],
      condition: (context) => {
        const validation = context.stepResults.get('cross-validate') as any;
        return validation?.output?.verified === true;
      }
    }
  ],
  onError: 'continue',
  maxRetries: 1
};

/**
 * Knowledge Extraction Workflow
 *
 * Extracts knowledge from documents and indexes it
 */
export const KNOWLEDGE_EXTRACTION_WORKFLOW: WorkflowDefinition = {
  id: 'knowledge-extraction',
  name: 'Knowledge Extraction',
  description: 'Extract structured knowledge from unstructured documents',
  version: '1.0.0',
  steps: [
    {
      id: 'extract',
      name: 'Extract Knowledge',
      description: 'Extract entities and relationships',
      plugin: 'openevolve',
      action: 'bubblelabsKnowledgeExtract',
      input: {
        source_type: '$source_type',
        source_value: '$source_value'
      },
      outputMapping: {
        knowledge: 'extracted_knowledge'
      }
    },
    {
      id: 'validate',
      name: 'Validate Knowledge',
      description: 'Validate extracted knowledge',
      plugin: 'datapizza',
      action: 'isProcessableData',
      input: {
        data: '$extracted_knowledge'
      },
      dependsOn: ['extract'],
      retryOnFailure: true
    },
    {
      id: 'enrich',
      name: 'Enrich Knowledge',
      description: 'Add metadata and context',
      plugin: 'datapizza',
      action: 'processData',
      input: {
        data: '$extracted_knowledge',
        processingType: 'enrichment'
      },
      dependsOn: ['validate'],
      outputMapping: {
        enriched: 'enriched_knowledge'
      }
    },
    {
      id: 'index',
      name: 'Index Knowledge',
      description: 'Index in knowledge base',
      plugin: 'ragbits',
      action: 'ingest',
      input: {
        content: '$enriched_knowledge',
        metadata: {
          documentType: 'knowledge',
          enriched: true
        }
      },
      dependsOn: ['enrich']
    },
    {
      id: 'store',
      name: 'Store Artifact',
      description: 'Store in knowledge graph',
      plugin: 'openevolve',
      action: 'bubblelabsKnowledgeStore',
      input: {
        artifact: '$enriched_knowledge'
      },
      dependsOn: ['enrich']
    }
  ],
  onError: 'stop',
  maxRetries: 2
};

/**
 * Problem Solving Workflow
 *
 * Analyzes problems using ROMA and generates solutions
 */
export const PROBLEM_SOLVING_WORKFLOW: WorkflowDefinition = {
  id: 'problem-solving',
  name: 'Problem Solving',
  description: 'Analyze complex problems and generate solutions',
  version: '1.0.0',
  steps: [
    {
      id: 'analyze',
      name: 'Analyze Problem',
      description: 'Decompose problem using ROMA',
      plugin: 'openevolve',
      action: 'bubblelabsRomaAnalyze',
      input: {
        problem: '$problem',
        max_depth: 5
      },
      outputMapping: {
        decomposition: 'problem_decomposition'
      }
    },
    {
      id: 'search-solutions',
      name: 'Search for Solutions',
      description: 'Search knowledge base for similar problems',
      plugin: 'ragbits',
      action: 'search',
      input: {
        query: '$problem',
        topK: 5,
        searchType: 'hybrid'
      },
      outputMapping: {
        solutions: 'similar_solutions'
      }
    },
    {
      id: 'generate',
      name: 'Generate Solution',
      description: 'Generate solution based on analysis',
      plugin: 'datapizza',
      action: 'processData',
      input: {
        data: {
          problem: '$problem',
          decomposition: '$problem_decomposition',
          similar: '$similar_solutions'
        },
        processingType: 'synthesis'
      },
      dependsOn: ['analyze', 'search-solutions'],
      outputMapping: {
        solution: 'generated_solution'
      }
    },
    {
      id: 'verify',
      name: 'Verify Solution',
      description: 'Verify solution using formal methods',
      plugin: 'openevolve',
      action: 'bubblelabsZ3Solve',
      input: {
        variables: [],
        constraints: ['$generated_solution']
      },
      dependsOn: ['generate'],
      retryOnFailure: true,
      outputMapping: {
        verification: 'solution_verification'
      }
    }
  ],
  onError: 'continue',
  maxRetries: 2
};

/**
 * All workflow templates
 */
export const WORKFLOW_TEMPLATES: Record<string, WorkflowDefinition> = {
  'research-assistant': RESEARCH_ASSISTANT_WORKFLOW,
  'data-analysis-pipeline': DATA_ANALYSIS_PIPELINE,
  'proof-verification': PROOF_VERIFICATION_WORKFLOW,
  'knowledge-extraction': KNOWLEDGE_EXTRACTION_WORKFLOW,
  'problem-solving': PROBLEM_SOLVING_WORKFLOW
};

/**
 * Get workflow template by ID
 */
export function getWorkflowTemplate(id: string): WorkflowDefinition | undefined {
  return WORKFLOW_TEMPLATES[id];
}

/**
 * Get all workflow templates
 */
export function getAllWorkflowTemplates(): WorkflowDefinition[] {
  return Object.values(WORKFLOW_TEMPLATES);
}

/**
 * Get workflow templates by category
 */
export function getWorkflowTemplatesByCategory(category: string): WorkflowDefinition[] {
  // Define categories
  const categories: Record<string, string[]> = {
    knowledge: ['research-assistant', 'knowledge-extraction'],
    analysis: ['data-analysis-pipeline'],
    verification: ['proof-verification'],
    problem: ['problem-solving']
  };

  const ids = categories[category] || [];
  return ids.map(id => WORKFLOW_TEMPLATES[id]).filter(Boolean) as WorkflowDefinition[];
}
