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
 * Gauntlet Execution Workflow
 *
 * Executes a gauntlet with multiple rounds, team validation, and quorum logic
 */
export const GAUNTLET_EXECUTION_WORKFLOW: WorkflowDefinition = {
  id: 'gauntlet-execution',
  name: 'Gauntlet Execution',
  description: 'Execute gauntlet rounds with team validation, quorum logic, and formal verification',
  version: '1.0.0',
  steps: [
    {
      id: 'initialize-gauntlet',
      name: 'Initialize Gauntlet',
      description: 'Load gauntlet configuration and prepare for execution',
      plugin: 'openevolve',
      action: 'getGauntlet',
      input: {
        gauntlet_name: '$gauntlet_name'
      },
      outputMapping: {
        gauntlet_config: 'gauntlet_config'
      }
    },
    {
      id: 'prepare-content',
      name: 'Prepare Content',
      description: 'Prepare content for gauntlet evaluation',
      plugin: 'openevolve',
      action: 'bubblelabsKnowledgeExtract',
      input: {
        source_type: '$content_type',
        source_value: '$content_value'
      },
      outputMapping: {
        prepared_content: 'prepared_content'
      }
    },
    {
      id: 'execute-rounds',
      name: 'Execute Gauntlet Rounds',
      description: 'Execute all gauntlet rounds sequentially with validation',
      plugin: 'openevolve',
      action: 'startEvolutionRun',
      input: {
        content: '$prepared_content',
        content_type: '$content_type',
        gauntlet_name: '$gauntlet_name',
        evolution_mode: '$evolution_mode',
        use_decomposition: false,
        parameters: {
          max_iterations: '$max_iterations',
          temperature: 0.7
        }
      },
      dependsOn: ['initialize-gauntlet', 'prepare-content'],
      outputMapping: {
        execution_id: 'gauntlet_execution_id',
        results: 'gauntlet_results'
      }
    },
    {
      id: 'formal-verification',
      name: 'Formal Verification (Optional)',
      description: 'Run formal verification if enabled in gauntlet',
      plugin: 'openevolve',
      action: 'bubblelabsZ3Prove',
      input: {
        theorem: '$prepared_content'
      },
      dependsOn: ['execute-rounds'],
      condition: (context) => {
        const gauntletConfig = context.stepResults.get('initialize-gauntlet') as any;
        return gauntletConfig?.output?.formal_verification_enabled === true;
      },
      outputMapping: {
        verification_result: 'formal_verification_result'
      }
    },
    {
      id: 'lean-verification',
      name: 'LeanAide Verification (Optional)',
      description: 'Run LeanAide verification if enabled',
      plugin: 'openevolve',
      action: 'bubblelabsLeanAideProve',
      input: {
        theorem: '$prepared_content'
      },
      dependsOn: ['execute-rounds'],
      condition: (context) => {
        const gauntletConfig = context.stepResults.get('initialize-gauntlet') as any;
        return gauntletConfig?.output?.proof_verification_enabled === true;
      },
      outputMapping: {
        lean_result: 'lean_verification_result'
      }
    },
    {
      id: 'store-results',
      name: 'Store Gauntlet Results',
      description: 'Store execution results in knowledge base',
      plugin: 'ragbits',
      action: 'ingest',
      input: {
        content: '$gauntlet_results',
        metadata: {
          documentType: 'gauntlet_result',
          gauntlet_name: '$gauntlet_name',
          execution_id: '$gauntlet_execution_id',
          timestamp: new Date().toISOString()
        }
      },
      dependsOn: ['execute-rounds']
    },
    {
      id: 'track-analytics',
      name: 'Track Analytics',
      description: 'Track gauntlet execution metrics',
      plugin: 'openevolve',
      action: 'bubblelabsAnalyticsTrack',
      input: {
        workflow_id: '$gauntlet_execution_id',
        metrics: {
          gauntlet_name: '$gauntlet_name',
          round_count: '$gauntlet_config.rounds.length',
          formal_verification: '$formal_verification_result',
          lean_verification: '$lean_verification_result'
        }
      },
      dependsOn: ['execute-rounds', 'formal-verification', 'lean-verification']
    }
  ],
  onError: 'continue',
  maxRetries: 2
};

/**
 * Decomposition Execution Workflow
 *
 * Decomposes complex problems and executes sub-problems with proper dependency management
 */
export const DECOMPOSITION_EXECUTION_WORKFLOW: WorkflowDefinition = {
  id: 'decomposition-execution',
  name: 'Decomposition Execution',
  description: 'Decompose complex problems into sub-problems, execute in dependency order, and reassemble',
  version: '1.0.0',
  steps: [
    {
      id: 'analyze-problem',
      name: 'Analyze Problem',
      description: 'Use ROMA to analyze problem structure and complexity',
      plugin: 'openevolve',
      action: 'bubblelabsRomaAnalyze',
      input: {
        problem: '$problem_statement',
        max_depth: 5
      },
      outputMapping: {
        problem_analysis: 'problem_analysis'
      }
    },
    {
      id: 'create-decomposition-plan',
      name: 'Create Decomposition Plan',
      description: 'Create workflow decomposition plan with sub-problems',
      plugin: 'openevolve',
      action: 'startEvolutionRun',
      input: {
        content: '$problem_statement',
        content_type: '$content_type',
        use_decomposition: true,
        parameters: {
          decomposition_method: '$decomposition_method',
          granularity: '$granularity',
          max_depth: '$max_depth',
          max_sub_problems: '$max_sub_problems'
        }
      },
      dependsOn: ['analyze-problem'],
      outputMapping: {
        decomposition_plan: 'decomposition_plan',
        workflow_id: 'decomposition_workflow_id'
      }
    },
    {
      id: 'get-dependency-graph',
      name: 'Get Dependency Graph',
      description: 'Retrieve dependency graph for execution ordering',
      plugin: 'openevolve',
      action: 'getWorkflowPlan',
      input: {
        workflow_id: '$decomposition_workflow_id'
      },
      dependsOn: ['create-decomposition-plan'],
      outputMapping: {
        dependency_graph: 'dependency_graph',
        execution_order: 'execution_order'
      }
    },
    {
      id: 'execute-sub-problems',
      name: 'Execute Sub-Problems',
      description: 'Execute sub-problems in dependency order',
      plugin: 'openevolve',
      action: 'createWorkflow',
      input: {
        problem_statement: '$problem_statement',
        content_analyzer_team: '$content_analyzer_team',
        planner_team: '$planner_team',
        solver_team: '$solver_team',
        patcher_team: '$patcher_team',
        assembler_team: '$assembler_team',
        sub_problem_red_gauntlet: '$red_gauntlet',
        sub_problem_gold_gauntlet: '$gold_gauntlet',
        final_red_gauntlet: '$final_red_gauntlet',
        final_gold_gauntlet: '$final_gold_gauntlet',
        max_refinement_loops: '$max_refinement_loops',
        mdap_enabled: '$mdap_enabled',
        maker_enabled: '$maker_enabled'
      },
      dependsOn: ['get-dependency-graph'],
      outputMapping: {
        workflow_id: 'sub_problem_workflow_id'
      }
    },
    {
      id: 'search-knowledge',
      name: 'Search Knowledge Base',
      description: 'Search for similar problems and solutions',
      plugin: 'ragbits',
      action: 'search',
      input: {
        query: '$problem_statement',
        topK: 5,
        searchType: 'hybrid'
      },
      outputMapping: {
        similar_solutions: 'similar_solutions'
      }
    },
    {
      id: 'reassemble-solution',
      name: 'Reassemble Solution',
      description: 'Reassemble sub-problem solutions into final result',
      plugin: 'datapizza',
      action: 'processData',
      input: {
        data: {
          problem_analysis: '$problem_analysis',
          decomposition_plan: '$decomposition_plan',
          sub_problem_results: '$sub_problem_results',
          similar_solutions: '$similar_solutions'
        },
        processingType: 'reassembly'
      },
      dependsOn: ['execute-sub-problems', 'search-knowledge'],
      outputMapping: {
        final_solution: 'final_solution'
      }
    },
    {
      id: 'validate-solution',
      name: 'Validate Final Solution',
      description: 'Run final validation through gauntlets',
      plugin: 'openevolve',
      action: 'startEvolutionRun',
      input: {
        content: '$final_solution',
        content_type: 'text_general',
        gauntlet_name: '$final_gold_gauntlet',
        evolution_mode: 'standard',
        parameters: {
          max_iterations: 3
        }
      },
      dependsOn: ['reassemble-solution'],
      outputMapping: {
        validation_result: 'validation_result'
      }
    },
    {
      id: 'store-results',
      name: 'Store Results',
      description: 'Store decomposition and solution in knowledge base',
      plugin: 'ragbits',
      action: 'ingest',
      input: {
        content: '$final_solution',
        metadata: {
          documentType: 'decomposition_result',
          workflow_id: '$decomposition_workflow_id',
          problem_statement: '$problem_statement',
          validation_passed: '$validation_result.approved',
          timestamp: new Date().toISOString()
        }
      },
      dependsOn: ['validate-solution']
    }
  ],
  onError: 'continue',
  maxRetries: 3
};

/**
 * Integrated Gauntlet + Decomposition Workflow
 *
 * Combines decomposition planning with gauntlet validation for complex problem solving
 */
export const GAUNTLET_DECOMPOSITION_WORKFLOW: WorkflowDefinition = {
  id: 'gauntlet-decomposition-integrated',
  name: 'Gauntlet + Decomposition Integration',
  description: 'Decompose problems, solve sub-problems with gauntlet validation, and reassemble',
  version: '1.0.0',
  steps: [
    {
      id: 'analyze-and-decompose',
      name: 'Analyze and Decompose',
      description: 'Analyze problem using ROMA and create decomposition plan',
      plugin: 'openevolve',
      action: 'bubblelabsRomaAnalyze',
      input: {
        problem: '$problem_statement',
        max_depth: '$max_depth'
      },
      outputMapping: {
        decomposition: 'problem_decomposition',
        sub_problems: 'sub_problems'
      }
    },
    {
      id: 'create-workflow',
      name: 'Create Workflow',
      description: 'Create workflow with decomposition plan',
      plugin: 'openevolve',
      action: 'createWorkflow',
      input: {
        problem_statement: '$problem_statement',
        content_analyzer_team: '$content_analyzer_team',
        planner_team: '$planner_team',
        solver_team: '$solver_team',
        patcher_team: '$patcher_team',
        assembler_team: '$assembler_team',
        sub_problem_red_gauntlet: '$sub_problem_red_gauntlet',
        sub_problem_gold_gauntlet: '$sub_problem_gold_gauntlet',
        final_red_gauntlet: '$final_red_gauntlet',
        final_gold_gauntlet: '$final_gold_gauntlet',
        max_refinement_loops: '$max_refinement_loops',
        mdap_enabled: true,
        maker_enabled: false
      },
      dependsOn: ['analyze-and-decompose'],
      outputMapping: {
        workflow_id: 'main_workflow_id'
      }
    },
    {
      id: 'get-workflow-plan',
      name: 'Get Workflow Plan',
      description: 'Retrieve detailed decomposition plan',
      plugin: 'openevolve',
      action: 'getWorkflowPlan',
      input: {
        workflow_id: '$main_workflow_id'
      },
      dependsOn: ['create-workflow'],
      outputMapping: {
        plan: 'decomposition_plan',
        dependency_graph: 'dependency_graph'
      }
    },
    {
      id: 'execute-sub-problem-gauntlets',
      name: 'Execute Sub-Problem Gauntlets',
      description: 'Execute each sub-problem through its assigned gauntlet',
      plugin: 'openevolve',
      action: 'startEvolutionRun',
      input: {
        content: '$sub_problem',
        content_type: '$content_type',
        gauntlet_name: '$sub_problem_gold_gauntlet',
        evolution_mode: '$evolution_mode',
        use_decomposition: false,
        parameters: {
          max_iterations: 3
        }
      },
      dependsOn: ['get-workflow-plan'],
      outputMapping: {
        sub_problem_results: 'sub_problem_results'
      },
      retryOnFailure: true
    },
    {
      id: 'get-workflow-results',
      name: 'Get Workflow Results',
      description: 'Retrieve complete workflow results',
      plugin: 'openevolve',
      action: 'getWorkflowResults',
      input: {
        workflow_id: '$main_workflow_id'
      },
      dependsOn: ['execute-sub-problem-gauntlets'],
      outputMapping: {
        workflow_results: 'complete_results'
      }
    },
    {
      id: 'final-validation',
      name: 'Final Validation Gauntlet',
      description: 'Run final solution through gold gauntlet',
      plugin: 'openevolve',
      action: 'startEvolutionRun',
      input: {
        content: '$complete_results.final_solution',
        content_type: 'text_general',
        gauntlet_name: '$final_gold_gauntlet',
        evolution_mode: 'standard',
        parameters: {
          max_iterations: 5
        }
      },
      dependsOn: ['get-workflow-results'],
      outputMapping: {
        final_validation: 'final_validation_result'
      }
    },
    {
      id: 'formal-verification',
      name: 'Formal Verification (Optional)',
      description: 'Verify solution using formal methods if applicable',
      plugin: 'openevolve',
      action: 'bubblelabsZ3Prove',
      input: {
        theorem: '$complete_results.final_solution'
      },
      dependsOn: ['get-workflow-results'],
      condition: (context) => {
        const params = context.parameters as any;
        return params?.enable_formal_verification === true;
      },
      outputMapping: {
        formal_verification: 'formal_verification_result'
      }
    },
    {
      id: 'store-and-track',
      name: 'Store and Track Results',
      description: 'Store final results and track analytics',
      plugin: 'openevolve',
      action: 'bubblelabsAnalyticsTrack',
      input: {
        workflow_id: '$main_workflow_id',
        metrics: {
          problem_statement: '$problem_statement',
          sub_problems_count: '$decomposition_plan.sub_problems.length',
          final_validation_passed: '$final_validation.approved',
          formal_verified: '$formal_verification.verified',
          total_execution_time: '$complete_results.execution_time'
        }
      },
      dependsOn: ['final-validation', 'formal-verification']
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
  'problem-solving': PROBLEM_SOLVING_WORKFLOW,
  'gauntlet-execution': GAUNTLET_EXECUTION_WORKFLOW,
  'decomposition-execution': DECOMPOSITION_EXECUTION_WORKFLOW,
  'gauntlet-decomposition-integrated': GAUNTLET_DECOMPOSITION_WORKFLOW
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
    problem: ['problem-solving'],
    gauntlet: ['gauntlet-execution'],
    decomposition: ['decomposition-execution'],
    integrated: ['gauntlet-decomposition-integrated']
  };

  const ids = categories[category] || [];
  return ids.map(id => WORKFLOW_TEMPLATES[id]).filter(Boolean) as WorkflowDefinition[];
}
