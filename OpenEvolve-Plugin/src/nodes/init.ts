// @ts-nocheck
/**
 * Auto-register all OpenEvolve nodes
 *
 * This file ensures all core nodes are registered when the nodes module
 * is imported. Users can also manually register additional nodes.
 *
 * @module init
 * @version 1.0.0
 */

import { DecompositionNode } from './DecompositionNode';
import { SolutionNode } from './SolutionNode';
import { VerificationNode } from './VerificationNode';
import { EvolutionNode } from './EvolutionNode';
import { AdversarialNode } from './AdversarialNode';
import { KnowledgeQueryNode } from './KnowledgeQueryNode';
import { LeanAideNode } from './LeanAIDENode';
import { CrewAINode } from './CrewAINode';
import { MDAPNode } from './MDAPNode';
import { MAKERNode } from './MAKERNode';
import { ROMANode } from './ROMANode';
import { InventionNode } from './InventionNode';
import { SubProblemNode } from './SubProblemNode';
import { GauntletNode } from './GauntletNode';
import { AssemblyNode } from './AssemblyNode';
import { OutputNode } from './OutputNode';
import { KnowledgeExtractionNode } from './KnowledgeExtractionNode';
import { ResearchQuestNode } from './ResearchQuestNode';
import { PyGraphistryNode } from './PyGraphistryNode';
import { registerNodes } from './registry';

// Register all core nodes
registerNodes({
  Decomposition: DecompositionNode,
  Solution: SolutionNode,
  Verification: VerificationNode,
  Evolution: EvolutionNode,
  Adversarial: AdversarialNode,
  KnowledgeQuery: KnowledgeQueryNode,
  LeanAIDE: LeanAideNode,
  CrewAI: CrewAINode,
  MDAP: MDAPNode,
  MAKER: MAKERNode,
  ROMA: ROMANode,
  Invention: InventionNode,
  SubProblem: SubProblemNode,
  Gauntlet: GauntletNode,
  Assembly: AssemblyNode,
  Output: OutputNode,
  KnowledgeExtraction: KnowledgeExtractionNode,
  ResearchQuest: ResearchQuestNode,
  PyGraphistry: PyGraphistryNode,
}, {
  source: 'core',
  validate: true,
  allowDuplicates: false,
});

// Export for convenience
export {
  DecompositionNode,
  SolutionNode,
  VerificationNode,
  EvolutionNode,
  AdversarialNode,
  KnowledgeQueryNode,
  LeanAideNode,
  CrewAINode,
  MDAPNode,
  MAKERNode,
  ROMANode,
  InventionNode,
  SubProblemNode,
  GauntletNode,
  AssemblyNode,
  OutputNode,
  KnowledgeExtractionNode,
  ResearchQuestNode,
  PyGraphistryNode
};
