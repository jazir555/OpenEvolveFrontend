/**
 * OpenEvolve Parameter Schemas
 *
 * Exports all parameter schemas for OpenEvolve services
 */

export { evolutionParameters } from './evolution';
export { adversarialParameters } from './adversarial';
export { makerParameters } from './maker';
export { mdapParameters } from './mdap';
export { decompositionParameters } from './decomposition';
export { knowledgeParameters } from './knowledge';
export { leanaideParameters } from './leanaide';
export { crewaiParameters } from './crewai';
export { romaParameters } from './roma';
export { inventionParameters } from './invention';
export { researchQuestConfigSchema as researchQuestParameters } from './researchQuest';
export { pyGraphistryConfigSchema as pyGraphistryParameters } from './pyGraphistry';

import { evolutionParameters } from './evolution';
import { adversarialParameters } from './adversarial';
import { makerParameters } from './maker';
import { mdapParameters } from './mdap';
import { decompositionParameters } from './decomposition';
import { knowledgeParameters } from './knowledge';
import { leanaideParameters } from './leanaide';
import { crewaiParameters } from './crewai';
import { romaParameters } from './roma';
import { inventionParameters } from './invention';
import { researchQuestConfigSchema as researchQuestParameters } from './researchQuest';
import { pyGraphistryConfigSchema as pyGraphistryParameters } from './pyGraphistry';

/**
 * Get parameter schema for a specific service
 */
export function getParametersForService(serviceId: string) {
  const schemas: Record<string, any[]> = {
    evolution: evolutionParameters,
    adversarial: adversarialParameters,
    maker: makerParameters,
    mdap: mdapParameters,
    decomposition: decompositionParameters,
    knowledge: knowledgeParameters,
    leanaide: leanaideParameters,
    crewai: crewaiParameters,
    roma: romaParameters,
    invention: inventionParameters,
    researchQuest: researchQuestParameters,
    pyGraphistry: pyGraphistryParameters,
  };

  return schemas[serviceId] || [];
}
