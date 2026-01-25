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
export { hephaestusParameters } from './hephaestus';
export { romaParameters } from './roma';
export { inventionParameters } from './invention';
/**
 * Get parameter schema for a specific service
 */
export declare function getParametersForService(serviceId: string): any[];
