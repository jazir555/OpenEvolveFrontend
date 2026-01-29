/**
 * Hephaestus Bubble ()
 *
 * Used for backward compatibility. CrewAI replaces Hephaestus as the
 * orchestration layer; use CrewAIBubble for new integrations.
 */

export { CrewAIBubble as HephaestusBubble } from './crewai-bubble';
export type { CrewAIParams as HephaestusParams, CrewAIResult as HephaestusResult } from './crewai-bubble';
