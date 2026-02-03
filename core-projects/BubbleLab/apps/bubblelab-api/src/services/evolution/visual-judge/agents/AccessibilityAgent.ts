import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';
import { BaseVisualJudge } from '../base-visual-judge.js';
import { accessibilityPrompt } from '../prompts.js';

export class AccessibilityAgent extends BaseVisualJudge {
  constructor() {
    super({
      agentName: 'AccessibilityAgent',
      provider: 'anthropic',
      systemPrompt: accessibilityPrompt,
      model: RECOMMENDED_MODELS.PRO_ALT,
      temperature: 0.2,
    });
  }
}
