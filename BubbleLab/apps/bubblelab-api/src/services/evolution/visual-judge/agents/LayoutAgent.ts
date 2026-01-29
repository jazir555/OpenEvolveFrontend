import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';
import { BaseVisualJudge } from '../base-visual-judge.js';
import { layoutPrompt } from '../prompts.js';

export class LayoutAgent extends BaseVisualJudge {
  constructor() {
    super({
      agentName: 'LayoutAgent',
      provider: 'openai',
      systemPrompt: layoutPrompt,
      model: RECOMMENDED_MODELS.BEST_ALT,
      temperature: 0.2,
    });
  }
}
