import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';
import { BaseVisualJudge } from '../base-visual-judge.js';
import { brandPrompt } from '../prompts.js';

export class BrandAgent extends BaseVisualJudge {
  constructor() {
    super({
      agentName: 'BrandAgent',
      provider: 'google',
      systemPrompt: brandPrompt,
      model: RECOMMENDED_MODELS.PRO,
      temperature: 0.2,
    });
  }
}
