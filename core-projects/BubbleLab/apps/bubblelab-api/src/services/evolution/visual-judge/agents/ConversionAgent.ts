import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';
import { BaseVisualJudge } from '../base-visual-judge.js';
import { conversionPrompt } from '../prompts.js';

export class ConversionAgent extends BaseVisualJudge {
  constructor() {
    super({
      agentName: 'ConversionAgent',
      provider: 'openai',
      systemPrompt: conversionPrompt,
      model: RECOMMENDED_MODELS.FAST_ALT,
      temperature: 0.2,
    });
  }
}
