import { randomUUID } from 'crypto';
import type {
  EvolutionResult,
  EvolutionRunResponse,
  EvolutionStartRequest,
} from './types.js';
import { mutateDesign } from './services/mutation.js';
import { renderHtml } from './services/screenshot.js';
import { judgeDesign } from './services/judge.js';
import { EvolutionEventBus } from './services/event-bus.js';

export class EvolutionOrchestrator {
  private eventBus = new EvolutionEventBus();

  get events() {
    return this.eventBus;
  }

  async runEvolution(request: EvolutionStartRequest): Promise<EvolutionRunResponse> {
    const runId = randomUUID();
    const iterations = request.iterations ?? 5;

    let current: EvolutionResult = {
      html: request.html,
      css: request.css,
      score: 0,
      changes: ['Initial design'],
    };

    const history: EvolutionResult[] = [current];

    try {
      for (let i = 0; i < iterations; i += 1) {
        this.eventBus.emit({
          type: 'generation_start',
          payload: { runId, iteration: i + 1, totalIterations: iterations },
        });

        const mutation = await mutateDesign(current.html, current.css);
        const screenshot = await renderHtml(mutation.html);
        const judgment = await judgeDesign(
          screenshot.image_base64,
          request.criteria
        );

        const candidate: EvolutionResult = {
          html: mutation.html,
          css: mutation.css,
          score: judgment.score,
          changes: mutation.changes,
        };

        this.eventBus.emit({
          type: 'design_evaluated',
          payload: {
            runId,
            iteration: i + 1,
            score: candidate.score,
            changes: candidate.changes,
          },
        });

        history.push(candidate);
        if (candidate.score >= current.score) {
          current = candidate;
        }

        this.eventBus.emit({
          type: 'generation_complete',
          payload: {
            runId,
            iteration: i + 1,
            bestScore: current.score,
          },
        });
      }
    } catch (error) {
      this.eventBus.emit({
        type: 'error',
        payload: {
          runId,
          message: error instanceof Error ? error.message : 'Unknown error',
        },
      });
      throw error;
    }

    this.eventBus.emit({
      type: 'evolution_complete',
      payload: { runId, bestScore: current.score, iterations },
    });

    return {
      runId,
      best: current,
      history,
    };
  }
}
