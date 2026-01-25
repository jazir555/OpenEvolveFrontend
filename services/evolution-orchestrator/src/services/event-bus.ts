import { EventEmitter } from 'events';

export type EvolutionEventType =
  | 'generation_start'
  | 'design_evaluated'
  | 'generation_complete'
  | 'evolution_complete'
  | 'error';

export type EvolutionEvent = {
  type: EvolutionEventType;
  version: number;
  timestamp: string;
  payload: Record<string, unknown>;
};

export class EvolutionEventBus {
  private emitter = new EventEmitter();
  private version = 1;

  emit(event: Omit<EvolutionEvent, 'version' | 'timestamp'>) {
    const payload: EvolutionEvent = {
      ...event,
      version: this.version,
      timestamp: new Date().toISOString(),
    };
    this.emitter.emit(event.type, payload);
    this.emitter.emit('event', payload);
  }

  on(type: EvolutionEventType | 'event', listener: (event: EvolutionEvent) => void) {
    this.emitter.on(type, listener);
  }
}
