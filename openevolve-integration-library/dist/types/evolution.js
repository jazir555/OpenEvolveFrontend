"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
    ** Defense;
strategy * /;
defenseStrategy ?  : 'adversarial-training' | 'distillation' | 'ensemble';
metrics: string[];
    ** Robustness;
score * /;
robustnessScore: number;
metricsHistory: MetricsSnapshot[];
metadata: EvolutionMetadata;
    ** Confidence;
before * /;
confidenceBefore ?  : number;
confidenceAfter ?  : number;
//# sourceMappingURL=evolution.js.map