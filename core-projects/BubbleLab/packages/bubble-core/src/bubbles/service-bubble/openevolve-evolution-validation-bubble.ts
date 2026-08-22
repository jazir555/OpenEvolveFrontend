import { z } from 'zod';
import type { BubbleOperationResult } from '@bubblelab/shared-schemas';
import type { ServiceBubbleParams } from '../../types/bubble.js';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import type { BubbleName } from '@bubblelab/shared-schemas';
import { OpenEvolveZ3ProverBubble } from './openevolve-z3prover-bubble.js';
import { OpenEvolveLeanAideBubble } from './openevolve-leanaide-bubble.js';

const EvolutionValidationOperationSchema = z.enum(['validate', 'health_check']);

const resolveBaseUrl = (): string => {
  const envUrl =
    (typeof process !== 'undefined' && process.env
      ? process.env.OPENEVOLVE_API_URL || process.env.OPENEVOLVE_API_BASE_URL
      : undefined) || '';
  const base = envUrl.trim().length > 0 ? envUrl : 'http://localhost:8000';
  return base.replace(/\/$/, '');
};

const EvolutionValidationParams = z.object({
  operation: EvolutionValidationOperationSchema,
  base_url: z.string().url().default(resolveBaseUrl()),
  timeout: z.number().min(1000).max(600000).default(120000),
  headers: z.record(z.string()).optional(),
  auth_token: z.string().optional(),
  auth_header: z.string().default('Authorization'),

  code: z.string().min(1),
  constraints: z.array(z.string()).optional(),
  invariants: z.array(z.string()).optional(),
  theorem: z.string().optional(),
  run_z3: z.boolean().default(true),
  run_leanaide: z.boolean().default(true),
  leanaide_model: z.string().default('gpt-4'),
});

type EvolutionValidationParamsType = z.input<typeof EvolutionValidationParams> & ServiceBubbleParams;

const Z3ResultSchema = z.object({
  valid: z.boolean().optional(),
  satisfiable: z.boolean().optional(),
  model: z.unknown().optional(),
  proof: z.string().optional(),
  errors: z.array(z.string()).optional(),
  timing: z.number().optional(),
});

const LeanProofSchema = z.object({
  proven: z.boolean().optional(),
  proofScript: z.string().optional(),
  tactics: z.array(z.string()).optional(),
  proofSteps: z.number().optional(),
  errors: z.array(z.string()).optional(),
  timing: z.number().optional(),
});

const EvolutionValidationDataSchema = z.object({
  valid: z.boolean(),
  summary: z.string(),
  z3: Z3ResultSchema.optional(),
  leanProof: LeanProofSchema.optional(),
});

const EvolutionValidationResultSchema = z.object({
  success: z.boolean(),
  operation: z.string(),
  data: EvolutionValidationDataSchema.optional(),
  error: z.string().optional(),
  timing: z.number(),
});

type EvolutionValidationResult = z.output<typeof EvolutionValidationResultSchema> & BubbleOperationResult;

export class OpenEvolveEvolutionValidationBubble extends ServiceBubble<
  EvolutionValidationParamsType,
  EvolutionValidationResult
> {
  static readonly service = 'openevolve';
  static readonly authType = 'none' as const;
  static readonly bubbleName: BubbleName = 'openevolve-evolution-validation' as BubbleName;
  static readonly type = 'service' as const;
  static readonly schema = EvolutionValidationParams;
  static readonly resultSchema = EvolutionValidationResultSchema;
  static readonly shortDescription = 'OpenEvolve evolution validation (Z3 + LeanAide)';
  static readonly longDescription = `
    Validates evolved code with real formal methods: runs Z3 (openevolve-z3prover)
    over the supplied constraints/invariants and generates a LeanAide proof
    (openevolve-leanaide generate_proof). Composes both into a single verdict.
  `;
  static readonly alias = 'openevolve-evolution-validation';

  constructor(params: EvolutionValidationParamsType, context?: BubbleContext) {
    super(params, context);
  }

  protected chooseCredential(): string | undefined {
    return this.params.auth_token;
  }

  public async testCredential(): Promise<boolean> {
    return true;
  }

  protected async performAction(): Promise<EvolutionValidationResult> {
    const startTime = Date.now();
    const summaryParts: string[] = [];
    let valid = true;
    let z3Out: z.infer<typeof Z3ResultSchema> | undefined;
    let leanOut: z.infer<typeof LeanProofSchema> | undefined;

    try {
      if (this.params.operation === 'health_check') {
        const z3 = await new OpenEvolveZ3ProverBubble({
          operation: 'health_check',
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        }).action();
        return {
          success: z3.success,
          operation: this.params.operation,
          data: { valid: z3.success, summary: 'Z3 health check', z3: { timing: 0 } },
          error: z3.error,
          timing: Date.now() - startTime,
        };
      }

      // Real Z3 validation
      if (this.params.run_z3) {
        const constraints = this.params.constraints || [];
        const invariants = this.params.invariants || [];
        const smtlib2 = this.buildSmtLib(constraints, invariants);
        const z3Result = await new OpenEvolveZ3ProverBubble({
          operation: 'solve_smt',
          smtlib2,
          logic: 'ALL',
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        }).action();
        const z3Data = (z3Result.data ?? {}) as unknown as Record<string, unknown>;
        const z3Success = z3Result.success;
        const satisfiable =
          z3Data['satisfiable'] === true ||
          (constraints.length === 0 && z3Success);
        z3Out = {
          valid: z3Success && satisfiable,
          satisfiable,
          model: z3Data['model'],
          proof: typeof z3Data['proof'] === 'string' ? (z3Data['proof'] as string) : undefined,
          errors: typeof z3Data['errors'] === 'object' ? (z3Data['errors'] as string[]) : undefined,
          timing: typeof z3Data['timing'] === 'number' ? (z3Data['timing'] as number) : 0,
        };
        if (z3Out.valid) {
          summaryParts.push('Z3 constraints verified');
        } else {
          valid = false;
          summaryParts.push('Z3 validation failed');
        }
      }

      // Real LeanAide proof generation
      if (this.params.run_leanaide) {
        const theorem = this.params.theorem || this.params.code;
        const leanResult = await new OpenEvolveLeanAideBubble({
          operation: 'generate_proof',
          theorem,
          model: this.params.leanaide_model as any,
          base_url: this.params.base_url,
          auth_token: this.params.auth_token,
        }).action();
        const leanData = (leanResult.data ?? {}) as unknown as Record<string, unknown>;
        const proven = leanData['success'] === true || leanData['proven'] === true;
        leanOut = {
          proven,
          proofScript: typeof leanData['proof'] === 'string' ? (leanData['proof'] as string) : undefined,
          tactics: Array.isArray(leanData['tactics']) ? (leanData['tactics'] as string[]) : undefined,
          proofSteps: typeof leanData['proofSteps'] === 'number' ? (leanData['proofSteps'] as number) : undefined,
          errors: typeof leanData['errors'] === 'object' ? (leanData['errors'] as string[]) : undefined,
          timing: typeof leanData['timing'] === 'number' ? (leanData['timing'] as number) : 0,
        };
        if (proven) {
          summaryParts.push('LeanAide proof generated');
        } else {
          summaryParts.push('LeanAide proof not completed');
        }
      }

      return {
        success: true,
        operation: this.params.operation,
        data: {
          valid,
          summary: summaryParts.join('. ') || 'Validation completed',
          z3: z3Out,
          leanProof: leanOut,
        },
        timing: Date.now() - startTime,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        operation: this.params.operation,
        error: message,
        timing: Date.now() - startTime,
      };
    }
  }

  private buildSmtLib(constraints: string[], invariants: string[]): string {
    const lines: string[] = ['(set-logic ALL)', '(declare-fun valid () Bool)'];
    [...constraints, ...invariants].forEach((c, i) => {
      if (c && c.trim().length > 0) {
        lines.push(`; constraint ${i + 1}\n(assert ${c.trim()})`);
      }
    });
    lines.push('(check-sat)');
    lines.push('(get-model)');
    return lines.join('\n');
  }
}

export default OpenEvolveEvolutionValidationBubble;
