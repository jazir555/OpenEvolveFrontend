export enum ConstraintType {
  HARD = 'hard',
  SOFT = 'soft',
}

export enum ConstraintCategoryInternal {}

export interface Constraint {
  constraint_id: string;
  type: ConstraintType;
  category: ConstraintCategoryInternal;
  description: string;
  dependencies: string[];
  created_at: Date;
}

export class SymbolicConstraintEngine {
  constructor();
  performEpistemicAudit(
    problemDescription: string,
    failurePatterns: Array<{
      pattern_description: string;
      failure_rate: number;
      data_points: number;
    }>,
    correlationId: string
  ): Promise<any>;
  addConstraint(constraint: Constraint, correlationId: string): Promise<any>;
  removeConstraint(constraintId: string, correlationId: string): Promise<any>;
  getConstraint(constraintId: string): Constraint | null;
  getAllConstraints(): Constraint[];
  detectContradictions(correlationId: string): Promise<any>;
  getStats(): any;
  resetCircuitBreakers(): void;
}
