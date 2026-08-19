export type ConstraintCategory = any;

export type EpistemicAuditResult = any;

export function validateEpistemicAuditResult(data: unknown): {
  success: boolean;
  data?: EpistemicAuditResult;
  errors?: string[];
};
