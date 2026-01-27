/**
 * Normalizes braceless control flow statements by wrapping them with braces.
 * This prevents issues when injecting logging or other statements into the code.
 *
 * Transforms:
 *   if (x) doSomething();
 *   else doOther();
 *
 * Into:
 *   if (x) { doSomething(); }
 *   else { doOther(); }
 */
export declare function normalizeBracelessControlFlow(code: string): string;
//# sourceMappingURL=normalize-control-flow.d.ts.map