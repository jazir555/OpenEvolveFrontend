/**
 * Utility functions for formatting bubble parameters
 */
import { BubbleParameter, ParsedBubbleWithInfo } from '@bubblelab/shared-schemas';
/**
 * Check if a string contains function literal patterns.
 * When function literals are present, the source code must be preserved as-is
 * because functions cannot be safely serialized or condensed to single-line.
 */
export declare function containsFunctionLiteral(value: string): boolean;
export declare function buildParametersObject(parameters: BubbleParameter[], variableId?: number, includeLoggerConfig?: boolean, dependencyGraphLiteral?: string, currentUniqueId?: string): string;
/**
 * Format a parameter value based on its type
 */
export declare function formatParameterValue(value: unknown, type: string): string;
/**
 * Try to parse a tools parameter that may be provided as JSON or a JS-like array literal.
 * Returns an array of objects with at least a name field, or null if parsing fails.
 */
export declare function parseToolsParamValue(raw: unknown): Array<Record<string, unknown>> | null;
/**
 * Condense a parameters string to a single line.
 * Used when parameters don't contain function literals.
 */
export declare function condenseToSingleLine(input: string): string;
/**
 * Replace a bubble instantiation with updated parameters
 *
 * This function:
 * 1. Replaces the bubble instantiation line with updated parameters
 * 2. Preserves multi-line structure when function literals are present
 * 3. Condenses to single-line otherwise
 * 4. Uses bubble.location.endLine to know exactly where to stop deleting
 */
export declare function replaceBubbleInstantiation(lines: string[], bubble: ParsedBubbleWithInfo): void;
//# sourceMappingURL=parameter-formatter.d.ts.map