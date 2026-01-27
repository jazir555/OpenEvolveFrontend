/**
 * Utility functions for parsing and working with cron expressions
 * Supports standard 5-field cron format: minute hour day month day-of-week
 */
export interface CronExpression {
    minute: string;
    hour: string;
    dayOfMonth: string;
    month: string;
    dayOfWeek: string;
    original: string;
}
export interface CronScheduleInfo {
    expression: CronExpression;
    description: string;
    nextRun?: Date;
    isValid: boolean;
    error?: string;
}
/**
 * Parse a cron expression string into its components
 * @param cronString - Cron expression (e.g., "0 0 * * *")
 * @returns Parsed cron expression object
 */
export declare function parseCronExpression(cronString: string): CronExpression;
/**
 * Validate a cron expression
 * @param cronString - Cron expression to validate
 * @returns Object with validation result
 */
export declare function validateCronExpression(cronString: string): {
    valid: boolean;
    error?: string;
};
/**
 * Generate a human-readable description of a cron expression
 * @param cronString - Cron expression to describe
 * @returns Human-readable description
 */
export declare function describeCronExpression(cronString: string): string;
/**
 * Get schedule information for a cron expression
 * @param cronString - Cron expression
 * @returns Schedule information including description and validation status
 */
export declare function getCronScheduleInfo(cronString: string): CronScheduleInfo;
//# sourceMappingURL=cron-utils.d.ts.map