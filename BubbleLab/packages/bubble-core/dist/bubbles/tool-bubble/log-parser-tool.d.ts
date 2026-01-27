/**
 * LOG PARSER TOOL
 *
 * A comprehensive tool for parsing, analyzing, and extracting structured data from log files.
 * Supports multiple log formats and provides powerful filtering and aggregation capabilities.
 *
 * Features:
 * - Multi-format log parsing (Apache, Nginx, JSON, CSV, Custom)
 * - Pattern matching with regex
 * - Log level filtering
 * - Time-based filtering
 * - Aggregation and statistics
 * - Error detection and alerting
 * - Log enrichment
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parse operation schema
 */
declare const LogParserToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["parse", "filter", "aggregate", "detect", "enrich", "transform", "analyze"]>;
    logData: z.ZodString;
    format: z.ZodOptional<z.ZodDefault<z.ZodEnum<["apache", "nginx", "json", "csv", "syslog", "custom", "auto"]>>>;
    customPattern: z.ZodOptional<z.ZodString>;
    timestampFormat: z.ZodOptional<z.ZodString>;
    filterLevel: z.ZodOptional<z.ZodArray<z.ZodEnum<["DEBUG", "INFO", "WARN", "ERROR", "FATAL", "TRACE"]>, "many">>;
    filterSource: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    filterTimeRange: z.ZodOptional<z.ZodObject<{
        start: z.ZodOptional<z.ZodString>;
        end: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        start?: string | undefined;
        end?: string | undefined;
    }, {
        start?: string | undefined;
        end?: string | undefined;
    }>>;
    filterPattern: z.ZodOptional<z.ZodString>;
    aggregateBy: z.ZodOptional<z.ZodArray<z.ZodEnum<["level", "source", "hour", "day"]>, "many">>;
    detectErrors: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    detectAnomalies: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    enrichWithGeo: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    includeRaw: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    operation: "filter" | "parse" | "transform" | "aggregate" | "detect" | "enrich" | "analyze";
    logData: string;
    format?: "custom" | "json" | "auto" | "csv" | "apache" | "nginx" | "syslog" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
    limit?: number | undefined;
    customPattern?: string | undefined;
    timestampFormat?: string | undefined;
    filterLevel?: ("ERROR" | "INFO" | "DEBUG" | "WARN" | "FATAL" | "TRACE")[] | undefined;
    filterSource?: string[] | undefined;
    filterTimeRange?: {
        start?: string | undefined;
        end?: string | undefined;
    } | undefined;
    filterPattern?: string | undefined;
    aggregateBy?: ("hour" | "source" | "day" | "level")[] | undefined;
    detectErrors?: boolean | undefined;
    detectAnomalies?: boolean | undefined;
    enrichWithGeo?: boolean | undefined;
    includeRaw?: boolean | undefined;
}, {
    operation: "filter" | "parse" | "transform" | "aggregate" | "detect" | "enrich" | "analyze";
    logData: string;
    format?: "custom" | "json" | "auto" | "csv" | "apache" | "nginx" | "syslog" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
    limit?: number | undefined;
    customPattern?: string | undefined;
    timestampFormat?: string | undefined;
    filterLevel?: ("ERROR" | "INFO" | "DEBUG" | "WARN" | "FATAL" | "TRACE")[] | undefined;
    filterSource?: string[] | undefined;
    filterTimeRange?: {
        start?: string | undefined;
        end?: string | undefined;
    } | undefined;
    filterPattern?: string | undefined;
    aggregateBy?: ("hour" | "source" | "day" | "level")[] | undefined;
    detectErrors?: boolean | undefined;
    detectAnomalies?: boolean | undefined;
    enrichWithGeo?: boolean | undefined;
    includeRaw?: boolean | undefined;
}>;
/**
 * Result schema
 */
declare const LogParserToolResultSchema: z.ZodObject<{
    operation: z.ZodString;
    entries: z.ZodArray<z.ZodObject<{
        timestamp: z.ZodOptional<z.ZodString>;
        level: z.ZodOptional<z.ZodString>;
        message: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        lineNumber: z.ZodNumber;
        raw: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }, {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }>, "many">;
    statistics: z.ZodObject<{
        totalEntries: z.ZodNumber;
        byLevel: z.ZodRecord<z.ZodString, z.ZodNumber>;
        bySource: z.ZodRecord<z.ZodString, z.ZodNumber>;
        timeRange: z.ZodObject<{
            start: z.ZodOptional<z.ZodString>;
            end: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            start?: string | undefined;
            end?: string | undefined;
        }, {
            start?: string | undefined;
            end?: string | undefined;
        }>;
        errorsCount: z.ZodNumber;
        warningsCount: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        timeRange: {
            start?: string | undefined;
            end?: string | undefined;
        };
        totalEntries: number;
        byLevel: Record<string, number>;
        bySource: Record<string, number>;
        errorsCount: number;
        warningsCount: number;
    }, {
        timeRange: {
            start?: string | undefined;
            end?: string | undefined;
        };
        totalEntries: number;
        byLevel: Record<string, number>;
        bySource: Record<string, number>;
        errorsCount: number;
        warningsCount: number;
    }>;
    errors: z.ZodOptional<z.ZodArray<z.ZodObject<{
        timestamp: z.ZodOptional<z.ZodString>;
        level: z.ZodOptional<z.ZodString>;
        message: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        lineNumber: z.ZodNumber;
        raw: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }, {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }>, "many">>;
    anomalies: z.ZodOptional<z.ZodArray<z.ZodObject<{
        entry: z.ZodObject<{
            timestamp: z.ZodOptional<z.ZodString>;
            level: z.ZodOptional<z.ZodString>;
            message: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            lineNumber: z.ZodNumber;
            raw: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }, {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }>;
        reason: z.ZodString;
        confidence: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        confidence: number;
        entry: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        };
        reason: string;
    }, {
        confidence: number;
        entry: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        };
        reason: string;
    }>, "many">>;
    aggregations: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodNumber>>>;
    metadata: z.ZodObject<{
        format: z.ZodString;
        entriesParsed: z.ZodNumber;
        entriesFiltered: z.ZodNumber;
        parseErrors: z.ZodNumber;
        parseTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        format: string;
        entriesParsed: number;
        entriesFiltered: number;
        parseErrors: number;
        parseTime: number;
    }, {
        format: string;
        entriesParsed: number;
        entriesFiltered: number;
        parseErrors: number;
        parseTime: number;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    entries: {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }[];
    success: boolean;
    operation: string;
    metadata: {
        format: string;
        entriesParsed: number;
        entriesFiltered: number;
        parseErrors: number;
        parseTime: number;
    };
    statistics: {
        timeRange: {
            start?: string | undefined;
            end?: string | undefined;
        };
        totalEntries: number;
        byLevel: Record<string, number>;
        bySource: Record<string, number>;
        errorsCount: number;
        warningsCount: number;
    };
    errors?: {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }[] | undefined;
    aggregations?: Record<string, Record<string, number>> | undefined;
    anomalies?: {
        confidence: number;
        entry: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        };
        reason: string;
    }[] | undefined;
}, {
    error: string;
    entries: {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }[];
    success: boolean;
    operation: string;
    metadata: {
        format: string;
        entriesParsed: number;
        entriesFiltered: number;
        parseErrors: number;
        parseTime: number;
    };
    statistics: {
        timeRange: {
            start?: string | undefined;
            end?: string | undefined;
        };
        totalEntries: number;
        byLevel: Record<string, number>;
        bySource: Record<string, number>;
        errorsCount: number;
        warningsCount: number;
    };
    errors?: {
        message: string;
        raw: string;
        lineNumber: number;
        timestamp?: string | undefined;
        metadata?: Record<string, unknown> | undefined;
        source?: string | undefined;
        level?: string | undefined;
    }[] | undefined;
    aggregations?: Record<string, Record<string, number>> | undefined;
    anomalies?: {
        confidence: number;
        entry: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        };
        reason: string;
    }[] | undefined;
}>;
type LogParserToolParams = z.output<typeof LogParserToolParamsSchema>;
type LogParserToolResult = z.output<typeof LogParserToolResultSchema>;
type LogParserToolParamsInput = z.input<typeof LogParserToolParamsSchema>;
/**
 * Log Parser Tool
 * Parses and analyzes log files
 */
export declare class LogParserTool extends ToolBubble<LogParserToolParams, LogParserToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["parse", "filter", "aggregate", "detect", "enrich", "transform", "analyze"]>;
        logData: z.ZodString;
        format: z.ZodOptional<z.ZodDefault<z.ZodEnum<["apache", "nginx", "json", "csv", "syslog", "custom", "auto"]>>>;
        customPattern: z.ZodOptional<z.ZodString>;
        timestampFormat: z.ZodOptional<z.ZodString>;
        filterLevel: z.ZodOptional<z.ZodArray<z.ZodEnum<["DEBUG", "INFO", "WARN", "ERROR", "FATAL", "TRACE"]>, "many">>;
        filterSource: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        filterTimeRange: z.ZodOptional<z.ZodObject<{
            start: z.ZodOptional<z.ZodString>;
            end: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            start?: string | undefined;
            end?: string | undefined;
        }, {
            start?: string | undefined;
            end?: string | undefined;
        }>>;
        filterPattern: z.ZodOptional<z.ZodString>;
        aggregateBy: z.ZodOptional<z.ZodArray<z.ZodEnum<["level", "source", "hour", "day"]>, "many">>;
        detectErrors: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        detectAnomalies: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        enrichWithGeo: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        includeRaw: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        operation: "filter" | "parse" | "transform" | "aggregate" | "detect" | "enrich" | "analyze";
        logData: string;
        format?: "custom" | "json" | "auto" | "csv" | "apache" | "nginx" | "syslog" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
        limit?: number | undefined;
        customPattern?: string | undefined;
        timestampFormat?: string | undefined;
        filterLevel?: ("ERROR" | "INFO" | "DEBUG" | "WARN" | "FATAL" | "TRACE")[] | undefined;
        filterSource?: string[] | undefined;
        filterTimeRange?: {
            start?: string | undefined;
            end?: string | undefined;
        } | undefined;
        filterPattern?: string | undefined;
        aggregateBy?: ("hour" | "source" | "day" | "level")[] | undefined;
        detectErrors?: boolean | undefined;
        detectAnomalies?: boolean | undefined;
        enrichWithGeo?: boolean | undefined;
        includeRaw?: boolean | undefined;
    }, {
        operation: "filter" | "parse" | "transform" | "aggregate" | "detect" | "enrich" | "analyze";
        logData: string;
        format?: "custom" | "json" | "auto" | "csv" | "apache" | "nginx" | "syslog" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
        limit?: number | undefined;
        customPattern?: string | undefined;
        timestampFormat?: string | undefined;
        filterLevel?: ("ERROR" | "INFO" | "DEBUG" | "WARN" | "FATAL" | "TRACE")[] | undefined;
        filterSource?: string[] | undefined;
        filterTimeRange?: {
            start?: string | undefined;
            end?: string | undefined;
        } | undefined;
        filterPattern?: string | undefined;
        aggregateBy?: ("hour" | "source" | "day" | "level")[] | undefined;
        detectErrors?: boolean | undefined;
        detectAnomalies?: boolean | undefined;
        enrichWithGeo?: boolean | undefined;
        includeRaw?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodString;
        entries: z.ZodArray<z.ZodObject<{
            timestamp: z.ZodOptional<z.ZodString>;
            level: z.ZodOptional<z.ZodString>;
            message: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            lineNumber: z.ZodNumber;
            raw: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }, {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }>, "many">;
        statistics: z.ZodObject<{
            totalEntries: z.ZodNumber;
            byLevel: z.ZodRecord<z.ZodString, z.ZodNumber>;
            bySource: z.ZodRecord<z.ZodString, z.ZodNumber>;
            timeRange: z.ZodObject<{
                start: z.ZodOptional<z.ZodString>;
                end: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                start?: string | undefined;
                end?: string | undefined;
            }, {
                start?: string | undefined;
                end?: string | undefined;
            }>;
            errorsCount: z.ZodNumber;
            warningsCount: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            timeRange: {
                start?: string | undefined;
                end?: string | undefined;
            };
            totalEntries: number;
            byLevel: Record<string, number>;
            bySource: Record<string, number>;
            errorsCount: number;
            warningsCount: number;
        }, {
            timeRange: {
                start?: string | undefined;
                end?: string | undefined;
            };
            totalEntries: number;
            byLevel: Record<string, number>;
            bySource: Record<string, number>;
            errorsCount: number;
            warningsCount: number;
        }>;
        errors: z.ZodOptional<z.ZodArray<z.ZodObject<{
            timestamp: z.ZodOptional<z.ZodString>;
            level: z.ZodOptional<z.ZodString>;
            message: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            lineNumber: z.ZodNumber;
            raw: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }, {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }>, "many">>;
        anomalies: z.ZodOptional<z.ZodArray<z.ZodObject<{
            entry: z.ZodObject<{
                timestamp: z.ZodOptional<z.ZodString>;
                level: z.ZodOptional<z.ZodString>;
                message: z.ZodString;
                source: z.ZodOptional<z.ZodString>;
                metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                lineNumber: z.ZodNumber;
                raw: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                message: string;
                raw: string;
                lineNumber: number;
                timestamp?: string | undefined;
                metadata?: Record<string, unknown> | undefined;
                source?: string | undefined;
                level?: string | undefined;
            }, {
                message: string;
                raw: string;
                lineNumber: number;
                timestamp?: string | undefined;
                metadata?: Record<string, unknown> | undefined;
                source?: string | undefined;
                level?: string | undefined;
            }>;
            reason: z.ZodString;
            confidence: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            confidence: number;
            entry: {
                message: string;
                raw: string;
                lineNumber: number;
                timestamp?: string | undefined;
                metadata?: Record<string, unknown> | undefined;
                source?: string | undefined;
                level?: string | undefined;
            };
            reason: string;
        }, {
            confidence: number;
            entry: {
                message: string;
                raw: string;
                lineNumber: number;
                timestamp?: string | undefined;
                metadata?: Record<string, unknown> | undefined;
                source?: string | undefined;
                level?: string | undefined;
            };
            reason: string;
        }>, "many">>;
        aggregations: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodNumber>>>;
        metadata: z.ZodObject<{
            format: z.ZodString;
            entriesParsed: z.ZodNumber;
            entriesFiltered: z.ZodNumber;
            parseErrors: z.ZodNumber;
            parseTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            format: string;
            entriesParsed: number;
            entriesFiltered: number;
            parseErrors: number;
            parseTime: number;
        }, {
            format: string;
            entriesParsed: number;
            entriesFiltered: number;
            parseErrors: number;
            parseTime: number;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        entries: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }[];
        success: boolean;
        operation: string;
        metadata: {
            format: string;
            entriesParsed: number;
            entriesFiltered: number;
            parseErrors: number;
            parseTime: number;
        };
        statistics: {
            timeRange: {
                start?: string | undefined;
                end?: string | undefined;
            };
            totalEntries: number;
            byLevel: Record<string, number>;
            bySource: Record<string, number>;
            errorsCount: number;
            warningsCount: number;
        };
        errors?: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }[] | undefined;
        aggregations?: Record<string, Record<string, number>> | undefined;
        anomalies?: {
            confidence: number;
            entry: {
                message: string;
                raw: string;
                lineNumber: number;
                timestamp?: string | undefined;
                metadata?: Record<string, unknown> | undefined;
                source?: string | undefined;
                level?: string | undefined;
            };
            reason: string;
        }[] | undefined;
    }, {
        error: string;
        entries: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }[];
        success: boolean;
        operation: string;
        metadata: {
            format: string;
            entriesParsed: number;
            entriesFiltered: number;
            parseErrors: number;
            parseTime: number;
        };
        statistics: {
            timeRange: {
                start?: string | undefined;
                end?: string | undefined;
            };
            totalEntries: number;
            byLevel: Record<string, number>;
            bySource: Record<string, number>;
            errorsCount: number;
            warningsCount: number;
        };
        errors?: {
            message: string;
            raw: string;
            lineNumber: number;
            timestamp?: string | undefined;
            metadata?: Record<string, unknown> | undefined;
            source?: string | undefined;
            level?: string | undefined;
        }[] | undefined;
        aggregations?: Record<string, Record<string, number>> | undefined;
        anomalies?: {
            confidence: number;
            entry: {
                message: string;
                raw: string;
                lineNumber: number;
                timestamp?: string | undefined;
                metadata?: Record<string, unknown> | undefined;
                source?: string | undefined;
                level?: string | undefined;
            };
            reason: string;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Parse and analyze log files";
    static readonly longDescription = "\n    Comprehensive log parsing and analysis tool.\n\n    Operations:\n    - parse: Parse logs into structured format\n    - filter: Filter logs by level, source, time, pattern\n    - aggregate: Aggregate logs by dimensions\n    - detect: Detect errors and anomalies\n    - enrich: Enrich logs with additional data\n    - transform: Transform log format\n    - analyze: Deep analysis with AI\n\n    Supported formats:\n    - Apache access logs\n    - Nginx access logs\n    - JSON logs\n    - CSV logs\n    - Syslog format\n    - Custom regex patterns\n\n    Features:\n    - Multi-format parsing\n    - Pattern matching\n    - Time-based filtering\n    - Error detection\n    - Anomaly detection\n    - Statistics generation\n    - Geo IP enrichment\n  ";
    static readonly alias = "log-parser";
    static readonly type = "tool";
    private readonly APACHE_PATTERN;
    private readonly NGINX_PATTERN;
    private readonly SYSLOG_PATTERN;
    constructor(params?: LogParserToolParamsInput, context?: BubbleContext);
    performAction(): Promise<LogParserToolResult>;
    private detectFormat;
    private parseLogs;
    private parseJSONLine;
    private parseApacheLine;
    private parseNginxLine;
    private parseSyslogLine;
    private parseCSVLine;
    /**
     * Parse CSV fields handling quoted strings with embedded delimiters
     * Supports:
     * - Quoted fields: "value,with,commas"
     * - Escaped quotes: "value with ""quotes"""
     * - Mixed quoted and unquoted fields
     */
    private parseCSVFields;
    private parseCustomLine;
    private extractLevel;
    private filterLogs;
    private calculateStatistics;
    private aggregateLogs;
    private detectErrors;
    private detectAnomalies;
    private enrichWithGeo;
    private getGeoIP;
    private isValidJSON;
    private isValidIP;
    private createErrorResult;
}
export {};
//# sourceMappingURL=log-parser-tool.d.ts.map