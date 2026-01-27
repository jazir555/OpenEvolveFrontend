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
import { CredentialType } from '@bubblelab/shared-schemas';
import { HttpBubble } from '../service-bubble/http.js';
import { AIAgentBubble } from '../service-bubble/ai-agent.js';
/**
 * Timezone utility for parsing timestamps with timezone support
 */
class TimezoneParser {
    static TIMEZONE_OFFSETS = {
        'UTC': 0,
        'GMT': 0,
        'EST': -5,
        'EDT': -4,
        'CST': -6,
        'CDT': -5,
        'MST': -7,
        'MDT': -6,
        'PST': -8,
        'PDT': -7,
    };
    /**
     * Parse timestamp with timezone support
     */
    static parseWithTimezone(timestamp, format) {
        // Try ISO format first
        const isoMatch = timestamp.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|([+-]\d{2}:\d{2}))?$/);
        if (isoMatch) {
            return new Date(timestamp);
        }
        // Try common log format [10/Oct/2023:13:55:36 +0000]
        const clfMatch = timestamp.match(/\[(\d{2}\/\w{3}\/\d{4}):(\d{2}:\d{2}:\d{2})\s+([+-]\d{4})\]/);
        if (clfMatch) {
            const [, datePart, timePart, tzOffset] = clfMatch;
            const offsetHours = parseInt(tzOffset.slice(0, 3));
            const offsetMinutes = parseInt(tzOffset.slice(0, 2) + tzOffset.slice(3));
            return new Date(`${datePart} ${timePart} GMT${tzOffset.slice(0, 3)}:${tzOffset.slice(3)}`);
        }
        // Default to Date constructor
        const parsed = new Date(timestamp);
        if (!isNaN(parsed.getTime())) {
            return parsed;
        }
        // Fallback: current time if parsing fails
        return new Date();
    }
    /**
     * Convert timestamp to UTC ISO string
     */
    static toUTC(timestamp) {
        return this.parseWithTimezone(timestamp).toISOString();
    }
}
/**
 * Field extractor for common log patterns
 */
class FieldExtractor {
    /**
     * Extract IP address from log line
     */
    static extractIP(line) {
        const ipPattern = /(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/;
        const match = line.match(ipPattern);
        return match ? match[1] : null;
    }
    /**
     * Extract HTTP status code
     */
    static extractStatus(line) {
        const statusPattern = /\s(\d{3})\s/;
        const match = line.match(statusPattern);
        return match ? parseInt(match[1]) : null;
    }
    /**
     * Extract response time (in seconds or milliseconds)
     */
    static extractResponseTime(line) {
        // Try milliseconds first
        const msPattern = /(\d+(?:\.\d+)?)\s*ms/;
        const msMatch = line.match(msPattern);
        if (msMatch)
            return parseFloat(msMatch[1]);
        // Try seconds
        const sPattern = /(\d+(?:\.\d+)?)\s*s/;
        const sMatch = line.match(sPattern);
        if (sMatch)
            return parseFloat(sMatch[1]) * 1000;
        // Try microseconds
        const usPattern = /(\d+(?:\.\d+)?)\s*µs/;
        const usMatch = line.match(usPattern);
        if (usMatch)
            return parseFloat(usMatch[1]) / 1000;
        return null;
    }
    /**
     * Extract user agent
     */
    static extractUserAgent(line) {
        const uaPattern = /"([^"]*"[^"]*)"/;
        const matches = line.match(uaPattern);
        if (matches && matches.length > 1) {
            // Last quoted field is often user agent
            return matches[matches.length - 1];
        }
        return null;
    }
    /**
     * Extract HTTP method
     */
    static extractMethod(line) {
        const methodPattern = /"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)\s/;
        const match = line.match(methodPattern);
        return match ? match[1] : null;
    }
    /**
     * Extract URL path
     */
    static extractPath(line) {
        const pathPattern = /"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)\s+(\S+)/;
        const match = line.match(pathPattern);
        return match ? match[1] : null;
    }
    /**
     * Extract all key-value pairs from log line
     */
    static extractKeyValuePairs(line) {
        const pairs = {};
        // Match key=value patterns
        const kvPattern = /(\w+)=(?:"([^"]*)"|([^\s]+))/g;
        let match;
        while ((match = kvPattern.exec(line)) !== null) {
            const [, key, quotedValue, unquotedValue] = match;
            pairs[key] = quotedValue || unquotedValue;
        }
        return pairs;
    }
}
/**
 * Log entry schema
 */
const LogEntrySchema = z.object({
    timestamp: z.string().optional().describe('Log timestamp'),
    level: z.string().optional().describe('Log level (DEBUG, INFO, WARN, ERROR)'),
    message: z.string().describe('Log message'),
    source: z.string().optional().describe('Log source or application'),
    metadata: z.record(z.unknown()).optional().describe('Additional metadata'),
    lineNumber: z.number().describe('Line number in original log'),
    raw: z.string().describe('Raw log line'),
});
/**
 * Log statistics schema
 */
const LogStatisticsSchema = z.object({
    totalEntries: z.number().describe('Total number of log entries'),
    byLevel: z.record(z.number()).describe('Count by log level'),
    bySource: z.record(z.number()).describe('Count by source'),
    timeRange: z.object({
        start: z.string().optional().describe('Start timestamp'),
        end: z.string().optional().describe('End timestamp'),
    }),
    errorsCount: z.number().describe('Number of error-level entries'),
    warningsCount: z.number().describe('Number of warning-level entries'),
});
/**
 * Parse operation schema
 */
const LogParserToolParamsSchema = z.object({
    operation: z
        .enum([
        'parse',
        'filter',
        'aggregate',
        'detect',
        'enrich',
        'transform',
        'analyze',
    ])
        .describe('Parse operation type'),
    // Input data
    logData: z
        .string()
        .describe('Raw log data as string (multiline with newlines)'),
    // Format specification
    format: z
        .enum(['apache', 'nginx', 'json', 'csv', 'syslog', 'custom', 'auto'])
        .default('auto')
        .optional()
        .describe('Log format'),
    customPattern: z
        .string()
        .optional()
        .describe('Custom regex pattern for format=custom'),
    timestampFormat: z
        .string()
        .optional()
        .describe('Timestamp format string (e.g., "%Y-%m-%d %H:%M:%S")'),
    // Filtering options
    filterLevel: z
        .array(z.enum(['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'TRACE']))
        .optional()
        .describe('Filter by log levels'),
    filterSource: z.array(z.string()).optional().describe('Filter by source'),
    filterTimeRange: z
        .object({
        start: z.string().optional().describe('Start time ISO string'),
        end: z.string().optional().describe('End time ISO string'),
    })
        .optional()
        .describe('Filter by time range'),
    filterPattern: z.string().optional().describe('Regex pattern to match'),
    // Aggregation options
    aggregateBy: z
        .array(z.enum(['level', 'source', 'hour', 'day']))
        .optional()
        .describe('Aggregation dimensions'),
    // Error detection
    detectErrors: z
        .boolean()
        .default(false)
        .optional()
        .describe('Detect and highlight errors'),
    detectAnomalies: z
        .boolean()
        .default(false)
        .optional()
        .describe('Detect anomalies using AI'),
    // Enrichment
    enrichWithGeo: z
        .boolean()
        .default(false)
        .optional()
        .describe('Enrich IP addresses with geo data'),
    // Output options
    includeRaw: z
        .boolean()
        .default(true)
        .optional()
        .describe('Include raw log lines in output'),
    limit: z
        .number()
        .min(1)
        .max(10000)
        .default(1000)
        .optional()
        .describe('Maximum number of entries to return'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Required credentials'),
    config: z.record(z.string(), z.unknown()).optional().describe('Additional config'),
});
/**
 * Result schema
 */
const LogParserToolResultSchema = z.object({
    operation: z.string().describe('Operation performed'),
    entries: z.array(LogEntrySchema).describe('Parsed log entries'),
    statistics: LogStatisticsSchema.describe('Log statistics'),
    errors: z
        .array(LogEntrySchema)
        .optional()
        .describe('Error entries if detected'),
    anomalies: z
        .array(z.object({
        entry: LogEntrySchema,
        reason: z.string(),
        confidence: z.number(),
    }))
        .optional()
        .describe('Anomalies if detected'),
    aggregations: z
        .record(z.record(z.number()))
        .optional()
        .describe('Aggregated data'),
    metadata: z.object({
        format: z.string(),
        entriesParsed: z.number(),
        entriesFiltered: z.number(),
        parseErrors: z.number(),
        parseTime: z.number(),
    }),
    success: z.boolean(),
    error: z.string(),
});
/**
 * Log Parser Tool
 * Parses and analyzes log files
 */
export class LogParserTool extends ToolBubble {
    static bubbleName = 'log-parser-tool';
    static schema = LogParserToolParamsSchema;
    static resultSchema = LogParserToolResultSchema;
    static shortDescription = 'Parse and analyze log files';
    static longDescription = `
    Comprehensive log parsing and analysis tool.

    Operations:
    - parse: Parse logs into structured format
    - filter: Filter logs by level, source, time, pattern
    - aggregate: Aggregate logs by dimensions
    - detect: Detect errors and anomalies
    - enrich: Enrich logs with additional data
    - transform: Transform log format
    - analyze: Deep analysis with AI

    Supported formats:
    - Apache access logs
    - Nginx access logs
    - JSON logs
    - CSV logs
    - Syslog format
    - Custom regex patterns

    Features:
    - Multi-format parsing
    - Pattern matching
    - Time-based filtering
    - Error detection
    - Anomaly detection
    - Statistics generation
    - Geo IP enrichment
  `;
    static alias = 'log-parser';
    static type = 'tool';
    APACHE_PATTERN = /^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+) "([^"]*)" "([^"]*)"/;
    NGINX_PATTERN = /^(\S+) - \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+) "([^"]*)" "([^"]*)"/;
    SYSLOG_PATTERN = /^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:]+):\s+(.*)/;
    constructor(params = {
        operation: 'parse',
        logData: '',
    }, context) {
        super(params, context);
    }
    async performAction() {
        const startTime = Date.now();
        try {
            const validatedParams = LogParserToolParamsSchema.parse(this.params);
            // Detect format if auto
            let format = validatedParams.format;
            if (format === 'auto') {
                const detectedFormat = this.detectFormat(validatedParams.logData);
                format = detectedFormat;
                if (!format) {
                    format = 'custom'; // Default to custom if detection fails
                }
            }
            // Parse logs
            const entries = this.parseLogs(validatedParams.logData, format ?? 'custom');
            // Filter if requested
            let filteredEntries = entries;
            if (validatedParams.operation === 'filter') {
                filteredEntries = this.filterLogs(entries, validatedParams);
            }
            // Limit results
            const limitedEntries = filteredEntries.slice(0, validatedParams.limit);
            // Calculate statistics
            const statistics = this.calculateStatistics(limitedEntries);
            // Prepare result
            const result = {
                operation: validatedParams.operation,
                entries: limitedEntries,
                statistics,
                metadata: {
                    format: format ?? 'custom',
                    entriesParsed: entries.length,
                    entriesFiltered: filteredEntries.length,
                    parseErrors: 0,
                    parseTime: Date.now() - startTime,
                },
                success: true,
                error: '',
            };
            // Handle specific operations
            switch (validatedParams.operation) {
                case 'aggregate':
                    result.aggregations = this.aggregateLogs(limitedEntries, validatedParams);
                    break;
                case 'detect':
                    if (validatedParams.detectErrors) {
                        result.errors = this.detectErrors(limitedEntries);
                    }
                    if (validatedParams.detectAnomalies) {
                        result.anomalies = await this.detectAnomalies(limitedEntries);
                    }
                    break;
                case 'enrich':
                    if (validatedParams.enrichWithGeo) {
                        result.entries = await this.enrichWithGeo(limitedEntries);
                    }
                    break;
                case 'analyze':
                    result.anomalies = await this.detectAnomalies(limitedEntries);
                    result.errors = this.detectErrors(limitedEntries);
                    result.aggregations = this.aggregateLogs(limitedEntries, validatedParams);
                    break;
            }
            return result;
        }
        catch (error) {
            return this.createErrorResult(error instanceof Error ? error.message : 'Unknown error occurred');
        }
    }
    detectFormat(logData) {
        const lines = logData.split('\n').slice(0, 10);
        // Check for JSON
        if (lines.every((line) => !line.trim() || this.isValidJSON(line))) {
            return 'json';
        }
        // Check for Apache
        if (lines.some((line) => this.APACHE_PATTERN.test(line))) {
            return 'apache';
        }
        // Check for Nginx
        if (lines.some((line) => this.NGINX_PATTERN.test(line))) {
            return 'nginx';
        }
        // Check for syslog
        if (lines.some((line) => this.SYSLOG_PATTERN.test(line))) {
            return 'syslog';
        }
        // Check for CSV
        if (lines[0]?.includes(',')) {
            return 'csv';
        }
        return 'custom';
    }
    parseLogs(logData, format) {
        const lines = logData.split('\n');
        const entries = [];
        lines.forEach((line, index) => {
            if (!line.trim())
                return;
            try {
                let entry;
                switch (format) {
                    case 'json':
                        entry = this.parseJSONLine(line, index);
                        break;
                    case 'apache':
                        entry = this.parseApacheLine(line, index);
                        break;
                    case 'nginx':
                        entry = this.parseNginxLine(line, index);
                        break;
                    case 'syslog':
                        entry = this.parseSyslogLine(line, index);
                        break;
                    case 'csv':
                        entry = this.parseCSVLine(line, index);
                        break;
                    default:
                        entry = this.parseCustomLine(line, index);
                }
                entries.push(entry);
            }
            catch (error) {
                // Skip unparseable lines
                console.warn(`Failed to parse line ${index + 1}:`, error);
            }
        });
        return entries;
    }
    parseJSONLine(line, index) {
        const data = JSON.parse(line);
        return {
            timestamp: data.timestamp || data.time || data['@timestamp'] || undefined,
            level: data.level || data.severity || undefined,
            message: data.message || data.msg || line,
            source: data.source || data.service || data.app || undefined,
            metadata: data,
            lineNumber: index + 1,
            raw: line,
        };
    }
    parseApacheLine(line, index) {
        const match = line.match(this.APACHE_PATTERN);
        if (!match) {
            throw new Error('Invalid Apache log format');
        }
        const [, ip, timestamp, method, path, protocol, status, bytes, referrer, userAgent] = match;
        // Convert timestamp to UTC
        const utcTimestamp = TimezoneParser.toUTC(timestamp);
        return {
            timestamp: utcTimestamp,
            level: parseInt(status) >= 400 ? 'ERROR' : 'INFO',
            message: `${method} ${path} ${protocol} - ${status}`,
            source: 'apache',
            metadata: {
                ip,
                method,
                path,
                protocol,
                status: parseInt(status),
                bytes: parseInt(bytes),
                referrer,
                userAgent,
                // Enhanced fields
                responseTime: FieldExtractor.extractResponseTime(line),
            },
            lineNumber: index + 1,
            raw: line,
        };
    }
    parseNginxLine(line, index) {
        const match = line.match(this.NGINX_PATTERN);
        if (!match) {
            throw new Error('Invalid Nginx log format');
        }
        const [, ip, timestamp, method, path, protocol, status, bytes, referrer, userAgent] = match;
        // Convert timestamp to UTC
        const utcTimestamp = TimezoneParser.toUTC(timestamp);
        return {
            timestamp: utcTimestamp,
            level: parseInt(status) >= 400 ? 'ERROR' : 'INFO',
            message: `${method} ${path} ${protocol} - ${status}`,
            source: 'nginx',
            metadata: {
                ip,
                method,
                path,
                protocol,
                status: parseInt(status),
                bytes: parseInt(bytes),
                referrer,
                userAgent,
                // Enhanced fields
                responseTime: FieldExtractor.extractResponseTime(line),
            },
            lineNumber: index + 1,
            raw: line,
        };
    }
    parseSyslogLine(line, index) {
        const match = line.match(this.SYSLOG_PATTERN);
        if (!match) {
            throw new Error('Invalid syslog format');
        }
        const [, timestamp, host, process, message] = match;
        return {
            timestamp,
            level: this.extractLevel(message),
            message,
            source: process,
            metadata: {
                host,
                process,
            },
            lineNumber: index + 1,
            raw: line,
        };
    }
    parseCSVLine(line, index) {
        // Enhanced CSV parsing that handles quoted fields with embedded commas
        const fields = this.parseCSVFields(line);
        return {
            timestamp: fields[0] || undefined,
            level: fields[1] || undefined,
            message: fields.slice(2).join(',') || line,
            source: 'csv',
            metadata: { fields },
            lineNumber: index + 1,
            raw: line,
        };
    }
    /**
     * Parse CSV fields handling quoted strings with embedded delimiters
     * Supports:
     * - Quoted fields: "value,with,commas"
     * - Escaped quotes: "value with ""quotes"""
     * - Mixed quoted and unquoted fields
     */
    parseCSVFields(line) {
        const fields = [];
        let current = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            const nextChar = line[i + 1];
            if (char === '"') {
                if (inQuotes && nextChar === '"') {
                    // Escaped quote within quoted field
                    current += '"';
                    i++; // Skip next quote
                }
                else {
                    // Toggle quote mode
                    inQuotes = !inQuotes;
                }
            }
            else if (char === ',' && !inQuotes) {
                // Field separator (only outside quotes)
                fields.push(current.trim());
                current = '';
            }
            else {
                current += char;
            }
        }
        // Add the last field
        fields.push(current.trim());
        return fields;
    }
    parseCustomLine(line, index) {
        // Use custom pattern if provided
        if (this.params.customPattern) {
            try {
                const regex = new RegExp(this.params.customPattern);
                const match = line.match(regex);
                if (match) {
                    return {
                        timestamp: match[1] || undefined,
                        level: match[2] || undefined,
                        message: match[3] || line,
                        source: match[4] || 'custom',
                        metadata: { match: match.slice(1) },
                        lineNumber: index + 1,
                        raw: line,
                    };
                }
            }
            catch (error) {
                console.warn('Invalid custom pattern:', error);
            }
        }
        // Fallback: basic parsing
        return {
            timestamp: undefined,
            level: this.extractLevel(line),
            message: line,
            source: 'unknown',
            lineNumber: index + 1,
            raw: line,
        };
    }
    extractLevel(message) {
        const upper = message.toUpperCase();
        if (upper.includes('ERROR') || upper.includes('FATAL'))
            return 'ERROR';
        if (upper.includes('WARN'))
            return 'WARN';
        if (upper.includes('DEBUG'))
            return 'DEBUG';
        if (upper.includes('TRACE'))
            return 'TRACE';
        return 'INFO';
    }
    filterLogs(entries, params) {
        let filtered = entries;
        // Filter by level
        if (params.filterLevel && params.filterLevel.length > 0) {
            filtered = filtered.filter((entry) => entry.level ? params.filterLevel.includes(entry.level) : false);
        }
        // Filter by source
        if (params.filterSource && params.filterSource.length > 0) {
            filtered = filtered.filter((entry) => entry.source && params.filterSource.includes(entry.source));
        }
        // Filter by time range
        if (params.filterTimeRange) {
            const { start, end } = params.filterTimeRange;
            filtered = filtered.filter((entry) => {
                if (!entry.timestamp)
                    return true;
                const entryTime = new Date(entry.timestamp);
                if (start && entryTime < new Date(start))
                    return false;
                if (end && entryTime > new Date(end))
                    return false;
                return true;
            });
        }
        // Filter by pattern
        if (params.filterPattern) {
            const pattern = new RegExp(params.filterPattern);
            filtered = filtered.filter((entry) => pattern.test(entry.message));
        }
        return filtered;
    }
    calculateStatistics(entries) {
        const byLevel = {};
        const bySource = {};
        let errorsCount = 0;
        let warningsCount = 0;
        const timestamps = [];
        entries.forEach((entry) => {
            // Count by level
            if (entry.level) {
                byLevel[entry.level] = (byLevel[entry.level] || 0) + 1;
                if (entry.level === 'ERROR' || entry.level === 'FATAL')
                    errorsCount++;
                if (entry.level === 'WARN')
                    warningsCount++;
            }
            // Count by source
            if (entry.source) {
                bySource[entry.source] = (bySource[entry.source] || 0) + 1;
            }
            // Collect timestamps
            if (entry.timestamp) {
                timestamps.push(entry.timestamp);
            }
        });
        // Sort timestamps
        timestamps.sort();
        return {
            totalEntries: entries.length,
            byLevel,
            bySource,
            timeRange: {
                start: timestamps[0],
                end: timestamps[timestamps.length - 1],
            },
            errorsCount,
            warningsCount,
        };
    }
    aggregateLogs(entries, params) {
        const aggregations = {};
        if (!params.aggregateBy || params.aggregateBy.length === 0) {
            return aggregations;
        }
        params.aggregateBy.forEach((dimension) => {
            aggregations[dimension] = {};
            entries.forEach((entry) => {
                let key = 'unknown';
                switch (dimension) {
                    case 'level':
                        key = entry.level || 'unknown';
                        break;
                    case 'source':
                        key = entry.source || 'unknown';
                        break;
                    case 'hour':
                        if (entry.timestamp) {
                            const date = new Date(entry.timestamp);
                            key = `${date.getHours()}:00`;
                        }
                        break;
                    case 'day':
                        if (entry.timestamp) {
                            const date = new Date(entry.timestamp);
                            key = date.toISOString().split('T')[0];
                        }
                        break;
                }
                aggregations[dimension][key] = (aggregations[dimension][key] || 0) + 1;
            });
        });
        return aggregations;
    }
    detectErrors(entries) {
        return entries.filter((entry) => entry.level === 'ERROR' || entry.level === 'FATAL');
    }
    async detectAnomalies(entries) {
        const anomalies = [];
        // Simple rule-based anomaly detection
        entries.forEach((entry) => {
            // Very long messages
            if (entry.message.length > 1000) {
                anomalies.push({
                    entry,
                    reason: 'Unusually long message',
                    confidence: 0.7,
                });
            }
            // High frequency of same error
            const sameErrors = entries.filter((e) => e.message === entry.message && e.lineNumber !== entry.lineNumber);
            if (sameErrors.length > 10) {
                anomalies.push({
                    entry,
                    reason: `Repeated error (${sameErrors.length} occurrences)`,
                    confidence: 0.9,
                });
            }
            // Unknown levels
            if (entry.level &&
                !['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'TRACE'].includes(entry.level)) {
                anomalies.push({
                    entry,
                    reason: 'Unknown log level',
                    confidence: 0.6,
                });
            }
        });
        // If AI credentials available, use AI for advanced detection
        const aiKey = this.params.credentials?.[CredentialType.OPENAI_CRED];
        if (aiKey && entries.length > 0) {
            try {
                const sampleSize = Math.min(20, entries.length);
                const sample = entries.slice(0, sampleSize);
                const prompt = `Analyze these log entries and identify any anomalies or patterns that indicate problems:

${sample.map((e) => `[${e.timestamp}] [${e.level}] ${e.message}`).join('\n')}

Respond with JSON array of anomalies with format: [{"lineNumber": number, "reason": string, "confidence": number}]`;
                const aiAgent = new AIAgentBubble({
                    name: 'Log Anomaly Detector',
                    message: prompt,
                    model: {
                        model: 'openai/gpt-4',
                        temperature: 0.3,
                    },
                    credentials: {
                        [CredentialType.OPENAI_CRED]: aiKey,
                    },
                });
                const result = await aiAgent.action();
                if (result.success && result.data?.response) {
                    const aiAnomalies = JSON.parse(result.data.response);
                    aiAnomalies.forEach((anomaly) => {
                        const entry = entries.find((e) => e.lineNumber === anomaly.lineNumber);
                        if (entry) {
                            anomalies.push({
                                entry,
                                reason: anomaly.reason,
                                confidence: anomaly.confidence,
                            });
                        }
                    });
                }
            }
            catch (error) {
                console.warn('AI anomaly detection failed:', error);
            }
        }
        return anomalies;
    }
    async enrichWithGeo(entries) {
        const enrichedEntries = [...entries];
        for (const entry of enrichedEntries) {
            const ip = entry.metadata?.ip;
            if (ip && this.isValidIP(ip)) {
                try {
                    const geoData = await this.getGeoIP(ip);
                    entry.metadata = {
                        ...entry.metadata,
                        geo: geoData,
                    };
                }
                catch (error) {
                    console.warn(`Failed to enrich IP ${ip}:`, error);
                }
            }
        }
        return enrichedEntries;
    }
    async getGeoIP(ip) {
        const httpBubble = new HttpBubble({
            url: `http://ip-api.com/json/${ip}`,
            method: 'GET',
            timeout: 5000,
        }, this.context);
        const result = await httpBubble.action();
        if (result.success && result.data?.json) {
            return result.data.json;
        }
        throw new Error('Geo IP lookup failed');
    }
    isValidJSON(str) {
        try {
            JSON.parse(str);
            return true;
        }
        catch {
            return false;
        }
    }
    isValidIP(ip) {
        const ipPattern = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
        return ipPattern.test(ip);
    }
    createErrorResult(errorMessage) {
        return {
            operation: this.params.operation,
            entries: [],
            statistics: {
                totalEntries: 0,
                byLevel: {},
                bySource: {},
                timeRange: {},
                errorsCount: 0,
                warningsCount: 0,
            },
            metadata: {
                format: this.params.format || 'auto',
                entriesParsed: 0,
                entriesFiltered: 0,
                parseErrors: 0,
                parseTime: 0,
            },
            success: false,
            error: errorMessage,
        };
    }
}
//# sourceMappingURL=log-parser-tool.js.map