"use strict";
/**
 * UTC Timestamp Utilities - Law of UTC (Section 1.6)
 * All Glue Code runs in UTC
 * Ingest timestamps → Convert to UTC ISO-8601 → Process
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Timer = void 0;
exports.toUtcIso = toUtcIso;
exports.nowUtc = nowUtc;
exports.parseUtc = parseUtc;
exports.isValidTimestamp = isValidTimestamp;
exports.durationMs = durationMs;
exports.formatDuration = formatDuration;
exports.addMs = addMs;
exports.addSeconds = addSeconds;
exports.addMinutes = addMinutes;
exports.addHours = addHours;
exports.addDays = addDays;
exports.isPast = isPast;
exports.isFuture = isFuture;
exports.compareTimestamps = compareTimestamps;
exports.toUnix = toUnix;
exports.fromUnix = fromUnix;
exports.truncateToSeconds = truncateToSeconds;
exports.toDatePart = toDatePart;
exports.toTimePart = toTimePart;
exports.toLocalDisplay = toLocalDisplay;
exports.isValidIsoUtc = isValidIsoUtc;
exports.ensureUtcIso = ensureUtcIso;
exports.startTimer = startTimer;
/**
 * Convert any timestamp to UTC ISO-8601 string
 */
function toUtcIso(timestamp = new Date()) {
    let date;
    if (timestamp instanceof Date) {
        date = timestamp;
    }
    else if (typeof timestamp === 'string') {
        // Parse and ensure we treat it as UTC
        date = new Date(timestamp);
    }
    else if (typeof timestamp === 'number') {
        date = new Date(timestamp);
    }
    else {
        date = new Date();
    }
    // Convert to UTC ISO-8601
    return date.toISOString();
}
/**
 * Get current UTC timestamp as ISO-8601 string
 */
function nowUtc() {
    return new Date().toISOString();
}
/**
 * Parse timestamp and return UTC Date object
 */
function parseUtc(timestamp) {
    let date;
    if (timestamp instanceof Date) {
        // Convert local date to UTC
        date = new Date(timestamp.toISOString());
    }
    else if (typeof timestamp === 'string') {
        date = new Date(timestamp);
    }
    else if (typeof timestamp === 'number') {
        date = new Date(timestamp);
    }
    else {
        date = new Date();
    }
    return date;
}
/**
 * Check if timestamp is valid
 */
function isValidTimestamp(timestamp) {
    if (!timestamp)
        return false;
    try {
        const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
        return !isNaN(date.getTime());
    }
    catch {
        return false;
    }
}
/**
 * Calculate duration between two timestamps in milliseconds
 */
function durationMs(start, end = new Date()) {
    const startDate = parseUtc(start);
    const endDate = parseUtc(end);
    return endDate.getTime() - startDate.getTime();
}
/**
 * Format duration in human-readable form
 */
function formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    if (days > 0) {
        return `${days}d ${hours % 24}h ${minutes % 60}m`;
    }
    if (hours > 0) {
        return `${hours}h ${minutes % 60}m`;
    }
    if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
}
/**
 * Add milliseconds to timestamp
 */
function addMs(timestamp, ms) {
    const date = parseUtc(timestamp);
    date.setTime(date.getTime() + ms);
    return date.toISOString();
}
/**
 * Add seconds to timestamp
 */
function addSeconds(timestamp, seconds) {
    return addMs(timestamp, seconds * 1000);
}
/**
 * Add minutes to timestamp
 */
function addMinutes(timestamp, minutes) {
    return addMs(timestamp, minutes * 60 * 1000);
}
/**
 * Add hours to timestamp
 */
function addHours(timestamp, hours) {
    return addMs(timestamp, hours * 60 * 60 * 1000);
}
/**
 * Add days to timestamp
 */
function addDays(timestamp, days) {
    return addMs(timestamp, days * 24 * 60 * 60 * 1000);
}
/**
 * Check if timestamp is in the past
 */
function isPast(timestamp) {
    const date = parseUtc(timestamp);
    return date.getTime() < Date.now();
}
/**
 * Check if timestamp is in the future
 */
function isFuture(timestamp) {
    const date = parseUtc(timestamp);
    return date.getTime() > Date.now();
}
/**
 * Compare two timestamps
 * Returns: -1 if first < second, 0 if equal, 1 if first > second
 */
function compareTimestamps(first, second) {
    const firstDate = parseUtc(first);
    const secondDate = parseUtc(second);
    if (firstDate.getTime() < secondDate.getTime())
        return -1;
    if (firstDate.getTime() > secondDate.getTime())
        return 1;
    return 0;
}
/**
 * Get Unix timestamp (seconds since epoch)
 */
function toUnix(timestamp = new Date()) {
    const date = parseUtc(timestamp);
    return Math.floor(date.getTime() / 1000);
}
/**
 * Create timestamp from Unix timestamp
 */
function fromUnix(unix) {
    return new Date(unix * 1000).toISOString();
}
/**
 * Truncate timestamp to seconds (remove milliseconds)
 */
function truncateToSeconds(timestamp) {
    const iso = toUtcIso(timestamp);
    return `${iso.substring(0, 19)}Z`;
}
/**
 * Extract date part from timestamp (YYYY-MM-DD)
 */
function toDatePart(timestamp) {
    const iso = toUtcIso(timestamp);
    return iso.substring(0, 10);
}
/**
 * Extract time part from timestamp (HH:mm:ss)
 */
function toTimePart(timestamp) {
    const iso = toUtcIso(timestamp);
    return iso.substring(11, 19);
}
/**
 * Format timestamp for display in user's local timezone
 * (But keep storage in UTC)
 */
function toLocalDisplay(timestamp) {
    const date = parseUtc(timestamp);
    return date.toLocaleString();
}
/**
 * Validate timestamp is in ISO-8601 UTC format
 */
function isValidIsoUtc(timestamp) {
    if (!timestamp)
        return false;
    // ISO-8601 UTC format should end with 'Z'
    if (!timestamp.endsWith('Z') && !timestamp.includes('+00:00')) {
        return false;
    }
    return isValidTimestamp(timestamp);
}
/**
 * Ensure timestamp is in UTC ISO-8601 format
 * If not, convert it
 */
function ensureUtcIso(timestamp) {
    if (typeof timestamp === 'string' && isValidIsoUtc(timestamp)) {
        return timestamp;
    }
    return toUtcIso(timestamp);
}
/**
 * Timer class for measuring operation duration
 */
class Timer {
    constructor() {
        this.startTime = Date.now();
    }
    /**
     * Stop the timer
     */
    stop() {
        this.endTime = Date.now();
        return this.elapsed();
    }
    /**
     * Get elapsed milliseconds
     */
    elapsed() {
        const end = this.endTime || Date.now();
        return end - this.startTime;
    }
    /**
     * Get elapsed as human-readable string
     */
    elapsedFormatted() {
        return formatDuration(this.elapsed());
    }
    /**
     * Get start timestamp as UTC ISO
     */
    getStartTimestamp() {
        return new Date(this.startTime).toISOString();
    }
    /**
     * Get end timestamp as UTC ISO (if stopped)
     */
    getEndTimestamp() {
        if (!this.endTime)
            return null;
        return new Date(this.endTime).toISOString();
    }
}
exports.Timer = Timer;
/**
 * Create a new timer
 */
function startTimer() {
    return new Timer();
}
//# sourceMappingURL=timestampUtils.js.map