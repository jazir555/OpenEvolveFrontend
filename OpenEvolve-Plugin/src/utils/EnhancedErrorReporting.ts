/**
 * Enhanced Error Reporting with Better Categorization
 * Provides sophisticated error classification, reporting, and analytics
 */

import { ExtendedErrorContext } from './EnhancedErrorContext';
import { errorLogger, ErrorSeverity } from './errorLogging';

// Define error categories
export enum ErrorCategory {
  NETWORK = 'network',
  AUTHENTICATION = 'authentication',
  VALIDATION = 'validation',
  TIMEOUT = 'timeout',
  DATABASE = 'database',
  MEMORY = 'memory',
  CONFIGURATION = 'configuration',
  BUSINESS_LOGIC = 'business_logic',
  USER_INPUT = 'user_input',
  PERMISSION = 'permission',
  INTEGRATION = 'integration',
  PERFORMANCE = 'performance',
  SECURITY = 'security',
  UNKNOWN = 'unknown'
}

// Define error subcategories
export enum ErrorSubcategory {
  // Network subcategories
  NETWORK_CONNECTION_FAILED = 'connection_failed',
  NETWORK_TIMEOUT = 'network_timeout',
  NETWORK_DNS_ERROR = 'dns_error',
  NETWORK_SSL_ERROR = 'ssl_error',
  
  // Authentication subcategories
  AUTH_TOKEN_EXPIRED = 'token_expired',
  AUTH_INVALID_CREDENTIALS = 'invalid_credentials',
  AUTH_INSUFFICIENT_PERMISSIONS = 'insufficient_permissions',
  AUTH_SESSION_EXPIRED = 'session_expired',
  
  // Validation subcategories
  VALIDATION_SCHEMA_ERROR = 'schema_error',
  VALIDATION_FORMAT_ERROR = 'format_error',
  VALIDATION_REQUIRED_FIELD_MISSING = 'required_field_missing',
  VALIDATION_RANGE_ERROR = 'range_error',
  
  // Database subcategories
  DB_CONNECTION_FAILED = 'connection_failed',
  DB_QUERY_TIMEOUT = 'query_timeout',
  DB_UNIQUE_CONSTRAINT_VIOLATION = 'unique_constraint_violation',
  DB_FOREIGN_KEY_VIOLATION = 'foreign_key_violation',
  
  // Performance subcategories
  PERFORMANCE_SLOW_OPERATION = 'slow_operation',
  PERFORMANCE_HIGH_MEMORY_USAGE = 'high_memory_usage',
  PERFORMANCE_LONG_RUNNING_TASK = 'long_running_task',
  
  // Security subcategories
  SECURITY_XSS_ATTEMPT = 'xss_attempt',
  SECURITY_CSRF_ATTEMPT = 'csrf_attempt',
  SECURITY_SQL_INJECTION_ATTEMPT = 'sql_injection_attempt',
}

// Enhanced error report interface
export interface EnhancedErrorReport {
  id: string;
  timestamp: Date;
  message: string;
  stack?: string;
  category: ErrorCategory;
  subcategory?: ErrorSubcategory;
  severity: ErrorSeverity;
  context: ExtendedErrorContext;
  tags: string[];
  userImpact: 'low' | 'medium' | 'high' | 'critical';
  frequency: number; // How often this error occurs
  affectedUsers: number; // How many users affected
  resolutionStatus: 'open' | 'investigating' | 'fixed' | 'wont_fix';
  priority: 'low' | 'medium' | 'high' | 'critical';
  relatedErrors: string[]; // IDs of related errors
  suggestedFix?: string;
  reproductionSteps?: string[];
  environment: string;
  version: string;
  resolvedAt?: Date;
  resolvedBy?: string;
}

// Error classifier interface
export interface ErrorClassifier {
  classify(error: any, context: ExtendedErrorContext): {
    category: ErrorCategory;
    subcategory?: ErrorSubcategory;
    severity: ErrorSeverity;
    userImpact: 'low' | 'medium' | 'high' | 'critical';
    priority: 'low' | 'medium' | 'high' | 'critical';
    suggestedFix?: string;
  };
}

/**
 * Enhanced Error Classifier
 * Provides sophisticated error classification based on multiple factors
 */
export class EnhancedErrorClassifier implements ErrorClassifier {
  private rules: Array<{
    condition: (error: any, context: ExtendedErrorContext) => boolean;
    category: ErrorCategory;
    subcategory?: ErrorSubcategory;
    severity: ErrorSeverity;
    userImpact: 'low' | 'medium' | 'high' | 'critical';
    priority: 'low' | 'medium' | 'high' | 'critical';
    suggestedFix?: string;
  }> = [];

  constructor() {
    this.initializeDefaultRules();
  }

  /**
   * Initialize default classification rules
   */
  private initializeDefaultRules(): void {
    // Network errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('network') ||
        error.message?.toLowerCase().includes('connection') ||
        error.message?.toLowerCase().includes('fetch') ||
        error.message?.toLowerCase().includes('failed to fetch') ||
        error.code === 'ECONNREFUSED' ||
        error.code === 'ENOTFOUND' ||
        error.code === 'ECONNABORTED',
      category: ErrorCategory.NETWORK,
      subcategory: ErrorSubcategory.NETWORK_CONNECTION_FAILED,
      severity: 'error',
      userImpact: 'high',
      priority: 'high',
      suggestedFix: 'Check network connectivity and API endpoint availability'
    });

    // Network timeout errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('timeout') ||
        error.message?.toLowerCase().includes('timed out') ||
        error.code === 'ETIMEDOUT',
      category: ErrorCategory.NETWORK,
      subcategory: ErrorSubcategory.NETWORK_TIMEOUT,
      severity: 'error',
      userImpact: 'medium',
      priority: 'medium',
      suggestedFix: 'Increase timeout values or optimize network operations'
    });

    // Authentication errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('auth') ||
        error.message?.toLowerCase().includes('token') ||
        error.message?.includes('401') ||
        error.message?.includes('403') ||
        error.status === 401 ||
        error.status === 403,
      category: ErrorCategory.AUTHENTICATION,
      subcategory: ErrorSubcategory.AUTH_INVALID_CREDENTIALS,
      severity: 'error',
      userImpact: 'high',
      priority: 'high',
      suggestedFix: 'Verify authentication credentials and token validity'
    });

    // Token expired errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('token') &&
        (error.message?.toLowerCase().includes('expired') || 
         error.message?.toLowerCase().includes('invalid')),
      category: ErrorCategory.AUTHENTICATION,
      subcategory: ErrorSubcategory.AUTH_TOKEN_EXPIRED,
      severity: 'error',
      userImpact: 'high',
      priority: 'high',
      suggestedFix: 'Refresh authentication token'
    });

    // Validation errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('validation') ||
        error.message?.toLowerCase().includes('invalid') ||
        error.message?.includes('422') ||
        error.status === 422,
      category: ErrorCategory.VALIDATION,
      subcategory: ErrorSubcategory.VALIDATION_SCHEMA_ERROR,
      severity: 'error',
      userImpact: 'medium',
      priority: 'medium',
      suggestedFix: 'Validate input data format and required fields'
    });

    // Required field errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('required') ||
        error.message?.toLowerCase().includes('missing'),
      category: ErrorCategory.VALIDATION,
      subcategory: ErrorSubcategory.VALIDATION_REQUIRED_FIELD_MISSING,
      severity: 'error',
      userImpact: 'medium',
      priority: 'medium',
      suggestedFix: 'Provide required field values'
    });

    // Database errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('database') ||
        error.message?.toLowerCase().includes('db') ||
        error.message?.toLowerCase().includes('sql') ||
        error.message?.toLowerCase().includes('mongo'),
      category: ErrorCategory.DATABASE,
      subcategory: ErrorSubcategory.DB_CONNECTION_FAILED,
      severity: 'error',
      userImpact: 'critical',
      priority: 'critical',
      suggestedFix: 'Check database connection and query syntax'
    });

    // Memory errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('memory') ||
        error.message?.toLowerCase().includes('heap') ||
        error.message?.toLowerCase().includes('out of memory'),
      category: ErrorCategory.MEMORY,
      severity: 'critical',
      userImpact: 'critical',
      priority: 'critical',
      suggestedFix: 'Optimize memory usage and check for memory leaks'
    });

    // Configuration errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('config') ||
        error.message?.toLowerCase().includes('configuration') ||
        error.message?.toLowerCase().includes('env') ||
        error.message?.toLowerCase().includes('environment'),
      category: ErrorCategory.CONFIGURATION,
      severity: 'error',
      userImpact: 'high',
      priority: 'high',
      suggestedFix: 'Verify configuration files and environment variables'
    });

    // Permission errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('permission') ||
        error.message?.includes('403') ||
        error.status === 403,
      category: ErrorCategory.PERMISSION,
      severity: 'error',
      userImpact: 'high',
      priority: 'high',
      suggestedFix: 'Check user permissions and access rights'
    });

    // Business logic errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('business') ||
        error.message?.toLowerCase().includes('constraint') ||
        error.message?.toLowerCase().includes('rule'),
      category: ErrorCategory.BUSINESS_LOGIC,
      severity: 'error',
      userImpact: 'medium',
      priority: 'medium',
      suggestedFix: 'Review business logic constraints and rules'
    });

    // Security errors
    this.rules.push({
      condition: (error, context) => 
        error.message?.toLowerCase().includes('security') ||
        error.message?.toLowerCase().includes('xss') ||
        error.message?.toLowerCase().includes('csrf') ||
        error.message?.toLowerCase().includes('injection'),
      category: ErrorCategory.SECURITY,
      severity: 'critical',
      userImpact: 'critical',
      priority: 'critical',
      suggestedFix: 'Review security measures and input sanitization'
    });
  }

  /**
   * Classify an error based on context and message
   */
  classify(error: any, context: ExtendedErrorContext): {
    category: ErrorCategory;
    subcategory?: ErrorSubcategory;
    severity: ErrorSeverity;
    userImpact: 'low' | 'medium' | 'high' | 'critical';
    priority: 'low' | 'medium' | 'high' | 'critical';
    suggestedFix?: string;
  } {
    // Try to match against rules
    for (const rule of this.rules) {
      if (rule.condition(error, context)) {
        return {
          category: rule.category,
          subcategory: rule.subcategory,
          severity: rule.severity,
          userImpact: rule.userImpact,
          priority: rule.priority,
          suggestedFix: rule.suggestedFix
        };
      }
    }

    // Default classification for unknown errors
    return {
      category: ErrorCategory.UNKNOWN,
      severity: 'error',
      userImpact: 'medium',
      priority: 'medium',
      suggestedFix: 'Investigate error details and context'
    };
  }

  /**
   * Add a custom classification rule
   */
  addRule(
    condition: (error: any, context: ExtendedErrorContext) => boolean,
    category: ErrorCategory,
    subcategory?: ErrorSubcategory,
    severity: ErrorSeverity = 'error',
    userImpact: 'low' | 'medium' | 'high' | 'critical' = 'medium',
    priority: 'low' | 'medium' | 'high' | 'critical' = 'medium',
    suggestedFix?: string
  ): void {
    this.rules.push({
      condition,
      category,
      subcategory,
      severity,
      userImpact,
      priority,
      suggestedFix
    });
  }
}

/**
 * Enhanced Error Reporter
 * Provides sophisticated error reporting with analytics and insights
 */
export class EnhancedErrorReporter {
  private classifier: EnhancedErrorClassifier;
  private reports: Map<string, EnhancedErrorReport> = new Map();
  private errorFrequency: Map<string, number> = new Map(); // Maps error signatures to frequency
  private affectedUsers: Map<string, Set<string>> = new Map(); // Maps error signatures to user IDs
  private relatedErrors: Map<string, Set<string>> = new Map(); // Maps error IDs to related error IDs
  private maxReports = 1000; // Maximum number of reports to keep in memory

  constructor() {
    this.classifier = new EnhancedErrorClassifier();
  }

  /**
   * Report an error with enhanced categorization
   */
  async reportError(
    error: any,
    context: ExtendedErrorContext,
    tags: string[] = []
  ): Promise<EnhancedErrorReport> {
    // Classify the error
    const classification = this.classifier.classify(error, context);

    // Create error signature for frequency tracking
    const errorSignature = this.createErrorSignature(error, context);
    
    // Update frequency
    const currentFreq = this.errorFrequency.get(errorSignature) || 0;
    this.errorFrequency.set(errorSignature, currentFreq + 1);
    
    // Update affected users
    if (context.userId) {
      if (!this.affectedUsers.has(errorSignature)) {
        this.affectedUsers.set(errorSignature, new Set());
      }
      this.affectedUsers.get(errorSignature)!.add(context.userId);
    }

    // Create enhanced report
    const report: EnhancedErrorReport = {
      id: this.generateErrorId(),
      timestamp: new Date(),
      message: error.message || String(error),
      stack: error.stack,
      category: classification.category,
      subcategory: classification.subcategory,
      severity: classification.severity,
      context,
      tags: [...tags, classification.category, ...(classification.subcategory ? [classification.subcategory] : [])],
      userImpact: classification.userImpact,
      frequency: currentFreq + 1,
      affectedUsers: this.affectedUsers.get(errorSignature)?.size || 0,
      resolutionStatus: 'open',
      priority: classification.priority,
      relatedErrors: [],
      suggestedFix: classification.suggestedFix,
      environment: context.environment || 'unknown',
      version: context.version || 'unknown'
    };

    // Store the report
    this.reports.set(report.id, report);

    // Maintain size limit
    if (this.reports.size > this.maxReports) {
      const firstKey = this.reports.keys().next().value;
      this.reports.delete(firstKey);
    }

    // Log to basic error logger as well
    errorLogger.logError(error, report.severity, context);

    return report;
  }

  /**
   * Create an error signature for grouping similar errors
   */
  private createErrorSignature(error: any, context: ExtendedErrorContext): string {
    // Create a signature based on error message, stack trace, and context
    const message = error.message || String(error);
    const component = context.component || 'unknown';
    const functionContext = context.function || 'unknown';
    
    // For stack traces, we'll use just the first few frames to avoid noise
    let stackSignature = '';
    if (error.stack) {
      const stackLines = error.stack.split('\n');
      // Take the first 3 lines after the error message
      stackSignature = stackLines.slice(1, 4).join('|').substring(0, 200);
    }
    
    return `${message.substring(0, 100)}|${component}|${functionContext}|${stackSignature}`;
  }

  /**
   * Generate a unique error ID
   */
  private generateErrorId(): string {
    return `err_enh_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get reports by category
   */
  getReportsByCategory(category: ErrorCategory): EnhancedErrorReport[] {
    return Array.from(this.reports.values()).filter(report => report.category === category);
  }

  /**
   * Get reports by severity
   */
  getReportsBySeverity(severity: ErrorSeverity): EnhancedErrorReport[] {
    return Array.from(this.reports.values()).filter(report => report.severity === severity);
  }

  /**
   * Get reports by priority
   */
  getReportsByPriority(priority: 'low' | 'medium' | 'high' | 'critical'): EnhancedErrorReport[] {
    return Array.from(this.reports.values()).filter(report => report.priority === priority);
  }

  /**
   * Get top frequent errors
   */
  getTopFrequentErrors(limit: number = 10): Array<{signature: string, frequency: number}> {
    const freqArray = Array.from(this.errorFrequency.entries())
      .map(([signature, frequency]) => ({ signature, frequency }))
      .sort((a, b) => b.frequency - a.frequency);
    
    return freqArray.slice(0, limit);
  }

  /**
   * Get top affected users
   */
  getTopAffectedUsers(limit: number = 10): Array<{userId: string, errorCount: number}> {
    const userErrorCounts = new Map<string, number>();
    
    this.affectedUsers.forEach((users, signature) => {
      users.forEach(userId => {
        userErrorCounts.set(userId, (userErrorCounts.get(userId) || 0) + 1);
      });
    });
    
    const userArray = Array.from(userErrorCounts.entries())
      .map(([userId, errorCount]) => ({ userId, errorCount }))
      .sort((a, b) => b.errorCount - a.errorCount);
    
    return userArray.slice(0, limit);
  }

  /**
   * Get error statistics
   */
  getErrorStatistics(): {
    totalErrors: number;
    byCategory: Record<ErrorCategory, number>;
    bySeverity: Record<ErrorSeverity, number>;
    byPriority: Record<'low' | 'medium' | 'high' | 'critical', number>;
    topFrequentErrors: Array<{signature: string, frequency: number}>;
    uniqueUsersAffected: number;
  } {
    const byCategory: Record<ErrorCategory, number> = {
      [ErrorCategory.NETWORK]: 0,
      [ErrorCategory.AUTHENTICATION]: 0,
      [ErrorCategory.VALIDATION]: 0,
      [ErrorCategory.TIMEOUT]: 0,
      [ErrorCategory.DATABASE]: 0,
      [ErrorCategory.MEMORY]: 0,
      [ErrorCategory.CONFIGURATION]: 0,
      [ErrorCategory.BUSINESS_LOGIC]: 0,
      [ErrorCategory.USER_INPUT]: 0,
      [ErrorCategory.PERMISSION]: 0,
      [ErrorCategory.INTEGRATION]: 0,
      [ErrorCategory.PERFORMANCE]: 0,
      [ErrorCategory.SECURITY]: 0,
      [ErrorCategory.UNKNOWN]: 0
    };
    
    const bySeverity: Record<ErrorSeverity, number> = {
      debug: 0,
      info: 0,
      warn: 0,
      error: 0,
      critical: 0
    };
    
    const byPriority: Record<'low' | 'medium' | 'high' | 'critical', number> = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0
    };
    
    Array.from(this.reports.values()).forEach(report => {
      byCategory[report.category]++;
      bySeverity[report.severity]++;
      byPriority[report.priority]++;
    });
    
    // Count unique users affected
    let uniqueUsers = 0;
    this.affectedUsers.forEach(users => {
      uniqueUsers += users.size;
    });
    
    return {
      totalErrors: this.reports.size,
      byCategory,
      bySeverity,
      byPriority,
      topFrequentErrors: this.getTopFrequentErrors(5),
      uniqueUsersAffected: uniqueUsers
    };
  }

  /**
   * Mark an error as resolved
   */
  markAsResolved(errorId: string, resolvedBy?: string): boolean {
    const report = this.reports.get(errorId);
    if (report) {
      report.resolutionStatus = 'fixed';
      report.resolvedAt = new Date();
      report.resolvedBy = resolvedBy;
      return true;
    }
    return false;
  }

  /**
   * Get all reports
   */
  getAllReports(): EnhancedErrorReport[] {
    return Array.from(this.reports.values());
  }

  /**
   * Get report by ID
   */
  getReportById(id: string): EnhancedErrorReport | undefined {
    return this.reports.get(id);
  }

  /**
   * Clear all reports
   */
  clearReports(): void {
    this.reports.clear();
    this.errorFrequency.clear();
    this.affectedUsers.clear();
    this.relatedErrors.clear();
  }

  /**
   * Add a custom classification rule
   */
  addClassificationRule(
    condition: (error: any, context: ExtendedErrorContext) => boolean,
    category: ErrorCategory,
    subcategory?: ErrorSubcategory,
    severity: ErrorSeverity = 'error',
    userImpact: 'low' | 'medium' | 'high' | 'critical' = 'medium',
    priority: 'low' | 'medium' | 'high' | 'critical' = 'medium',
    suggestedFix?: string
  ): void {
    this.classifier.addRule(condition, category, subcategory, severity, userImpact, priority, suggestedFix);
  }

  /**
   * Get the error classifier
   */
  getClassifier(): EnhancedErrorClassifier {
    return this.classifier;
  }
}

// Create a singleton instance
export const enhancedErrorReporter = new EnhancedErrorReporter();

/**
 * Helper function to report an error with enhanced categorization
 */
export async function reportEnhancedError(
  error: any,
  context: ExtendedErrorContext,
  tags: string[] = []
): Promise<EnhancedErrorReport> {
  return enhancedErrorReporter.reportError(error, context, tags);
}