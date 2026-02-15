/**
 * Contract Test for Structured Logger
 *
 * Tests compliance with Federation Constitution:
 * - Law of UTC: All timestamps in UTC ISO-8601
 * - Observability: JSON Lines format with required fields
 * - Correlation ID generation and propagation
 */

import { StructuredLogger, LogEntry } from './structuredLogger';

describe('StructuredLogger Contract Tests', () => {
  let logger: StructuredLogger;
  let consoleOutput: string[];
  let originalConsoleLog: typeof console.log;
  let originalConsoleWarn: typeof console.warn;
  let originalConsoleError: typeof console.error;
  let originalConsoleDebug: typeof console.debug;

  beforeEach(() => {
    // Capture console output
    consoleOutput = [];
    originalConsoleLog = console.log;
    originalConsoleWarn = console.warn;
    originalConsoleError = console.error;
    originalConsoleDebug = console.debug;

    console.log = (...args: unknown[]) => consoleOutput.push(args[0] as string);
    console.warn = (...args: unknown[]) => consoleOutput.push(args[0] as string);
    console.error = (...args: unknown[]) => consoleOutput.push(args[0] as string);
    console.debug = (...args: unknown[]) => consoleOutput.push(args[0] as string);

    logger = new StructuredLogger('test-service');
  });

  afterEach(() => {
    // Restore console
    console.log = originalConsoleLog;
    console.warn = originalConsoleWarn;
    console.error = originalConsoleError;
    console.debug = originalConsoleDebug;
  });

  describe('UTC Timestamp Compliance (Law 6)', () => {
    it('should generate UTC ISO-8601 timestamps', () => {
      logger.info('Test message');

      expect(consoleOutput.length).toBe(1);
      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);

      // ISO-8601 format check
      expect(logEntry.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);

      // Verify it ends with Z (UTC indicator)
      expect(logEntry.timestamp).toEndWith('Z');
    });

    it('should include timestamp in every log entry', () => {
      logger.debug('Debug message');
      logger.info('Info message');
      logger.warn('Warning message');
      logger.error('Error message');

      expect(consoleOutput.length).toBe(4);

      consoleOutput.forEach(output => {
        const entry: LogEntry = JSON.parse(output);
        expect(entry.timestamp).toBeDefined();
        expect(typeof entry.timestamp).toBe('string');
      });
    });
  });

  describe('Correlation ID Management', () => {
    it('should auto-generate correlation ID if not provided', () => {
      logger.info('Test without correlation ID');

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.correlation_id).toBeDefined();
      expect(typeof logEntry.correlation_id).toBe('string');
      expect(logEntry.correlation_id).toMatch(/^cid-\d+-[a-z0-9]+$/);
    });

    it('should use provided correlation ID', () => {
      const providedCorrelationId = 'test-correlation-123';
      logger.info('Test with correlation ID', {
        correlation_id: providedCorrelationId
      });

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.correlation_id).toBe(providedCorrelationId);
    });

    it('should propagate correlation ID in child logger', () => {
      const childLogger = logger.child({ correlation_id: 'parent-123' });
      childLogger.info('Child message');

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.correlation_id).toBe('parent-123');
    });
  });

  describe('Required Log Fields', () => {
    it('should include all required fields in log entry', () => {
      logger.info('Test message', {
        source_service: 'custom-service',
        target_service: 'target-service'
      });

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);

      // Required fields per CLAUDE.md Section 3.3
      expect(logEntry.timestamp).toBeDefined();
      expect(logEntry.level).toBeDefined();
      expect(logEntry.msg).toBeDefined();
      expect(logEntry.source_service).toBeDefined();
      expect(logEntry.correlation_id).toBeDefined();
    });

    it('should include target_service when provided in context', () => {
      logger.info('Test message', {
        target_service: 'external-api'
      });

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.target_service).toBe('external-api');
    });
  });

  describe('JSON Lines Format', () => {
    it('should output single-line JSON per entry', () => {
      logger.info('Test message');

      expect(consoleOutput.length).toBe(1);
      const output = consoleOutput[0];

      // Verify it's valid JSON
      expect(() => JSON.parse(output)).not.toThrow();

      // Verify it's a single line (no newlines within the JSON)
      const lines = output.split('\n');
      expect(lines.length).toBe(1);
    });

    it('should be parseable as JSONL', () => {
      logger.info('Message 1');
      logger.info('Message 2');
      logger.info('Message 3');

      expect(consoleOutput.length).toBe(3);

      // All lines should be valid JSON
      consoleOutput.forEach(output => {
        expect(() => JSON.parse(output)).not.toThrow();
      });
    });
  });

  describe('Log Levels', () => {
    it('should support all log levels', () => {
      logger.debug('Debug');
      logger.info('Info');
      logger.warn('Warning');
      logger.error('Error');

      expect(consoleOutput.length).toBe(4);

      const levels = consoleOutput.map(output => {
        const entry: LogEntry = JSON.parse(output);
        return entry.level;
      });

      expect(levels).toContain('debug');
      expect(levels).toContain('info');
      expect(levels).toContain('warn');
      expect(levels).toContain('error');
    });

    it('should respect minimum log level', () => {
      const strictLogger = new StructuredLogger('test', 'warn');

      strictLogger.debug('Should not appear');
      strictLogger.info('Should not appear');
      strictLogger.warn('Should appear');
      strictLogger.error('Should appear');

      expect(consoleOutput.length).toBe(2);
    });
  });

  describe('Error Handling', () => {
    it('should include error details when error object provided', () => {
      const testError = new Error('Test error message');
      testError.stack = 'Error stack trace';

      logger.error('Operation failed', testError);

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.error).toBeDefined();
      expect(logEntry.error?.message).toBe('Test error message');
      expect(logEntry.error?.stack).toBe('Error stack trace');
    });

    it('should include error code if present', () => {
      const errorWithCode = new Error('Error with code') as Error & { code?: string };
      errorWithCode.code = 'ERR_TIMEOUT';

      logger.error('Operation timed out', errorWithCode);

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.error?.code).toBe('ERR_TIMEOUT');
    });
  });

  describe('Context Merging', () => {
    it('should merge custom context fields', () => {
      logger.info('Test', {
        user_id: 'user-123',
        workflow_id: 'workflow-456',
        custom_field: 'custom_value'
      });

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.user_id).toBe('user-123');
      expect(logEntry.workflow_id).toBe('workflow-456');
      expect(logEntry.custom_field).toBe('custom_value');
    });

    it('should not include internal context fields as root', () => {
      logger.info('Test', {
        correlation_id: 'custom-123',
        source_service: 'custom-service',
        target_service: 'target-api'
      });

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      // These should be at root level, not duplicated
      expect(Object.keys(logEntry).filter(k =>
        ['correlation_id', 'source_service', 'target_service'].includes(k)
      ).length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('Service Name Handling', () => {
    it('should use provided service name', () => {
      const customLogger = new StructuredLogger('custom-service-name');
      customLogger.info('Test');

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.source_service).toBe('custom-service-name');
    });

    it('should override source_service from context', () => {
      logger.info('Test', {
        source_service: 'override-service'
      });

      const logEntry: LogEntry = JSON.parse(consoleOutput[0]);
      expect(logEntry.source_service).toBe('override-service');
    });
  });

  describe('Logger Configuration', () => {
    it('should allow changing minimum log level', () => {
      const testLogger = new StructuredLogger('test', 'info');

      testLogger.debug('Not visible');
      expect(consoleOutput.length).toBe(0);

      testLogger.setMinLevel('debug');
      testLogger.debug('Now visible');
      expect(consoleOutput.length).toBe(1);
    });

    it('should create independent child loggers', () => {
      const child1 = logger.child({ child_id: '1' });
      const child2 = logger.child({ child_id: '2' });

      child1.info('Child 1');
      child2.info('Child 2');

      expect(consoleOutput.length).toBe(2);

      const entry1: LogEntry = JSON.parse(consoleOutput[0]);
      const entry2: LogEntry = JSON.parse(consoleOutput[1]);

      expect(entry1.child_id).toBe('1');
      expect(entry2.child_id).toBe('2');
    });
  });

  describe('Default Logger Instances', () => {
    it('should export default logger instances', () => {
      // Test that default loggers are exported
      expect(() => {
        const { logger: defaultLogger, apiLogger, ragbitsLogger } = require('./structuredLogger');
        expect(defaultLogger).toBeInstanceOf(StructuredLogger);
        expect(apiLogger).toBeInstanceOf(StructuredLogger);
        expect(ragbitsLogger).toBeInstanceOf(StructuredLogger);
      }).not.toThrow();
    });
  });
});
