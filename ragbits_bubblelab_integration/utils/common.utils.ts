/**
 * Utility functions for Ragbits + BubbleLab Integration
 */

export interface LoggerConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  prefix?: string;
}

export class Logger {
  private config: LoggerConfig;

  constructor(config: LoggerConfig = { level: 'info' }) {
    this.config = config;
  }

  debug(message: string): void {
    if (this.shouldLog('debug')) {
      this.log('DEBUG', message);
    }
  }

  info(message: string): void {
    if (this.shouldLog('info')) {
      this.log('INFO', message);
    }
  }

  warn(message: string): void {
    if (this.shouldLog('warn')) {
      this.log('WARN', message);
    }
  }

  error(message: string): void {
    if (this.shouldLog('error')) {
      this.log('ERROR', message);
    }
  }

  private shouldLog(level: keyof typeof Logger.prototype): boolean {
    const levels = { debug: 0, info: 1, warn: 2, error: 3 };
    return levels[level] >= levels[this.config.level];
  }

  private log(level: string, message: string): void {
    const timestamp = new Date().toISOString();
    const prefix = this.config.prefix ? `[${this.config.prefix}] ` : '';
    console.log(`[${timestamp}] ${level} ${prefix}${message}`);
  }
}

export function generateId(prefix: string = 'id'): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

export function validateSchema<T>(obj: any, schema: any): obj is T {
  // Simple validation - in a real implementation, you'd use a proper validation library
  try {
    // This is a basic check - a real implementation would validate against the schema
    return obj !== null && typeof obj === 'object';
  } catch {
    return false;
  }
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return function executedFunction(...args: Parameters<T>): void {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}