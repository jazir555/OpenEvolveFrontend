export interface LoggerContext {
  correlation_id?: string;
  source_service?: string;
  target_service?: string;
  [key: string]: any;
}

export class Logger {
  constructor(serviceName?: string);
  debug(msg: string, context?: LoggerContext): void;
  info(msg: string, context?: LoggerContext): void;
  warn(msg: string, context?: LoggerContext): void;
  error(msg: string, error?: Error, context?: LoggerContext): void;
  child(context: LoggerContext): Logger;
}
