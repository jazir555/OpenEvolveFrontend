/**
 * Logger - Glue layer structured logger.
 *
 * Signature: error(msg, error?, context?), warn(msg, context?), info(msg, context?)
 * JSON Lines logging with component and correlation context.
 */

export type LogContext = Record<string, unknown>;

export class Logger {
  private readonly component: string;

  constructor(component: string) {
    this.component = component;
  }

  error(message: string, error?: Error, context?: LogContext): void {
    if (error) {
      // eslint-disable-next-line no-console
      console.error(
        JSON.stringify({
          level: 'error',
          component: this.component,
          message,
          error: error.message,
          stack: error.stack,
          ...context,
        })
      );
    } else {
      // eslint-disable-next-line no-console
      console.error(
        JSON.stringify({ level: 'error', component: this.component, message, ...context })
      );
    }
  }

  warn(message: string, context?: LogContext): void {
    // eslint-disable-next-line no-console
    console.warn(
      JSON.stringify({ level: 'warn', component: this.component, message, ...context })
    );
  }

  info(message: string, context?: LogContext): void {
    // eslint-disable-next-line no-console
    console.info(
      JSON.stringify({ level: 'info', component: this.component, message, ...context })
    );
  }
}
