/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Local compile-time/runtime stand-in for the shared Glue `Logger`
 * (mirrors the signature of `../../lib/logger`). The adapter's emitted
 * `require('../types/logger')` resolves to this module, so behavior is
 * preserved without depending on files outside `adapters/icr-adapter/`.
 */

export class Logger {
  constructor(public readonly name: string = 'unknown') {}

  info(msg: any, context?: any): void {
    this.write('info', msg, context);
  }

  warn(msg: any, context?: any): void {
    this.write('warn', msg, context);
  }

  error(msg: any, error?: any, context?: any): void {
    this.write('error', msg, { ...context, ...(error && {
      error_name: error.name,
      error_message: error.message,
      error_stack: error.stack
    }) });
  }

  private write(level: string, msg: any, context?: any): void {
    const entry = {
      level,
      msg,
      timestamp_utc: new Date().toISOString(),
      source_service: this.name,
      ...context
    };
    // eslint-disable-next-line no-console
    console.log(JSON.stringify(entry));
  }
}
