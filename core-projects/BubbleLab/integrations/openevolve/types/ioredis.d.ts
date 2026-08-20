/**
 * Ambient module declaration for `ioredis`.
 *
 * ioredis is only used by the Redis bubble. It is not installed in this
 * isolated package, so we declare the minimal surface the bubble uses to
 * keep typecheck self-contained.
 */
declare module 'ioredis' {
  export default class Redis {
    constructor(...args: any[]);
    on(...args: any[]): any;
    get(...args: any[]): any;
    set(...args: any[]): any;
    setex(...args: any[]): any;
    del(...args: any[]): any;
    ping(...args: any[]): any;
    info(...args: any[]): any;
    quit(...args: any[]): any;
    [key: string]: any;
  }
}
