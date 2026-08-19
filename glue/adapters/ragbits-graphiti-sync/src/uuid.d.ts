declare module 'uuid' {
  export function v4(options?: { random?: Uint8Array; rng?: () => Uint8Array }): string;
  export function v4(
    options: { random?: Uint8Array; rng?: () => Uint8Array },
    buffer: Uint8Array,
    offset?: number
  ): Uint8Array;
  export function v1(options?: any): string;
  export function v3(name: string, namespace: string | Uint8Array): string;
  export function v5(name: string, namespace: string | Uint8Array): string;
}
