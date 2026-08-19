/**
 * Local ambient type shims for the KarateClub adapter.
 *
 * `uuid@9` ships no bundled type declarations and `@types/uuid` is not
 * installed in this workspace, so declare the minimal surface the adapter
 * actually uses. This is a type-only declaration; runtime behavior is
 * unchanged (the real `uuid` package is still loaded at runtime).
 */

declare module 'uuid' {
  export function v4(): string;
}
