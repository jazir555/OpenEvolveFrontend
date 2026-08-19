/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Ambient module declarations for dependencies that are not resolvable within
 * this adapter's isolated tsconfig (they live outside `adapters/icr-adapter/`
 * or lack bundled type declarations). These are type-level only and do not
 * change runtime behavior.
 */

declare module 'uuid' {
  export function v4(): string;
  const _default: { v4: () => string };
  export default _default;
}

declare module 'axios' {
  export type AxiosInstance = any;
  export type AxiosError = any;
  export type AxiosResponse<T = any> = any;
  const axios: any;
  export default axios;
}

declare module 'express' {
  const express: any;
  export default express;
}
