/**
 * Minimal ambient `react` declaration.
 *
 * The real React types (`@types/react`) are a peer dependency supplied by the
 * host application. This package's isolated `tsc` component build
 * (`tsconfig.components.json`) type-checks against this permissive but
 * generic-aware declaration so it does not require `@types/react` to be installed
 * here. A host build that provides `@types/react` overrides this shim
 * automatically (the real types are more specific and take precedence via
 * normal module resolution).
 */

declare module "react" {
  export type ReactNode = any;
  export type Dispatch<A> = (value: A) => void;
  export type SetStateAction<S> = S | ((prevState: S) => S);
  export type DependencyList = ReadonlyArray<any>;

  export interface FC<P = any> {
    (props: P): ReactNode;
  }
  export const FC: any;

  export function useState<S>(
    initialState: S | (() => S)
  ): [S, Dispatch<SetStateAction<S>>];
  export function useState<S = undefined>(): [
    S | undefined,
    Dispatch<SetStateAction<S | undefined>>
  ];

  export function useEffect(
    effect: () => void | (() => void),
    deps?: DependencyList
  ): void;

  export function useMemo<T>(factory: () => T, deps?: DependencyList): T;
  export function useRef<T>(initialValue: T): { current: T };
  export function useRef<T>(initialValue: T | null): { current: T | null };
  export function useCallback<T extends (...args: any[]) => any>(
    callback: T,
    deps?: DependencyList
  ): T;
  export function useReducer<R extends (...args: any[]) => any>(
    reducer: R,
    initialState: any
  ): [any, Dispatch<any>];
  export function useContext<T>(context: any): T;
  export function useMemo<T>(factory: () => T, deps?: DependencyList): T;
  export function forwardRef<T, P = {}>(render: any): any;
  export function createElement(type: any, props?: any, ...children: any[]): any;
}

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
  interface Element {}
  interface ElementClass {}
}
