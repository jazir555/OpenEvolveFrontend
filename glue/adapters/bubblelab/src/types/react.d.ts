declare module 'react' {
  export type DependencyList = ReadonlyArray<unknown>;
  export type Dispatch<A> = (value: A) => void;
  export type SetStateAction<S> = S | ((prevState: S) => S);

  export function useState<S>(
    initialState: S | (() => S)
  ): [S, Dispatch<SetStateAction<S>>];

  export function useEffect(
    effect: () => void | (() => void),
    deps?: DependencyList
  ): void;
}
