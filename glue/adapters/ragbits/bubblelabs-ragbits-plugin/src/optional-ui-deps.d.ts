declare namespace React {
  type ReactNode = unknown;
  type SetStateAction<T> = T | ((prev: T) => T);
  type Dispatch<T> = (value: T) => void;
  type ComponentType<T = any> = (props?: any) => unknown;
  type FC<T = any> = ComponentType<T>;
  type ChangeEvent<T = any> = { target: T };
  type FormEvent<T = any> = { target: T; preventDefault(): void };
  type MouseEvent<T = any> = { currentTarget: T; preventDefault(): void };
  type KeyboardEvent<T = any> = {
    currentTarget: T;
    key: string;
    shiftKey?: boolean;
    preventDefault(): void;
  };

  function useState<T = any>(initial: T | (() => T)): [T, Dispatch<SetStateAction<T>>];
  function useEffect(effect: () => void | (() => void), deps?: readonly unknown[]): void;
  function useMemo<T>(factory: () => T, deps?: readonly unknown[]): T;
  function useCallback<T extends (...args: any[]) => any>(
    callback: T,
    deps?: readonly unknown[]
  ): T;
  function useRef<T = any>(value?: T): { current: T };
  function createContext<T = any>(value: T): any;
  function useContext<T = any>(context: any): T;
  function lazy<T extends ComponentType<any>>(
    factory: () => Promise<{ default: T }>
  ): T;
  function createElement(type: any, props?: any, ...children: any[]): unknown;
  const Suspense: FC<any>;
}

declare module 'react' {
  export = React;
  export as namespace React;
}

declare module 'react/jsx-runtime' {
  export const Fragment: unknown;
  export function jsx(type: unknown, props: unknown, key?: unknown): unknown;
  export function jsxs(type: unknown, props: unknown, key?: unknown): unknown;
}

declare namespace JSX {
  interface IntrinsicAttributes {
    key?: string | number;
  }

  interface IntrinsicElements {
    [elemName: string]: any;
  }
}

declare module 'react-toastify' {
  export interface ToastOptions {
    [key: string]: unknown;
  }

  export const toast: {
    success(message: unknown, options?: ToastOptions): unknown;
    error(message: unknown, options?: ToastOptions): unknown;
    info(message: unknown, options?: ToastOptions): unknown;
    warn(message: unknown, options?: ToastOptions): unknown;
    warning(message: unknown, options?: ToastOptions): unknown;
    loading?(message: unknown, options?: ToastOptions): unknown;
    dismiss?(id?: unknown): void;
  };
}

declare module 'lucide-react' {
  export type IconComponent = (props: Record<string, unknown>) => unknown;

  export const AlertCircle: IconComponent;
  export const AlertTriangle: IconComponent;
  export const BarChart2: IconComponent;
  export const BarChart3: IconComponent;
  export const Bot: IconComponent;
  export const Brain: IconComponent;
  export const Check: IconComponent;
  export const CheckCircle: IconComponent;
  export const Clock: IconComponent;
  export const Cpu: IconComponent;
  export const Database: IconComponent;
  export const ExternalLink: IconComponent;
  export const Eye: IconComponent;
  export const EyeOff: IconComponent;
  export const FileText: IconComponent;
  export const Filter: IconComponent;
  export const Info: IconComponent;
  export const Loader2: IconComponent;
  export const MathFunction: IconComponent;
  export const Network: IconComponent;
  export const Pause: IconComponent;
  export const Pipeline: IconComponent;
  export const Play: IconComponent;
  export const Plus: IconComponent;
  export const Puzzle: IconComponent;
  export const RefreshCw: IconComponent;
  export const RotateCcw: IconComponent;
  export const Save: IconComponent;
  export const Search: IconComponent;
  export const Server: IconComponent;
  export const Settings: IconComponent;
  export const Shield: IconComponent;
  export const Square: IconComponent;
  export const Stop: IconComponent;
  export const Tool: IconComponent;
  export const Trash2: IconComponent;
  export const Upload: IconComponent;
  export const X: IconComponent;
  export const XCircle: IconComponent;
}
