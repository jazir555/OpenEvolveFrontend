declare module '@headlessui/react' {
  export const RadioGroup: any;
  export const Switch: any;
  export const Listbox: any;
  export const Disclosure: any;
  export const Popover: any;
  export const Menu: any;
  export const Dialog: any;
  export const Transition: any;
  export const Tab: any;
}

declare module '@testing-library/react' {
  export const render: any;
  export const screen: any;
  export const waitFor: any;
  export const fireEvent: any;
  export const cleanup: any;
  export interface RenderOptions {}
}

declare module '@testing-library/jest-dom/matchers' {
  const matchers: any;
  export default matchers;
}

declare module 'react-router-dom' {
  export const MemoryRouter: any;
  export const Router: any;
  export const Route: any;
  export const Routes: any;
  export const Navigate: any;
  export const BrowserRouter: any;
  export const useLocation: any;
  export const useNavigate: any;
  export const useParams: any;
}

// Vitest globals
declare function describe(name: string, fn: () => void): void;
declare function it(name: string, fn: () => void | Promise<void>): void;
declare function test(name: string, fn: () => void | Promise<void>): void;
declare function expect(value: any): any;
declare const vi: any;
declare function beforeEach(fn: () => void): void;
declare function afterEach(fn: () => void): void;
declare function beforeAll(fn: () => void): void;
declare function afterAll(fn: () => void): void;
