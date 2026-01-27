/// <reference types="vitest/globals" />

declare module 'vitest' {
  interface Matchers<R = any> {
    toBeInTheDocument(): R;
    toBeVisible(): R;
    toHaveClass(...classes: string[]): R;
    toHaveTextContent(text: string | RegExp): R;
    toBeDisabled(): R;
    toBeEnabled(): R;
    toHaveAttribute(name: string, value?: string): R;
    toHaveStyle(css: Record<string, string>): R;
    toHaveValue(value: string | string[]): R;
    toBeChecked(): R;
    toContainHTML(html: string): R;
    toHaveFocus(): R;
  }
}
