/**
 * Jest Setup File
 *
 * Global setup for Jest tests.
 * This file is run before each test file.
 */
export {};
declare global {
    namespace NodeJS {
        interface Global {
            testUtils: {
                createTestProblem: (overrides?: any) => any;
                wait: (ms: number) => Promise<void>;
                retry: <T>(fn: () => Promise<T>, maxRetries?: number, delay?: number) => Promise<T>;
            };
        }
    }
}
//# sourceMappingURL=jest.setup.d.ts.map