"use strict";
/**
 * Contract Test for Circuit Breaker
 *
 * Tests compliance with Federation Constitution Section 2.3:
 * - System Failure → Circuit Breaker
 * - Stop hammering the dead service
 * - Wait for health check to pass
 * - Half-open state testing
 */
Object.defineProperty(exports, "__esModule", { value: true });
const circuitBreaker_1 = require("./circuitBreaker");
describe('CircuitBreaker Contract Tests', () => {
    describe('Configuration Compliance (Law 5)', () => {
        it('should accept explicit configuration', () => {
            const config = {
                failureThreshold: 5,
                successThreshold: 2,
                timeoutMs: 60000,
                monitoringPeriodMs: 10000
            };
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', config);
            expect(breaker).toBeDefined();
            expect(breaker.getState().state).toBe(circuitBreaker_1.CircuitState.CLOSED);
        });
        it('should use sensible defaults if not provided', () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 3
            });
            const stats = breaker.getState();
            expect(stats.state).toBe(circuitBreaker_1.CircuitState.CLOSED);
            expect(breaker).toBeDefined();
        });
    });
    describe('Circuit State Transitions', () => {
        it('should start in CLOSED state', () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 3,
                timeoutMs: 1000
            });
            expect(breaker.getState().state).toBe(circuitBreaker_1.CircuitState.CLOSED);
        });
        it('should transition to OPEN after failure threshold', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 2,
                timeoutMs: 100
            });
            // Fail twice to reach threshold
            try {
                await breaker.execute(async () => {
                    throw new Error('Simulated failure');
                });
            }
            catch (e) {
                // Expected
            }
            try {
                await breaker.execute(async () => {
                    throw new Error('Simulated failure');
                });
            }
            catch (e) {
                // Expected
            }
            // Should be open now
            const stats = breaker.getState();
            expect(stats.state).toBe(circuitBreaker_1.CircuitState.OPEN);
            expect(stats.failureCount).toBeGreaterThanOrEqual(2);
            expect(stats.openedAt).toBeDefined();
        });
        it('should transition to HALF-OPEN after timeout', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1,
                timeoutMs: 100, // Short timeout
                successThreshold: 2
            });
            // Trigger open state
            try {
                await breaker.execute(async () => {
                    throw new Error('Failure');
                });
            }
            catch (e) {
                // Expected
            }
            expect(breaker.getState().state).toBe(circuitBreaker_1.CircuitState.OPEN);
            // Wait for timeout
            await new Promise(resolve => setTimeout(resolve, 150));
            // Next execution should attempt half-open
            try {
                await breaker.execute(async () => {
                    throw new Error('Still failing');
                });
            }
            catch (e) {
                // Expected
            }
            const stats = breaker.getState();
            expect(stats.state).toBe(circuitBreaker_1.CircuitState.OPEN); // Still failing
            expect(stats.failureCount).toBeGreaterThan(0);
        });
        it('should transition back to CLOSED after success threshold in HALF-OPEN', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1,
                timeoutMs: 100,
                successThreshold: 2
            });
            // Trigger open state
            try {
                await breaker.execute(async () => {
                    throw new Error('Failure');
                });
            }
            catch (e) {
                // Expected
            }
            expect(breaker.getState().state).toBe(circuitBreaker_1.CircuitState.OPEN);
            // Wait for timeout
            await new Promise(resolve => setTimeout(resolve, 150));
            // Execute successful operations to close circuit
            const result1 = await breaker.execute(async () => 'success1');
            const result2 = await breaker.execute(async () => 'success2');
            expect(result1).toBe('success1');
            expect(result2).toBe('success2');
            const stats = breaker.getState();
            expect(stats.state).toBe(circuitBreaker_1.CircuitState.CLOSED);
            expect(stats.failureCount).toBe(0);
        });
    });
    describe('Request Rejection', () => {
        it('should reject requests when circuit is OPEN', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1,
                timeoutMs: 5000
            });
            // Open the circuit
            try {
                await breaker.execute(async () => {
                    throw new Error('Failure');
                });
            }
            catch (e) {
                // Expected
            }
            expect(breaker.getState().state).toBe(circuitBreaker_1.CircuitState.OPEN);
            // Try to execute while open
            await expectAsync(breaker.execute(async () => 'should not execute')).rejects.toThrow(circuitBreaker_1.CircuitBreakerOpenError);
        });
        it('should include next attempt time in rejection', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1,
                timeoutMs: 60000
            });
            // Open the circuit
            try {
                await breaker.execute(async () => {
                    throw new Error('Failure');
                });
            }
            catch (e) {
                // Expected
            }
            try {
                await breaker.execute(async () => 'should not execute');
            }
            catch (error) {
                expect(error).toBeInstanceOf(circuitBreaker_1.CircuitBreakerOpenError);
                expect(error.message).toContain('retry after');
            }
        });
    });
    describe('Failure Tracking', () => {
        it('should track failure count', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 5
            });
            // Execute 3 successful operations
            await breaker.execute(async () => 'ok');
            await breaker.execute(async () => 'ok');
            await breaker.execute(async () => 'ok');
            let stats = breaker.getState();
            expect(stats.failureCount).toBe(0);
            // Fail 2 times
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            stats = breaker.getState();
            expect(stats.failureCount).toBeGreaterThanOrEqual(2);
        });
        it('should track success count in HALF-OPEN state', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1,
                timeoutMs: 100,
                successThreshold: 3
            });
            // Open circuit
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            // Wait for timeout
            await new Promise(resolve => setTimeout(resolve, 150));
            // Execute 2 successful operations
            await breaker.execute(async () => 'success1');
            await breaker.execute(async () => 'success2');
            const stats = breaker.getState();
            expect(stats.successCount).toBeGreaterThanOrEqual(2);
        });
        it('should record last failure time', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 5
            });
            const beforeFail = Date.now();
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            const stats = breaker.getState();
            expect(stats.lastFailureTime).toBeGreaterThanOrEqual(beforeFail);
            expect(stats.lastFailureTime).toBeLessThanOrEqual(Date.now());
        });
    });
    describe('Circuit Breaker Stats', () => {
        it('should return comprehensive stats', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 3,
                timeoutMs: 5000,
                successThreshold: 2
            });
            // Generate some activity
            await breaker.execute(async () => 'ok');
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            const stats = breaker.getState();
            expect(stats).toBeDefined();
            expect(stats.state).toBeDefined();
            expect(typeof stats.failureCount).toBe('number');
            expect(typeof stats.successCount).toBe('number');
            expect(stats.lastFailureTime).toBeDefined();
            expect(stats.lastSuccessTime).toBeDefined();
        });
        it('should include next attempt time when OPEN', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1,
                timeoutMs: 60000
            });
            // Open circuit
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            const stats = breaker.getState();
            expect(stats.nextAttemptAt).toBeDefined();
            expect(stats.nextAttemptAt).toBeGreaterThan(Date.now());
        });
    });
    describe('Reset Functionality', () => {
        it('should reset to CLOSED state', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1
            });
            // Open the circuit
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            expect(breaker.getState().state).toBe(circuitBreaker_1.CircuitState.OPEN);
            // Reset
            breaker.reset();
            expect(breaker.getState().state).toBe(circuitBreaker_1.CircuitState.CLOSED);
            expect(breaker.getState().failureCount).toBe(0);
            expect(breaker.getState().openedAt).toBeUndefined();
        });
    });
    describe('CircuitBreakerRegistry', () => {
        it('should manage multiple circuit breakers', () => {
            const registry = new circuitBreaker_1.CircuitBreakerRegistry();
            const breaker1 = registry.get('service-1', {
                failureThreshold: 5
            });
            const breaker2 = registry.get('service-2', {
                failureThreshold: 3
            });
            expect(breaker1).toBeDefined();
            expect(breaker2).toBeDefined();
            // Should return same instance on subsequent calls
            const breaker1Again = registry.get('service-1');
            expect(breaker1Again).toBe(breaker1);
        });
        it('should get stats for all breakers', () => {
            const registry = new circuitBreaker_1.CircuitBreakerRegistry();
            registry.get('service-1');
            registry.get('service-2');
            registry.get('service-3');
            const allStats = registry.getAllStats();
            expect(allStats.size).toBe(3);
            expect(allStats.has('service-1')).toBe(true);
            expect(allStats.has('service-2')).toBe(true);
            expect(allStats.has('service-3')).toBe(true);
        });
        it('should reset all breakers', () => {
            const registry = new circuitBreaker_1.CircuitBreakerRegistry();
            const breaker1 = registry.get('service-1', {
                failureThreshold: 1
            });
            // Open circuit
            try {
                breaker1.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            expect(breaker1.getState().state).toBe(circuitBreaker_1.CircuitState.OPEN);
            // Reset all
            registry.resetAll();
            expect(breaker1.getState().state).toBe(circuitBreaker_1.CircuitState.CLOSED);
        });
        it('should remove specific breaker', () => {
            const registry = new circuitBreaker_1.CircuitBreakerRegistry();
            registry.get('service-1');
            expect(registry.getAllStats().has('service-1')).toBe(true);
            registry.remove('service-1');
            expect(registry.getAllStats().has('service-1')).toBe(false);
        });
    });
    describe('Error Types', () => {
        it('should throw CircuitBreakerOpenError when OPEN', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('test-service', {
                failureThreshold: 1,
                timeoutMs: 5000
            });
            // Open circuit
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            // Try to execute
            try {
                await breaker.execute(async () => 'should fail');
                fail('Should have thrown CircuitBreakerOpenError');
            }
            catch (error) {
                expect(error).toBeInstanceOf(circuitBreaker_1.CircuitBreakerOpenError);
                expect(error.name).toBe('CircuitBreakerOpenError');
            }
        });
        it('should include service name in error message', async () => {
            const breaker = new circuitBreaker_1.CircuitBreaker('my-service', {
                failureThreshold: 1,
                timeoutMs: 5000
            });
            // Open circuit
            try {
                await breaker.execute(async () => {
                    throw new Error('fail');
                });
            }
            catch (e) {
                // Expected
            }
            try {
                await breaker.execute(async () => 'fail');
                fail('Should have thrown');
            }
            catch (error) {
                expect(error.message).toContain('my-service');
            }
        });
    });
});
//# sourceMappingURL=circuitBreaker.contract.test.js.map