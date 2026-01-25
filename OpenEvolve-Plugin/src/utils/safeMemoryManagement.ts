/**
 * Safe Memory Management Utilities
 * 
 * Provides utilities for safe memory management with comprehensive error handling
 */

import { errorLogger } from './errorLogging';

/**
 * Safe memory cleanup utility
 */
export function safeCleanup(
  cleanupFn: () => void,
  options?: {
    onError?: (error: Error) => void;
  }
): void {
  const { onError } = options || {};
  
  try {
    cleanupFn();
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeMemoryManagement', 
        function: 'safeCleanup', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    
    onError?.(typedError);
  }
}

/**
 * Safe resource manager with error handling
 */
export class SafeResourceManager {
  private resources: Array<() => void> = [];
  private onError?: (error: Error) => void;

  constructor(options?: { onError?: (error: Error) => void }) {
    this.onError = options?.onError;
  }

  /**
   * Add a resource cleanup function
   */
  add(cleanupFn: () => void): void {
    try {
      this.resources.push(cleanupFn);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeResourceManager.add' 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Remove a resource cleanup function
   */
  remove(cleanupFn: () => void): boolean {
    try {
      const index = this.resources.indexOf(cleanupFn);
      if (index !== -1) {
        this.resources.splice(index, 1);
        return true;
      }
      return false;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeResourceManager.remove' 
        }
      );
      this.onError?.(error as Error);
      return false;
    }
  }

  /**
   * Clean up all resources
   */
  cleanup(): void {
    try {
      this.resources.forEach(cleanupFn => {
        try {
          cleanupFn();
        } catch (cleanupError) {
          errorLogger.logError(
            cleanupError instanceof Error ? cleanupError : new Error(String(cleanupError)),
            'error',
            { 
              component: 'SafeMemoryManagement', 
              function: 'SafeResourceManager.cleanup.resource' 
            }
          );
          this.onError?.(cleanupError as Error);
        }
      });
      this.resources = [];
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeResourceManager.cleanup' 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Get the number of managed resources
   */
  getResourceCount(): number {
    try {
      return this.resources.length;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeResourceManager.getResourceCount' 
        }
      );
      this.onError?.(error as Error);
      return 0;
    }
  }
}

/**
 * Safe memory monitor with error handling
 */
export class SafeMemoryMonitor {
  private intervalId: number | null = null;
  private onError?: (error: Error) => void;
  private onMemoryWarning?: (usage: { used: number; total: number; percentage: number }) => void;

  constructor(options?: {
    onError?: (error: Error) => void;
    onMemoryWarning?: (usage: { used: number; total: number; percentage: number }) => void;
  }) {
    this.onError = options?.onError;
    this.onMemoryWarning = options?.onMemoryWarning;
  }

  /**
   * Start monitoring memory usage
   */
  start(interval: number = 5000): void {
    try {
      if (this.intervalId !== null) {
        this.stop();
      }

      this.intervalId = window.setInterval(() => {
        try {
          this.checkMemoryUsage();
        } catch (monitorError) {
          errorLogger.logError(
            monitorError instanceof Error ? monitorError : new Error(String(monitorError)),
            'error',
            { 
              component: 'SafeMemoryManagement', 
              function: 'SafeMemoryMonitor.start.check' 
            }
          );
          this.onError?.(monitorError as Error);
        }
      }, interval);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryMonitor.start' 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Stop monitoring memory usage
   */
  stop(): void {
    try {
      if (this.intervalId !== null) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryMonitor.stop' 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Check current memory usage
   */
  checkMemoryUsage(): { used: number; total: number; percentage: number } | null {
    try {
      // Check for memory API availability
      if (performance && (performance as any).memory) {
        const memory = (performance as any).memory;
        const used = memory.usedJSHeapSize;
        const total = memory.jsHeapSizeLimit;
        const percentage = (used / total) * 100;

        // Trigger warning callback if threshold is exceeded
        if (percentage > 80) { // 80% threshold
          this.onMemoryWarning?.({ used, total, percentage });
        }

        return { used, total, percentage };
      }

      // Fallback for environments without memory API
      return { used: 0, total: 0, percentage: 0 };
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryMonitor.checkMemoryUsage' 
        }
      );
      this.onError?.(error as Error);
      return null;
    }
  }
}

/**
 * Safe object pool with error handling
 */
export class SafeObjectPool<T> {
  private pool: T[] = [];
  private factory: () => T;
  private resetFn?: (obj: T) => void;
  private onError?: (error: Error) => void;
  private maxSize: number;

  constructor(
    factory: () => T,
    options?: {
      resetFn?: (obj: T) => void;
      onError?: (error: Error) => void;
      maxSize?: number;
    }
  ) {
    this.factory = factory;
    this.resetFn = options?.resetFn;
    this.onError = options?.onError;
    this.maxSize = options?.maxSize || 100; // Default max size
  }

  /**
   * Get an object from the pool
   */
  acquire(): T {
    try {
      if (this.pool.length > 0) {
        return this.pool.pop()!;
      } else {
        return this.factory();
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeObjectPool.acquire' 
        }
      );
      this.onError?.(error as Error);
      return this.factory(); // Fallback to creating new object
    }
  }

  /**
   * Release an object back to the pool
   */
  release(obj: T): void {
    try {
      // Reset the object if reset function is provided
      if (this.resetFn) {
        this.resetFn(obj);
      }

      // Only add back to pool if we haven't exceeded max size
      if (this.pool.length < this.maxSize) {
        this.pool.push(obj);
      } else {
        // If pool is full, we could implement disposal logic here
        // For now, we just ignore the extra objects
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeObjectPool.release' 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Clear the pool
   */
  clear(): void {
    try {
      this.pool = [];
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeObjectPool.clear' 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Get the current pool size
   */
  size(): number {
    try {
      return this.pool.length;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeObjectPool.size' 
        }
      );
      this.onError?.(error as Error);
      return 0;
    }
  }
}

/**
 * Safe memory leak detector
 */
export class SafeMemoryLeakDetector {
  private trackedObjects: Map<string, WeakRef<object>> = new Map();
  private trackedArrays: Map<string, WeakRef<any[]>> = new Map();
  private trackedMaps: Map<string, WeakRef<Map<any, any>>> = new Map();
  private trackedSets: Map<string, WeakRef<Set<any>>> = new Map();
  private onError?: (error: Error) => void;

  constructor(options?: { onError?: (error: Error) => void }) {
    this.onError = options?.onError;
  }

  /**
   * Track an object for potential memory leaks
   */
  trackObject(id: string, obj: object): void {
    try {
      if (typeof WeakRef !== 'undefined') {
        this.trackedObjects.set(id, new WeakRef(obj));
      } else {
        // Fallback for environments without WeakRef
        console.warn('WeakRef not supported, memory leak detection may be limited');
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryLeakDetector.trackObject', 
          additionalData: { id } 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Track an array for potential memory leaks
   */
  trackArray(id: string, arr: any[]): void {
    try {
      if (typeof WeakRef !== 'undefined') {
        this.trackedArrays.set(id, new WeakRef(arr));
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryLeakDetector.trackArray', 
          additionalData: { id } 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Track a Map for potential memory leaks
   */
  trackMap(id: string, map: Map<any, any>): void {
    try {
      if (typeof WeakRef !== 'undefined') {
        this.trackedMaps.set(id, new WeakRef(map));
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryLeakDetector.trackMap', 
          additionalData: { id } 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Track a Set for potential memory leaks
   */
  trackSet(id: string, set: Set<any>): void {
    try {
      if (typeof WeakRef !== 'undefined') {
        this.trackedSets.set(id, new WeakRef(set));
      }
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryLeakDetector.trackSet', 
          additionalData: { id } 
        }
      );
      this.onError?.(error as Error);
    }
  }

  /**
   * Check for potential memory leaks
   */
  checkLeaks(): Array<{ id: string; type: string; isAlive: boolean }> {
    try {
      const results: Array<{ id: string; type: string; isAlive: boolean }> = [];

      // Check objects
      for (const [id, ref] of this.trackedObjects) {
        try {
          const isAlive = ref.deref() !== undefined;
          results.push({ id, type: 'object', isAlive });
          if (!isAlive) {
            this.trackedObjects.delete(id);
          }
        } catch (checkError) {
          errorLogger.logError(
            checkError instanceof Error ? checkError : new Error(String(checkError)),
            'error',
            { 
              component: 'SafeMemoryManagement', 
              function: 'SafeMemoryLeakDetector.checkLeaks.object', 
              additionalData: { id } 
            }
          );
          this.onError?.(checkError as Error);
        }
      }

      // Check arrays
      for (const [id, ref] of this.trackedArrays) {
        try {
          const isAlive = ref.deref() !== undefined;
          results.push({ id, type: 'array', isAlive });
          if (!isAlive) {
            this.trackedArrays.delete(id);
          }
        } catch (checkError) {
          errorLogger.logError(
            checkError instanceof Error ? checkError : new Error(String(checkError)),
            'error',
            { 
              component: 'SafeMemoryManagement', 
              function: 'SafeMemoryLeakDetector.checkLeaks.array', 
              additionalData: { id } 
            }
          );
          this.onError?.(checkError as Error);
        }
      }

      // Check maps
      for (const [id, ref] of this.trackedMaps) {
        try {
          const isAlive = ref.deref() !== undefined;
          results.push({ id, type: 'map', isAlive });
          if (!isAlive) {
            this.trackedMaps.delete(id);
          }
        } catch (checkError) {
          errorLogger.logError(
            checkError instanceof Error ? checkError : new Error(String(checkError)),
            'error',
            { 
              component: 'SafeMemoryManagement', 
              function: 'SafeMemoryLeakDetector.checkLeaks.map', 
              additionalData: { id } 
            }
          );
          this.onError?.(checkError as Error);
        }
      }

      // Check sets
      for (const [id, ref] of this.trackedSets) {
        try {
          const isAlive = ref.deref() !== undefined;
          results.push({ id, type: 'set', isAlive });
          if (!isAlive) {
            this.trackedSets.delete(id);
          }
        } catch (checkError) {
          errorLogger.logError(
            checkError instanceof Error ? checkError : new Error(String(checkError)),
            'error',
            { 
              component: 'SafeMemoryManagement', 
              function: 'SafeMemoryLeakDetector.checkLeaks.set', 
              additionalData: { id } 
            }
          );
          this.onError?.(checkError as Error);
        }
      }

      return results;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryLeakDetector.checkLeaks' 
        }
      );
      this.onError?.(error as Error);
      return [];
    }
  }

  /**
   * Clear all tracked objects
   */
  clear(): void {
    try {
      this.trackedObjects.clear();
      this.trackedArrays.clear();
      this.trackedMaps.clear();
      this.trackedSets.clear();
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeMemoryManagement', 
          function: 'SafeMemoryLeakDetector.clear' 
        }
      );
      this.onError?.(error as Error);
    }
  }
}

/**
 * Safe garbage collection helper (where available)
 */
export function safeGarbageCollect(
  options?: {
    onError?: (error: Error) => void;
  }
): void {
  const { onError } = options || {};
  
  try {
    // Try to trigger garbage collection if available (mainly for testing environments)
    if (typeof (globalThis as any).gc === 'function') {
      (globalThis as any).gc();
    } else if (typeof window !== 'undefined' && (window as any).gc) {
      (window as any).gc();
    }
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'SafeMemoryManagement', 
        function: 'safeGarbageCollect' 
      }
    );
    onError?.(error as Error);
  }
}

/**
 * Safe memory usage reporter
 */
export function safeGetMemoryUsage(): { used: number; total: number; percentage: number } | null {
  try {
    if (performance && (performance as any).memory) {
      const memory = (performance as any).memory;
      const used = memory.usedJSHeapSize;
      const total = memory.jsHeapSizeLimit;
      const percentage = (used / total) * 100;

      return { used, total, percentage };
    }

    // Fallback for environments without memory API
    return { used: 0, total: 0, percentage: 0 };
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'SafeMemoryManagement', 
        function: 'safeGetMemoryUsage' 
      }
    );
    return null;
  }
}