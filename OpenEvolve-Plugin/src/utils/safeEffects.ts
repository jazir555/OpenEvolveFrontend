/**
 * Safe Effect Hooks
 * 
 * Provides React effect hooks with comprehensive error handling
 */

import { useEffect, useLayoutEffect, DependencyList, EffectCallback } from 'react';
import { errorLogger } from './errorLogging';

/**
 * Safe useEffect with error handling
 */
export function useSafeEffect(effect: EffectCallback, deps?: DependencyList) {
  useEffect(() => {
    let effectResult: ReturnType<EffectCallback> | void;
    
    try {
      effectResult = effect();
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'useSafeEffect', 
          function: 'effect', 
          additionalData: { deps } 
        }
      );
    }

    // Handle cleanup function if returned
    if (typeof effectResult === 'function') {
      return () => {
        try {
          effectResult!();
        } catch (cleanupError) {
          errorLogger.logError(
            cleanupError instanceof Error ? cleanupError : new Error(String(cleanupError)),
            'error',
            { 
              component: 'useSafeEffect', 
              function: 'cleanup', 
              additionalData: { deps } 
            }
          );
        }
      };
    }
  }, deps);
}

/**
 * Safe useLayoutEffect with error handling
 */
export function useSafeLayoutEffect(effect: EffectCallback, deps?: DependencyList) {
  useLayoutEffect(() => {
    let effectResult: ReturnType<EffectCallback> | void;
    
    try {
      effectResult = effect();
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'useSafeLayoutEffect', 
          function: 'effect', 
          additionalData: { deps } 
        }
      );
    }

    // Handle cleanup function if returned
    if (typeof effectResult === 'function') {
      return () => {
        try {
          effectResult!();
        } catch (cleanupError) {
          errorLogger.logError(
            cleanupError instanceof Error ? cleanupError : new Error(String(cleanupError)),
            'error',
            { 
              component: 'useSafeLayoutEffect', 
              function: 'cleanup', 
              additionalData: { deps } 
            }
          );
        }
      };
    }
  }, deps);
}

/**
 * Safe effect that retries on failure
 */
export function useSafeEffectWithRetry(
  effect: EffectCallback,
  deps?: DependencyList,
  options: {
    maxRetries?: number;
    retryDelay?: number;
    onError?: (error: Error) => void;
  } = {}
) {
  const { maxRetries = 3, retryDelay = 1000, onError } = options;

  useEffect(() => {
    let effectResult: ReturnType<EffectCallback> | void;
    let retryCount = 0;
    
    const executeEffect = () => {
      try {
        effectResult = effect();
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'useSafeEffectWithRetry', 
            function: 'effect', 
            additionalData: { deps, retryCount } 
          }
        );
        
        onError?.(typedError);
        
        // Attempt retry if under max attempts
        if (retryCount < maxRetries) {
          retryCount++;
          setTimeout(executeEffect, retryDelay);
        }
      }
    };

    executeEffect();

    // Handle cleanup function if returned
    if (typeof effectResult === 'function') {
      return () => {
        try {
          effectResult!();
        } catch (cleanupError) {
          errorLogger.logError(
            cleanupError instanceof Error ? cleanupError : new Error(String(cleanupError)),
            'error',
            { 
              component: 'useSafeEffectWithRetry', 
              function: 'cleanup', 
              additionalData: { deps } 
            }
          );
        }
      };
    }
  }, deps);
}

/**
 * Safe effect that only runs once but with error handling
 */
export function useSafeEffectOnce(effect: EffectCallback) {
  useEffect(() => {
    let effectResult: ReturnType<EffectCallback> | void;
    let hasRun = false;
    
    if (!hasRun) {
      try {
        hasRun = true;
        effectResult = effect();
      } catch (error) {
        errorLogger.logError(
          error instanceof Error ? error : new Error(String(error)),
          'error',
          { 
            component: 'useSafeEffectOnce', 
            function: 'effect', 
            additionalData: { hasRun } 
          }
        );
      }
    }

    // Handle cleanup function if returned
    if (typeof effectResult === 'function') {
      return () => {
        try {
          effectResult!();
        } catch (cleanupError) {
          errorLogger.logError(
            cleanupError instanceof Error ? cleanupError : new Error(String(cleanupError)),
            'error',
            { 
              component: 'useSafeEffectOnce', 
              function: 'cleanup', 
              additionalData: { hasRun } 
            }
          );
        }
      };
    }
  }, []); // Empty dependency array ensures it only runs once
}

/**
 * Safe async effect with cancellation support
 */
export function useSafeAsyncEffect(
  effect: () => Promise<void> | (() => void | Promise<void>),
  deps?: DependencyList
) {
  useEffect(() => {
    let cancelled = false;
    let cleanupFn: (() => void | Promise<void>) | void;

    const executeEffect = async () => {
      try {
        const result = effect();
        if (typeof result === 'function') {
          cleanupFn = result;
        }
      } catch (error) {
        if (!cancelled) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { 
              component: 'useSafeAsyncEffect', 
              function: 'effect', 
              additionalData: { deps } 
            }
          );
        }
      }
    };

    executeEffect();

    return () => {
      cancelled = true;
      if (cleanupFn) {
        try {
          const cleanupResult = cleanupFn();
          if (cleanupResult instanceof Promise) {
            cleanupResult.catch(cleanupError => {
              errorLogger.logError(
                cleanupError instanceof Error ? cleanupError : new Error(String(cleanupError)),
                'error',
                { 
                  component: 'useSafeAsyncEffect', 
                  function: 'asyncCleanup', 
                  additionalData: { deps } 
                }
              );
            });
          }
        } catch (syncCleanupError) {
          errorLogger.logError(
            syncCleanupError instanceof Error ? syncCleanupError : new Error(String(syncCleanupError)),
            'error',
            { 
              component: 'useSafeAsyncEffect', 
              function: 'syncCleanup', 
              additionalData: { deps } 
            }
          );
        }
      }
    };
  }, deps);
}

/**
 * Safe effect that handles promise rejections
 */
export function useSafePromiseEffect(
  effect: () => Promise<any>,
  deps?: DependencyList,
  options: {
    onError?: (error: Error) => void;
    onSuccess?: (result: any) => void;
  } = {}
) {
  const { onError, onSuccess } = options;

  useEffect(() => {
    let cancelled = false;

    const executeEffect = async () => {
      try {
        const result = await effect();
        if (!cancelled && onSuccess) {
          onSuccess(result);
        }
      } catch (error) {
        if (!cancelled) {
          const typedError = error instanceof Error ? error : new Error(String(error));
          
          errorLogger.logError(
            typedError,
            'error',
            { 
              component: 'useSafePromiseEffect', 
              function: 'effect', 
              additionalData: { deps } 
            }
          );
          
          onError?.(typedError);
        }
      }
    };

    executeEffect();

    return () => {
      cancelled = true;
    };
  }, deps);
}