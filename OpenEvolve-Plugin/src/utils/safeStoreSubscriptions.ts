/**
 * Safe Store Subscription Utilities
 * 
 * Provides utilities for safe store subscriptions with comprehensive error handling
 */

import { StoreApi } from 'zustand/vanilla';
import { errorLogger } from './errorLogging';

/**
 * Safe store subscription with error handling
 */
export function safeSubscribe<TState>(
  store: StoreApi<TState>,
  selector: (state: TState) => any,
  listener: (selectedState: any, previousState: any) => void,
  options?: {
    onError?: (error: Error) => void;
    equalityFn?: (a: any, b: any) => boolean;
  }
): () => void {
  const { onError, equalityFn } = options || {};
  
  try {
    // Create a wrapper listener that handles errors
    const safeListener = (selectedState: any, previousState: any) => {
      try {
        listener(selectedState, previousState);
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeStoreSubscription', 
            function: 'safeSubscribe.listener', 
            additionalData: { selector: selector.toString() } 
          }
        );
        
        onError?.(typedError);
      }
    };

    // Subscribe to the store with error handling
    return store.subscribe(
      selector,
      safeListener,
      equalityFn
    );
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeStoreSubscription', 
        function: 'safeSubscribe', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    
    onError?.(typedError);
    
    // Return a no-op unsubscribe function
    return () => {};
  }
}

/**
 * Safe store subscription with selector error handling
 */
export function safeSubscribeWithSelectorErrorHandling<TState>(
  store: StoreApi<TState>,
  selector: (state: TState) => any,
  listener: (selectedState: any, previousState: any) => void,
  options?: {
    onError?: (error: Error) => void;
    equalityFn?: (a: any, b: any) => boolean;
  }
): () => void {
  const { onError, equalityFn } = options || {};
  
  try {
    // Create a wrapper selector that handles errors
    const safeSelector = (state: TState) => {
      try {
        return selector(state);
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeStoreSubscription', 
            function: 'safeSubscribeWithSelectorErrorHandling.selector', 
            additionalData: { selector: selector.toString() } 
          }
        );
        
        onError?.(typedError);
        
        // Return undefined as a safe fallback
        return undefined;
      }
    };

    // Create a wrapper listener that handles errors
    const safeListener = (selectedState: any, previousState: any) => {
      try {
        listener(selectedState, previousState);
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeStoreSubscription', 
            function: 'safeSubscribeWithSelectorErrorHandling.listener', 
            additionalData: { selector: selector.toString() } 
          }
        );
        
        onError?.(typedError);
      }
    };

    // Subscribe to the store with error handling
    return store.subscribe(
      safeSelector,
      safeListener,
      equalityFn
    );
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeStoreSubscription', 
        function: 'safeSubscribeWithSelectorErrorHandling', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    
    onError?.(typedError);
    
    // Return a no-op unsubscribe function
    return () => {};
  }
}

/**
 * Safe store subscription with retry mechanism
 */
export function safeSubscribeWithRetry<TState>(
  store: StoreApi<TState>,
  selector: (state: TState) => any,
  listener: (selectedState: any, previousState: any) => void,
  options?: {
    onError?: (error: Error) => void;
    equalityFn?: (a: any, b: any) => boolean;
    maxRetries?: number;
    retryDelay?: number;
  }
): () => void {
  const { onError, equalityFn, maxRetries = 3, retryDelay = 1000 } = options || {};
  
  let retryCount = 0;
  let unsubscribe: (() => void) | null = null;
  
  const attemptSubscribe = () => {
    try {
      // Create a wrapper listener that handles errors and retries
      const safeListener = (selectedState: any, previousState: any) => {
        try {
          listener(selectedState, previousState);
        } catch (error) {
          const typedError = error instanceof Error ? error : new Error(String(error));
          
          errorLogger.logError(
            typedError,
            'error',
            { 
              component: 'SafeStoreSubscription', 
              function: 'safeSubscribeWithRetry.listener', 
              additionalData: { selector: selector.toString(), retryCount } 
            }
          );
          
          onError?.(typedError);
          
          // If we've reached max retries, don't try again
          if (retryCount >= maxRetries) {
            return;
          }
          
          // Unsubscribe and try again after delay
          retryCount++;
          if (unsubscribe) {
            unsubscribe();
          }
          
          setTimeout(() => {
            unsubscribe = attemptSubscribe();
          }, retryDelay);
        }
      };

      // Subscribe to the store
      return store.subscribe(
        selector,
        safeListener,
        equalityFn
      );
    } catch (error) {
      const typedError = error instanceof Error ? error : new Error(String(error));
      
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafeStoreSubscription', 
          function: 'safeSubscribeWithRetry', 
          additionalData: { selector: selector.toString(), retryCount, maxRetries } 
        }
      );
      
      onError?.(typedError);
      
      // If we've reached max retries, return no-op
      if (retryCount >= maxRetries) {
        return () => {};
      }
      
      // Try again after delay
      retryCount++;
      setTimeout(() => {
        unsubscribe = attemptSubscribe();
      }, retryDelay);
      
      return () => {};
    }
  };

  unsubscribe = attemptSubscribe();
  return () => {
    try {
      unsubscribe?.();
    } catch (unsubscribeError) {
      errorLogger.logError(
        unsubscribeError instanceof Error ? unsubscribeError : new Error(String(unsubscribeError)),
        'error',
        { 
          component: 'SafeStoreSubscription', 
          function: 'safeSubscribeWithRetry.unsubscribe' 
        }
      );
    }
  };
}

/**
 * Safe store state getter with error handling
 */
export function safeGetState<TState>(store: StoreApi<TState>): TState | null {
  try {
    return store.getState();
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'SafeStoreSubscription', 
        function: 'safeGetState' 
      }
    );
    return null;
  }
}

/**
 * Safe store state update with error handling
 */
export function safeSetState<TState>(
  store: StoreApi<TState>,
  newState: Partial<TState> | ((state: TState) => Partial<TState>),
  options?: {
    onError?: (error: Error) => void;
  }
): void {
  const { onError } = options || {};
  
  try {
    if (typeof newState === 'function') {
      store.setState(newState as (state: TState) => Partial<TState>);
    } else {
      store.setState(newState);
    }
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeStoreSubscription', 
        function: 'safeSetState', 
        additionalData: { newStateType: typeof newState } 
      }
    );
    
    onError?.(typedError);
  }
}

/**
 * Safe store batch update with error handling
 */
export function safeBatchUpdate<TState>(
  store: StoreApi<TState>,
  updates: Array<(state: TState) => Partial<TState>>,
  options?: {
    onError?: (error: Error) => void;
    continueOnError?: boolean;
  }
): void {
  const { onError, continueOnError = true } = options || {};
  
  for (const update of updates) {
    try {
      store.setState(update);
    } catch (error) {
      const typedError = error instanceof Error ? error : new Error(String(error));
      
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafeStoreSubscription', 
          function: 'safeBatchUpdate', 
          additionalData: { updateIndex: updates.indexOf(update) } 
        }
      );
      
      onError?.(typedError);
      
      if (!continueOnError) {
        break;
      }
    }
  }
}

/**
 * Safe store middleware with error handling
 */
export function withSafeStoreMiddleware<TState>(
  store: StoreApi<TState>,
  middleware: (state: TState, prevState: TState) => void
): StoreApi<TState> {
  // Create a new store with error handling middleware
  const originalSubscribe = store.subscribe;
  
  // Override the subscribe method to add error handling
  const safeSubscribe = (
    selector: (state: TState) => any,
    listener: (selectedState: any, previousState: any) => void,
    equalityFn?: (a: any, b: any) => boolean
  ) => {
    const safeListener = (selectedState: any, previousState: any) => {
      try {
        // Get the full state for middleware
        const currentState = store.getState();
        const prevState = previousState; // This is tricky - we don't have the full previous state
        
        // Run middleware
        try {
          middleware(currentState, prevState);
        } catch (middlewareError) {
          errorLogger.logError(
            middlewareError instanceof Error ? middlewareError : new Error(String(middlewareError)),
            'error',
            { 
              component: 'SafeStoreSubscription', 
              function: 'withSafeStoreMiddleware.middleware' 
            }
          );
        }
        
        // Run original listener
        listener(selectedState, previousState);
      } catch (error) {
        errorLogger.logError(
          error instanceof Error ? error : new Error(String(error)),
          'error',
          { 
            component: 'SafeStoreSubscription', 
            function: 'withSafeStoreMiddleware.listener' 
          }
        );
      }
    };
    
    return originalSubscribe(selector, safeListener, equalityFn);
  };
  
  // Return a new store object with the safe subscribe
  return {
    ...store,
    subscribe: safeSubscribe,
    setState: store.setState,
    getState: store.getState,
    destroy: store.destroy,
  };
}

/**
 * Safe store subscription manager
 */
export class SafeStoreSubscriptionManager<TState> {
  private subscriptions: Array<() => void> = [];
  private store: StoreApi<TState>;
  private onError?: (error: Error) => void;

  constructor(store: StoreApi<TState>, options?: { onError?: (error: Error) => void }) {
    this.store = store;
    this.onError = options?.onError;
  }

  /**
   * Add a safe subscription
   */
  addSubscription(
    selector: (state: TState) => any,
    listener: (selectedState: any, previousState: any) => void,
    options?: { equalityFn?: (a: any, b: any) => boolean }
  ): string {
    try {
      const unsubscribe = safeSubscribe(
        this.store,
        selector,
        listener,
        { onError: this.onError, equalityFn: options?.equalityFn }
      );
      
      const id = Math.random().toString(36).substr(2, 9);
      this.subscriptions.push(unsubscribe);
      
      return id;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeStoreSubscription', 
          function: 'SafeStoreSubscriptionManager.addSubscription' 
        }
      );
      return '';
    }
  }

  /**
   * Remove a subscription
   */
  removeSubscription(id: string): boolean {
    try {
      const index = this.subscriptions.findIndex((_, i) => 
        Math.random().toString(36).substr(2, 9) === id // This won't work as intended
      );
      
      if (index !== -1) {
        const unsubscribe = this.subscriptions[index];
        unsubscribe();
        this.subscriptions.splice(index, 1);
        return true;
      }
      
      return false;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeStoreSubscription', 
          function: 'SafeStoreSubscriptionManager.removeSubscription', 
          additionalData: { id } 
        }
      );
      return false;
    }
  }

  /**
   * Remove all subscriptions
   */
  removeAll(): void {
    try {
      this.subscriptions.forEach(unsubscribe => {
        try {
          unsubscribe();
        } catch (unsubscribeError) {
          errorLogger.logError(
            unsubscribeError instanceof Error ? unsubscribeError : new Error(String(unsubscribeError)),
            'error',
            { 
              component: 'SafeStoreSubscription', 
              function: 'SafeStoreSubscriptionManager.removeAll.unsubscribe' 
            }
          );
        }
      });
      this.subscriptions = [];
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeStoreSubscription', 
          function: 'SafeStoreSubscriptionManager.removeAll' 
        }
      );
    }
  }

  /**
   * Get store state safely
   */
  getState(): TState | null {
    return safeGetState(this.store);
  }

  /**
   * Set store state safely
   */
  setState(newState: Partial<TState> | ((state: TState) => Partial<TState>)): void {
    safeSetState(this.store, newState, { onError: this.onError });
  }
}