/**
 * Safe Event Listener Utilities
 * 
 * Provides utilities for safe event handling with comprehensive error handling
 */

import { errorLogger } from './errorLogging';

/**
 * Safe event listener with error handling
 */
export function addSafeEventListener<K extends keyof HTMLElementEventMap>(
  element: HTMLElement,
  type: K,
  listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => any,
  options?: boolean | AddEventListenerOptions
): () => void {
  // Create a wrapper listener that handles errors
  const safeListener = function (this: HTMLElement, event: HTMLElementEventMap[K]) {
    try {
      return listener.call(this, event);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeEventListener', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  };

  // Add the event listener
  element.addEventListener(type, safeListener as EventListener, options);

  // Return a function to remove the event listener
  return () => {
    try {
      element.removeEventListener(type, safeListener as EventListener, options);
    } catch (removeError) {
      errorLogger.logError(
        removeError instanceof Error ? removeError : new Error(String(removeError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeEventListener.remove', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  };
}

/**
 * Safe event listener for window object
 */
export function addSafeWindowEventListener<K extends keyof WindowEventMap>(
  type: K,
  listener: (this: Window, ev: WindowEventMap[K]) => any,
  options?: boolean | AddEventListenerOptions
): () => void {
  // Create a wrapper listener that handles errors
  const safeListener = function (this: Window, event: WindowEventMap[K]) {
    try {
      return listener.call(this, event);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeWindowEventListener', 
          additionalData: { eventType: type } 
        }
      );
    }
  };

  // Add the event listener
  window.addEventListener(type, safeListener as EventListener, options);

  // Return a function to remove the event listener
  return () => {
    try {
      window.removeEventListener(type, safeListener as EventListener, options);
    } catch (removeError) {
      errorLogger.logError(
        removeError instanceof Error ? removeError : new Error(String(removeError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeWindowEventListener.remove', 
          additionalData: { eventType: type } 
        }
      );
    }
  };
}

/**
 * Safe event listener for document object
 */
export function addSafeDocumentEventListener<K extends keyof DocumentEventMap>(
  type: K,
  listener: (this: Document, ev: DocumentEventMap[K]) => any,
  options?: boolean | AddEventListenerOptions
): () => void {
  // Create a wrapper listener that handles errors
  const safeListener = function (this: Document, event: DocumentEventMap[K]) {
    try {
      return listener.call(this, event);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeDocumentEventListener', 
          additionalData: { eventType: type } 
        }
      );
    }
  };

  // Add the event listener
  document.addEventListener(type, safeListener as EventListener, options);

  // Return a function to remove the event listener
  return () => {
    try {
      document.removeEventListener(type, safeListener as EventListener, options);
    } catch (removeError) {
      errorLogger.logError(
        removeError instanceof Error ? removeError : new Error(String(removeError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeDocumentEventListener.remove', 
          additionalData: { eventType: type } 
        }
      );
    }
  };
}

/**
 * Safe event listener manager
 */
export class SafeEventListenerManager {
  private listeners: Array<() => void> = [];

  /**
   * Add a safe event listener
   */
  add<K extends keyof HTMLElementEventMap>(
    element: HTMLElement,
    type: K,
    listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => any,
    options?: boolean | AddEventListenerOptions
  ): void {
    try {
      const removeListener = addSafeEventListener(element, type, listener, options);
      this.listeners.push(removeListener);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'SafeEventListenerManager.add', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  }

  /**
   * Add a safe window event listener
   */
  addWindow<K extends keyof WindowEventMap>(
    type: K,
    listener: (this: Window, ev: WindowEventMap[K]) => any,
    options?: boolean | AddEventListenerOptions
  ): void {
    try {
      const removeListener = addSafeWindowEventListener(type, listener, options);
      this.listeners.push(removeListener);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'SafeEventListenerManager.addWindow', 
          additionalData: { eventType: type } 
        }
      );
    }
  }

  /**
   * Add a safe document event listener
   */
  addDocument<K extends keyof DocumentEventMap>(
    type: K,
    listener: (this: Document, ev: DocumentEventMap[K]) => any,
    options?: boolean | AddEventListenerOptions
  ): void {
    try {
      const removeListener = addSafeDocumentEventListener(type, listener, options);
      this.listeners.push(removeListener);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'SafeEventListenerManager.addDocument', 
          additionalData: { eventType: type } 
        }
      );
    }
  }

  /**
   * Remove all event listeners
   */
  removeAll(): void {
    try {
      this.listeners.forEach(removeListener => {
        try {
          removeListener();
        } catch (removeError) {
          errorLogger.logError(
            removeError instanceof Error ? removeError : new Error(String(removeError)),
            'error',
            { 
              component: 'SafeEventListener', 
              function: 'SafeEventListenerManager.removeAll.listener' 
            }
          );
        }
      });
      this.listeners = [];
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'SafeEventListenerManager.removeAll' 
        }
      );
    }
  }

  /**
   * Get the number of active listeners
   */
  getListenerCount(): number {
    try {
      return this.listeners.length;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'SafeEventListenerManager.getListenerCount' 
        }
      );
      return 0;
    }
  }
}

/**
 * Safe event dispatcher with error handling
 */
export function safeDispatchEvent(
  target: EventTarget,
  event: Event,
  options?: {
    onError?: (error: Error) => void;
  }
): boolean {
  const { onError } = options || {};
  
  try {
    return target.dispatchEvent(event);
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeEventListener', 
        function: 'safeDispatchEvent', 
        additionalData: { eventType: event.type } 
      }
    );
    
    onError?.(typedError);
    return false;
  }
}

/**
 * Safe event handler factory with error handling
 */
export function createSafeEventHandler<T extends Event>(
  handler: (event: T) => void,
  options?: {
    onError?: (error: Error) => void;
    preventDefaultOnError?: boolean;
  }
): (event: T) => void {
  const { onError, preventDefaultOnError = false } = options || {};
  
  return (event: T) => {
    try {
      handler(event);
    } catch (error) {
      const typedError = error instanceof Error ? error : new Error(String(error));
      
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'createSafeEventHandler', 
          additionalData: { eventType: event.type } 
        }
      );
      
      onError?.(typedError);
      
      if (preventDefaultOnError && event.cancelable) {
        event.preventDefault();
      }
    }
  };
}

/**
 * Safe event listener with retry mechanism
 */
export function addSafeEventListenerWithRetry<K extends keyof HTMLElementEventMap>(
  element: HTMLElement,
  type: K,
  listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => any,
  options?: boolean | AddEventListenerOptions,
  retryOptions?: {
    maxRetries?: number;
    retryDelay?: number;
  }
): () => void {
  const { maxRetries = 3, retryDelay = 1000 } = retryOptions || {};
  let retryCount = 0;
  
  const safeListener = function (this: HTMLElement, event: HTMLElementEventMap[K]) {
    try {
      return listener.call(this, event);
    } catch (error) {
      const typedError = error instanceof Error ? error : new Error(String(error));
      
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeEventListenerWithRetry', 
          additionalData: { eventType: type, elementTag: element.tagName, retryCount } 
        }
      );
      
      // If we've reached max retries, don't try again
      if (retryCount >= maxRetries) {
        return;
      }
      
      // Retry the listener after a delay
      retryCount++;
      setTimeout(() => {
        try {
          listener.call(this, event);
        } catch (retryError) {
          errorLogger.logError(
            retryError instanceof Error ? retryError : new Error(String(retryError)),
            'error',
            { 
              component: 'SafeEventListener', 
              function: 'addSafeEventListenerWithRetry.retry', 
              additionalData: { eventType: type, elementTag: element.tagName, retryCount } 
            }
          );
        }
      }, retryDelay);
    }
  };

  // Add the event listener
  element.addEventListener(type, safeListener as EventListener, options);

  // Return a function to remove the event listener
  return () => {
    try {
      element.removeEventListener(type, safeListener as EventListener, options);
    } catch (removeError) {
      errorLogger.logError(
        removeError instanceof Error ? removeError : new Error(String(removeError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeEventListenerWithRetry.remove', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  };
}

/**
 * Safe event listener with timeout
 */
export function addSafeEventListenerWithTimeout<K extends keyof HTMLElementEventMap>(
  element: HTMLElement,
  type: K,
  listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => any,
  timeout: number,
  options?: boolean | AddEventListenerOptions
): () => void {
  let timeoutId: number | null = null;
  
  const safeListener = function (this: HTMLElement, event: HTMLElementEventMap[K]) {
    try {
      // Clear the timeout when the event is handled
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
      
      return listener.call(this, event);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeEventListenerWithTimeout', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  };

  // Add the event listener
  element.addEventListener(type, safeListener as EventListener, options);

  // Set up the timeout
  timeoutId = window.setTimeout(() => {
    try {
      element.removeEventListener(type, safeListener as EventListener, options);
    } catch (removeError) {
      errorLogger.logError(
        removeError instanceof Error ? removeError : new Error(String(removeError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeEventListenerWithTimeout.timeout', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  }, timeout);

  // Return a function to remove the event listener and clear the timeout
  return () => {
    try {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      element.removeEventListener(type, safeListener as EventListener, options);
    } catch (removeError) {
      errorLogger.logError(
        removeError instanceof Error ? removeError : new Error(String(removeError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeEventListenerWithTimeout.remove', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  };
}

/**
 * Safe event listener with debounce
 */
export function addSafeDebouncedEventListener<K extends keyof HTMLElementEventMap>(
  element: HTMLElement,
  type: K,
  listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => any,
  delay: number,
  options?: boolean | AddEventListenerOptions
): () => void {
  let timeoutId: number | null = null;
  
  const safeListener = function (this: HTMLElement, event: HTMLElementEventMap[K]) {
    try {
      // Clear the previous timeout
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
      
      // Set a new timeout to execute the listener
      timeoutId = window.setTimeout(() => {
        try {
          listener.call(this, event);
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { 
              component: 'SafeEventListener', 
              function: 'addSafeDebouncedEventListener.timeout', 
              additionalData: { eventType: type, elementTag: element.tagName } 
            }
          );
        }
      }, delay);
    } catch (setupError) {
      errorLogger.logError(
        setupError instanceof Error ? setupError : new Error(String(setupError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeDebouncedEventListener', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  };

  // Add the event listener
  element.addEventListener(type, safeListener as EventListener, options);

  // Return a function to remove the event listener and clear the timeout
  return () => {
    try {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      element.removeEventListener(type, safeListener as EventListener, options);
    } catch (removeError) {
      errorLogger.logError(
        removeError instanceof Error ? removeError : new Error(String(removeError)),
        'error',
        { 
          component: 'SafeEventListener', 
          function: 'addSafeDebouncedEventListener.remove', 
          additionalData: { eventType: type, elementTag: element.tagName } 
        }
      );
    }
  };
}