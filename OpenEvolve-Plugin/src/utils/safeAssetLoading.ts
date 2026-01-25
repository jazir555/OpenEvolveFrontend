/**
 * Safe Asset Loading Utilities
 * 
 * Provides asset loading utilities with comprehensive error handling
 */

import { errorLogger } from './errorLogging';

/**
 * Safe image loading with error handling
 */
export function safeLoadImage(
  src: string,
  options?: {
    timeout?: number;
    onError?: (error: Error) => void;
    onSuccess?: (image: HTMLImageElement) => void;
    fallbackSrc?: string;
  }
): Promise<HTMLImageElement> {
  const { timeout = 10000, onError, onSuccess, fallbackSrc } = options || {};
  
  return new Promise((resolve, reject) => {
    const image = new Image();
    const timeoutId = setTimeout(() => {
      const error = new Error(`Image load timeout: ${src}`);
      errorLogger.logError(error, 'error', {
        component: 'SafeAssetLoading',
        function: 'safeLoadImage',
        additionalData: { src, timeout }
      });
      
      onError?.(error);
      
      if (fallbackSrc) {
        safeLoadImage(fallbackSrc, options)
          .then(resolve)
          .catch(reject);
      } else {
        reject(error);
      }
    }, timeout);

    image.onload = () => {
      clearTimeout(timeoutId);
      onSuccess?.(image);
      resolve(image);
    };

    image.onerror = () => {
      clearTimeout(timeoutId);
      const error = new Error(`Failed to load image: ${src}`);
      errorLogger.logError(error, 'error', {
        component: 'SafeAssetLoading',
        function: 'safeLoadImage',
        additionalData: { src }
      });
      
      onError?.(error);
      
      if (fallbackSrc) {
        safeLoadImage(fallbackSrc, options)
          .then(resolve)
          .catch(reject);
      } else {
        reject(error);
      }
    };

    image.src = src;
  });
}

/**
 * Safe script loading with error handling
 */
export function safeLoadScript(
  src: string,
  options?: {
    timeout?: number;
    onError?: (error: Error) => void;
    onSuccess?: (script: HTMLScriptElement) => void;
    attributes?: Record<string, string>;
  }
): Promise<HTMLScriptElement> {
  const { timeout = 10000, onError, onSuccess, attributes = {} } = options || {};
  
  return new Promise((resolve, reject) => {
    // Check if script is already loaded
    const existingScript = document.querySelector(`script[src="${src}"]`) as HTMLScriptElement;
    if (existingScript) {
      onSuccess?.(existingScript);
      resolve(existingScript);
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    
    // Apply additional attributes
    Object.entries(attributes).forEach(([key, value]) => {
      script.setAttribute(key, value);
    });

    const timeoutId = setTimeout(() => {
      const error = new Error(`Script load timeout: ${src}`);
      errorLogger.logError(error, 'error', {
        component: 'SafeAssetLoading',
        function: 'safeLoadScript',
        additionalData: { src, timeout, attributes }
      });
      
      onError?.(error);
      reject(error);
      
      // Clean up the script element
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
    }, timeout);

    script.onload = () => {
      clearTimeout(timeoutId);
      onSuccess?.(script);
      resolve(script);
    };

    script.onerror = () => {
      clearTimeout(timeoutId);
      const error = new Error(`Failed to load script: ${src}`);
      errorLogger.logError(error, 'error', {
        component: 'SafeAssetLoading',
        function: 'safeLoadScript',
        additionalData: { src, attributes }
      });
      
      onError?.(error);
      reject(error);
    };

    document.head.appendChild(script);
  });
}

/**
 * Safe stylesheet loading with error handling
 */
export function safeLoadStylesheet(
  href: string,
  options?: {
    timeout?: number;
    onError?: (error: Error) => void;
    onSuccess?: (link: HTMLLinkElement) => void;
    attributes?: Record<string, string>;
  }
): Promise<HTMLLinkElement> {
  const { timeout = 10000, onError, onSuccess, attributes = {} } = options || {};
  
  return new Promise((resolve, reject) => {
    // Check if stylesheet is already loaded
    const existingLink = document.querySelector(`link[href="${href}"]`) as HTMLLinkElement;
    if (existingLink) {
      onSuccess?.(existingLink);
      resolve(existingLink);
      return;
    }

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    
    // Apply additional attributes
    Object.entries(attributes).forEach(([key, value]) => {
      link.setAttribute(key, value);
    });

    const timeoutId = setTimeout(() => {
      const error = new Error(`Stylesheet load timeout: ${href}`);
      errorLogger.logError(error, 'error', {
        component: 'SafeAssetLoading',
        function: 'safeLoadStylesheet',
        additionalData: { href, timeout, attributes }
      });
      
      onError?.(error);
      reject(error);
      
      // Clean up the link element
      if (link.parentNode) {
        link.parentNode.removeChild(link);
      }
    }, timeout);

    link.onload = () => {
      clearTimeout(timeoutId);
      onSuccess?.(link);
      resolve(link);
    };

    link.onerror = () => {
      clearTimeout(timeoutId);
      const error = new Error(`Failed to load stylesheet: ${href}`);
      errorLogger.logError(error, 'error', {
        component: 'SafeAssetLoading',
        function: 'safeLoadStylesheet',
        additionalData: { href, attributes }
      });
      
      onError?.(error);
      reject(error);
    };

    document.head.appendChild(link);
  });
}

/**
 * Safe asset preloader with error handling
 */
export class SafeAssetPreloader {
  private loadedAssets: Map<string, any> = new Map();
  private loadingPromises: Map<string, Promise<any>> = new Map();

  /**
   * Preload an image asset
   */
  async preloadImage(src: string, options?: Parameters<typeof safeLoadImage>[1]): Promise<HTMLImageElement> {
    if (this.loadedAssets.has(src)) {
      return this.loadedAssets.get(src);
    }

    if (this.loadingPromises.has(src)) {
      return this.loadingPromises.get(src);
    }

    const loadPromise = safeLoadImage(src, options)
      .then(image => {
        this.loadedAssets.set(src, image);
        this.loadingPromises.delete(src);
        return image;
      })
      .catch(error => {
        this.loadingPromises.delete(src);
        throw error;
      });

    this.loadingPromises.set(src, loadPromise);
    return loadPromise;
  }

  /**
   * Preload a script asset
   */
  async preloadScript(src: string, options?: Parameters<typeof safeLoadScript>[1]): Promise<HTMLScriptElement> {
    if (this.loadedAssets.has(src)) {
      return this.loadedAssets.get(src);
    }

    if (this.loadingPromises.has(src)) {
      return this.loadingPromises.get(src);
    }

    const loadPromise = safeLoadScript(src, options)
      .then(script => {
        this.loadedAssets.set(src, script);
        this.loadingPromises.delete(src);
        return script;
      })
      .catch(error => {
        this.loadingPromises.delete(src);
        throw error;
      });

    this.loadingPromises.set(src, loadPromise);
    return loadPromise;
  }

  /**
   * Preload a stylesheet asset
   */
  async preloadStylesheet(href: string, options?: Parameters<typeof safeLoadStylesheet>[1]): Promise<HTMLLinkElement> {
    if (this.loadedAssets.has(href)) {
      return this.loadedAssets.get(href);
    }

    if (this.loadingPromises.has(href)) {
      return this.loadingPromises.get(href);
    }

    const loadPromise = safeLoadStylesheet(href, options)
      .then(link => {
        this.loadedAssets.set(href, link);
        this.loadingPromises.delete(href);
        return link;
      })
      .catch(error => {
        this.loadingPromises.delete(href);
        throw error;
      });

    this.loadingPromises.set(href, loadPromise);
    return loadPromise;
  }

  /**
   * Preload multiple assets
   */
  async preloadAssets(assets: Array<{
    type: 'image' | 'script' | 'stylesheet';
    src: string;
    options?: any;
  }>): Promise<void> {
    const promises = assets.map(asset => {
      switch (asset.type) {
        case 'image':
          return this.preloadImage(asset.src, asset.options);
        case 'script':
          return this.preloadScript(asset.src, asset.options);
        case 'stylesheet':
          return this.preloadStylesheet(asset.src, asset.options);
        default:
          const error = new Error(`Unknown asset type: ${(asset as any).type}`);
          errorLogger.logError(error, 'error', {
            component: 'SafeAssetLoading',
            function: 'SafeAssetPreloader.preloadAssets',
            additionalData: { asset }
          });
          return Promise.reject(error);
      }
    });

    try {
      await Promise.all(promises);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeAssetLoading', 
          function: 'SafeAssetPreloader.preloadAssets', 
          additionalData: { assetCount: assets.length } 
        }
      );
      throw error;
    }
  }

  /**
   * Check if an asset is loaded
   */
  isLoaded(src: string): boolean {
    try {
      return this.loadedAssets.has(src);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeAssetLoading', 
          function: 'SafeAssetPreloader.isLoaded', 
          additionalData: { src } 
        }
      );
      return false;
    }
  }

  /**
   * Get a loaded asset
   */
  getAsset(src: string): any | null {
    try {
      return this.loadedAssets.get(src) || null;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeAssetLoading', 
          function: 'SafeAssetPreloader.getAsset', 
          additionalData: { src } 
        }
      );
      return null;
    }
  }

  /**
   * Clear loaded assets
   */
  clear(): void {
    try {
      this.loadedAssets.clear();
      // Don't clear loadingPromises as ongoing loads should complete
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeAssetLoading', 
          function: 'SafeAssetPreloader.clear' 
        }
      );
    }
  }
}

/**
 * Safe asset loader with retry and fallback
 */
export async function safeLoadAssetWithRetry(
  loader: () => Promise<any>,
  options?: {
    maxRetries?: number;
    retryDelay?: number;
    fallback?: () => Promise<any>;
  }
): Promise<any> {
  const { maxRetries = 3, retryDelay = 1000, fallback } = options || {};
  
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await loader();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      errorLogger.logError(
        lastError,
        'warn', // Use warn for retries
        { 
          component: 'SafeAssetLoading', 
          function: 'safeLoadAssetWithRetry', 
          additionalData: { attempt, maxRetries } 
        }
      );
      
      if (attempt < maxRetries) {
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }
  
  // If all retries failed and fallback is provided, try the fallback
  if (fallback) {
    try {
      return await fallback();
    } catch (fallbackError) {
      errorLogger.logError(
        fallbackError instanceof Error ? fallbackError : new Error(String(fallbackError)),
        'error',
        { 
          component: 'SafeAssetLoading', 
          function: 'safeLoadAssetWithRetry.fallback' 
        }
      );
    }
  }
  
  // If we get here, all attempts failed
  throw lastError;
}

/**
 * Safe asset bundle loader
 */
export async function safeLoadAssetBundle(
  assets: Array<{
    id: string;
    type: 'image' | 'script' | 'stylesheet';
    src: string;
    options?: any;
  }>,
  options?: {
    parallel?: boolean;
    onProgress?: (loaded: number, total: number) => void;
    onError?: (error: Error, assetId: string) => void;
  }
): Promise<Record<string, any>> {
  const { parallel = true, onProgress, onError } = options || {};
  const results: Record<string, any> = {};
  const total = assets.length;
  let loaded = 0;

  const updateProgress = () => {
    loaded++;
    onProgress?.(loaded, total);
  };

  if (parallel) {
    // Load all assets in parallel
    const promises = assets.map(async (asset) => {
      try {
        let result;
        switch (asset.type) {
          case 'image':
            result = await safeLoadImage(asset.src, asset.options);
            break;
          case 'script':
            result = await safeLoadScript(asset.src, asset.options);
            break;
          case 'stylesheet':
            result = await safeLoadStylesheet(asset.src, asset.options);
            break;
          default:
            throw new Error(`Unknown asset type: ${asset.type}`);
        }
        results[asset.id] = result;
        updateProgress();
        return result;
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeAssetLoading', 
            function: 'safeLoadAssetBundle.parallel', 
            additionalData: { assetId: asset.id, assetType: asset.type, assetSrc: asset.src } 
          }
        );
        onError?.(typedError, asset.id);
        throw typedError;
      }
    });

    try {
      await Promise.all(promises);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeAssetLoading', 
          function: 'safeLoadAssetBundle.parallel', 
          additionalData: { totalAssets: assets.length } 
        }
      );
      throw error;
    }
  } else {
    // Load assets sequentially
    for (const asset of assets) {
      try {
        let result;
        switch (asset.type) {
          case 'image':
            result = await safeLoadImage(asset.src, asset.options);
            break;
          case 'script':
            result = await safeLoadScript(asset.src, asset.options);
            break;
          case 'stylesheet':
            result = await safeLoadStylesheet(asset.src, asset.options);
            break;
          default:
            throw new Error(`Unknown asset type: ${asset.type}`);
        }
        results[asset.id] = result;
        updateProgress();
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeAssetLoading', 
            function: 'safeLoadAssetBundle.sequential', 
            additionalData: { assetId: asset.id, assetType: asset.type, assetSrc: asset.src } 
          }
        );
        onError?.(typedError, asset.id);
        throw typedError;
      }
    }
  }

  return results;
}