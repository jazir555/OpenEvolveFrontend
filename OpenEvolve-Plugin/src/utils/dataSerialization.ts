/**
 * Data Serialization Utilities
 * 
 * Provides safe serialization and deserialization functions with comprehensive error handling
 */

import { errorLogger } from './errorLogging';

/**
 * Safe JSON stringify with error handling
 */
export function safeStringify(
  data: any, 
  options: {
    space?: string | number;
    fallbackValue?: string;
    errorHandler?: (error: Error) => any;
  } = {}
): string {
  const { space, fallbackValue = '{}', errorHandler } = options;
  
  try {
    return JSON.stringify(data, null, space);
  } catch (error) {
    const serializationError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      serializationError,
      'error',
      { 
        component: 'DataSerialization', 
        function: 'safeStringify', 
        additionalData: { 
          dataType: typeof data,
          dataLength: typeof data === 'string' ? data.length : Array.isArray(data) ? data.length : Object.keys(data || {}).length
        } 
      }
    );
    
    if (errorHandler) {
      try {
        return JSON.stringify(errorHandler(serializationError));
      } catch {
        return fallbackValue;
      }
    }
    
    return fallbackValue;
  }
}

/**
 * Safe JSON parse with error handling
 */
export function safeParse<T = any>(
  jsonString: string, 
  options: {
    fallbackValue?: T;
    errorHandler?: (error: Error) => T;
  } = {}
): T {
  const { fallbackValue, errorHandler } = options;
  
  try {
    return JSON.parse(jsonString) as T;
  } catch (error) {
    const parsingError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      parsingError,
      'error',
      { 
        component: 'DataSerialization', 
        function: 'safeParse', 
        additionalData: { 
          jsonStringLength: jsonString?.length,
          jsonStringPreview: jsonString?.substring(0, 100) + (jsonString?.length > 100 ? '...' : '')
        } 
      }
    );
    
    if (errorHandler) {
      return errorHandler(parsingError);
    }
    
    return fallbackValue as T;
  }
}

/**
 * Serialize with circular reference handling
 */
export function serializeWithCircularRef(
  data: any,
  options: {
    space?: string | number;
    fallbackValue?: string;
  } = {}
): string {
  const { space, fallbackValue = '{}' } = options;
  
  try {
    const seen = new WeakSet();
    return JSON.stringify(data, (key, val) => {
      if (val != null && typeof val === 'object') {
        if (seen.has(val)) {
          return '[Circular]';
        }
        seen.add(val);
      }
      return val;
    }, space);
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'DataSerialization', 
        function: 'serializeWithCircularRef', 
        additionalData: { 
          dataType: typeof data,
          hasCircularRefs: checkForCircularReferences(data)
        } 
      }
    );
    
    return fallbackValue;
  }
}

/**
 * Check if object has circular references
 */
function checkForCircularReferences(obj: any, seen = new WeakSet()): boolean {
  if (obj && typeof obj === 'object') {
    if (seen.has(obj)) {
      return true;
    }
    seen.add(obj);
    
    for (const key in obj) {
      if (obj.hasOwnProperty(key) && checkForCircularReferences(obj[key], seen)) {
        return true;
      }
    }
    
    seen.delete(obj);
  }
  return false;
}

/**
 * Deep clone with error handling
 */
export function safeClone<T>(obj: T): T | null {
  try {
    if (obj === null || obj === undefined) {
      return obj;
    }
    
    // Handle primitive types
    if (typeof obj !== 'object') {
      return obj;
    }
    
    // Handle dates
    if (obj instanceof Date) {
      return new Date(obj.getTime()) as any;
    }
    
    // Handle arrays
    if (Array.isArray(obj)) {
      return obj.map(item => safeClone(item)) as any;
    }
    
    // Handle objects
    const cloned: any = {};
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = safeClone(obj[key]);
      }
    }
    
    return cloned;
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'DataSerialization', 
        function: 'safeClone', 
        additionalData: { 
          objectType: typeof obj,
          isArray: Array.isArray(obj)
        } 
      }
    );
    
    return null;
  }
}

/**
 * Serialize to URL parameters with error handling
 */
export function serializeToUrlParams(
  params: Record<string, any>,
  options: {
    arrayFormat?: 'brackets' | 'indices' | 'repeat' | 'comma';
    fallbackValue?: string;
  } = {}
): string {
  const { arrayFormat = 'brackets', fallbackValue = '' } = options;
  
  try {
    const searchParams = new URLSearchParams();
    
    for (const [key, value] of Object.entries(params)) {
      if (value === null || value === undefined) {
        continue;
      }
      
      if (Array.isArray(value)) {
        switch (arrayFormat) {
          case 'brackets':
            value.forEach(v => searchParams.append(`${key}[]`, String(v)));
            break;
          case 'indices':
            value.forEach((v, i) => searchParams.append(`${key}[${i}]`, String(v)));
            break;
          case 'repeat':
            value.forEach(v => searchParams.append(key, String(v)));
            break;
          case 'comma':
            searchParams.append(key, value.join(','));
            break;
        }
      } else {
        searchParams.append(key, String(value));
      }
    }
    
    return searchParams.toString();
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'DataSerialization', 
        function: 'serializeToUrlParams', 
        additionalData: { params } 
      }
    );
    
    return fallbackValue;
  }
}

/**
 * Deserialize from URL parameters with error handling
 */
export function deserializeFromUrlParams(
  paramString: string
): Record<string, any> {
  try {
    const searchParams = new URLSearchParams(paramString);
    const result: Record<string, any> = {};
    
    for (const [key, value] of searchParams.entries()) {
      // Check if this is an array parameter (ends with [])
      if (key.endsWith('[]')) {
        const arrayKey = key.slice(0, -2);
        if (!result[arrayKey]) {
          result[arrayKey] = [];
        }
        result[arrayKey].push(value);
      } else {
        // Check if there are multiple values for the same key
        const allValues = searchParams.getAll(key);
        if (allValues.length > 1) {
          result[key] = allValues;
        } else {
          result[key] = value;
        }
      }
    }
    
    return result;
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'DataSerialization', 
        function: 'deserializeFromUrlParams', 
        additionalData: { paramString } 
      }
    );
    
    return {};
  }
}

/**
 * Serialize to FormData with error handling
 */
export function serializeToFormData(
  data: Record<string, any>,
  options: {
    arrayFormat?: 'brackets' | 'indices' | 'repeat';
    fileHandling?: 'include' | 'exclude' | 'error';
  } = {}
): FormData {
  const { arrayFormat = 'brackets', fileHandling = 'include' } = options;
  
  try {
    const formData = new FormData();
    
    for (const [key, value] of Object.entries(data)) {
      if (value === null || value === undefined) {
        continue;
      }
      
      if (Array.isArray(value)) {
        switch (arrayFormat) {
          case 'brackets':
            value.forEach(v => {
              if (fileHandling === 'exclude' && (v instanceof File || v instanceof Blob)) {
                return;
              }
              if (fileHandling === 'error' && (v instanceof File || v instanceof Blob)) {
                throw new Error(`File objects not allowed in FormData: ${key}`);
              }
              formData.append(`${key}[]`, v);
            });
            break;
          case 'indices':
            value.forEach((v, i) => {
              if (fileHandling === 'exclude' && (v instanceof File || v instanceof Blob)) {
                return;
              }
              if (fileHandling === 'error' && (v instanceof File || v instanceof Blob)) {
                throw new Error(`File objects not allowed in FormData: ${key}[${i}]`);
              }
              formData.append(`${key}[${i}]`, v);
            });
            break;
          case 'repeat':
            value.forEach(v => {
              if (fileHandling === 'exclude' && (v instanceof File || v instanceof Blob)) {
                return;
              }
              if (fileHandling === 'error' && (v instanceof File || v instanceof Blob)) {
                throw new Error(`File objects not allowed in FormData: ${key}`);
              }
              formData.append(key, v);
            });
            break;
        }
      } else {
        if (fileHandling === 'exclude' && (value instanceof File || value instanceof Blob)) {
          continue;
        }
        if (fileHandling === 'error' && (value instanceof File || value instanceof Blob)) {
          throw new Error(`File objects not allowed in FormData: ${key}`);
        }
        formData.append(key, value);
      }
    }
    
    return formData;
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'DataSerialization', 
        function: 'serializeToFormData', 
        additionalData: { data, options } 
      }
    );
    
    return new FormData();
  }
}

/**
 * Safe serialization for storage (localStorage/sessionStorage)
 */
export function serializeForStorage(data: any): string {
  try {
    // First try regular serialization
    const serialized = safeStringify(data);
    
    // Check if the serialized data is too large for storage
    const sizeInBytes = new Blob([serialized]).size;
    const maxSize = 5 * 1024 * 1024; // 5MB limit
    
    if (sizeInBytes > maxSize) {
      throw new Error(`Serialized data too large for storage: ${sizeInBytes} bytes (max: ${maxSize})`);
    }
    
    return serialized;
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'DataSerialization', 
        function: 'serializeForStorage', 
        additionalData: { dataSize: JSON.stringify(data)?.length } 
      }
    );
    
    return '{}';
  }
}

/**
 * Safe deserialization from storage
 */
export function deserializeFromStorage<T = any>(serializedData: string): T | null {
  try {
    if (!serializedData) {
      return null;
    }
    
    return safeParse<T>(serializedData);
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'DataSerialization', 
        function: 'deserializeFromStorage', 
        additionalData: { serializedDataLength: serializedData?.length } 
      }
    );
    
    return null;
  }
}

/**
 * Create a serializer with custom error handling
 */
export function createSerializer(options: {
  errorHandler?: (error: Error, operation: string, data?: any) => any;
  fallbackValue?: any;
} = {}) {
  const { errorHandler, fallbackValue } = options;
  
  return {
    stringify: (data: any, space?: string | number) => {
      try {
        return safeStringify(data, { space, fallbackValue, errorHandler: errorHandler ? () => errorHandler(new Error('Serialization failed'), 'stringify', data) : undefined });
      } catch (error) {
        return fallbackValue ?? '{}';
      }
    },
    
    parse: <T = any>(jsonString: string) => {
      try {
        return safeParse<T>(jsonString, { fallbackValue, errorHandler: errorHandler ? (error) => errorHandler(error, 'parse', jsonString) : undefined });
      } catch (error) {
        return fallbackValue;
      }
    },
    
    clone: <T>(obj: T) => {
      try {
        return safeClone(obj);
      } catch (error) {
        if (errorHandler) {
          errorHandler(error as Error, 'clone', obj);
        }
        return fallbackValue;
      }
    },
  };
}

// Export a default safe serializer instance
export const safeSerializer = createSerializer({
  fallbackValue: null,
});