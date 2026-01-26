/**
 * Object Utilities
 * Helper functions for object manipulation
 */

/**
 * Deep clone object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as T;
  if (obj instanceof Array) return obj.map((item) => deepClone(item)) as T;
  if (obj instanceof Object) {
    const clonedObj = {} as T;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        (clonedObj as Record<string, unknown>)[key] = deepClone(
          (obj as Record<string, unknown>)[key]
        );
      }
    }
    return clonedObj;
  }
  return obj;
}

/**
 * Deep merge objects
 */
export function deepMerge<T extends Record<string, unknown>>(target: T, ...sources: Partial<T>[]): T {
  if (!sources.length) return target;
  const source = sources.shift();

  if (isObject(target) && isObject(source)) {
    for (const key in source) {
      if (isObject(source[key])) {
        if (!target[key]) Object.assign(target, { [key]: {} });
        deepMerge(
          target[key] as Record<string, unknown>,
          source[key] as Record<string, unknown>
        );
      } else {
        Object.assign(target, { [key]: source[key] });
      }
    }
  }

  return deepMerge(target, ...sources);
}

/**
 * Check if value is plain object
 */
export function isObject(item: unknown): item is Record<string, unknown> {
  return Boolean(item && typeof item === 'object' && !Array.isArray(item));
}

/**
 * Check if object is empty
 */
export function isEmpty(obj: Record<string, unknown>): boolean {
  return Object.keys(obj).length === 0;
}

/**
 * Get nested object value by path
 */
export function get(obj: Record<string, unknown>, path: string, defaultValue?: unknown): unknown {
  const keys = path.split('.');
  let result: unknown = obj;

  for (const key of keys) {
    if (isObject(result)) {
      result = result[key];
    } else {
      return defaultValue;
    }
  }

  return result !== undefined ? result : defaultValue;
}

/**
 * Set nested object value by path
 */
export function set(obj: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split('.');
  const lastKey = keys.pop()!;
  let target: Record<string, unknown> = obj;

  for (const key of keys) {
    if (!isObject(target[key])) {
      target[key] = {};
    }
    target = target[key] as Record<string, unknown>;
  }

  target[lastKey] = value;
}

/**
 * Remove keys from object
 */
export function omit<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };
  keys.forEach((key) => delete result[key]);
  return result;
}

/**
 * Pick keys from object
 */
export function pick<T extends Record<string, unknown>, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>;
  keys.forEach((key) => {
    if (key in obj) {
      result[key] = obj[key];
    }
  });
  return result;
}

/**
 * Get all keys of object (including nested)
 */
export function deepKeys(obj: Record<string, unknown>, prefix = ''): string[] {
  const keys: string[] = [];

  for (const key in obj) {
    const path = prefix ? `${prefix}.${key}` : key;
    keys.push(path);

    if (isObject(obj[key])) {
      keys.push(...deepKeys(obj[key] as Record<string, unknown>, path));
    }
  }

  return keys;
}

/**
 * Invert object (swap keys and values)
 */
export function invert<T extends Record<string, string>>(obj: T): Record<string, string> {
  return Object.entries(obj).reduce((acc, [key, value]) => {
    acc[value] = key;
    return acc;
  }, {} as Record<string, string>);
}

/**
 * Transform object values
 */
export function mapValues<T extends Record<string, unknown>, U>(
  obj: T,
  mapper: (value: T[keyof T], key: keyof T) => U
): Record<keyof T, U> {
  return Object.entries(obj).reduce((acc, [key, value]) => {
    acc[key as keyof T] = mapper(value as T[keyof T], key as keyof T);
    return acc;
  }, {} as Record<keyof T, U>);
}

/**
 * Transform object keys
 */
export function mapKeys<T extends Record<string, unknown>>(
  obj: T,
  mapper: (key: keyof T) => string
): Record<string, unknown> {
  return Object.entries(obj).reduce((acc, [key, value]) => {
    acc[mapper(key as keyof T)] = value;
    return acc;
  }, {} as Record<string, unknown>);
}

/**
 * Freeze object recursively
 */
export function deepFreeze<T>(obj: T): T {
  if (isObject(obj)) {
    Object.keys(obj).forEach((key) => {
      const value = obj[key];
      if (isObject(value) || Array.isArray(value)) {
        deepFreeze(value);
      }
    });
    Object.freeze(obj);
  }
  return obj;
}
