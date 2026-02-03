type CacheEntry = {
  value: string;
  expiresAt?: number;
};

export class CacheService {
  private readonly store = new Map<string, CacheEntry>();

  constructor(private readonly defaultTtlMs = 15 * 60 * 1000) {}

  get(key: string): string | null {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (entry.expiresAt && entry.expiresAt < Date.now()) {
      this.store.delete(key);
      return null;
    }
    return entry.value;
  }

  set(key: string, value: string, ttlMs?: number): void {
    const ttl = ttlMs ?? this.defaultTtlMs;
    const expiresAt = ttl > 0 ? Date.now() + ttl : undefined;
    this.store.set(key, { value, expiresAt });
  }

  delete(key: string): void {
    this.store.delete(key);
  }

  getJson<T>(key: string): T | null {
    const value = this.get(key);
    if (!value) return null;
    try {
      return JSON.parse(value) as T;
    } catch {
      this.delete(key);
      return null;
    }
  }

  setJson<T>(key: string, value: T, ttlMs?: number): void {
    this.set(key, JSON.stringify(value), ttlMs);
  }
}

export const cacheService = new CacheService();
