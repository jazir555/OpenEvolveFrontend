/**
 * URL Utilities
 * URL parsing and manipulation helpers
 */

/**
 * Get query param value
 */
export function getQueryParam(param: string, url?: string): string | null {
  const searchParams = new URLSearchParams(url ? new URL(url).search : window.location.search);
  return searchParams.get(param);
}

/**
 * Get all query params
 */
export function getQueryParams(url?: string): Record<string, string> {
  const searchParams = new URLSearchParams(url ? new URL(url).search : window.location.search);
  const params: Record<string, string> = {};

  searchParams.forEach((value, key) => {
    params[key] = value;
  });

  return params;
}

/**
 * Set query param
 */
export function setQueryParam(param: string, value: string): void {
  const url = new URL(window.location.href);
  url.searchParams.set(param, value);
  window.history.replaceState({}, '', url.toString());
}

/**
 * Remove query param
 */
export function removeQueryParam(param: string): void {
  const url = new URL(window.location.href);
  url.searchParams.delete(param);
  window.history.replaceState({}, '', url.toString());
}

/**
 * Build URL with query params
 */
export function buildUrl(base: string, params: Record<string, string | number | boolean | undefined>): string {
  const url = new URL(base);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

/**
 * Parse URL into parts
 */
export function parseUrl(url: string): {
  protocol: string;
  hostname: string;
  port?: string;
  pathname: string;
  search: string;
  hash: string;
} {
  const parsed = new URL(url);
  return {
    protocol: parsed.protocol,
    hostname: parsed.hostname,
    port: parsed.port,
    pathname: parsed.pathname,
    search: parsed.search,
    hash: parsed.hash,
  };
}

/**
 * Check if URL is absolute
 */
export function isAbsoluteUrl(url: string): boolean {
  return /^https?:\/\//i.test(url);
}

/**
 * Check if URL is relative
 */
export function isRelativeUrl(url: string): boolean {
  return !isAbsoluteUrl(url);
}

/**
 * Join URL parts
 */
export function joinUrl(...parts: string[]): string {
  return parts
    .map((part, index) => {
      if (index === 0) return part.replace(/\/+$/, '');
      return part.replace(/^\/+/, '').replace(/\/+$/, '');
    })
    .filter(Boolean)
    .join('/');
}

/**
 * Get file extension from URL
 */
export function getUrlExtension(url: string): string {
  const pathname = new URL(url).pathname;
  const match = pathname.match(/\.([^.]+)$/);
  return match ? match[1] : '';
}

/**
 * Get filename from URL
 */
export function getUrlFilename(url: string): string {
  const pathname = new URL(url).pathname;
  const parts = pathname.split('/');
  return parts[parts.length - 1] || '';
}

/**
 * Check if URLs are the same (ignoring hash and search)
 */
export function isSameUrl(url1: string, url2: string): boolean {
  const u1 = new URL(url1);
  const u2 = new URL(url2);

  return (
    u1.protocol === u2.protocol &&
    u1.hostname === u2.hostname &&
    u1.port === u2.port &&
    u1.pathname === u2.pathname
  );
}

/**
 * Add protocol if missing
 */
export function ensureProtocol(url: string, protocol = 'https:'): string {
  if (!url.match(/^https?:/i)) {
    return `${protocol}//${url}`;
  }
  return url;
}

/**
 * Validate URL
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Encode URI component safely
 */
export function encodeUriComponent(component: string): string {
  return encodeURIComponent(component);
}

/**
 * Decode URI component safely
 */
export function decodeUriComponent(component: string): string {
  try {
    return decodeURIComponent(component);
  } catch {
    return component;
  }
}
