/**
 * Utility Modules Index
 * Central export point for all utility functions
 */

// Array utilities
export * from './array';

// Math utilities
export * from './math';

// DOM utilities
export * from './dom';

// Number utilities
export * from './numbers';

// Object utilities
export * from './object';

// URL utilities
export * from './url';

// Crypto utilities
export * from './crypto';

// Date utilities
export * from './date';

// String utilities (exclude conflicting exports)
export {
  capitalize,
  camelCase,
  kebabCase,
  snakeCase,
  generateStringId,
  pluralizeCount,
  truncateString
} from './string';

// Format utilities (exclude formatBytes which is defined in numbers.ts)
export {
  formatNumber,
  formatCurrency,
  formatPercentage,
  truncate,
  formatList,
  highlightTerms,
  pluralize,
  titleCase,
  formatPhoneNumber,
  formatSSN,
  stripHtml,
  escapeHtml
} from './format';

// Validation utilities
export * from './validation';

// Storage utilities
export * from './storage';

// Test utilities
export * from './test';

// Debounce utilities
export * from './debounce';

// Clipboard utilities
export * from './clipboard';
