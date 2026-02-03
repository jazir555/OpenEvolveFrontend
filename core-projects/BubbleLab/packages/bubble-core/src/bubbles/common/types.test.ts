/**
 * Comprehensive tests for common type utilities
 */

import { describe, it, expect } from 'vitest';
import {
  Result,
  ok,
  err,
  unwrap,
  CredentialType,
  type Credential,
  type RequestOptions,
  type PaginationOptions,
  type PaginatedResponse,
  type SortOptions,
  type FilterOptions,
  type QueryOptions,
  type DateRange,
  type TimeRange,
  type Coordinate,
  type BoundingBox,
  type Address,
  type Money,
  type PersonName,
  type ContactInfo,
  type UserProfile,
  type OperationMetadata,
  isResult,
  isOk,
  isErr,
  isPlainObject,
  isIsoTimestamp,
  isNonEmptyString,
  isPositiveNumber,
  isArray,
  createMoneySchema,
  createCoordinateSchema,
  createPersonNameSchema,
  deepClone,
  deepMerge
} from './types.js';
import { z } from 'zod';

describe('types utilities', () => {
  describe('Result type', () => {
    describe('ok', () => {
      it('should create successful result', () => {
        const result = ok('success');

        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data).toBe('success');
        }
      });

      it('should work with various data types', () => {
        expect(ok(42)).toEqual({ success: true, data: 42 });
        expect(ok({ key: 'value' })).toEqual({ success: true, data: { key: 'value' } });
        expect(ok(null)).toEqual({ success: true, data: null });
        expect(ok(undefined)).toEqual({ success: true, data: undefined });
      });
    });

    describe('err', () => {
      it('should create error result', () => {
        const error = new Error('Test error');
        const result = err(error);

        expect(result.success).toBe(false);
        if (!result.success) {
          expect(result.error).toBe(error);
        }
      });

      it('should work with various error types', () => {
        const error1 = new Error('Error 1');
        const error2 = 'String error';
        const error3 = 42;

        expect(err(error1)).toEqual({ success: false, error: error1 });
        expect(err(error2)).toEqual({ success: false, error: error2 });
        expect(err(error3)).toEqual({ success: false, error: error3 });
      });
    });

    describe('unwrap', () => {
      it('should return data from successful result', () => {
        const result = ok('success');
        expect(unwrap(result)).toBe('success');
      });

      it('should throw error from failed result', () => {
        const error = new Error('Test error');
        const result = err(error);

        expect(() => unwrap(result)).toThrow('Test error');
      });
    });
  });

  describe('CredentialType enum', () => {
    it('should have all expected values', () => {
      expect(CredentialType.API_KEY).toBe('api_key');
      expect(CredentialType.OAUTH_TOKEN).toBe('oauth_token');
      expect(CredentialType.BASIC_AUTH).toBe('basic_auth');
      expect(CredentialType.BEARER_TOKEN).toBe('bearer_token');
      expect(CredentialType.DATABASE_CRED).toBe('database_cred');
      expect(CredentialType.CUSTOM_AUTH_KEY).toBe('custom_auth_key');
      expect(CredentialType.SLACK_CRED).toBe('slack_cred');
      expect(CredentialType.STRIPE_CRED).toBe('stripe_cred');
      expect(CredentialType.AIRTABLE_CRED).toBe('airtable_cred');
      expect(CredentialType.GMAIL_CRED).toBe('gmail_cred');
      expect(CredentialType.GOOGLE_CALENDAR_CRED).toBe('google_calendar_cred');
      expect(CredentialType.SHEETS_CRED).toBe('sheets_cred');
      expect(CredentialType.NOTION_CRED).toBe('notion_cred');
      expect(CredentialType.POSTGRES_CRED).toBe('postgres_cred');
      expect(CredentialType.REDIS_CRED).toBe('redis_cred');
      expect(CredentialType.MONGODB_CRED).toBe('mongodb_cred');
      expect(CredentialType.S3_CRED).toBe('s3_cred');
      expect(CredentialType.AWS_CRED).toBe('aws_cred');
    });
  });

  describe('Credential interface', () => {
    it('should accept valid credential object', () => {
      const credential: Credential = {
        type: CredentialType.API_KEY,
        value: 'secret-key',
        expiresAt: new Date(),
        metadata: { service: 'test' }
      };

      expect(credential.type).toBe(CredentialType.API_KEY);
      expect(credential.value).toBe('secret-key');
      expect(credential.expiresAt).toBeInstanceOf(Date);
      expect(credential.metadata).toEqual({ service: 'test' });
    });

    it('should accept credential without optional fields', () => {
      const credential: Credential = {
        type: CredentialType.BEARER_TOKEN,
        value: 'token'
      };

      expect(credential.expiresAt).toBeUndefined();
      expect(credential.metadata).toBeUndefined();
    });
  });

  describe('RequestOptions interface', () => {
    it('should accept all options', () => {
      const options: RequestOptions = {
        timeout: 5000,
        headers: { 'Authorization': 'Bearer token' },
        retries: 3,
        signal: new AbortController().signal
      };

      expect(options.timeout).toBe(5000);
      expect(options.headers).toEqual({ 'Authorization': 'Bearer token' });
      expect(options.retries).toBe(3);
      expect(options.signal).toBeInstanceOf(AbortSignal);
    });
  });

  describe('PaginationOptions interface', () => {
    it('should accept various pagination styles', () => {
      const offsetPagination: PaginationOptions = {
        limit: 10,
        offset: 20
      };

      const cursorPagination: PaginationOptions = {
        limit: 10,
        cursor: 'abc123'
      };

      const pagePagination: PaginationOptions = {
        limit: 10,
        page: 2
      };

      expect(offsetPagination.offset).toBe(20);
      expect(cursorPagination.cursor).toBe('abc123');
      expect(pagePagination.page).toBe(2);
    });
  });

  describe('PaginatedResponse interface', () => {
    it('should create valid paginated response', () => {
      const response: PaginatedResponse<string> = {
        data: ['item1', 'item2'],
        pagination: {
          total: 100,
          limit: 10,
          offset: 0,
          hasMore: true,
          nextCursor: 'next123'
        }
      };

      expect(response.data).toHaveLength(2);
      expect(response.pagination.total).toBe(100);
      expect(response.pagination.hasMore).toBe(true);
      expect(response.pagination.nextCursor).toBe('next123');
    });
  });

  describe('QueryOptions interface', () => {
    it('should combine pagination, sort, and filter', () => {
      const options: QueryOptions = {
        limit: 10,
        offset: 0,
        sort: { field: 'createdAt', direction: 'desc' },
        filter: { status: 'active' }
      };

      expect(options.sort?.field).toBe('createdAt');
      expect(options.sort?.direction).toBe('desc');
      expect(options.filter).toEqual({ status: 'active' });
    });
  });

  describe('DateRange and TimeRange interfaces', () => {
    it('should create valid date range', () => {
      const range: DateRange = {
        start: new Date('2024-01-01'),
        end: new Date('2024-12-31')
      };

      expect(range.start).toBeInstanceOf(Date);
      expect(range.end).toBeInstanceOf(Date);
    });

    it('should create valid time range', () => {
      const range: TimeRange = {
        start: '2024-01-01T00:00:00Z',
        end: '2024-12-31T23:59:59Z'
      };

      expect(range.start).toMatch(/^\d{4}-\d{2}-\d{2}T/);
      expect(range.end).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });
  });

  describe('Coordinate and BoundingBox interfaces', () => {
    it('should create valid coordinate', () => {
      const coord: Coordinate = {
        latitude: 40.7128,
        longitude: -74.0060
      };

      expect(coord.latitude).toBe(40.7128);
      expect(coord.longitude).toBe(-74.0060);
    });

    it('should create valid bounding box', () => {
      const bbox: BoundingBox = {
        north: 40.9,
        south: 40.5,
        east: -73.8,
        west: -74.2
      };

      expect(bbox.north).toBeGreaterThan(bbox.south);
      expect(bbox.east).toBeGreaterThan(bbox.west);
    });
  });

  describe('Address interface', () => {
    it('should create complete address', () => {
      const address: Address = {
        street: '123 Main St',
        city: 'New York',
        state: 'NY',
        postalCode: '10001',
        country: 'USA',
        latitude: 40.7128,
        longitude: -74.0060
      };

      expect(address.street).toBe('123 Main St');
      expect(address.city).toBe('New York');
      expect(address.latitude).toBeDefined();
      expect(address.longitude).toBeDefined();
    });
  });

  describe('Money interface', () => {
    it('should represent amount in minor units', () => {
      const money: Money = {
        amount: 1099, // $10.99
        currency: 'USD'
      };

      expect(money.amount).toBe(1099);
      expect(money.currency).toBe('USD');
    });
  });

  describe('PersonName interface', () => {
    it('should create complete person name', () => {
      const name: PersonName = {
        prefix: 'Dr.',
        firstName: 'John',
        middleName: 'A.',
        lastName: 'Doe',
        suffix: 'Jr.',
        fullName: 'Dr. John A. Doe Jr.'
      };

      expect(name.firstName).toBe('John');
      expect(name.lastName).toBe('Doe');
      expect(name.fullName).toBeDefined();
    });
  });

  describe('UserProfile interface', () => {
    it('should create complete user profile', () => {
      const profile: UserProfile = {
        id: 'user-123',
        name: {
          firstName: 'John',
          lastName: 'Doe'
        },
        contact: {
          email: 'john@example.com',
          phone: '+1234567890'
        },
        timezone: 'America/New_York',
        locale: 'en-US',
        metadata: { role: 'admin' }
      };

      expect(profile.id).toBe('user-123');
      expect(profile.name.firstName).toBe('John');
      expect(profile.contact?.email).toBe('john@example.com');
      expect(profile.timezone).toBe('America/New_York');
    });
  });

  describe('type guards', () => {
    describe('isResult', () => {
      it('should identify Result objects', () => {
        expect(isResult(ok('success'))).toBe(true);
        expect(isResult(err(new Error()))).toBe(true);
        expect(isResult({ success: true, data: null })).toBe(true);
        expect(isResult({ success: false, error: null })).toBe(true);
      });

      it('should reject non-Result objects', () => {
        expect(isResult(null)).toBe(false);
        expect(isResult(undefined)).toBe(false);
        expect(isResult({})).toBe(false);
        expect(isResult({ success: 'yes' })).toBe(false);
        expect(isResult('string')).toBe(false);
      });
    });

    describe('isOk', () => {
      it('should identify successful results', () => {
        expect(isOk(ok('success'))).toBe(true);
        expect(isOk(ok(42))).toBe(true);
      });

      it('should reject failed results', () => {
        expect(isOk(err(new Error()))).toBe(false);
        expect(isOk({ success: false, error: null } as any)).toBe(false);
      });
    });

    describe('isErr', () => {
      it('should identify failed results', () => {
        expect(isErr(err(new Error()))).toBe(true);
        expect(isErr({ success: false, error: null } as any)).toBe(true);
      });

      it('should reject successful results', () => {
        expect(isErr(ok('success'))).toBe(false);
        expect(isErr({ success: true, data: null } as any)).toBe(false);
      });
    });

    describe('isPlainObject', () => {
      it('should identify plain objects', () => {
        expect(isPlainObject({})).toBe(true);
        expect(isPlainObject({ key: 'value' })).toBe(true);
        expect(isPlainObject({ a: 1, b: 2 })).toBe(true);
      });

      it('should reject non-plain-objects', () => {
        expect(isPlainObject(null)).toBe(false);
        expect(isPlainObject(undefined)).toBe(false);
        expect(isPlainObject([])).toBe(false);
        expect(isPlainObject(new Date())).toBe(false);
        expect(isPlainObject(new RegExp('test'))).toBe(false);
        expect(isPlainObject(new Error('test'))).toBe(false);
        expect(isPlainObject('string')).toBe(false);
        expect(isPlainObject(42)).toBe(false);
        expect(isPlainObject(true)).toBe(false);
      });
    });

    describe('isIsoTimestamp', () => {
      it('should identify valid ISO 8601 timestamps', () => {
        expect(isIsoTimestamp('2024-01-01T00:00:00Z')).toBe(true);
        expect(isIsoTimestamp('2024-01-01T00:00:00.123Z')).toBe(true);
        expect(isIsoTimestamp('2024-01-01T00:00:00+05:30')).toBe(true);
        expect(isIsoTimestamp('2024-12-31T23:59:59-08:00')).toBe(true);
      });

      it('should reject invalid timestamps', () => {
        expect(isIsoTimestamp('2024-01-01')).toBe(false);
        expect(isIsoTimestamp('01-01-2024')).toBe(false);
        expect(isIsoTimestamp('invalid')).toBe(false);
        expect(isIsoTimestamp('')).toBe(false);
        expect(isIsoTimestamp(null)).toBe(false);
        expect(isIsoTimestamp(undefined)).toBe(false);
        expect(isIsoTimestamp(123)).toBe(false);
      });
    });

    describe('isNonEmptyString', () => {
      it('should identify non-empty strings', () => {
        expect(isNonEmptyString('hello')).toBe(true);
        expect(isNonEmptyString('  hello  ')).toBe(true);
        expect(isNonEmptyString('a')).toBe(true);
      });

      it('should reject empty or whitespace strings', () => {
        expect(isNonEmptyString('')).toBe(false);
        expect(isNonEmptyString('   ')).toBe(false);
        expect(isNonEmptyString('\t\n')).toBe(false);
      });

      it('should reject non-strings', () => {
        expect(isNonEmptyString(null)).toBe(false);
        expect(isNonEmptyString(undefined)).toBe(false);
        expect(isNonEmptyString(123)).toBe(false);
        expect(isNonEmptyString({})).toBe(false);
      });
    });

    describe('isPositiveNumber', () => {
      it('should identify positive numbers', () => {
        expect(isPositiveNumber(1)).toBe(true);
        expect(isPositiveNumber(0.5)).toBe(true);
        expect(isPositiveNumber(1000)).toBe(true);
      });

      it('should reject non-positive numbers', () => {
        expect(isPositiveNumber(0)).toBe(false);
        expect(isPositiveNumber(-1)).toBe(false);
        expect(isPositiveNumber(-0.5)).toBe(false);
      });

      it('should reject non-numbers', () => {
        expect(isPositiveNumber(NaN)).toBe(false);
        expect(isPositiveNumber(null)).toBe(false);
        expect(isPositiveNumber(undefined)).toBe(false);
        expect(isPositiveNumber('123')).toBe(false);
      });
    });

    describe('isArray', () => {
      it('should identify arrays', () => {
        expect(isArray([])).toBe(true);
        expect(isArray([1, 2, 3])).toBe(true);
        expect(isArray(['a', 'b'])).toBe(true);
        expect(isArray([null, undefined])).toBe(true);
      });

      it('should reject non-arrays', () => {
        expect(isArray(null)).toBe(false);
        expect(isArray(undefined)).toBe(false);
        expect(isArray({})).toBe(false);
        expect(isArray('string')).toBe(false);
        expect(isArray(123)).toBe(false);
      });
    });
  });

  describe('Zod schema creators', () => {
    describe('createMoneySchema', () => {
      it('should validate Money objects', () => {
        const schema = createMoneySchema();

        const validMoney = { amount: 1099, currency: 'USD' };
        expect(() => schema.parse(validMoney)).not.toThrow();

        const invalidAmount = { amount: -100, currency: 'USD' };
        expect(() => schema.parse(invalidAmount)).toThrow();

        const invalidCurrency = { amount: 100, currency: 'XXXX' };
        expect(() => schema.parse(invalidCurrency)).toThrow();

        const invalidCurrencyLength = { amount: 100, currency: 'US' };
        expect(() => schema.parse(invalidCurrencyLength)).toThrow();
      });
    });

    describe('createCoordinateSchema', () => {
      it('should validate Coordinate objects', () => {
        const schema = createCoordinateSchema();

        const validCoord = { latitude: 40.7128, longitude: -74.0060 };
        expect(() => schema.parse(validCoord)).not.toThrow();

        const invalidLat = { latitude: 91, longitude: 0 };
        expect(() => schema.parse(invalidLat)).toThrow();

        const invalidLng = { latitude: 0, longitude: 181 };
        expect(() => schema.parse(invalidLng)).toThrow();
      });
    });

    describe('createPersonNameSchema', () => {
      it('should validate PersonName objects', () => {
        const schema = createPersonNameSchema();

        const validName = {
          prefix: 'Dr.',
          firstName: 'John',
          lastName: 'Doe'
        };
        expect(() => schema.parse(validName)).not.toThrow();

        const minimalName = {};
        expect(() => schema.parse(minimalName)).not.toThrow();
      });
    });
  });

  describe('deepClone', () => {
    it('should clone primitives', () => {
      expect(deepClone(null)).toBe(null);
      expect(deepClone(undefined)).toBe(undefined);
      expect(deepClone('string')).toBe('string');
      expect(deepClone(42)).toBe(42);
      expect(deepClone(true)).toBe(true);
    });

    it('should clone Date objects', () => {
      const date = new Date('2024-01-01');
      const cloned = deepClone(date);

      expect(cloned).toEqual(date);
      expect(cloned).not.toBe(date);
      expect(cloned.getTime()).toBe(date.getTime());
    });

    it('should clone arrays', () => {
      const arr = [1, 2, { a: 3 }];
      const cloned = deepClone(arr);

      expect(cloned).toEqual(arr);
      expect(cloned).not.toBe(arr);
      expect(cloned[2]).not.toBe(arr[2]);
    });

    it('should clone objects', () => {
      const obj = { a: 1, b: { c: 2 } };
      const cloned = deepClone(obj);

      expect(cloned).toEqual(obj);
      expect(cloned).not.toBe(obj);
      expect(cloned.b).not.toBe(obj.b);
    });
  });

  describe('deepMerge', () => {
    it('should merge shallow objects', () => {
      const target = { a: 1, b: 2 };
      const source = { b: 3, c: 4 };
      const result = deepMerge(target, source);

      expect(result).toEqual({ a: 1, b: 3, c: 4 });
    });

    it('should merge nested objects', () => {
      const target = { a: { x: 1, y: 2 } };
      const source = { a: { y: 3, z: 4 }, b: 5 };
      const result = deepMerge(target, source);

      expect(result).toEqual({ a: { x: 1, y: 3, z: 4 }, b: 5 });
    });

    it('should not mutate target', () => {
      const target = { a: 1, b: 2 };
      const source = { b: 3 };
      const result = deepMerge(target, source);

      expect(target).toEqual({ a: 1, b: 2 });
      expect(result).toEqual({ a: 1, b: 3 });
    });
  });
});
