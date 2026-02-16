import fetch, { Headers, Request, Response } from 'node-fetch';

if (typeof globalThis.fetch !== 'function') {
  globalThis.fetch = fetch as unknown as typeof globalThis.fetch;
}

if (typeof globalThis.Headers === 'undefined') {
  globalThis.Headers = Headers as unknown as typeof globalThis.Headers;
}

if (typeof globalThis.Request === 'undefined') {
  globalThis.Request = Request as unknown as typeof globalThis.Request;
}

if (typeof globalThis.Response === 'undefined') {
  globalThis.Response = Response as unknown as typeof globalThis.Response;
}
