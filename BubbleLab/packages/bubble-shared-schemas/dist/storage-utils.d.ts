export type StorableValue = {
    truncated: boolean;
    preview: string | unknown;
    sizeBytes: number;
};
/**
 * Prepare an object for storage with a size cap. If the JSON stringified
 * representation exceeds maxBytes, return a preview marker with metadata.
 * Also logs a warning when truncation happens.
 *
 * Returns a consistent object structure when truncated to ensure compatibility
 * with database schemas expecting JSON objects (jsonb/text with mode: 'json').
 */
export declare function prepareForStorage(value: unknown, options?: {
    maxBytes?: number;
    previewBytes?: number;
}): StorableValue;
export declare function cleanUpObjectForDisplayAndStorage(obj: unknown, maxBytes?: number): unknown;
//# sourceMappingURL=storage-utils.d.ts.map