export interface ApiError {
    status: number;
    data: unknown;
}
export declare class ApiHttpError extends Error implements ApiError {
    status: number;
    data: unknown;
    constructor(status: number, data: unknown);
}
