export interface ApiError {
  status: number;
  data: unknown;
}

export class ApiHttpError extends Error implements ApiError {
  status: number;
  data: unknown;

  constructor(status: number, data: unknown) {
    super(`HTTP ${status}: ${JSON.stringify(data)}`);
    this.name = 'ApiHttpError';
    this.status = status;
    this.data = data;
  }
}