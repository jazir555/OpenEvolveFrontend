export class ValidationError extends Error {
  constructor(message: string, public field?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ValidationError';
    if (details) {
      Object.assign(this, details);
    }
  }
}

export class NetworkError extends Error {
  constructor(message: string = 'Network error', details?: Record<string, unknown>) {
    super(message);
    this.name = 'NetworkError';
    if (details) {
      Object.assign(this, details);
    }
  }
}
