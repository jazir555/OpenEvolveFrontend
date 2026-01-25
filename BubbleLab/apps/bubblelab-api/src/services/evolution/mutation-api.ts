import { env } from '../../config/env.js';

const BASE_URL = env.MUTATION_ENGINE_URL || 'http://localhost:8002';

const request = async <T>(path: string, payload: unknown): Promise<T> => {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Mutation engine error: ${message}`);
  }

  return (await response.json()) as T;
};

export const mutationApi = {
  mutate<T>(payload: unknown) {
    return request<T>('/mutate', payload);
  },
  mutateBatch<T>(payload: unknown) {
    return request<T>('/mutate/batch', payload);
  },
};
