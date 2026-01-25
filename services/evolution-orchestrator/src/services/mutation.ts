const MUTATION_URL =
  process.env.MUTATION_ENGINE_URL || 'http://localhost:8002';

export type MutationResult = {
  html: string;
  css?: string;
  changes: string[];
};

export async function mutateDesign(html: string, css?: string) {
  const response = await fetch(`${MUTATION_URL}/mutate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ design: { html, css } }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return (await response.json()) as MutationResult;
}
