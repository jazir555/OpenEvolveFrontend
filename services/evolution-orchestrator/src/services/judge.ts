const JUDGE_URL =
  process.env.JUDGE_URL || 'http://localhost:3001/evolution-judge/judge';
const JUDGE_API_TOKEN = process.env.JUDGE_API_TOKEN;

export type JudgeAggregate = {
  score: number;
  weights: Record<string, number>;
  agents: Array<{ score: number; reasoning: string }>;
};

export async function judgeDesign(imageBase64: string, criteria?: string) {
  const response = await fetch(JUDGE_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(JUDGE_API_TOKEN ? { Authorization: `Bearer ${JUDGE_API_TOKEN}` } : {}),
    },
    body: JSON.stringify({
      input: {
        image: { type: 'base64', data: imageBase64, mimeType: 'image/png' },
        criteria,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return (await response.json()) as JudgeAggregate;
}
