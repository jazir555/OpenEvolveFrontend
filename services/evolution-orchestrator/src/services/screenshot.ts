const SCREENSHOT_URL =
  process.env.SCREENSHOT_RENDERER_URL || 'http://localhost:8001';

export type RenderResponse = {
  image_base64: string;
  mime_type: string;
  width: number;
  height: number;
  duration_ms: number;
};

export async function renderHtml(html: string): Promise<RenderResponse> {
  const response = await fetch(`${SCREENSHOT_URL}/render`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ html }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return (await response.json()) as RenderResponse;
}
