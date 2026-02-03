export interface HtmlPreviewOptions {
  width?: number;
  height?: number;
  backgroundColor?: string;
  timeoutMs?: number;
}

const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 768;
const DEFAULT_TIMEOUT_MS = 600;

const waitForFrameReady = (iframe: HTMLIFrameElement, timeoutMs: number) =>
  new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    iframe.onload = finish;
    window.setTimeout(finish, timeoutMs);
  });

export async function renderHtmlToPngBase64(
  html: string,
  options: HtmlPreviewOptions = {}
): Promise<string | null> {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return null;
  }

  const width = options.width ?? DEFAULT_WIDTH;
  const height = options.height ?? DEFAULT_HEIGHT;
  const backgroundColor = options.backgroundColor ?? '#ffffff';
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  const iframe = document.createElement('iframe');
  iframe.style.position = 'fixed';
  iframe.style.left = '-99999px';
  iframe.style.top = '0';
  iframe.style.width = `${width}px`;
  iframe.style.height = `${height}px`;
  iframe.style.border = '0';
  iframe.style.opacity = '0';
  iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin');

  document.body.appendChild(iframe);

  try {
    iframe.srcdoc = html;
    await waitForFrameReady(iframe, timeoutMs);

    const doc = iframe.contentDocument;
    if (!doc || !doc.body) return null;

    doc.documentElement.style.width = `${width}px`;
    doc.documentElement.style.height = `${height}px`;
    doc.body.style.margin = '0';
    doc.body.style.width = `${width}px`;
    doc.body.style.height = `${height}px`;
    doc.body.style.overflow = 'hidden';

    const { toPng } = await import('html-to-image');
    const dataUrl = await toPng(doc.body, {
      width,
      height,
      backgroundColor,
      pixelRatio: 1,
      cacheBust: true,
    });

    const prefix = 'data:image/png;base64,';
    return dataUrl.startsWith(prefix) ? dataUrl.slice(prefix.length) : dataUrl;
  } catch {
    return null;
  } finally {
    iframe.remove();
  }
}
