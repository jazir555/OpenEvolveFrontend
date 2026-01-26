/**
 * Clipboard Utilities
 * Copy to clipboard and paste from clipboard
 */

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }

    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
      document.execCommand('copy');
      textArea.remove();
      return true;
    } catch (error) {
      console.error('Fallback copy failed', error);
      textArea.remove();
      return false;
    }
  } catch (error) {
    console.error('Copy failed', error);
    return false;
  }
}

/**
 * Read text from clipboard
 */
export async function readFromClipboard(): Promise<string> {
  try {
    if (navigator.clipboard && navigator.clipboard.readText) {
      return await navigator.clipboard.readText();
    }

    throw new Error('Clipboard API not available');
  } catch (error) {
    console.error('Read from clipboard failed', error);
    return '';
  }
}

/**
 * Copy rich text/HTML to clipboard
 */
export async function copyHtmlToClipboard(html: string): Promise<boolean> {
  try {
    if (navigator.clipboard && navigator.clipboard.write) {
      const blob = new Blob([html], { type: 'text/html' });
      const textBlob = new Blob([html.replace(/<[^>]*>/g, '')], { type: 'text/plain' });

      await navigator.clipboard.write([
        new ClipboardItem({
          'text/html': blob,
          'text/plain': textBlob,
        }),
      ]);

      return true;
    }

    return false;
  } catch (error) {
    console.error('Copy HTML failed', error);
    return false;
  }
}
