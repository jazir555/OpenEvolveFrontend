type MockHandler = ((event: any) => void) | null;

export class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;
  onopen: MockHandler = null;
  onmessage: MockHandler = null;
  onerror: MockHandler = null;
  onclose: MockHandler = null;
  private sentMessages: any[] = [];

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      if (url.includes('invalid')) {
        this.readyState = MockWebSocket.CLOSED;
        this.onerror?.(new Event('error'));
        this.onclose?.(new CloseEvent('close'));
      } else {
        this.readyState = MockWebSocket.OPEN;
        this.onopen?.(new Event('open'));
      }
    }, 0);
  }

  send(data: string) {
    try {
      const parsed = JSON.parse(data);
      if (parsed && parsed.type && Object.prototype.hasOwnProperty.call(parsed, 'data')) {
        this.sentMessages.push({ type: parsed.type, data: parsed.data });
        return;
      }
      this.sentMessages.push(parsed);
    } catch {
      this.sentMessages.push(data);
    }
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new CloseEvent('close'));
  }

  receiveMessage(message: any) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }

  getSentMessages() {
    return this.sentMessages;
  }
}
