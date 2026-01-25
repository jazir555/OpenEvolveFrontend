export interface LogEntry {
    timestamp: string;
    level: 'info' | 'warn' | 'error' | 'debug';
    message: string;
    source?: string;
}
interface LiveLogViewerProps {
    logs: LogEntry[];
    className?: string;
    maxHeight?: string;
}
export declare function LiveLogViewer({ logs, className, maxHeight }: LiveLogViewerProps): import("react/jsx-runtime").JSX.Element;
export {};
