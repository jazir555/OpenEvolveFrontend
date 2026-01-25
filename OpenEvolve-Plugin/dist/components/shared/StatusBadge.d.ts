type Status = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';
interface StatusBadgeProps {
    status?: Status;
    isConnected?: boolean;
    className?: string;
}
export declare function StatusBadge({ status, isConnected, className }: StatusBadgeProps): import("react/jsx-runtime").JSX.Element;
export {};
