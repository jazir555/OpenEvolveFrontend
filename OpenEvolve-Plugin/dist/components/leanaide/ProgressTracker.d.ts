export interface ProgressStep {
    id: string;
    label: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    timestamp?: string;
    duration?: number;
}
interface ProgressTrackerProps {
    steps: ProgressStep[];
    className?: string;
}
export declare function ProgressTracker({ steps, className }: ProgressTrackerProps): import("react/jsx-runtime").JSX.Element;
export {};
