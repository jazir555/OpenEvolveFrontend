interface ProgressBarProps {
    progress: number;
    label?: string;
    showPercentage?: boolean;
    className?: string;
    color?: 'blue' | 'green' | 'yellow' | 'red';
}
export declare function ProgressBar({ progress, label, showPercentage, className, color, }: ProgressBarProps): import("react/jsx-runtime").JSX.Element;
export {};
