export interface VerificationError {
    line: number;
    column: number;
    message: string;
    severity: 'error' | 'warning' | 'info';
}
export interface VerificationResult {
    status: 'pending' | 'running' | 'success' | 'failed';
    errors: VerificationError[];
    warnings: VerificationError[];
    output: string;
    duration?: number;
}
interface VerificationDisplayProps {
    result: VerificationResult;
    className?: string;
}
export declare function VerificationDisplay({ result, className }: VerificationDisplayProps): import("react/jsx-runtime").JSX.Element;
export {};
