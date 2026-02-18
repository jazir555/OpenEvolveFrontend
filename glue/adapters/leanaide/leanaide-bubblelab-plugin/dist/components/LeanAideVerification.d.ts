import { LeanAideTaskResponse } from '../lib/leanaideClient';
export interface LeanAideVerificationProps {
    problemStatement: string;
    solutionCode?: string;
    onVerificationResult?: (result: LeanAideTaskResponse) => void;
    mode?: 'theorem' | 'definition' | 'verification' | 'query' | 'elaboration';
    className?: string;
}
export declare function LeanAideVerification({ problemStatement, solutionCode, onVerificationResult, mode, className, }: LeanAideVerificationProps): any;
