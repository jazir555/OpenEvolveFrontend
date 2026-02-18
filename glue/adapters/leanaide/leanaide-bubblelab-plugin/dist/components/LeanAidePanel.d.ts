import { LeanAideTaskResponse } from '../lib/leanaideClient';
export interface LeanAidePanelProps {
    isOpen: boolean;
    onClose: () => void;
    problemStatement: string;
    solutionCode?: string;
    onVerificationResult?: (result: LeanAideTaskResponse) => void;
    className?: string;
}
export declare function LeanAidePanel({ isOpen, onClose, problemStatement, solutionCode, onVerificationResult, className, }: LeanAidePanelProps): any;
