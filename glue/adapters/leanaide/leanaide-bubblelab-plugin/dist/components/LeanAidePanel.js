import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { Brain, X, MathFunction } from 'lucide-react';
import { LeanAideVerification } from './LeanAideVerification';
export function LeanAidePanel({ isOpen, onClose, problemStatement, solutionCode, onVerificationResult, className = '', }) {
    const [mode, setMode] = useState('verification');
    if (!isOpen)
        return null;
    return (_jsxs("div", { className: `flex flex-col h-full bg-background border-l overflow-hidden ${className}`, children: [_jsxs("div", { className: "flex items-center justify-between p-3 border-b", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx(Brain, { className: "h-5 w-5 text-blue-500" }), _jsx("h3", { className: "font-semibold text-sm", children: "LeanAIDE Verification" })] }), _jsx("button", { onClick: onClose, className: "text-muted-foreground hover:text-foreground", "aria-label": "Close LeanAIDE panel", children: _jsx(X, { className: "h-4 w-4" }) })] }), _jsx("div", { className: "p-3 border-b", children: _jsx("div", { className: "flex gap-2 overflow-x-auto pb-1", children: [
                        { value: 'verification', label: 'Verify Solution' },
                        { value: 'theorem', label: 'Translate Theorem' },
                        { value: 'definition', label: 'Translate Definition' },
                        { value: 'query', label: 'Math Query' },
                        { value: 'elaboration', label: 'Elaborate Code' },
                    ].map((item) => (_jsxs("button", { onClick: () => setMode(item.value), className: `px-3 py-1 rounded-md text-sm font-medium transition-colors flex items-center gap-1 whitespace-nowrap ${mode === item.value
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground'}`, children: [_jsx(MathFunction, { className: "h-3 w-3" }), item.label] }, item.value))) }) }), _jsx("div", { className: "flex-1 overflow-auto p-3", children: _jsx(LeanAideVerification, { problemStatement: problemStatement, solutionCode: solutionCode, mode: mode, onVerificationResult: onVerificationResult }) }), _jsxs("div", { className: "p-3 border-t text-xs text-muted-foreground", children: [_jsxs("p", { className: "mb-1", children: [_jsx("strong", { children: "LeanAIDE" }), " provides formal verification using Lean 4 theorem prover."] }), _jsx("p", { children: "Results are mathematically proven and can be used for rigorous validation." })] })] }));
}
//# sourceMappingURL=LeanAidePanel.js.map