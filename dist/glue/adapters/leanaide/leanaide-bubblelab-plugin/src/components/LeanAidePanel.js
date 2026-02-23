"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LeanAidePanel = LeanAidePanel;
const react_1 = require("react");
const lucide_react_1 = require("lucide-react");
const LeanAideVerification_1 = require("./LeanAideVerification");
function LeanAidePanel({ isOpen, onClose, problemStatement, solutionCode, onVerificationResult, className = '', }) {
    const [mode, setMode] = (0, react_1.useState)('verification');
    if (!isOpen)
        return null;
    return (<div className={`flex flex-col h-full bg-background border-l overflow-hidden ${className}`}>
      <div className="flex items-center justify-between p-3 border-b">
        <div className="flex items-center gap-2">
          <lucide_react_1.Brain className="h-5 w-5 text-blue-500"/>
          <h3 className="font-semibold text-sm">LeanAIDE Verification</h3>
        </div>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground" aria-label="Close LeanAIDE panel">
          <lucide_react_1.X className="h-4 w-4"/>
        </button>
      </div>

      <div className="p-3 border-b">
        <div className="flex gap-2 overflow-x-auto pb-1">
          {[
            { value: 'verification', label: 'Verify Solution' },
            { value: 'theorem', label: 'Translate Theorem' },
            { value: 'definition', label: 'Translate Definition' },
            { value: 'query', label: 'Math Query' },
            { value: 'elaboration', label: 'Elaborate Code' },
        ].map((item) => (<button key={item.value} onClick={() => setMode(item.value)} className={`px-3 py-1 rounded-md text-sm font-medium transition-colors flex items-center gap-1 whitespace-nowrap ${mode === item.value
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground'}`}>
              <lucide_react_1.MathFunction className="h-3 w-3"/>
              {item.label}
            </button>))}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-3">
        <LeanAideVerification_1.LeanAideVerification problemStatement={problemStatement} solutionCode={solutionCode} mode={mode} onVerificationResult={onVerificationResult}/>
      </div>

      <div className="p-3 border-t text-xs text-muted-foreground">
        <p className="mb-1">
          <strong>LeanAIDE</strong> provides formal verification using Lean 4 theorem prover.
        </p>
        <p>
          Results are mathematically proven and can be used for rigorous validation.
        </p>
      </div>
    </div>);
}
//# sourceMappingURL=LeanAidePanel.js.map