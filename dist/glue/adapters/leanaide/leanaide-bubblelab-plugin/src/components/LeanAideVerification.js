"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LeanAideVerification = LeanAideVerification;
const react_1 = require("react");
const lucide_react_1 = require("lucide-react");
const react_toastify_1 = require("react-toastify");
const leanaideService_1 = require("../services/leanaideService");
function LeanAideVerification({ problemStatement, solutionCode, onVerificationResult, mode = 'verification', className = '', }) {
    const [isVerifying, setIsVerifying] = (0, react_1.useState)(false);
    const [verificationResult, setVerificationResult] = (0, react_1.useState)(null);
    const [error, setError] = (0, react_1.useState)(null);
    const [showDetails, setShowDetails] = (0, react_1.useState)(false);
    (0, react_1.useEffect)(() => {
        // Reset state when problem changes
        setVerificationResult(null);
        setError(null);
        setShowDetails(false);
    }, [problemStatement, solutionCode]);
    const handleVerification = async () => {
        if (!(0, leanaideService_1.isLeanAideAvailable)()) {
            react_toastify_1.toast.error('LeanAIDE service is not available. Please configure LeanAIDE integration.');
            return;
        }
        if (!problemStatement.trim()) {
            react_toastify_1.toast.warning('Please provide a problem statement for verification.');
            return;
        }
        setIsVerifying(true);
        setError(null);
        try {
            let result;
            switch (mode) {
                case 'theorem':
                    result = await (0, leanaideService_1.translateTheorem)(problemStatement);
                    break;
                case 'definition':
                    result = await (0, leanaideService_1.translateDefinition)(problemStatement);
                    break;
                case 'verification':
                    if (!solutionCode) {
                        throw new Error('Solution code is required for verification mode');
                    }
                    result = await (0, leanaideService_1.verifySolution)(problemStatement, solutionCode);
                    break;
                case 'query':
                    result = await (0, leanaideService_1.mathQuery)(problemStatement);
                    break;
                case 'elaboration':
                    if (!solutionCode) {
                        throw new Error('Lean code is required for elaboration mode');
                    }
                    result = await (0, leanaideService_1.elaborateCode)(solutionCode);
                    break;
                default:
                    throw new Error(`Unknown mode: ${mode}`);
            }
            setVerificationResult(result);
            if (onVerificationResult) {
                onVerificationResult(result);
            }
            if (!result.success && result.error) {
                setError(result.error);
                react_toastify_1.toast.error(`LeanAIDE verification failed: ${result.error}`);
            }
            else {
                react_toastify_1.toast.success('LeanAIDE verification completed successfully');
            }
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            setError(errorMessage);
            react_toastify_1.toast.error(`LeanAIDE error: ${errorMessage}`);
            console.error('LeanAIDE verification error:', error);
        }
        finally {
            setIsVerifying(false);
        }
    };
    const getModeLabel = () => {
        switch (mode) {
            case 'theorem':
                return 'Translate Theorem';
            case 'definition':
                return 'Translate Definition';
            case 'verification':
                return 'Verify Solution';
            case 'query':
                return 'Math Query';
            case 'elaboration':
                return 'Elaborate Code';
            default:
                return 'LeanAIDE Operation';
        }
    };
    const getResultIcon = () => {
        if (isVerifying)
            return <lucide_react_1.Loader2 className="h-4 w-4 animate-spin"/>;
        if (error)
            return <lucide_react_1.AlertCircle className="h-4 w-4 text-red-500"/>;
        if (verificationResult?.success)
            return <lucide_react_1.Check className="h-4 w-4 text-green-500"/>;
        if (verificationResult)
            return <lucide_react_1.X className="h-4 w-4 text-yellow-500"/>;
        return <lucide_react_1.MathFunction className="h-4 w-4"/>;
    };
    const getResultStatus = () => {
        if (isVerifying)
            return 'Verifying...';
        if (error)
            return 'Error';
        if (verificationResult?.success)
            return 'Verified';
        if (verificationResult)
            return 'Failed';
        return 'Ready';
    };
    return (<div className={`border rounded-lg p-4 bg-background/50 backdrop-blur-sm ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <lucide_react_1.Brain className="h-5 w-5 text-blue-500"/>
          <h3 className="font-semibold text-sm">LeanAIDE Verification</h3>
        </div>
        <div className="flex items-center gap-2">
          {getResultIcon()}
          <span className="text-xs text-muted-foreground">
            {getResultStatus()}
          </span>
        </div>
      </div>

      <div className="mb-3">
        <p className="text-xs text-muted-foreground mb-2">
          {getModeLabel()} using Lean 4 theorem prover
        </p>
        <div className="bg-muted/30 rounded-md p-2 text-xs overflow-auto max-h-32">
          <code className="whitespace-pre-wrap">{problemStatement}</code>
        </div>
      </div>

      <button onClick={handleVerification} disabled={isVerifying} className={`w-full flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${isVerifying
            ? 'bg-muted text-muted-foreground cursor-not-allowed'
            : 'bg-primary text-primary-foreground hover:bg-primary/90'}`}>
        {isVerifying ? (<>
            <lucide_react_1.Loader2 className="h-4 w-4 animate-spin"/>
            Verifying...
          </>) : (<>
            <lucide_react_1.MathFunction className="h-4 w-4"/>
            {getModeLabel()}
          </>)}
      </button>

      {verificationResult && (<div className="mt-3 border-t pt-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-sm">Results</h4>
            <button onClick={() => setShowDetails(!showDetails)} className="text-xs text-muted-foreground hover:text-foreground">
              {showDetails ? 'Hide' : 'Show'} Details
            </button>
          </div>

          {verificationResult.data && (<div className="bg-muted/20 rounded-md p-2 text-xs overflow-auto max-h-48">
              <pre className="whitespace-pre-wrap break-words">
                {JSON.stringify(verificationResult.data, null, 2)}
              </pre>
            </div>)}

          {showDetails && verificationResult.logs && (<div className="mt-2 bg-muted/20 rounded-md p-2 text-xs overflow-auto max-h-32">
              <pre className="whitespace-pre-wrap break-words text-muted-foreground">
                {verificationResult.logs}
              </pre>
            </div>)}

          {error && (<div className="mt-2 bg-destructive/10 border border-destructive/20 rounded-md p-2 text-xs text-destructive">
              <p className="font-medium">Error:</p>
              <p>{error}</p>
            </div>)}
        </div>)}
    </div>);
}
//# sourceMappingURL=LeanAideVerification.js.map