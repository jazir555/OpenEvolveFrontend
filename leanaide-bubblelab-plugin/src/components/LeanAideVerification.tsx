import { useState, useEffect } from 'react';
import { Check, X, AlertCircle, Loader2, Brain, MathFunction } from 'lucide-react';
import { toast } from 'react-toastify';
import {
  translateTheorem,
  translateDefinition,
  verifySolution,
  elaborateCode,
  mathQuery,
  isLeanAideAvailable,
} from '../services/leanaideService';
import { LeanAideTaskResponse } from '../lib/leanaideClient';

export interface LeanAideVerificationProps {
  problemStatement: string;
  solutionCode?: string;
  onVerificationResult?: (result: LeanAideTaskResponse) => void;
  mode?: 'theorem' | 'definition' | 'verification' | 'query' | 'elaboration';
  className?: string;
}

export function LeanAideVerification({
  problemStatement,
  solutionCode,
  onVerificationResult,
  mode = 'verification',
  className = '',
}: LeanAideVerificationProps) {
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<LeanAideTaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    // Reset state when problem changes
    setVerificationResult(null);
    setError(null);
    setShowDetails(false);
  }, [problemStatement, solutionCode]);

  const handleVerification = async () => {
    if (!isLeanAideAvailable()) {
      toast.error('LeanAIDE service is not available. Please configure LeanAIDE integration.');
      return;
    }

    if (!problemStatement.trim()) {
      toast.warning('Please provide a problem statement for verification.');
      return;
    }

    setIsVerifying(true);
    setError(null);

    try {
      let result: LeanAideTaskResponse;

      switch (mode) {
        case 'theorem':
          result = await translateTheorem(problemStatement);
          break;
        case 'definition':
          result = await translateDefinition(problemStatement);
          break;
        case 'verification':
          if (!solutionCode) {
            throw new Error('Solution code is required for verification mode');
          }
          result = await verifySolution(problemStatement, solutionCode);
          break;
        case 'query':
          result = await mathQuery(problemStatement);
          break;
        case 'elaboration':
          if (!solutionCode) {
            throw new Error('Lean code is required for elaboration mode');
          }
          result = await elaborateCode(solutionCode);
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
        toast.error(`LeanAIDE verification failed: ${result.error}`);
      } else {
        toast.success('LeanAIDE verification completed successfully');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      toast.error(`LeanAIDE error: ${errorMessage}`);
      console.error('LeanAIDE verification error:', error);
    } finally {
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
    if (isVerifying) return <Loader2 className="h-4 w-4 animate-spin" />;
    if (error) return <AlertCircle className="h-4 w-4 text-red-500" />;
    if (verificationResult?.success) return <Check className="h-4 w-4 text-green-500" />;
    if (verificationResult) return <X className="h-4 w-4 text-yellow-500" />;
    return <MathFunction className="h-4 w-4" />;
  };

  const getResultStatus = () => {
    if (isVerifying) return 'Verifying...';
    if (error) return 'Error';
    if (verificationResult?.success) return 'Verified';
    if (verificationResult) return 'Failed';
    return 'Ready';
  };

  return (
    <div className={`border rounded-lg p-4 bg-background/50 backdrop-blur-sm ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-blue-500" />
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

      <button
        onClick={handleVerification}
        disabled={isVerifying}
        className={`w-full flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
          isVerifying
            ? 'bg-muted text-muted-foreground cursor-not-allowed'
            : 'bg-primary text-primary-foreground hover:bg-primary/90'
        }`}
      >
        {isVerifying ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Verifying...
          </>
        ) : (
          <>
            <MathFunction className="h-4 w-4" />
            {getModeLabel()}
          </>
        )}
      </button>

      {verificationResult && (
        <div className="mt-3 border-t pt-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-sm">Results</h4>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              {showDetails ? 'Hide' : 'Show'} Details
            </button>
          </div>

          {verificationResult.data && (
            <div className="bg-muted/20 rounded-md p-2 text-xs overflow-auto max-h-48">
              <pre className="whitespace-pre-wrap break-words">
                {JSON.stringify(verificationResult.data, null, 2)}
              </pre>
            </div>
          )}

          {showDetails && verificationResult.logs && (
            <div className="mt-2 bg-muted/20 rounded-md p-2 text-xs overflow-auto max-h-32">
              <pre className="whitespace-pre-wrap break-words text-muted-foreground">
                {verificationResult.logs}
              </pre>
            </div>
          )}

          {error && (
            <div className="mt-2 bg-destructive/10 border border-destructive/20 rounded-md p-2 text-xs text-destructive">
              <p className="font-medium">Error:</p>
              <p>{error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}