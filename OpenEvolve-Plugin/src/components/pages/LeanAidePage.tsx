// @ts-nocheck
import { useMemo, useState } from 'react';
import { ProofEditor } from '@/components/leanaide/ProofEditor';
import { ModelSelector } from '@/components/leanaide/ModelSelector';
import { VerificationDisplay } from '@/components/leanaide/VerificationDisplay';
import { ProgressTracker, ProgressStep } from '@/components/leanaide/ProgressTracker';
import { BubbleButton, BubbleCard } from '@/components/bubblelab';
import { useLeanAide } from '@/services/hooks/useApi';
import { PageErrorBoundary } from '@/components/shared/PageErrorBoundary';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

const defaultSteps: ProgressStep[] = [
  { id: '1', label: 'Parse Lean code', status: 'pending' },
  { id: '2', label: 'Type check', status: 'pending' },
  { id: '3', label: 'Generate proof', status: 'pending' },
  { id: '4', label: 'Verify proof', status: 'pending' },
];

const fallbackModels = [
  {
    id: 'lean-4-prover',
    name: 'Lean 4 Prover',
    provider: 'OpenAI',
    description: 'GPT-4 based Lean theorem prover',
    capabilities: ['Type Checking', 'Proof Generation', 'Tactic Suggestion'],
  },
  {
    id: 'claude-lean',
    name: 'Claude Lean',
    provider: 'Anthropic',
    description: 'Claude-based Lean assistant',
    capabilities: ['Code Completion', 'Proof Strategy', 'Error Analysis'],
  },
];

function LeanAidePageBase() {
  const [leanCode, setLeanCode] = useState('');
  const [selectedModel, setSelectedModel] = useState('lean-4-prover');
  const [verificationResult, setVerificationResult] = useState<any>(null);
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>(defaultSteps);
  const [pageError, setPageError] = useState<string | null>(null);

  const { models, verifyProof, isVerifying } = useLeanAide();

  const availableModels = useMemo(() => {
    if (!models || models.length === 0) {
      return fallbackModels;
    }

    return models.flatMap((provider) =>
      provider.models.map((model) => ({
        id: `${provider.provider}-${model}`,
        name: model,
        provider: provider.provider,
        description: `${provider.provider} ${model}`,
        capabilities: ['Proof Generation', 'Verification'],
      }))
    );
  }, [models]);

  const handleVerify = async () => {
    setProgressSteps(
      defaultSteps.map((step) => ({ ...step, status: 'in_progress' as const }))
    );
    setVerificationResult({
      status: 'running',
      errors: [],
      warnings: [],
      output: '',
    });

    try {
      setPageError(null);
      const result = await verifyProof(leanCode);
      setVerificationResult({
        status: result?.success ? 'success' : 'failed',
        errors: result?.errors || [],
        warnings: result?.warnings || [],
        output: result?.output || '',
        duration: result?.elapsed_time || 0,
      });
      setProgressSteps((prev) =>
        prev.map((step) => ({ ...step, status: 'completed' as const }))
      );
    } catch (error: any) {
      setPageError(error?.message || 'Verification failed.');
      setVerificationResult({
        status: 'failed',
        errors: [error?.message || 'Verification failed.'],
        warnings: [],
        output: '',
        duration: 0,
      });
      setProgressSteps((prev) =>
        prev.map((step, index) => ({
          ...step,
          status: index === prev.length - 1 ? 'failed' : 'completed',
        }))
      );
    }
  };

  return (
    <PageErrorBoundary label="LeanAide">
      <div className="leanaide-page flex h-screen bg-slate-50">
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">LeanAide - Proof Assistant</h1>
              <p className="text-sm text-slate-600">Generate and verify Lean 4 proofs</p>
            </div>
            <div className="flex gap-2">
              <BubbleButton onClick={() => setLeanCode('')} variant="secondary">
                Clear
              </BubbleButton>
              <BubbleButton
                onClick={handleVerify}
                disabled={isVerifying || !leanCode.trim()}
                variant="primary"
              >
                {isVerifying ? 'Verifying...' : 'Verify Proof'}
              </BubbleButton>
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-6">
          {pageError && (
            <div className="mb-6 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {pageError}
            </div>
          )}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
            <div className="flex flex-col space-y-4">
              <BubbleCard title="Model Selection">
                <ModelSelector
                  models={availableModels}
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                />
              </BubbleCard>

              <BubbleCard title="Proof Editor" className="flex-1 overflow-hidden">
                <ProofEditor
                  value={leanCode}
                  onChange={setLeanCode}
                  language="lean"
                  className="h-full"
                />
              </BubbleCard>
            </div>

            <div className="flex flex-col space-y-4">
              <BubbleCard title="Verification Progress">
                <ProgressTracker steps={progressSteps} />
              </BubbleCard>

              {verificationResult && <VerificationDisplay result={verificationResult} />}

              <BubbleCard title="Example Proofs" className="flex-1 overflow-auto">
                <div className="space-y-2">
                  {[
                    {
                      name: 'Basic Proposition',
                      code: 'theorem test (p : Prop) : p -> p := by\n  intro h\n  exact h',
                    },
                    {
                      name: 'Addition Commutative',
                      code: 'example : forall a b : Nat, a + b = b + a := by\n  intro a b\n  exact Nat.add_comm a b',
                    },
                  ].map((example, i) => (
                    <BubbleButton
                      key={i}
                      onClick={() => setLeanCode(example.code)}
                      variant="ghost"
                      className="w-full text-left justify-start rounded-lg bg-slate-50 px-3 py-2 hover:bg-slate-100"
                    >
                      <div className="font-medium text-slate-900">{example.name}</div>
                      <div className="mt-1 truncate font-mono text-xs text-slate-600">
                        {example.code}
                      </div>
                    </BubbleButton>
                  ))}
                </div>
              </BubbleCard>
            </div>
          </div>
        </div>
      </main>
    </div>
    </PageErrorBoundary>
  );
}

export const LeanAidePage = withComponentBoundary(LeanAidePageBase, 'LeanAidePage');
