// @ts-nocheck
import { useEffect, useMemo, useRef, useState } from 'react';
import { ProviderSettingsPanel } from '@/components/config/ProviderSettingsPanel';
import {
  BubbleButton,
  BubbleCard,
  BubbleField,
  BubbleInput,
  BubbleSelect,
  BubbleTextArea,
  BubbleToggle,
} from '@/components/bubblelab';
import { LiveLogViewer, LogEntry } from '@/components/shared/LiveLogViewer';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { useAdversarialTest, useConfig } from '@/services/hooks/useApi';
import { AdversarialViz } from 'openevolve-pygraphistry-plugin';
import { VizErrorBoundary } from '@/components/shared/VizErrorBoundary';
import { PageErrorBoundary } from '@/components/shared/PageErrorBoundary';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

const attackOptions = [
  'prompt_injection',
  'jailbreak',
  'data_exfiltration',
  'system_prompt_leak',
  'tool_misuse',
];

function AdversarialPageBase() {
  const { providers } = useConfig();
  const [content, setContent] = useState('');
  const [selectedAttacks, setSelectedAttacks] = useState<string[]>(['prompt_injection']);
  const [numRounds, setNumRounds] = useState(5);
  const [testId, setTestId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [pageError, setPageError] = useState<string | null>(null);
  const lastStatusRef = useRef<string | null>(null);

  const [redTeamModels, setRedTeamModels] = useState<Array<{ provider: string; model: string }>>([
    { provider: 'openai', model: 'gpt-4' },
  ]);
  const [blueTeamModels, setBlueTeamModels] = useState<Array<{ provider: string; model: string }>>([
    { provider: 'anthropic', model: 'claude-3-sonnet' },
  ]);

  const providerOptions = useMemo(() => {
    if (!providers || providers.length === 0) {
      return [
        { provider: 'openai', name: 'OpenAI', models: ['gpt-4'] },
        { provider: 'anthropic', name: 'Anthropic', models: ['claude-3-sonnet'] },
      ];
    }
    return providers;
  }, [providers]);

  const {
    test,
    startTest,
    stopTest,
    isLoading,
    error,
  } = useAdversarialTest(testId || undefined);

  const status = test?.status || 'idle';

  useEffect(() => {
    if (!status || status === lastStatusRef.current) {
      return;
    }
    lastStatusRef.current = status;
    setLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toISOString(),
        level: 'info',
        message: `Adversarial status: ${status}`,
      },
    ]);
  }, [status]);

  const toggleAttack = (attack: string) => {
    setSelectedAttacks((prev) =>
      prev.includes(attack) ? prev.filter((item) => item !== attack) : [...prev, attack]
    );
  };

  const updateTeam = (
    team: 'red' | 'blue',
    index: number,
    updates: Partial<{ provider: string; model: string }>
  ) => {
    const setTeam = team === 'red' ? setRedTeamModels : setBlueTeamModels;
    const current = team === 'red' ? redTeamModels : blueTeamModels;
    setTeam(current.map((item, idx) => (idx === index ? { ...item, ...updates } : item)));
  };

  const addTeamMember = (team: 'red' | 'blue') => {
    const setTeam = team === 'red' ? setRedTeamModels : setBlueTeamModels;
    const defaultProvider = providerOptions[0];
    setTeam((prev) => [
      ...prev,
      { provider: defaultProvider?.provider || 'openai', model: defaultProvider?.models?.[0] || 'gpt-4' },
    ]);
  };

  const removeTeamMember = (team: 'red' | 'blue', index: number) => {
    const setTeam = team === 'red' ? setRedTeamModels : setBlueTeamModels;
    setTeam((prev) => prev.filter((_, idx) => idx !== index));
  };

  const handleStart = async () => {
    if (!content.trim()) {
      alert('Please enter content to test');
      return;
    }

    setLogs([
      {
        timestamp: new Date().toISOString(),
        level: 'info',
        message: 'Starting adversarial test...',
      },
    ]);

    try {
      setPageError(null);
      const response = await startTest({
        content,
        attack_modes: selectedAttacks,
        parameters: {
          num_rounds: numRounds,
          red_team_models: redTeamModels,
          blue_team_models: blueTeamModels,
        },
      });

      if (response?.test_id) {
        setTestId(response.test_id);
      }
    } catch (err) {
      setPageError(err instanceof Error ? err.message : 'Failed to start adversarial test.');
    }
  };

  const handleReset = () => {
    setTestId(null);
    setLogs([]);
  };

  const handleStop = async () => {
    if (!testId) return;
    try {
      setPageError(null);
      await stopTest(testId);
    } catch (err) {
      setPageError(err instanceof Error ? err.message : 'Failed to stop adversarial test.');
    }
  };

  return (
    <PageErrorBoundary label="Adversarial testing">
      <div className="flex h-screen bg-slate-50">
      <ProviderSettingsPanel />

      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <h1 className="text-2xl font-bold text-slate-900">Adversarial Testing</h1>
          <p className="text-sm text-slate-600">Run red/blue team validation workflows.</p>
        </header>

        <div className="flex-1 overflow-auto p-6 space-y-6">
          {(pageError || error) && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {pageError || (error instanceof Error ? error.message : 'Failed to load adversarial status.')}
            </div>
          )}
          <BubbleCard title="Test Setup">
            <div className="space-y-4">
              <BubbleField label="Content Under Test">
                <BubbleTextArea
                  rows={6}
                  value={content}
                  onChange={(event) => setContent(event.target.value)}
                  placeholder="Paste content or prompts for adversarial testing..."
                />
              </BubbleField>
              <BubbleField label="Attack Modes">
                <div className="flex flex-wrap gap-2">
                  {attackOptions.map((attack) => (
                    <BubbleToggle
                      key={attack}
                      checked={selectedAttacks.includes(attack)}
                      onChange={() => toggleAttack(attack)}
                      label={attack.replace(/_/g, ' ')}
                    />
                  ))}
                </div>
              </BubbleField>
              <BubbleField label="Rounds">
                <BubbleInput
                  type="number"
                  min={1}
                  value={numRounds}
                  onChange={(event) => setNumRounds(parseInt(event.target.value, 10) || 1)}
                />
              </BubbleField>
            </div>
          </BubbleCard>

          <BubbleCard title="Team Models">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-700">Red Team Models</h3>
                  <BubbleButton onClick={() => addTeamMember('red')} variant="secondary">
                    Add
                  </BubbleButton>
                </div>
                {redTeamModels.map((item, index) => (
                  <div key={`red-${index}`} className="grid grid-cols-6 gap-2">
                    <BubbleSelect
                      className="col-span-2"
                      value={item.provider}
                      onChange={(event) => updateTeam('red', index, { provider: event.target.value })}
                    >
                      {providerOptions.map((provider) => (
                        <option key={provider.provider} value={provider.provider}>
                          {provider.name || provider.provider}
                        </option>
                      ))}
                    </BubbleSelect>
                    <BubbleSelect
                      className="col-span-3"
                      value={item.model}
                      onChange={(event) => updateTeam('red', index, { model: event.target.value })}
                    >
                      {(providerOptions.find((p) => p.provider === item.provider)?.models || []).map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </BubbleSelect>
                    <BubbleButton onClick={() => removeTeamMember('red', index)} variant="secondary">
                      Remove
                    </BubbleButton>
                  </div>
                ))}
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-700">Blue Team Models</h3>
                  <BubbleButton onClick={() => addTeamMember('blue')} variant="secondary">
                    Add
                  </BubbleButton>
                </div>
                {blueTeamModels.map((item, index) => (
                  <div key={`blue-${index}`} className="grid grid-cols-6 gap-2">
                    <BubbleSelect
                      className="col-span-2"
                      value={item.provider}
                      onChange={(event) => updateTeam('blue', index, { provider: event.target.value })}
                    >
                      {providerOptions.map((provider) => (
                        <option key={provider.provider} value={provider.provider}>
                          {provider.name || provider.provider}
                        </option>
                      ))}
                    </BubbleSelect>
                    <BubbleSelect
                      className="col-span-3"
                      value={item.model}
                      onChange={(event) => updateTeam('blue', index, { model: event.target.value })}
                    >
                      {(providerOptions.find((p) => p.provider === item.provider)?.models || []).map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </BubbleSelect>
                    <BubbleButton onClick={() => removeTeamMember('blue', index)} variant="secondary">
                      Remove
                    </BubbleButton>
                  </div>
                ))}
              </div>
            </div>
          </BubbleCard>

          <BubbleCard title="Execution Controls">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <StatusBadge status={status as any} />
              <div className="flex flex-wrap gap-2">
                {!testId && (
                  <BubbleButton onClick={handleStart}>Start Test</BubbleButton>
                )}
                {testId && status !== 'stopped' && (
                  <BubbleButton onClick={handleStop} variant="secondary">
                    Stop Test
                  </BubbleButton>
                )}
                <BubbleButton onClick={handleReset} variant="secondary">
                  Reset
                </BubbleButton>
              </div>
            </div>
          </BubbleCard>

          <BubbleCard title="Adversarial Visualization" description="Graphistry-backed red/blue team outcomes.">
            <VizErrorBoundary label="Adversarial visualization">
              <AdversarialViz content={content} />
            </VizErrorBoundary>
          </BubbleCard>

          <BubbleCard title="Live Logs">
            <LiveLogViewer logs={logs} maxHeight="260px" />
            {isLoading && (
              <p className="mt-3 text-xs text-slate-400">Fetching adversarial status...</p>
            )}
          </BubbleCard>
        </div>
      </main>
    </div>
    </PageErrorBoundary>
  );
}

export const AdversarialPage = withComponentBoundary(AdversarialPageBase, 'AdversarialPage');
