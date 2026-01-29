import { Link } from '@tanstack/react-router';
import { BarChart3, Clock, Sparkles, Shield, Workflow } from 'lucide-react';
import { useEvolutionSettingsStore } from '@/stores/evolutionSettingsStore';

const formatValue = (value: unknown) => {
  if (typeof value === 'boolean') {
    return value ? 'Enabled' : 'Disabled';
  }
  if (value === undefined || value === null || value === '') {
    return 'Not set';
  }
  return String(value);
};

export function EvolutionInsightsPage() {
  const { snapshots } = useEvolutionSettingsStore();
  const latest = snapshots[0];

  return (
    <div className="h-full bg-[#0a0a0a] overflow-auto">
      <div className="max-w-6xl mx-auto px-8 py-10 space-y-10">
        <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center">
              <BarChart3 className="h-5 w-5 text-purple-300" />
            </div>
            <div>
              <h1 className="text-3xl font-semibold text-white">
                Evolution Insights
              </h1>
              <p className="text-sm text-neutral-400 mt-1">
                Review saved snapshots and key configuration highlights.
              </p>
            </div>
          </div>
          <Link
            to="/evolution"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-neutral-700 text-neutral-200 text-sm hover:border-neutral-500 transition-colors"
          >
            Back to Settings
          </Link>
        </header>

        {!latest ? (
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-10 text-center">
            <p className="text-neutral-200 text-sm font-medium">
              No evolution snapshots yet.
            </p>
            <p className="text-neutral-500 text-xs mt-2">
              Save a snapshot from the Evolution Studio to populate insights.
            </p>
            <Link
              to="/evolution"
              className="inline-flex items-center gap-2 px-4 py-2 mt-4 rounded-lg bg-blue-600 text-white text-sm hover:bg-blue-500 transition-colors"
            >
              Save your first snapshot
            </Link>
          </div>
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-5 space-y-3">
                <div className="flex items-center gap-2 text-white">
                  <Sparkles className="h-4 w-4 text-blue-300" />
                  <h2 className="text-sm font-semibold">Evolution Summary</h2>
                </div>
                <div className="text-xs text-neutral-400 space-y-1">
                  <p>
                    Iterations:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.evolutionInputs.iterations)}
                    </span>
                  </p>
                  <p>
                    Population:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.evolutionInputs.populationSize)}
                    </span>
                  </p>
                  <p>
                    Temperature:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.evolutionInputs.temperature)}
                    </span>
                  </p>
                </div>
              </div>
              <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-5 space-y-3">
                <div className="flex items-center gap-2 text-white">
                  <Shield className="h-4 w-4 text-emerald-300" />
                  <h2 className="text-sm font-semibold">Adversarial Profile</h2>
                </div>
                <div className="text-xs text-neutral-400 space-y-1">
                  <p>
                    Attack Mode:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.adversarialInputs.attackMode)}
                    </span>
                  </p>
                  <p>
                    Rounds:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.adversarialInputs.rounds)}
                    </span>
                  </p>
                  <p>
                    Reporting:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.adversarialInputs.enableReporting)}
                    </span>
                  </p>
                </div>
              </div>
              <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-5 space-y-3">
                <div className="flex items-center gap-2 text-white">
                  <Workflow className="h-4 w-4 text-purple-300" />
                  <h2 className="text-sm font-semibold">Decomposition Scope</h2>
                </div>
                <div className="text-xs text-neutral-400 space-y-1">
                  <p>
                    Method:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.decompositionInputs.decompositionMethod)}
                    </span>
                  </p>
                  <p>
                    Granularity:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.decompositionInputs.granularity)}
                    </span>
                  </p>
                  <p>
                    Output:{' '}
                    <span className="text-neutral-200">
                      {formatValue(latest.decompositionInputs.outputFormat)}
                    </span>
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-6">
              <div className="flex items-center gap-2 text-white mb-4">
                <Clock className="h-4 w-4 text-neutral-400" />
                <h2 className="text-sm font-semibold">Recent Snapshots</h2>
              </div>
              <div className="space-y-3">
                {snapshots.slice(0, 5).map((snapshot) => (
                  <div
                    key={snapshot.id}
                    className="flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-950/30 px-4 py-3"
                  >
                    <div>
                      <p className="text-xs text-neutral-200 font-medium">
                        {new Date(snapshot.createdAt).toLocaleString()}
                      </p>
                      <p className="text-[11px] text-neutral-500">
                        Evolution: {formatValue(snapshot.evolutionInputs.model)} ·
                        Adversarial: {formatValue(snapshot.adversarialInputs.attackMode)} ·
                        Decomposition: {formatValue(snapshot.decompositionInputs.decompositionMethod)}
                      </p>
                    </div>
                    <span className="text-[10px] text-neutral-500">
                      {snapshot.id}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
