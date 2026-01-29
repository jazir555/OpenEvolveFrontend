// @ts-nocheck
import { BubbleBadge, BubbleCard } from '@/components/bubblelab';
import { useMonitoring } from '@/services/hooks/useApi';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

const statusTone = (status?: string) => {
  if (!status) return 'neutral';
  const normalized = status.toLowerCase();
  if (normalized === 'healthy') return 'success';
  if (normalized === 'degraded') return 'warning';
  return 'danger';
};

function SystemHealthPanelBase() {
  const { health, isLoading } = useMonitoring(10000);

  return (
    <BubbleCard title="System Health" description="Live service and resource status.">
      {isLoading && <p className="text-sm text-slate-500">Loading system status...</p>}
      {!isLoading && health && (
        <div className="space-y-4 text-sm text-slate-600">
          <div className="flex items-center justify-between">
            <span className="font-medium text-slate-700">Overall Status</span>
            <BubbleBadge tone={statusTone(health.status) as any}>
              {health.status || 'unknown'}
            </BubbleBadge>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <div className="text-slate-500">CPU</div>
              <div className="text-slate-800">{health.resource_usage?.cpu_percent ?? 0}%</div>
            </div>
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <div className="text-slate-500">Memory</div>
              <div className="text-slate-800">{health.resource_usage?.memory_percent ?? 0}%</div>
            </div>
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <div className="text-slate-500">Disk</div>
              <div className="text-slate-800">{health.resource_usage?.disk_percent ?? 0}%</div>
            </div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">Active Operations</div>
            <div className="mt-1 flex items-center gap-3 text-sm">
              <span>Evolutions: {health.active_operations?.evolutions_running ?? 0}</span>
              <span>Adversarial: {health.active_operations?.adversarial_tests_running ?? 0}</span>
            </div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">Services</div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              {Object.entries(health.services || {}).map(([service, status]) => (
                <div key={service} className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-2">
                  <span className="text-slate-600">{service}</span>
                  <BubbleBadge tone={statusTone(status) as any}>{status}</BubbleBadge>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      {!isLoading && !health && (
        <p className="text-sm text-slate-500">System health data unavailable.</p>
      )}
    </BubbleCard>
  );
}

export const SystemHealthPanel = withComponentBoundary(
  SystemHealthPanelBase,
  'SystemHealthPanel'
);
