import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2 } from 'lucide-react';
import { backendsApi, type BackendStatus } from '@/services/backendsApi';

export const Route = createFileRoute('/backends')({
  component: BackendsPage,
});

function StatusBadge({ running }: { running: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium ${
        running
          ? 'bg-green-100 text-green-700'
          : 'bg-gray-100 text-gray-500'
      }`}
    >
      <span
        className={`h-2 w-2 rounded-full ${running ? 'bg-green-500' : 'bg-gray-400'}`}
      />
      {running ? 'Running' : 'Stopped'}
    </span>
  );
}

function BackendsPage() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const query = useQuery({
    queryKey: ['backends'],
    queryFn: backendsApi.list,
    refetchInterval: 5000,
  });

  const backends = query.data?.backends ?? [];

  const startMutation = useMutation({
    mutationFn: (name: string) => backendsApi.start(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['backends'] }),
  });

  const stopMutation = useMutation({
    mutationFn: (name: string) => backendsApi.stop(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['backends'] }),
  });

  const toggle = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const runBulk = (action: 'start' | 'stop') => {
    const names = [...selected];
    if (names.length === 0) return;
    names.forEach((name) => {
      if (action === 'start') startMutation.mutate(name);
      else stopMutation.mutate(name);
    });
  };

  const anyPending = startMutation.isPending || stopMutation.isPending;

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Backend Servers</h1>
          <p className="mt-1 text-sm text-gray-600">
            Start, stop, and check the status of the backend processes BubbleLab
            depends on. Changes are applied on the host running the BubbleLab API.
          </p>
        </div>
        <button
          className="rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-50"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          {query.isFetching ? (
            <Loader2 className="inline h-4 w-4 animate-spin" />
          ) : (
            'Refresh'
          )}
        </button>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <span className="text-sm text-gray-500">
          {selected.size} selected
        </span>
        <button
          className="rounded-md bg-green-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          onClick={() => runBulk('start')}
          disabled={anyPending || selected.size === 0}
        >
          Start selected
        </button>
        <button
          className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          onClick={() => runBulk('stop')}
          disabled={anyPending || selected.size === 0}
        >
          Stop selected
        </button>
      </div>

      <div className="overflow-hidden rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="w-12 px-4 py-3" />
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                Backend
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                Port
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold uppercase text-gray-500">
                PID
              </th>
              <th className="px-4 py-3 text-right text-xs font-semibold uppercase text-gray-500">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {backends.map((b: BackendStatus) => (
              <tr key={b.name} className="hover:bg-gray-50">
                <td className="px-4 py-3">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300"
                    checked={selected.has(b.name)}
                    onChange={() => toggle(b.name)}
                    aria-label={`Select ${b.label}`}
                  />
                </td>
                <td className="px-4 py-3">
                  <div className="font-medium text-gray-900">{b.label}</div>
                  <div className="max-w-md text-xs text-gray-500">
                    {b.description}
                  </div>
                  {b.error && (
                    <div className="mt-1 text-xs text-red-600">{b.error}</div>
                  )}
                </td>
                <td className="px-4 py-3 font-mono text-sm text-gray-700">
                  {b.port}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge running={b.running} />
                </td>
                <td className="px-4 py-3 font-mono text-sm text-gray-700">
                  {b.pid ?? '—'}
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="flex justify-end gap-2">
                    <button
                      className="rounded-md bg-green-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
                      onClick={() => startMutation.mutate(b.name)}
                      disabled={anyPending || b.running}
                    >
                      Start
                    </button>
                    <button
                      className="rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
                      onClick={() => stopMutation.mutate(b.name)}
                      disabled={anyPending || !b.running}
                    >
                      Stop
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {backends.length === 0 && !query.isLoading && (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-sm text-gray-500">
                  No backends found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {query.isLoading && (
        <div className="mt-4 flex items-center gap-2 text-sm text-gray-500">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading backend status…
        </div>
      )}
      {query.isError && (
        <div className="mt-4 rounded-md bg-red-50 p-3 text-sm text-red-700">
          Failed to load backends: {(query.error as Error)?.message}
        </div>
      )}
    </div>
  );
}
