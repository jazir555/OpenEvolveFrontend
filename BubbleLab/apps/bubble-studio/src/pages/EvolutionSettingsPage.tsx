import { Link } from '@tanstack/react-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeftRight,
  BarChart3,
  Download,
  GitCompare,
  Image,
  Play,
  RefreshCw,
  RotateCcw,
  Save,
  Shield,
  Sparkles,
  Trash2,
  Trophy,
  Workflow,
} from 'lucide-react';
import { CredentialType } from '@bubblelab/shared-schemas';
import { toast } from 'react-toastify';
import {
  evolutionParameters,
  evolutionConstraintParameters,
  evolutionPreviewParameters,
  adversarialParameters,
  decompositionParameters,
} from '@/lib/evolution/schemas';
import { EvolutionParameterForm } from '@/components/evolution/EvolutionParameterForm';
import { useEvolutionSettingsStore } from '@/stores/evolutionSettingsStore';
import { useEvolutionRuntimeStore } from '@/stores/evolutionRuntimeStore';
import {
  useEvolutionGraphStore,
  type EvolutionGraphNode,
  type EvolutionNodeData,
} from '@/stores/evolutionGraphStore';
import { evolutionApi } from '@/services/evolutionApi';
import { evolutionGraphApi } from '@/services/evolutionGraphApi';
import { credentialsApi } from '@/services/credentialsApi';
import { useEvolutionWebSocket } from '@/hooks/useEvolutionWebSocket';
import { useCredentials } from '@/hooks/useCredentials';
import { API_BASE_URL } from '@/env';
import { renderHtmlToPngBase64 } from '@/utils/evolutionPreview';
import type { EvolutionStartPayload } from '@/types/evolution';
import type {
  EvolutionRunRecord,
  EvolutionRunUsage,
} from '@/services/evolutionGraphApi';

const PROVIDER_OPTIONS = [
  {
    value: 'openai',
    label: 'OpenAI',
    credentialType: CredentialType.OPENAI_CRED,
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    credentialType: CredentialType.ANTHROPIC_CRED,
  },
  {
    value: 'google',
    label: 'Google',
    credentialType: CredentialType.GOOGLE_GEMINI_CRED,
  },
  {
    value: 'openrouter',
    label: 'OpenRouter',
    credentialType: CredentialType.OPENROUTER_CRED,
  },
  {
    value: 'deepseek',
    label: 'DeepSeek',
    credentialType: CredentialType.DEEPSEEK_CRED,
  },
] as const;

const MODEL_OPTIONS_BY_PROVIDER: Record<
  string,
  Array<{ value: string; label: string }>
> = {
  openai: [
    { value: 'gpt-4', label: 'GPT-4' },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
  ],
  anthropic: [
    { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' },
    { value: 'claude-3-opus', label: 'Claude 3 Opus' },
    { value: 'claude-3-haiku', label: 'Claude 3 Haiku' },
  ],
  google: [{ value: 'gemini-pro', label: 'Gemini Pro' }],
  openrouter: [
    { value: 'gpt-4', label: 'GPT-4' },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
    { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' },
    { value: 'claude-3-opus', label: 'Claude 3 Opus' },
    { value: 'claude-3-haiku', label: 'Claude 3 Haiku' },
    { value: 'gemini-pro', label: 'Gemini Pro' },
    { value: 'deepseek-chat', label: 'DeepSeek Chat' },
    { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner' },
  ],
  deepseek: [
    { value: 'deepseek-chat', label: 'DeepSeek Chat' },
    { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner' },
  ],
};

const DEFAULT_MODEL_BY_PROVIDER: Record<string, string> = {
  openai: 'gpt-4',
  anthropic: 'claude-3-sonnet',
  google: 'gemini-pro',
  openrouter: 'claude-3-sonnet',
  deepseek: 'deepseek-chat',
};

const STATUS_BADGE_STYLES: Record<EvolutionNodeData['status'], string> = {
  survived: 'bg-emerald-500/20 text-emerald-200 border-emerald-500/40',
  killed: 'bg-red-500/20 text-red-200 border-red-500/40',
  evaluating: 'bg-amber-500/20 text-amber-200 border-amber-500/40',
};

const getCredentialTypeForProvider = (provider: string) => {
  return (
    PROVIDER_OPTIONS.find((option) => option.value === provider)?.credentialType ??
    null
  );
};

const constraintKeys = evolutionConstraintParameters.map((param) => param.name);

const formatBytes = (value: number) => {
  if (!Number.isFinite(value)) return '--';
  if (value < 1024) return `${value} B`;
  const units = ['KB', 'MB', 'GB'];
  let size = value;
  let unitIndex = -1;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const precision = size >= 10 ? 1 : 2;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
};

const formatMetric = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) return '--';
  return value.toFixed(2);
};

const extractCss = (html: string): string | null => {
  const match = html.match(/<style[^>]*>([\s\S]*?)<\/style>/i);
  return match ? match[1].trim() : null;
};

const getNodeRankValue = (node: EvolutionGraphNode) => {
  const data = node.data as EvolutionNodeData;
  return data.fitness ?? data.score ?? -Infinity;
};

const getNodeLabel = (nodeData: EvolutionNodeData) =>
  nodeData.label || `Generation ${nodeData.generation}`;

const COST_PER_VARIANT_USD = 0.0025;

interface EvolutionSettingsPageProps {
  onCollapsePanel?: () => void;
}

export function EvolutionSettingsPage({
  onCollapsePanel,
}: EvolutionSettingsPageProps) {
  const {
    evolutionInputs,
    adversarialInputs,
    decompositionInputs,
    setEvolutionInput,
    setAdversarialInput,
    setDecompositionInput,
    resetEvolution,
    resetAdversarial,
    resetDecomposition,
    resetAll,
    addSnapshot,
    snapshots,
  } = useEvolutionSettingsStore();
  const {
    evolutionId,
    status,
    progress,
    statusMessage,
    latestGeneration,
    latestFitness,
    error,
    socketStatus,
    setStart,
    setStatus,
    setError,
    clearError,
    triggerReconnect,
    reset: resetRuntime,
  } = useEvolutionRuntimeStore();
  const runId = useEvolutionGraphStore((s) => s.runId);
  const nodes = useEvolutionGraphStore((s) => s.nodes);
  const setActiveRun = useEvolutionGraphStore((s) => s.setActiveRun);
  const clearGraph = useEvolutionGraphStore((s) => s.clearGraph);
  const setNodes = useEvolutionGraphStore((s) => s.setNodes);
  const previewMode = useEvolutionGraphStore((s) => s.previewMode);
  const setPreviewMode = useEvolutionGraphStore((s) => s.setPreviewMode);
  const openModal = useEvolutionGraphStore((s) => s.openModal);

  const [usage, setUsage] = useState<EvolutionRunUsage | null>(null);
  const [usageLoading, setUsageLoading] = useState(false);
  const [usageError, setUsageError] = useState<string | null>(null);
  const [isOffline, setIsOffline] = useState(
    typeof navigator !== 'undefined' ? !navigator.onLine : false
  );
  const [runs, setRuns] = useState<EvolutionRunRecord[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runFilter, setRunFilter] = useState('all');
  const [runPage, setRunPage] = useState(1);
  const [showStartConfirm, setShowStartConfirm] = useState(false);
  const [pendingStartPayload, setPendingStartPayload] =
    useState<EvolutionStartPayload | null>(null);
  const lastStartPayloadRef = useRef<EvolutionStartPayload | null>(null);
  const [nodeHtmlCache, setNodeHtmlCache] = useState<Record<string, string>>(
    {}
  );
  const [comparePrimaryId, setComparePrimaryId] = useState<string | null>(null);
  const [compareSecondaryId, setCompareSecondaryId] = useState<string | null>(
    null
  );
  const [showWinnerSource, setShowWinnerSource] = useState(false);
  const [winnerSourceLoading, setWinnerSourceLoading] = useState(false);

  useEvolutionWebSocket();

  const credentialsQuery = useCredentials(API_BASE_URL);
  const credentials = credentialsQuery.data ?? [];
  const credentialsUpdatedAt = credentialsQuery.dataUpdatedAt;

  const latestSnapshot = snapshots[0];
  const apiKey =
    typeof evolutionInputs.apiKey === 'string' ? evolutionInputs.apiKey : '';
  const content =
    typeof evolutionInputs.content === 'string' ? evolutionInputs.content : '';
  const cachePreviewsEnabled = Boolean(evolutionInputs.cachePreviews ?? true);
  const isRunning = status === 'running' || status === 'starting';
  const provider = String(evolutionInputs.provider ?? 'anthropic');
  const iterationsValue = Number(evolutionInputs.iterations ?? 10);
  const populationValue = Number(evolutionInputs.populationSize ?? 10);
  const maxBudgetValue =
    typeof evolutionInputs.maxBudget === 'number'
      ? evolutionInputs.maxBudget
      : 0;
  const estimatedCost = Number.isFinite(iterationsValue) && Number.isFinite(populationValue)
    ? iterationsValue * populationValue * COST_PER_VARIANT_USD
    : 0;
  const isBudgetOver = maxBudgetValue > 0 && estimatedCost > maxBudgetValue;
  const hasRetryPayload = Boolean(lastStartPayloadRef.current);
  const apiKeyRef = useRef(apiKey);
  const lastAutoFillRef = useRef<{ provider: string; value: string } | null>(
    null
  );

  useEffect(() => {
    apiKeyRef.current = apiKey;
  }, [apiKey]);

  useEffect(() => {
    const update = () => {
      setIsOffline(!navigator.onLine);
    };
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    return () => {
      window.removeEventListener('online', update);
      window.removeEventListener('offline', update);
    };
  }, []);

  const scoredNodes = useMemo(() => {
    return nodes
      .filter((node) => getNodeRankValue(node) > -Infinity)
      .sort((a, b) => {
        const rankDiff = getNodeRankValue(b) - getNodeRankValue(a);
        if (rankDiff !== 0) return rankDiff;
        const timeA = a.data.createdAt ? Date.parse(a.data.createdAt) : 0;
        const timeB = b.data.createdAt ? Date.parse(b.data.createdAt) : 0;
        if (timeA !== timeB) return timeA - timeB;
        return a.data.id.localeCompare(b.data.id);
      });
  }, [nodes]);

  const winnerNode = scoredNodes[0] ?? null;
  const winnerHtml = winnerNode
    ? winnerNode.data.html ?? nodeHtmlCache[winnerNode.data.id] ?? null
    : null;
  const winnerCss = winnerHtml ? extractCss(winnerHtml) : null;

  const fitnessTimeline = useMemo(() => {
    const bestByGeneration = new Map<number, number>();
    nodes.forEach((node) => {
      if (node.data.fitness == null || Number.isNaN(node.data.fitness)) return;
      const current = bestByGeneration.get(node.data.generation);
      if (current == null || node.data.fitness > current) {
        bestByGeneration.set(node.data.generation, node.data.fitness);
      }
    });
    return Array.from(bestByGeneration.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([generation, fitness]) => ({ generation, fitness }));
  }, [nodes]);

  const maxTimelineFitness = useMemo(() => {
    return fitnessTimeline.reduce(
      (max, point) => Math.max(max, point.fitness),
      0
    );
  }, [fitnessTimeline]);

  const lineageHighlights = useMemo(() => {
    const bestByGeneration = new Map<number, EvolutionGraphNode>();
    nodes.forEach((node) => {
      const rank = getNodeRankValue(node);
      if (rank === -Infinity) return;
      const current = bestByGeneration.get(node.data.generation);
      if (!current || rank > getNodeRankValue(current)) {
        bestByGeneration.set(node.data.generation, node);
      }
    });
    return Array.from(bestByGeneration.values()).sort(
      (a, b) => a.data.generation - b.data.generation
    );
  }, [nodes]);

  const comparisonCandidates = useMemo(() => {
    return [...nodes].sort((a, b) => {
      if (a.data.generation !== b.data.generation) {
        return a.data.generation - b.data.generation;
      }
      const rankDiff = getNodeRankValue(b) - getNodeRankValue(a);
      if (rankDiff !== 0) return rankDiff;
      const timeA = a.data.createdAt ? Date.parse(a.data.createdAt) : 0;
      const timeB = b.data.createdAt ? Date.parse(b.data.createdAt) : 0;
      if (timeA !== timeB) return timeA - timeB;
      return a.data.id.localeCompare(b.data.id);
    });
  }, [nodes]);

  const comparePrimaryNode = useMemo(
    () => comparisonCandidates.find((node) => node.id === comparePrimaryId) ?? null,
    [comparisonCandidates, comparePrimaryId]
  );

  const compareSecondaryNode = useMemo(
    () => comparisonCandidates.find((node) => node.id === compareSecondaryId) ?? null,
    [comparisonCandidates, compareSecondaryId]
  );

  useEffect(() => {
    if (!comparisonCandidates.length) {
      setComparePrimaryId(null);
      setCompareSecondaryId(null);
      return;
    }

    const availableIds = new Set(comparisonCandidates.map((node) => node.id));
    const nextPrimary =
      comparePrimaryId && availableIds.has(comparePrimaryId)
        ? comparePrimaryId
        : comparisonCandidates[0].id;

    let nextSecondary =
      compareSecondaryId && availableIds.has(compareSecondaryId)
        ? compareSecondaryId
        : null;

    if (nextSecondary === nextPrimary) {
      nextSecondary = null;
    }

    if (!nextSecondary) {
      const fallback = comparisonCandidates.find(
        (node) => node.id !== nextPrimary
      );
      nextSecondary = fallback ? fallback.id : null;
    }

    if (nextPrimary !== comparePrimaryId) {
      setComparePrimaryId(nextPrimary);
    }
    if (nextSecondary !== compareSecondaryId) {
      setCompareSecondaryId(nextSecondary);
    }
  }, [comparisonCandidates, comparePrimaryId, compareSecondaryId]);

  useEffect(() => {
    setShowWinnerSource(false);
  }, [winnerNode?.id]);

  const fetchUsage = useCallback(
    async (targetRunId: number | null = runId, silent = false) => {
      if (!targetRunId) {
        setUsage(null);
        setUsageError(null);
        return;
      }

      setUsageLoading(true);
      setUsageError(null);

      try {
        const response = await evolutionGraphApi.getRunUsage(targetRunId);
        setUsage(response);
      } catch (usageErr) {
        const message =
          usageErr instanceof Error
            ? usageErr.message
            : 'Failed to load storage usage';
        setUsageError(message);
        if (!silent) {
          toast.error(message);
        }
      } finally {
        setUsageLoading(false);
      }
    },
    [runId]
  );

  const fetchRuns = useCallback(async () => {
    setRunsLoading(true);
    setRunsError(null);
    try {
      const data = await evolutionGraphApi.listRuns();
      setRuns(data);
    } catch (runsErr) {
      const message =
        runsErr instanceof Error
          ? runsErr.message
          : 'Failed to load evolution runs';
      setRunsError(message);
    } finally {
      setRunsLoading(false);
    }
  }, []);

  const fetchNodeHtml = useCallback(
    async (node: EvolutionGraphNode | null) => {
      if (!node) return null;
      if (node.data.html) return node.data.html;
      const cached = nodeHtmlCache[node.data.id];
      if (cached) return cached;
      if (!node.data.htmlUrl) return null;
      try {
        const res = await fetch(node.data.htmlUrl);
        if (!res.ok) {
          throw new Error('Failed to load HTML content.');
        }
        const text = await res.text();
        setNodeHtmlCache((prev) => ({ ...prev, [node.data.id]: text }));
        return text;
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to load HTML content.';
        toast.error(message);
        return null;
      }
    },
    [nodeHtmlCache]
  );

  const downloadBlob = useCallback((blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }, []);

  const handleDownloadNodeHtml = useCallback(
    async (node: EvolutionGraphNode | null) => {
      const html = await fetchNodeHtml(node);
      if (!html || !node) {
        toast.error('HTML not available for this node.');
        return;
      }
      downloadBlob(new Blob([html], { type: 'text/html' }), `${node.data.id}.html`);
      toast.success('HTML download started.');
    },
    [downloadBlob, fetchNodeHtml]
  );

  const handleDownloadNodeCss = useCallback(
    async (node: EvolutionGraphNode | null) => {
      const html = await fetchNodeHtml(node);
      if (!html || !node) {
        toast.error('HTML not available for this node.');
        return;
      }
      const css = extractCss(html);
      if (!css) {
        toast.info('No embedded CSS found in this HTML.');
        return;
      }
      downloadBlob(new Blob([css], { type: 'text/css' }), `${node.data.id}.css`);
      toast.success('CSS download started.');
    },
    [downloadBlob, fetchNodeHtml]
  );

  const handleDownloadNodePreview = useCallback(
    async (node: EvolutionGraphNode | null) => {
      if (!node) return;
      try {
        if (node.data.thumbnailUrl) {
          const response = await fetch(node.data.thumbnailUrl);
          if (!response.ok) {
            throw new Error('Failed to download cached preview.');
          }
          const blob = await response.blob();
          downloadBlob(blob, `${node.data.id}-preview.png`);
          toast.success('Preview download started.');
          return;
        }

        const html = await fetchNodeHtml(node);
        if (!html) {
          toast.error('HTML not available for preview export.');
          return;
        }
        const base64 = await renderHtmlToPngBase64(html, {
          width: 1280,
          height: 720,
        });
        if (!base64) {
          toast.error('Unable to render preview image.');
          return;
        }
        const blob = await fetch(`data:image/png;base64,${base64}`).then(
          (res) => res.blob()
        );
        downloadBlob(blob, `${node.data.id}-preview.png`);
        toast.success('Preview download started.');
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : 'Failed to download preview image.';
        toast.error(message);
      }
    },
    [downloadBlob, fetchNodeHtml]
  );

  const renderPreview = useCallback(
    (
      node: EvolutionGraphNode | null,
      heightClass: string,
      emptyLabel: string
    ) => {
      if (!node) {
        return (
          <div
            className={`${heightClass} rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400 text-sm`}
          >
            {emptyLabel}
          </div>
        );
      }

      if (previewMode === 'live') {
        if (node.data.htmlUrl || node.data.html) {
          return (
            <iframe
              title={node.data.label || 'Evolution preview'}
              src={node.data.htmlUrl || undefined}
              srcDoc={node.data.htmlUrl ? undefined : node.data.html || undefined}
              className={`w-full ${heightClass} rounded-xl border border-neutral-800 bg-white`}
              sandbox="allow-scripts allow-same-origin"
            />
          );
        }
        return (
          <div
            className={`${heightClass} rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400 text-sm`}
          >
            Live HTML preview not available yet.
          </div>
        );
      }

      if (!cachePreviewsEnabled) {
        return (
          <div
            className={`${heightClass} rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400 text-sm`}
          >
            Cached previews are disabled in settings.
          </div>
        );
      }

      if (node.data.thumbnailUrl) {
        return (
          <img
            src={node.data.thumbnailUrl}
            alt={node.data.label || 'Cached preview'}
            className={`w-full ${heightClass} rounded-xl border border-neutral-800 object-cover`}
          />
        );
      }

      return (
        <div
          className={`${heightClass} rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400 text-sm`}
        >
          Cached preview not available yet.
        </div>
      );
    },
    [cachePreviewsEnabled, previewMode]
  );

  const handleToggleWinnerSource = useCallback(async () => {
    if (!winnerNode) return;
    const nextValue = !showWinnerSource;
    setShowWinnerSource(nextValue);
    if (!nextValue) return;
    if (winnerNode.data.html || nodeHtmlCache[winnerNode.data.id]) return;
    setWinnerSourceLoading(true);
    await fetchNodeHtml(winnerNode);
    setWinnerSourceLoading(false);
  }, [fetchNodeHtml, nodeHtmlCache, showWinnerSource, winnerNode]);

  const handleSwapCompare = useCallback(() => {
    if (!comparePrimaryId || !compareSecondaryId) return;
    setComparePrimaryId(compareSecondaryId);
    setCompareSecondaryId(comparePrimaryId);
  }, [comparePrimaryId, compareSecondaryId]);

  const renderComparisonCard = useCallback(
    (node: EvolutionGraphNode | null, label: string) => {
      const statusBadge = node
        ? STATUS_BADGE_STYLES[node.data.status ?? 'evaluating']
        : 'border-neutral-700 text-neutral-500';
      return (
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-4 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
                {label}
              </p>
              <p className="text-sm font-semibold text-white">
                {node ? getNodeLabel(node.data) : 'Select a node'}
              </p>
              {node && (
                <p className="text-[11px] text-neutral-500">
                  Generation {node.data.generation}
                </p>
              )}
            </div>
            <span
              className={`px-2 py-1 rounded-full border text-[10px] uppercase tracking-[0.18em] ${statusBadge}`}
            >
              {node?.data.status ?? 'empty'}
            </span>
          </div>
          {renderPreview(node, 'h-[220px]', 'Pick a node to compare.')}
          <div className="grid grid-cols-2 gap-3 text-xs text-neutral-400">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                Fitness
              </p>
              <p className="text-sm font-semibold text-white">
                {formatMetric(node?.data.fitness)}
              </p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                Score
              </p>
              <p className="text-sm font-semibold text-white">
                {formatMetric(node?.data.score)}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => node && openModal(node.id)}
            disabled={!node}
            className="w-full px-3 py-2 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Open Full Preview
          </button>
        </div>
      );
    },
    [openModal, renderPreview]
  );

  useEffect(() => {
    void fetchUsage(runId, true);
  }, [fetchUsage, runId]);

  useEffect(() => {
    void fetchRuns();
  }, [fetchRuns, runId]);

  const providerOptions = useMemo(() => {
    if (!credentials.length) {
      return PROVIDER_OPTIONS.map(({ value, label }) => ({ value, label }));
    }

    const available = PROVIDER_OPTIONS.filter((option) =>
      credentials.some(
        (credential) => credential.credentialType === option.credentialType
      )
    );

    return (available.length ? available : PROVIDER_OPTIONS).map(
      ({ value, label }) => ({ value, label })
    );
  }, [credentials]);

  const modelOptions = useMemo(() => {
    return MODEL_OPTIONS_BY_PROVIDER[provider] ?? MODEL_OPTIONS_BY_PROVIDER.anthropic;
  }, [provider]);

  const RUNS_PER_PAGE = 5;
  const filteredRuns = useMemo(() => {
    if (runFilter === 'all') return runs;
    return runs.filter((run) => run.status === runFilter);
  }, [runFilter, runs]);
  const totalRunPages = Math.max(
    1,
    Math.ceil(filteredRuns.length / RUNS_PER_PAGE)
  );
  const pagedRuns = useMemo(() => {
    const start = (runPage - 1) * RUNS_PER_PAGE;
    return filteredRuns.slice(start, start + RUNS_PER_PAGE);
  }, [filteredRuns, runPage]);
  const runStatusOptions = useMemo(() => {
    const base = ['running', 'completed', 'paused', 'stopped', 'error'];
    const fromRuns = Array.from(new Set(runs.map((run) => run.status)));
    return ['all', ...Array.from(new Set([...fromRuns, ...base]))];
  }, [runs]);

  const evolutionParameterOverrides = useMemo(
    () =>
      evolutionParameters.map((parameter) =>
        parameter.name === 'provider'
          ? { ...parameter, options: providerOptions }
          : parameter.name === 'model'
            ? { ...parameter, options: modelOptions }
            : parameter
      ),
    [modelOptions, providerOptions]
  );

  const selectedCredentialType = useMemo(
    () => getCredentialTypeForProvider(provider),
    [provider]
  );

  useEffect(() => {
    if (!providerOptions.length) return;
    if (providerOptions.some((option) => option.value === provider)) return;
    setEvolutionInput('provider', providerOptions[0].value);
  }, [provider, providerOptions, setEvolutionInput]);

  useEffect(() => {
    setRunPage(1);
  }, [runFilter]);

  useEffect(() => {
    if (runPage > totalRunPages) {
      setRunPage(totalRunPages);
    }
  }, [runPage, totalRunPages]);

  useEffect(() => {
    const modelValue = String(evolutionInputs.model ?? '');
    const modelIsValid = modelOptions.some(
      (option) => option.value === modelValue
    );
    if (modelIsValid) return;
    const fallback =
      DEFAULT_MODEL_BY_PROVIDER[provider] ?? modelOptions[0]?.value;
    if (fallback) {
      setEvolutionInput('model', fallback);
    }
  }, [evolutionInputs.model, modelOptions, provider, setEvolutionInput]);

  useEffect(() => {
    if (!selectedCredentialType) return;

    let cancelled = false;

    credentialsApi
      .getDefaultCredentialValue(selectedCredentialType)
      .then((response) => {
        if (cancelled) return;

        if (response.value) {
          if (apiKeyRef.current !== response.value) {
            setEvolutionInput('apiKey', response.value);
          }
          lastAutoFillRef.current = { provider, value: response.value };
          return;
        }

        const lastAutoFill = lastAutoFillRef.current;
        if (
          lastAutoFill &&
          lastAutoFill.provider !== provider &&
          apiKeyRef.current === lastAutoFill.value
        ) {
          setEvolutionInput('apiKey', '');
          lastAutoFillRef.current = null;
        }
      })
      .catch(() => {
        // Ignore credential lookup failures; user can still paste a key.
      });

    return () => {
      cancelled = true;
    };
  }, [
    provider,
    selectedCredentialType,
    setEvolutionInput,
    credentialsUpdatedAt,
  ]);

  useEffect(() => {
    const lastAutoFill = lastAutoFillRef.current;
    if (!lastAutoFill) return;
    if (apiKey && apiKey !== lastAutoFill.value) {
      lastAutoFillRef.current = null;
    }
  }, [apiKey]);

  const handleClearGraph = async () => {
    if (!runId) {
      clearGraph();
      return;
    }

    try {
      await evolutionGraphApi.clearRunNodes(runId);
      clearGraph();
      void fetchUsage(runId, true);
      toast.success('Evolution graph cleared.');
    } catch (clearError) {
      toast.error(
        clearError instanceof Error
          ? clearError.message
          : 'Failed to clear evolution graph'
      );
    }
  };

  const handleClearThumbnails = async () => {
    if (!runId) {
      toast.error('No evolution run selected.');
      return;
    }

    try {
      const response = await evolutionGraphApi.clearRunThumbnails(runId);
      setNodes(
        nodes.map((node) => ({
          ...node.data,
          thumbnailUrl: null,
        }))
      );
      void fetchUsage(runId, true);
      toast.success(
        response.removedCount
          ? `Cleared ${response.removedCount} cached previews.`
          : 'No cached previews found.'
      );
    } catch (clearError) {
      toast.error(
        clearError instanceof Error
          ? clearError.message
          : 'Failed to clear cached previews'
      );
    }
  };

  const handleSelectRun = (run: EvolutionRunRecord) => {
    clearGraph();
    setActiveRun(run.id, run.evolutionId);
    void fetchUsage(run.id, true);
  };

  const handleDeleteRun = async (run: EvolutionRunRecord) => {
    try {
      await evolutionGraphApi.deleteRun(run.id);
      if (runId === run.id) {
        clearGraph();
        setActiveRun(null, null);
        setUsage(null);
      }
      await fetchRuns();
      toast.success('Evolution run deleted.');
    } catch (deleteError) {
      toast.error(
        deleteError instanceof Error
          ? deleteError.message
          : 'Failed to delete evolution run'
      );
    }
  };

  const performStart = async (payload: EvolutionStartPayload) => {
    try {
      const response = await evolutionApi.startEvolution(payload);
      setStart(response.evolution_id, response.websocket_url);

      try {
        const run = await evolutionGraphApi.createRun({
          evolutionId: response.evolution_id,
          status: 'running',
          name:
            content.trim().length > 48
              ? `${content.trim().slice(0, 48)}...`
              : content.trim() || 'Evolution Run',
          config: {
            evolutionInputs,
            adversarialInputs,
            decompositionInputs,
          },
        });
        clearGraph();
        setActiveRun(run.id, run.evolutionId);
        void fetchUsage(run.id, true);
      } catch (graphError) {
        toast.error(
          graphError instanceof Error
            ? graphError.message
            : 'Evolution started, but graph persistence failed'
        );
      }

      toast.success('Evolution started.');
    } catch (startError) {
      setError(
        startError instanceof Error
          ? startError.message
          : 'Failed to start evolution'
      );
    }
  };

  const handleStartEvolution = async () => {
    if (!content.trim()) {
      toast.error('Please add content to evolve before starting.');
      return;
    }
    if (!apiKey.trim()) {
      toast.error('Please provide a model API key to start evolution.');
      return;
    }
    if (isBudgetOver) {
      toast.error('Estimated cost exceeds the max budget. Adjust settings.');
      return;
    }

    const constraints = constraintKeys.reduce<Record<string, unknown>>(
      (acc, key) => {
        const value = evolutionInputs[key];
        if (value !== undefined && value !== null && value !== '') {
          acc[key] = value;
        }
        return acc;
      },
      {}
    );

    const payload: EvolutionStartPayload = {
      content,
      mode: 'standard',
      parameters: {
        max_iterations: Number(evolutionInputs.iterations ?? 10),
        population_size: Number(evolutionInputs.populationSize ?? 10),
        temperature: Number(evolutionInputs.temperature ?? 0.7),
        top_p: 0.9,
        mutation_rate: Number(evolutionInputs.mutationRate ?? 0.1),
        crossover_rate: Number(evolutionInputs.crossoverRate ?? 0.7),
        branching_mode: String(evolutionInputs.branchingMode ?? 'lineage') as
          | 'root'
          | 'lineage',
        children_per_parent: Number(evolutionInputs.childrenPerParent ?? 3),
        survival_threshold: Number(evolutionInputs.survivalThreshold ?? 0.6),
      },
      models: [
        {
          provider: String(evolutionInputs.provider ?? 'anthropic'),
          model: String(evolutionInputs.model ?? 'claude-3-sonnet'),
          api_key: apiKey.trim(),
        },
      ],
      constraints: Object.keys(constraints).length ? constraints : undefined,
    };

    setPendingStartPayload(payload);
    setShowStartConfirm(true);
  };

  const handleRetryStart = () => {
    const payload = lastStartPayloadRef.current;
    if (!payload) {
      toast.error('No previous start payload to retry.');
      return;
    }
    clearError();
    setStatus('starting');
    void performStart(payload);
  };

  const handleConfirmStart = () => {
    if (!pendingStartPayload) {
      setShowStartConfirm(false);
      return;
    }
    setShowStartConfirm(false);
    setStatus('starting');
    addSnapshot();
    lastStartPayloadRef.current = pendingStartPayload;
    void performStart(pendingStartPayload);
    setPendingStartPayload(null);
  };

  const handleCancelStart = () => {
    setShowStartConfirm(false);
    setPendingStartPayload(null);
  };

  const handlePauseEvolution = async () => {
    if (!evolutionId) {
      toast.error('No evolution is running.');
      return;
    }
    try {
      await evolutionApi.pauseEvolution(evolutionId);
      setStatus('paused');
      toast.success('Evolution paused.');
    } catch (pauseError) {
      toast.error(
        pauseError instanceof Error
          ? pauseError.message
          : 'Failed to pause evolution'
      );
    }
  };

  const handleResumeEvolution = async () => {
    if (!evolutionId) {
      toast.error('No evolution to resume.');
      return;
    }
    try {
      await evolutionApi.resumeEvolution(evolutionId);
      setStatus('running');
      toast.success('Evolution resumed.');
    } catch (resumeError) {
      toast.error(
        resumeError instanceof Error
          ? resumeError.message
          : 'Failed to resume evolution'
      );
    }
  };

  return (
    <div className="h-full bg-[#0a0a0a] overflow-auto">
      <div className="max-w-6xl mx-auto px-8 py-10 space-y-10">
        <header className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-blue-300" />
              </div>
              <div>
                <h1 className="text-3xl font-semibold text-white">
                  Evolution Studio
                </h1>
                <p className="text-sm text-neutral-400 mt-1">
                  Configure OpenEvolve parameters and capture snapshots for review.
                </p>
              </div>
            </div>
            <p className="text-xs text-neutral-500 mt-3">
              {latestSnapshot
                ? `Latest snapshot saved at ${new Date(
                    latestSnapshot.createdAt
                  ).toLocaleString()}`
                : 'No snapshots yet. Save one to populate insights.'}
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/evolution/insights"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-neutral-700 text-neutral-200 text-sm hover:border-neutral-500 transition-colors"
            >
              <BarChart3 className="h-4 w-4" />
              View Insights
            </Link>
            <button
              type="button"
              onClick={handleStartEvolution}
              disabled={isRunning}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm hover:bg-emerald-500 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <Play className="h-4 w-4" />
              Start Evolution
            </button>
            <button
              type="button"
              onClick={addSnapshot}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white text-sm hover:bg-blue-500 transition-colors"
            >
              <Save className="h-4 w-4" />
              Save Snapshot
            </button>
            <button
              type="button"
              onClick={handleClearGraph}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-red-500/40 text-red-200 text-sm hover:border-red-400 transition-colors"
            >
              <Trash2 className="h-4 w-4" />
              Clear Graph
            </button>
            <button
              type="button"
              onClick={() => {
                resetAll();
                resetRuntime();
              }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-neutral-700 text-neutral-200 text-sm hover:border-neutral-500 transition-colors"
            >
              <RotateCcw className="h-4 w-4" />
              Reset All
            </button>
            {onCollapsePanel && (
              <button
                type="button"
                onClick={onCollapsePanel}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-neutral-700 text-neutral-200 text-sm hover:border-neutral-500 transition-colors"
              >
                Collapse Panel
              </button>
            )}
          </div>
        </header>

        {isOffline && (
          <div className="rounded-2xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-xs text-amber-200">
            You are offline. Evolution updates may pause until the connection is restored.
          </div>
        )}

        <section className="rounded-2xl border border-neutral-800 bg-neutral-900/50 p-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">
                Evolution Run Status
              </h2>
              <p className="text-xs text-neutral-400">
                {statusMessage} {socketStatus !== 'connected' ? `(${socketStatus})` : ''}
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs text-neutral-300">
              <div>
                Generation:{' '}
                <span className="text-neutral-100 font-semibold">
                  {latestGeneration ?? '--'}
                </span>
              </div>
              <div>
                Fitness:{' '}
                <span className="text-neutral-100 font-semibold">
                  {latestFitness !== null ? latestFitness.toFixed(2) : '--'}
                </span>
              </div>
              <div className="uppercase text-[10px] tracking-wide text-neutral-400">
                {status}
              </div>
              {status === 'running' && (
                <button
                  type="button"
                  onClick={handlePauseEvolution}
                  className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 text-[11px] hover:border-neutral-500 transition-colors"
                >
                  Pause
                </button>
              )}
              {status === 'paused' && (
                <button
                  type="button"
                  onClick={handleResumeEvolution}
                  className="px-3 py-1.5 rounded-lg border border-emerald-500/40 text-emerald-200 text-[11px] hover:border-emerald-400 transition-colors"
                >
                  Resume
                </button>
              )}
            </div>
          </div>
          <div className="mt-4">
            <div className="h-2 w-full rounded-full bg-neutral-800">
              <div
                className="h-2 rounded-full bg-blue-500 transition-all"
                style={{ width: `${Math.min(progress, 100)}%` }}
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-[11px] text-neutral-500">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-neutral-400">
            <button
              type="button"
              onClick={triggerReconnect}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Reconnect WebSocket
            </button>
            <span>Use this if updates stop flowing.</span>
          </div>
          {error && (
            <div className="mt-3 text-xs text-red-300 border border-red-500/30 bg-red-500/10 px-3 py-2 rounded-lg flex flex-wrap items-center justify-between gap-2">
              <span>{error}</span>
              {hasRetryPayload && (
                <button
                  type="button"
                  onClick={handleRetryStart}
                  className="px-3 py-1.5 rounded-lg border border-red-400/40 text-red-100 text-[11px] hover:border-red-300 transition-colors"
                >
                  Retry Start
                </button>
              )}
            </div>
          )}
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <Trophy className="h-4 w-4 text-amber-300" />
                <h2 className="text-xl font-semibold">Evolution Results</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Highlight the strongest variant and track fitness improvements.
              </p>
            </div>
            {winnerNode && (
              <div className="text-xs text-neutral-400">
                Best fitness{' '}
                <span className="text-neutral-100 font-semibold">
                  {formatMetric(winnerNode.data.fitness ?? winnerNode.data.score)}
                </span>
              </div>
            )}
          </div>

          {!nodes.length ? (
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-6 text-xs text-neutral-400">
              Start an evolution run to generate results and fitness data.
            </div>
          ) : !winnerNode ? (
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-6 text-xs text-neutral-400">
              No scored variants yet. Fitness metrics will appear as the judge
              evaluates descendants.
            </div>
          ) : (
            <div className="grid gap-4 xl:grid-cols-2">
              <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 space-y-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
                      Winner
                    </p>
                    <h3 className="text-lg font-semibold text-white">
                      {getNodeLabel(winnerNode.data)}
                    </h3>
                    <p className="text-[11px] text-neutral-500">
                      Generation {winnerNode.data.generation}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-1 rounded-full border text-[10px] uppercase tracking-[0.18em] ${
                      STATUS_BADGE_STYLES[winnerNode.data.status ?? 'evaluating']
                    }`}
                  >
                    {winnerNode.data.status}
                  </span>
                </div>
                {renderPreview(winnerNode, 'h-[240px]', 'No winner available.')}
                <div className="grid grid-cols-2 gap-3 text-xs text-neutral-400">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                      Fitness
                    </p>
                    <p className="text-sm font-semibold text-white">
                      {formatMetric(winnerNode.data.fitness)}
                    </p>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                      Score
                    </p>
                    <p className="text-sm font-semibold text-white">
                      {formatMetric(winnerNode.data.score)}
                    </p>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                      Status
                    </p>
                    <p className="text-sm font-semibold text-white">
                      {winnerNode.data.status}
                    </p>
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                      Created
                    </p>
                    <p className="text-sm font-semibold text-white">
                      {winnerNode.data.createdAt
                        ? new Date(winnerNode.data.createdAt).toLocaleString()
                        : '--'}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => openModal(winnerNode.data.id)}
                    className="px-3 py-1.5 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors"
                  >
                    Open Full Preview
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDownloadNodeHtml(winnerNode)}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" />
                    HTML
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDownloadNodeCss(winnerNode)}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" />
                    CSS
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDownloadNodePreview(winnerNode)}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Preview
                  </button>
                </div>
                <button
                  type="button"
                  onClick={handleToggleWinnerSource}
                  className="inline-flex items-center gap-2 text-xs text-neutral-300 hover:text-white transition-colors"
                >
                  {showWinnerSource ? 'Hide' : 'Show'} HTML/CSS
                </button>
                {showWinnerSource && (
                  <div className="rounded-xl border border-neutral-800 bg-neutral-950/60 p-4 space-y-3">
                    {winnerSourceLoading && (
                      <p className="text-xs text-neutral-400">
                        Loading HTML source…
                      </p>
                    )}
                    <div>
                      <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                        HTML
                      </p>
                      {winnerHtml ? (
                        <pre className="mt-2 text-[11px] text-neutral-200 bg-neutral-900/60 border border-neutral-800 rounded-lg p-3 max-h-48 overflow-auto whitespace-pre-wrap">
                          {winnerHtml}
                        </pre>
                      ) : (
                        <p className="mt-2 text-xs text-neutral-500">
                          HTML not available yet.
                        </p>
                      )}
                    </div>
                    <div>
                      <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                        CSS
                      </p>
                      {winnerCss ? (
                        <pre className="mt-2 text-[11px] text-neutral-200 bg-neutral-900/60 border border-neutral-800 rounded-lg p-3 max-h-40 overflow-auto whitespace-pre-wrap">
                          {winnerCss}
                        </pre>
                      ) : (
                        <p className="mt-2 text-xs text-neutral-500">
                          No embedded CSS found yet.
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
              <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 space-y-4">
                <div>
                  <p className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
                    Fitness Progression
                  </p>
                  <p className="text-xs text-neutral-400 mt-1">
                    Best fitness score per generation.
                  </p>
                </div>
                {fitnessTimeline.length ? (
                  <div className="h-32 flex items-end gap-2">
                    {fitnessTimeline.map((point) => {
                      const height = maxTimelineFitness
                        ? Math.max(12, (point.fitness / maxTimelineFitness) * 100)
                        : 12;
                      return (
                        <div
                          key={`gen-${point.generation}`}
                          className="flex flex-col items-center gap-1"
                        >
                          <div
                            className="w-6 rounded-md bg-blue-500/70"
                            style={{ height: `${height}%` }}
                            title={`Gen ${point.generation}: ${point.fitness.toFixed(2)}`}
                          />
                          <span className="text-[10px] text-neutral-500">
                            G{point.generation}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-xs text-neutral-500">
                    No fitness metrics yet.
                  </div>
                )}
                <div className="rounded-xl border border-neutral-800 bg-neutral-950/60 p-3 space-y-2">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                    Lineage Highlights
                  </p>
                  {lineageHighlights.length ? (
                    <div className="space-y-2">
                      {lineageHighlights.slice(0, 6).map((node) => (
                        <div
                          key={node.id}
                          className="flex items-center justify-between text-xs text-neutral-300"
                        >
                          <span>
                            Gen {node.data.generation} · {getNodeLabel(node.data)}
                          </span>
                          <span>{formatMetric(node.data.fitness ?? node.data.score)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-neutral-500">
                      No lineage highlights yet.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <GitCompare className="h-4 w-4 text-sky-300" />
                <h2 className="text-xl font-semibold">Comparison View</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Compare two variants side-by-side before promoting a branch.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                Preview
              </span>
              <div className="flex rounded-full border border-neutral-700 overflow-hidden text-xs">
                {(['live', 'cached'] as const).map((mode) => {
                  const isDisabled = mode === 'cached' && !cachePreviewsEnabled;
                  return (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setPreviewMode(mode)}
                      disabled={isDisabled}
                      aria-pressed={previewMode === mode}
                      className={`px-3 py-1.5 uppercase tracking-[0.16em] ${
                        previewMode === mode
                          ? 'bg-blue-500/20 text-blue-200'
                          : 'text-neutral-400 hover:text-neutral-200'
                      } ${isDisabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                    >
                      {mode}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
          {comparisonCandidates.length < 2 ? (
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-6 text-xs text-neutral-400">
              Generate at least two descendants to enable comparison.
            </div>
          ) : (
            <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4 space-y-4">
              <div className="flex flex-wrap items-center gap-3 text-xs text-neutral-400">
                <label className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
                  Primary
                </label>
                <select
                  aria-label="Primary comparison node"
                  value={comparePrimaryId ?? ''}
                  onChange={(event) => setComparePrimaryId(event.target.value)}
                  className="min-w-[220px] bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-200 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  {comparisonCandidates.map((node) => (
                    <option key={node.id} value={node.id}>
                      {getNodeLabel(node.data)} · Gen {node.data.generation} ·{' '}
                      {formatMetric(node.data.fitness ?? node.data.score)}
                    </option>
                  ))}
                </select>
                <label className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
                  Compare
                </label>
                <select
                  aria-label="Secondary comparison node"
                  value={compareSecondaryId ?? ''}
                  onChange={(event) =>
                    setCompareSecondaryId(event.target.value)
                  }
                  className="min-w-[220px] bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-200 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  {comparisonCandidates.map((node) => (
                    <option key={node.id} value={node.id}>
                      {getNodeLabel(node.data)} · Gen {node.data.generation} ·{' '}
                      {formatMetric(node.data.fitness ?? node.data.score)}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={handleSwapCompare}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors"
                >
                  <ArrowLeftRight className="h-3.5 w-3.5" />
                  Swap
                </button>
              </div>
              {!cachePreviewsEnabled && previewMode === 'cached' && (
                <p className="text-[11px] text-neutral-500">
                  Cached previews are disabled in settings.
                </p>
              )}
              <div className="grid gap-4 md:grid-cols-2">
                {renderComparisonCard(comparePrimaryNode, 'Primary')}
                {renderComparisonCard(compareSecondaryNode, 'Compare')}
              </div>
            </div>
          )}
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <Sparkles className="h-4 w-4 text-blue-300" />
                <h2 className="text-xl font-semibold">Evolution Engine</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Core genetic optimization settings and model configuration.
              </p>
            </div>
            <button
              type="button"
              onClick={resetEvolution}
              className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-neutral-700 text-neutral-300 hover:border-neutral-500 transition-colors"
            >
              <RotateCcw className="h-3 w-3" />
              Reset Evolution
            </button>
          </div>
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4 flex flex-wrap items-center justify-between gap-3 text-xs text-neutral-300">
            <div>
              <p className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
                Estimated Cost
              </p>
              <p className="mt-1 text-lg font-semibold text-white">
                ${estimatedCost.toFixed(2)}
              </p>
              <p className="text-[11px] text-neutral-500">
                Based on population × iterations.
              </p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
                Max Budget
              </p>
              <p
                className={`mt-1 text-lg font-semibold ${
                  isBudgetOver ? 'text-red-300' : 'text-white'
                }`}
              >
                ${maxBudgetValue.toFixed(2)}
              </p>
              {isBudgetOver && (
                <p className="text-[11px] text-red-300">
                  Estimate exceeds budget.
                </p>
              )}
            </div>
          </div>
          <EvolutionParameterForm
            parameters={evolutionParameterOverrides}
            values={evolutionInputs}
            onChange={setEvolutionInput}
          />
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <Sparkles className="h-4 w-4 text-pink-300" />
                <h2 className="text-xl font-semibold">Design Constraints</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Brand, palette, and layout constraints for the judge.
              </p>
            </div>
          </div>
          <EvolutionParameterForm
            parameters={evolutionConstraintParameters}
            values={evolutionInputs}
            onChange={setEvolutionInput}
          />
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <Image className="h-4 w-4 text-sky-300" />
                <h2 className="text-xl font-semibold">Preview & Storage</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Control cached thumbnails to balance speed and disk usage.
              </p>
            </div>
          </div>
          <EvolutionParameterForm
            parameters={evolutionPreviewParameters}
            values={evolutionInputs}
            onChange={setEvolutionInput}
          />
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-white">
                  Cached Preview Storage
                </p>
                <p className="text-xs text-neutral-400">
                  Tracks local thumbnail usage for the active evolution run.
                </p>
              </div>
              <button
                type="button"
                onClick={() => void fetchUsage(runId)}
                disabled={!runId || usageLoading}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-neutral-700 text-neutral-200 text-xs hover:border-neutral-500 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh Usage
              </button>
            </div>
            {!runId && (
              <div className="text-xs text-neutral-500">
                Start an evolution run to capture storage usage metrics.
              </div>
            )}
            {runId && (
              <div className="grid gap-3 md:grid-cols-3 text-xs text-neutral-300">
                <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-3">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                    Thumbnails
                  </p>
                  <p className="mt-1 text-lg font-semibold text-white">
                    {usage ? usage.thumbnailCount : '--'}
                  </p>
                  <p className="text-[11px] text-neutral-400">
                    {usage ? formatBytes(usage.thumbnailBytes) : '--'}
                  </p>
                </div>
                <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-3">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                    HTML Assets
                  </p>
                  <p className="mt-1 text-lg font-semibold text-white">
                    {usage ? usage.htmlCount : '--'}
                  </p>
                  <p className="text-[11px] text-neutral-400">
                    {usage ? formatBytes(usage.htmlBytes) : '--'}
                  </p>
                </div>
                <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-3">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-neutral-500">
                    Total Usage
                  </p>
                  <p className="mt-1 text-lg font-semibold text-white">
                    {usage ? formatBytes(usage.totalBytes) : '--'}
                  </p>
                  <p className="text-[11px] text-neutral-400">
                    {usage ? `${usage.totalAssets} assets` : '--'}
                  </p>
                </div>
              </div>
            )}
            {usageError && (
              <div className="text-xs text-red-300 border border-red-500/30 bg-red-500/10 px-3 py-2 rounded-lg">
                {usageError}
              </div>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={handleClearThumbnails}
                disabled={
                  !runId ||
                  usageLoading ||
                  !usage ||
                  usage.thumbnailCount === 0
                }
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-red-500/40 text-red-200 text-xs hover:border-red-400 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Purge Cached Previews
              </button>
              <span className="text-[11px] text-neutral-500">
                Removes stored thumbnail images without deleting run data.
              </span>
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <BarChart3 className="h-4 w-4 text-indigo-300" />
                <h2 className="text-xl font-semibold">Evolution History</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Review recent runs, switch between lineages, or delete old runs.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void fetchRuns()}
              disabled={runsLoading}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-neutral-700 text-neutral-200 text-xs hover:border-neutral-500 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh History
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-neutral-400">
            <label className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">
              Status Filter
            </label>
            <select
              value={runFilter}
              onChange={(event) => setRunFilter(event.target.value)}
              className="bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-200 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {runStatusOptions.map((statusOption) => (
                <option key={statusOption} value={statusOption}>
                  {statusOption === 'all'
                    ? 'All statuses'
                    : statusOption.replace('-', ' ')}
                </option>
              ))}
            </select>
          </div>
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-4 space-y-3">
            {runsLoading && (
              <div className="text-xs text-neutral-500">Loading evolution runs…</div>
            )}
            {!runsLoading && runsError && (
              <div className="text-xs text-red-300 border border-red-500/30 bg-red-500/10 px-3 py-2 rounded-lg">
                {runsError}
              </div>
            )}
            {!runsLoading && !runsError && filteredRuns.length === 0 && (
              <div className="text-xs text-neutral-500">
                No evolution runs found for this filter.
              </div>
            )}
            {pagedRuns.map((run) => {
              const isActive = runId === run.id;
              return (
                <div
                  key={run.id}
                  className={`rounded-xl border px-4 py-3 flex flex-wrap items-center justify-between gap-3 ${
                    isActive
                      ? 'border-blue-500/50 bg-blue-500/10'
                      : 'border-neutral-800 bg-neutral-900/40'
                  }`}
                >
                  <div>
                    <p className="text-sm font-semibold text-white">
                      {run.name || `Evolution ${run.evolutionId.slice(0, 8)}`}
                    </p>
                    <p className="text-[11px] text-neutral-500">
                      {new Date(run.createdAt).toLocaleString()} · {run.status}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleSelectRun(run)}
                      className="px-3 py-1.5 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors"
                    >
                      {isActive ? 'Viewing' : 'Load'}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDeleteRun(run)}
                      className="px-3 py-1.5 rounded-lg border border-red-500/40 text-xs text-red-200 hover:border-red-400 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              );
            })}
            {filteredRuns.length > RUNS_PER_PAGE && (
              <div className="flex items-center justify-between text-xs text-neutral-400">
                <button
                  type="button"
                  onClick={() => setRunPage((page) => Math.max(1, page - 1))}
                  disabled={runPage <= 1}
                  className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  Prev
                </button>
                <span>
                  Page {runPage} of {totalRunPages}
                </span>
                <button
                  type="button"
                  onClick={() =>
                    setRunPage((page) => Math.min(totalRunPages, page + 1))
                  }
                  disabled={runPage >= totalRunPages}
                  className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <Shield className="h-4 w-4 text-emerald-300" />
                <h2 className="text-xl font-semibold">Adversarial Testing</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Configure red/blue team simulations and attack strategies.
              </p>
            </div>
            <button
              type="button"
              onClick={resetAdversarial}
              className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-neutral-700 text-neutral-300 hover:border-neutral-500 transition-colors"
            >
              <RotateCcw className="h-3 w-3" />
              Reset Adversarial
            </button>
          </div>
          <EvolutionParameterForm
            parameters={adversarialParameters}
            values={adversarialInputs}
            onChange={setAdversarialInput}
          />
        </section>

        <section className="space-y-4 pb-6">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-white">
                <Workflow className="h-4 w-4 text-purple-300" />
                <h2 className="text-xl font-semibold">Decomposition Workflow</h2>
              </div>
              <p className="text-xs text-neutral-400">
                Plan task decomposition parameters and output formats.
              </p>
            </div>
            <button
              type="button"
              onClick={resetDecomposition}
              className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-neutral-700 text-neutral-300 hover:border-neutral-500 transition-colors"
            >
              <RotateCcw className="h-3 w-3" />
              Reset Decomposition
            </button>
          </div>
          <EvolutionParameterForm
            parameters={decompositionParameters}
            values={decompositionInputs}
            onChange={setDecompositionInput}
          />
        </section>
      </div>
      {showStartConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl border border-neutral-800 bg-[#0b0f1a] p-6 shadow-2xl">
            <h3 className="text-lg font-semibold text-white">
              Confirm Evolution Start
            </h3>
            <p className="mt-2 text-sm text-neutral-400">
              Review the estimated cost before launching this run.
            </p>
            <div className="mt-4 rounded-xl border border-neutral-800 bg-neutral-900/60 p-4 text-sm text-neutral-200">
              <div className="flex items-center justify-between">
                <span>Estimated cost</span>
                <span className="font-semibold text-white">
                  ${estimatedCost.toFixed(2)}
                </span>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-neutral-400">
                <span>Max budget</span>
                <span
                  className={isBudgetOver ? 'text-red-300' : 'text-neutral-200'}
                >
                  ${maxBudgetValue.toFixed(2)}
                </span>
              </div>
              <div className="mt-2 text-[11px] text-neutral-500">
                Population {populationValue} × Iterations {iterationsValue}
              </div>
            </div>
            <div className="mt-5 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={handleCancelStart}
                className="px-4 py-2 rounded-lg border border-neutral-700 text-neutral-200 text-sm hover:border-neutral-500 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmStart}
                className="px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm hover:bg-emerald-500 transition-colors"
              >
                Start Evolution
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
