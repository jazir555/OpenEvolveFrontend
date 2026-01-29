import { useEffect, useMemo, useState } from 'react';
import JSZip from 'jszip';
import { toast } from 'react-toastify';
import {
  useEvolutionGraphStore,
  type EvolutionNodeData,
} from '@/stores/evolutionGraphStore';
import { useEvolutionSettingsStore } from '@/stores/evolutionSettingsStore';

interface EvolutionPreviewModalProps {
  node: EvolutionNodeData;
  onClose: () => void;
}

type PreviewTab = 'preview' | 'html' | 'meta';

export function EvolutionPreviewModal({
  node,
  onClose,
}: EvolutionPreviewModalProps) {
  const [activeTab, setActiveTab] = useState<PreviewTab>('preview');
  const previewMode = useEvolutionGraphStore((s) => s.previewMode);
  const setPreviewMode = useEvolutionGraphStore((s) => s.setPreviewMode);
  const nodes = useEvolutionGraphStore((s) => s.nodes);
  const openModal = useEvolutionGraphStore((s) => s.openModal);
  const cachePreviewsEnabled = useEvolutionSettingsStore((s) =>
    Boolean(s.evolutionInputs.cachePreviews ?? true)
  );
  const [htmlSource, setHtmlSource] = useState<string | null>(null);
  const [htmlError, setHtmlError] = useState<string | null>(null);

  const resolveHtml = async (): Promise<string | null> => {
    if (node.html) return node.html;
    if (htmlSource) return htmlSource;
    if (!node.htmlUrl) return null;
    try {
      const res = await fetch(node.htmlUrl);
      const text = await res.text();
      setHtmlSource(text);
      return text;
    } catch {
      toast.error('Failed to load HTML content.');
      return null;
    }
  };

  const extractCss = (html: string): string | null => {
    const match = html.match(/<style[^>]*>([\s\S]*?)<\/style>/i);
    return match ? match[1].trim() : null;
  };

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadHtml = async () => {
    const html = await resolveHtml();
    if (!html) {
      toast.error('HTML not available for this node.');
      return;
    }
    downloadBlob(new Blob([html], { type: 'text/html' }), `${node.id}.html`);
    toast.success('HTML download started.');
  };

  const handleDownloadCss = async () => {
    const html = await resolveHtml();
    if (!html) {
      toast.error('HTML not available for this node.');
      return;
    }
    const css = extractCss(html);
    if (!css) {
      toast.info('No embedded CSS found in this HTML.');
      return;
    }
    downloadBlob(new Blob([css], { type: 'text/css' }), `${node.id}.css`);
    toast.success('CSS download started.');
  };

  const handleDownloadMetadata = () => {
    downloadBlob(
      new Blob([metadata], { type: 'application/json' }),
      `${node.id}.json`
    );
    toast.success('Metadata download started.');
  };

  const handleExportZip = async () => {
    const html = await resolveHtml();
    if (!html) {
      toast.error('HTML not available for this node.');
      return;
    }
    const css = extractCss(html);
    const zip = new JSZip();
    zip.file(`${node.id}.html`, html);
    if (css) {
      zip.file(`${node.id}.css`, css);
    }
    zip.file(`${node.id}.json`, metadata);
    if (node.thumbnailUrl) {
      try {
        const response = await fetch(node.thumbnailUrl);
        const blob = await response.blob();
        zip.file(`${node.id}-preview.png`, blob);
      } catch {
        // Ignore missing preview assets.
      }
    }
    const blob = await zip.generateAsync({ type: 'blob' });
    downloadBlob(blob, `${node.id}.zip`);
    toast.success('ZIP export started.');
  };

  const metadata = useMemo(() => {
    const payload = {
      id: node.id,
      generation: node.generation,
      status: node.status,
      fitness: node.fitness,
      score: node.score,
      label: node.label,
      metadata: node.metadata,
    };
    return JSON.stringify(payload, null, 2);
  }, [node]);

  const hasHtml = Boolean(node.htmlUrl || node.html);
  const hasCached = cachePreviewsEnabled && Boolean(node.thumbnailUrl);
  const stats = useMemo(
    () => ({
      status: node.status ?? 'evaluating',
      fitness:
        node.fitness == null || Number.isNaN(node.fitness)
          ? '--'
          : node.fitness.toFixed(2),
      score:
        node.score == null || Number.isNaN(node.score)
          ? '--'
          : node.score.toFixed(2),
    }),
    [node.fitness, node.score, node.status]
  );
  const siblings = useMemo(() => {
    if (!nodes.length) return [];
    const group = nodes.filter((item) => item.data.parentId === node.parentId);
    return [...group].sort((a, b) => {
      const timeA = a.data.createdAt ? Date.parse(a.data.createdAt) : 0;
      const timeB = b.data.createdAt ? Date.parse(b.data.createdAt) : 0;
      if (timeA !== timeB) return timeA - timeB;
      return a.data.id.localeCompare(b.data.id);
    });
  }, [nodes, node.parentId]);
  const currentIndex = siblings.findIndex((item) => item.data.id === node.id);
  const previousNode = currentIndex > 0 ? siblings[currentIndex - 1] : null;
  const nextNode =
    currentIndex >= 0 && currentIndex < siblings.length - 1
      ? siblings[currentIndex + 1]
      : null;
  const agentScores = Array.isArray(node.metadata?.scores)
    ? (node.metadata?.scores as Array<{
        name?: string;
        score?: number;
        reasoning?: string;
      }>)
    : null;
  const improvements = Array.isArray(node.metadata?.improvements)
    ? (node.metadata?.improvements as string[])
    : null;

  useEffect(() => {
    if (activeTab !== 'html') return;
    if (node.html) {
      setHtmlSource(node.html);
      setHtmlError(null);
      return;
    }
    if (!node.htmlUrl) {
      setHtmlSource(null);
      setHtmlError(null);
      return;
    }
    let cancelled = false;
    setHtmlError(null);
    fetch(node.htmlUrl)
      .then((res) => res.text())
      .then((text) => {
        if (cancelled) return;
        setHtmlSource(text);
      })
      .catch(() => {
        if (cancelled) return;
        setHtmlError('Failed to load HTML source.');
      });
    return () => {
      cancelled = true;
    };
  }, [activeTab, node.html, node.htmlUrl]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-5xl max-h-[90vh] bg-[#0b0f1a] border border-neutral-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800">
          <div>
            <p className="text-sm text-neutral-400">Evolution Variant</p>
            <h3 className="text-lg font-semibold text-white">
              {node.label || `Generation ${node.generation}`}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => previousNode && openModal(previousNode.id)}
              disabled={!previousNode}
              className="px-3 py-1.5 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Prev
            </button>
            <button
              type="button"
              onClick={() => nextNode && openModal(nextNode.id)}
              disabled={!nextNode}
              className="px-3 py-1.5 rounded-lg border border-neutral-700 text-xs text-neutral-200 hover:border-neutral-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
            <button
              type="button"
              onClick={onClose}
              className="text-neutral-400 hover:text-white transition-colors"
            >
              Close
            </button>
          </div>
        </div>
        <div className="px-6 py-3 border-b border-neutral-800 flex flex-wrap items-center gap-3 text-xs text-neutral-300">
          <span className="uppercase text-[10px] tracking-[0.2em] text-neutral-500">
            Status
          </span>
          <span className="font-semibold text-neutral-100">{stats.status}</span>
          <span className="ml-3 uppercase text-[10px] tracking-[0.2em] text-neutral-500">
            Fitness
          </span>
          <span className="font-semibold text-neutral-100">{stats.fitness}</span>
          <span className="ml-3 uppercase text-[10px] tracking-[0.2em] text-neutral-500">
            Score
          </span>
          <span className="font-semibold text-neutral-100">{stats.score}</span>
        </div>
        <div className="px-6 py-3 border-b border-neutral-800 flex items-center gap-2">
          {(['preview', 'html', 'meta'] as PreviewTab[]).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold uppercase tracking-wide transition-colors ${
                activeTab === tab
                  ? 'bg-blue-500/20 text-blue-200 border border-blue-500/40'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              {tab}
            </button>
          ))}
          <div className="ml-auto flex items-center gap-3">
            <div className="flex rounded-full border border-neutral-700 overflow-hidden text-[10px] uppercase tracking-[0.18em]">
              {(['live', 'cached'] as const).map((mode) => {
                const isDisabled = mode === 'cached' && !cachePreviewsEnabled;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setPreviewMode(mode)}
                    disabled={isDisabled}
                    className={`px-3 py-1.5 ${
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
            {!cachePreviewsEnabled && (
              <span className="text-[10px] text-neutral-500">
                Cached previews disabled
              </span>
            )}
          </div>
        </div>
        <div className="px-6 py-3 border-b border-neutral-800 flex flex-wrap items-center gap-2 text-xs">
          <button
            type="button"
            onClick={() => void handleDownloadHtml()}
            className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors"
          >
            Download HTML
          </button>
          <button
            type="button"
            onClick={() => void handleDownloadCss()}
            className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors"
          >
            Download CSS
          </button>
          <button
            type="button"
            onClick={() => void handleExportZip()}
            className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors"
          >
            Export ZIP
          </button>
          <button
            type="button"
            onClick={handleDownloadMetadata}
            className="px-3 py-1.5 rounded-lg border border-neutral-700 text-neutral-200 hover:border-neutral-500 transition-colors"
          >
            Download Metadata
          </button>
        </div>
        <div className="flex-1 min-h-0 overflow-auto">
          {activeTab === 'preview' && (
            <div className="p-6">
              {previewMode === 'live' ? (
                hasHtml ? (
                  <iframe
                    title={node.label || 'Evolution HTML preview'}
                    src={node.htmlUrl || undefined}
                    srcDoc={node.htmlUrl ? undefined : node.html || undefined}
                    className="w-full h-[520px] rounded-xl border border-neutral-800 bg-white"
                    sandbox="allow-scripts allow-same-origin"
                  />
                ) : (
                  <div className="h-[360px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400">
                    Live HTML preview not available yet.
                  </div>
                )
              ) : !cachePreviewsEnabled ? (
                <div className="h-[360px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400">
                  Cached previews are disabled in settings.
                </div>
              ) : hasCached ? (
                <img
                  src={node.thumbnailUrl ?? undefined}
                  alt={node.label || 'Preview'}
                  className="w-full rounded-xl border border-neutral-800"
                />
              ) : (
                <div className="h-[360px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400">
                  Cached preview not available yet.
                </div>
              )}
            </div>
          )}
          {activeTab === 'html' && (
            <div className="p-6">
              {htmlError ? (
                <div className="h-[360px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400">
                  {htmlError}
                </div>
              ) : hasHtml ? (
                <pre className="text-xs text-neutral-200 bg-neutral-900/60 border border-neutral-800 rounded-xl p-4 whitespace-pre-wrap">
                  {htmlSource ?? 'Loading HTML source...'}
                </pre>
              ) : (
                <div className="h-[360px] rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-400">
                  HTML source not available yet.
                </div>
              )}
            </div>
          )}
          {activeTab === 'meta' && (
            <div className="p-6 space-y-6">
              <div>
                <h4 className="text-xs uppercase tracking-[0.2em] text-neutral-400">
                  Agent Scores
                </h4>
                {agentScores && agentScores.length ? (
                  <div className="mt-3 space-y-3">
                    {agentScores.map((score, index) => (
                      <div
                        key={`${score.name ?? 'agent'}-${index}`}
                        className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3"
                      >
                        <div className="flex items-center justify-between text-xs text-neutral-300">
                          <span className="font-semibold text-neutral-100">
                            {score.name ?? `Agent ${index + 1}`}
                          </span>
                          <span>{score.score?.toFixed(2) ?? '--'}</span>
                        </div>
                        {score.reasoning && (
                          <p className="mt-2 text-[11px] text-neutral-400 whitespace-pre-wrap">
                            {score.reasoning}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-neutral-500">
                    No agent scores available yet.
                  </p>
                )}
              </div>
              <div>
                <h4 className="text-xs uppercase tracking-[0.2em] text-neutral-400">
                  Improvement Suggestions
                </h4>
                {improvements && improvements.length ? (
                  <ul className="mt-3 space-y-2 text-xs text-neutral-300">
                    {improvements.map((item, index) => (
                      <li
                        key={`${item}-${index}`}
                        className="rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-2"
                      >
                        {item}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-xs text-neutral-500">
                    No improvement suggestions captured.
                  </p>
                )}
              </div>
              <pre className="text-xs text-neutral-200 bg-neutral-900/60 border border-neutral-800 rounded-xl p-4 whitespace-pre-wrap">
                {metadata}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
