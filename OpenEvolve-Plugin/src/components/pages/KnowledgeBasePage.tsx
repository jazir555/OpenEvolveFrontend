// @ts-nocheck
import React, { useState, useEffect, useCallback } from 'react';
import { 
  BubbleCard, 
  BubbleButton, 
  BubbleInput, 
  BubbleBadge,
  BubbleTabs,
  BubbleTable,
  BubbleLoading
} from '../bubblelab';
import { useKnowledgeEngine } from '@/hooks/useKnowledgeEngine';
import { ArtifactList } from '@/components/knowledge/ArtifactList';
import { KnowledgeSearch } from '@/components/knowledge/KnowledgeSearch';
import { ArtifactEditor } from '@/components/knowledge/ArtifactEditor';
import { ArtifactDetail } from '@/components/knowledge/ArtifactDetail';
import { PyGraphistryViz } from '../visualization/PyGraphistryViz';
import { toast } from 'react-toastify';
import { PageErrorBoundary } from '@/components/shared/PageErrorBoundary';
import { withPageBoundary } from '../shared/PageErrorBoundary';

/**
 * Unified Knowledge Base Page
 * Integrates Factual, Semantic, Agentic, and Temporal memory systems.
 */
const KnowledgeBasePageBase: React.FC = () => {
  const [activeTab, setActiveTab] = useState('list');
  const [view, setView] = useState<'list' | 'search' | 'create' | 'detail' | 'graph'>('list');
  const [selectedArtifact, setSelectedArtifact] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [unifiedResults, setUnifiedResults] = useState<any>(null);
  const [isResearching, setIsResearching] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);

  const { 
    artifacts, 
    graphData, 
    loading, 
    error, 
    getArtifacts, 
    getGraph, 
    ingest,
    updateArtifact,
    deleteArtifact,
    unifiedSearch, 
    distillSkill, 
    deepResearch, 
    verifyFact,
    selfHeal,
    synthesize
  } = useKnowledgeEngine();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      await getArtifacts();
      await getGraph();
    } catch (err) {
      setPageError('Failed to load knowledge data.');
    }
  };

  const handleVerify = async (text: string) => {
    toast.info('Starting Lean 4 formalization...');
    try {
      const result = await verifyFact(text);
      if (result && result.success) {
        toast.success('Fact formalized and verified against Lean 4 kernel');
      } else {
        toast.error('Verification failed: claim is too ambiguous or service unavailable');
      }
    } catch (err) {
      toast.error('Verification failed: service unavailable');
    }
  };

  const handleUnifiedSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const results = await unifiedSearch(searchQuery);
      if (results) {
        setUnifiedResults(results);
        setActiveTab('search');
        setView('search');
      }
    } catch (err) {
      toast.error('Unified search failed.');
    }
  };

  const handleDeepResearch = async () => {
    if (!searchQuery.trim()) return;
    setIsResearching(true);
    toast.info('Starting Deep Research Agent... This may take a minute.');
    try {
      const result = await deepResearch(searchQuery);
      if (result && result.success) {
        toast.success(`Research complete! Found ${result.findings_count} findings.`);
        handleUnifiedSearch();
      } else {
        toast.error(`Research failed: ${result?.message || 'Unknown error'}`);
      }
    } finally {
      setIsResearching(false);
    }
  };

  const handleDistill = async (id: string) => {
    const success = await distillSkill(id);
    if (success) {
      toast.success('Strategy distilled into ACE skillbook');
    } else {
      toast.error('Distillation failed: insufficient quality or missing ACE');
    }
  };

  const handleSelfHeal = async () => {
    toast.info('Triggering autonomous self-healing...');
    const result = await selfHeal();
    if (result && result.healed_count > 0) {
      toast.success(`Self-healing complete: repaired ${result.healed_count} entities.`);
      loadData();
    } else {
      toast.info('Self-healing finished: no weak entities found.');
    }
  };

  const handleSynthesize = async () => {
    toast.info('Starting recursive knowledge synthesis...');
    const result = await synthesize();
    if (result && result.success) {
      toast.success(`Synthesis complete: created ${result.meta_nodes_created} Meta-Nodes.`);
      loadData();
    } else {
      toast.error('Synthesis failed.');
    }
  };

  const handleDelete = async (artifactId: string) => {
    if (confirm('Are you sure you want to delete this artifact?')) {
      try {
        await deleteArtifact(artifactId);
        toast.success('Artifact deleted.');
        const selectedId = selectedArtifact?.id || selectedArtifact?.artifact_id;
        if (selectedId === artifactId) {
          setSelectedArtifact(null);
          setView('list');
        }
      } catch (err) {
        toast.error('Failed to delete artifact.');
      }
    }
  };

  const handleSaveArtifact = async (data: any) => {
    try {
      const artifactId = selectedArtifact?.id || selectedArtifact?.artifact_id;
      if (artifactId) {
        await updateArtifact(artifactId, {
          ...data,
          title: data.name,
        });
        toast.success('Artifact updated.');
      } else {
        await ingest({
          title: data.name,
          content: data.content,
          tags: data.tags,
          metadata: {
            description: data.description,
            type: data.type,
          },
        });
        toast.success('Artifact created.');
      }
      setSelectedArtifact(null);
      setView('list');
    } catch (err) {
      toast.error('Failed to save artifact.');
    }
  };

  const tabs = [
    { id: 'list', label: 'Entity List', icon: 'list' },
    { id: 'graph', label: 'Graph View', icon: 'share-2' },
    { id: 'skills', label: 'ACE Skills', icon: 'zap' },
    { id: 'search', label: 'Unified Search', icon: 'search' }
  ];

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    if (tabId === 'list') setView('list');
    if (tabId === 'graph') setView('graph');
    if (tabId === 'search') setView('search');
  };

  return (
    <PageErrorBoundary label="Knowledge base">
      <div className="knowledge-base-page min-h-screen bg-slate-50">
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Knowledge Base</h1>
              <p className="text-sm text-slate-600">Unified memory, deep research, and agentic skills</p>
            </div>

            <div className="flex items-center gap-2">
              <div className="relative w-64">
                <BubbleInput 
                  placeholder="Search facts, skills, solutions..." 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleUnifiedSearch()}
                />
              </div>
              <BubbleButton onClick={handleUnifiedSearch} disabled={loading || isResearching}>
                Search
              </BubbleButton>
              <BubbleButton variant="secondary" onClick={handleDeepResearch} disabled={loading || isResearching || !searchQuery.trim()}>
                {isResearching ? 'Researching...' : 'Deep Research'}
              </BubbleButton>
              <BubbleButton
                onClick={() => {
                  setSelectedArtifact(null);
                  setView('create');
                }}
                variant="primary"
              >
                + New
              </BubbleButton>
            </div>
          </div>
        </header>

        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <BubbleTabs 
              tabs={tabs} 
              activeTab={activeTab} 
              onChange={handleTabChange} 
            />
            
            <div className="flex gap-2">
              <BubbleButton size="sm" variant="ghost" onClick={handleSelfHeal}>
                Self-Heal
              </BubbleButton>
              <BubbleButton size="sm" variant="ghost" onClick={handleSynthesize}>
                Synthesize
              </BubbleButton>
            </div>
          </div>

          {(pageError || error) && (
            <div className="mb-6 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {pageError || (error instanceof Error ? error.message : String(error))}
            </div>
          )}

          {view === 'list' && activeTab === 'list' && (
            <ArtifactList
              artifacts={artifacts}
              onArtifactSelect={(artifact) => {
                setSelectedArtifact(artifact);
                setView('detail');
              }}
              onArtifactDelete={handleDelete}
              onVerify={handleVerify}
              onDistill={handleDistill}
            />
          )}

          {view === 'graph' && activeTab === 'graph' && (
            <BubbleCard className="h-[700px] p-0 overflow-hidden relative">
              <div className="absolute top-4 left-4 z-10 bg-white/80 backdrop-blur p-2 rounded border text-xs">
                Interactive Graphistry Visualization
              </div>
              <PyGraphistryViz
                nodes={graphData?.nodes ?? []}
                edges={graphData?.edges ?? []}
                height="100%"
              />
            </BubbleCard>
          )}

          {activeTab === 'skills' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <BubbleCard title="Learned Strategies" description="Skills distilled from successful executions.">
                <div className="space-y-4">
                  {(unifiedResults?.agentic_skills || []).length > 0 ? (
                    unifiedResults.agentic_skills.map((skill: any, i: number) => (
                    <div key={i} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                      <p className="text-sm font-medium text-slate-800">{skill.strategy}</p>
                      <div className="mt-2 flex items-center gap-2">
                        <BubbleBadge tone="success">Usage: {skill.helpful}</BubbleBadge>
                        <span className="text-xs text-slate-400">Distilled Strategy</span>
                      </div>
                    </div>
                  ))
                  ) : (
                    <p className="text-sm text-slate-500 italic">
                      Perform a unified search to see relevant skills.
                    </p>
                  )}
                </div>
              </BubbleCard>
              
              <BubbleCard title="ACE Engine Status" description="Learning and adaptation metrics.">
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-sm text-slate-600">Active Learning</span>
                    <BubbleBadge tone="success">ENABLED</BubbleBadge>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-sm text-slate-600">Skillbook Version</span>
                    <span className="text-sm font-mono font-bold">v2.1 (Toon)</span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-sm text-slate-600">Improvement Target</span>
                    <span className="text-sm font-bold text-indigo-600">20% - 35%</span>
                  </div>
                </div>
              </BubbleCard>
            </div>
          )}

          {view === 'search' && (
            <div className="space-y-6">
              {unifiedResults && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <BubbleCard title="Facts (Graph Entities)">
                    <div className="space-y-3">
                      {(unifiedResults.factual_entities || []).length > 0 ? (
                        unifiedResults.factual_entities.map((e: any) => (
                        <div key={e.id} className="p-2 border rounded hover:bg-slate-50 cursor-pointer" onClick={() => { setSelectedArtifact(e); setView('detail'); }}>
                          <p className="font-bold text-sm">{e.id}</p>
                          <p className="text-xs text-slate-500 truncate">{e.description || e.content || JSON.stringify(e)}</p>
                        </div>
                      ))
                      ) : (
                        <p className="text-xs text-slate-400 italic">No facts found for this query.</p>
                      )}
                    </div>
                  </BubbleCard>
                  
                  <BubbleCard title="Expertise (ACE Skills)">
                    <div className="space-y-3">
                      {(unifiedResults.agentic_skills || []).length > 0 ? (
                        unifiedResults.agentic_skills.map((s: any, i: number) => (
                        <div key={i} className="p-2 bg-indigo-50 border border-indigo-100 rounded">
                          <p className="text-sm text-indigo-900">{s.strategy}</p>
                        </div>
                      ))
                      ) : (
                        <p className="text-xs text-slate-400 italic">No skills found for this query.</p>
                      )}
                    </div>
                  </BubbleCard>

                  <BubbleCard title="Research (Ragbits RAG)">
                    <div className="space-y-3">
                      {(unifiedResults.semantic_solutions || []).length > 0 ? (
                        unifiedResults.semantic_solutions.map((r: any, i: number) => (
                        <div key={i} className="p-2 border rounded">
                          <p className="text-sm line-clamp-3">{r.content}</p>
                          <p className="text-[10px] text-slate-400 mt-1">
                            Relevance: {Math.round((r.score ?? 0) * 100)}%
                          </p>
                        </div>
                      ))
                      ) : (
                        <p className="text-xs text-slate-400 italic">No research snippets found.</p>
                      )}
                    </div>
                  </BubbleCard>

                  <BubbleCard title="Temporal Insights (Graphiti)">
                    <div className="space-y-3">
                      {(unifiedResults.temporal_insights?.nodes || []).length > 0 ? (
                        unifiedResults.temporal_insights.nodes.map((n: any) => (
                        <div key={n.id} className="p-2 bg-amber-50 border border-amber-100 rounded">
                          <p className="font-bold text-sm text-amber-900">{n.label}</p>
                          <p className="text-xs text-amber-700">{n.summary}</p>
                        </div>
                      ))
                      ) : (
                        <p className="text-xs text-slate-400 italic">
                          No temporal patterns found for this query.
                        </p>
                      )}
                    </div>
                  </BubbleCard>
                </div>
              )}
              {!unifiedResults && <div className="text-center py-20 bg-white rounded-xl border border-dashed border-slate-300">
                <p className="text-slate-400 italic">Enter a query above to perform a unified search.</p>
              </div>}
            </div>
          )}

          {view === 'create' && (
            <div className="max-w-3xl mx-auto space-y-4">
              <BubbleButton onClick={() => setView('list')} variant="ghost">
                {'<- Back'}
              </BubbleButton>
              <BubbleCard title="Create New Artifact">
                <ArtifactEditor
                  artifact={selectedArtifact ?? undefined}
                  onSave={handleSaveArtifact}
                  onCancel={() => setView('list')}
                  types={['Evolution', 'Workflow', 'Proof', 'Model', 'Dataset']}
                />
              </BubbleCard>
            </div>
          )}

          {view === 'detail' && selectedArtifact && (
            <div className="max-w-4xl mx-auto space-y-4">
              <BubbleButton onClick={() => setView('list')} variant="ghost">
                {'<- Back'}
              </BubbleButton>
              <ArtifactDetail
                artifact={selectedArtifact}
                onEdit={() => setView('create')}
                onDelete={() => handleDelete(selectedArtifact.id || selectedArtifact.artifact_id)}
              />
            </div>
          )}

          {loading && (
            <div className="fixed inset-0 bg-white/50 backdrop-blur-sm flex items-center justify-center z-50">
              <BubbleLoading label="Syncing with Knowledge Engine..." />
            </div>
          )}
        </div>
      </div>
    </PageErrorBoundary>
  );
};

export const KnowledgeBasePage = withPageBoundary(KnowledgeBasePageBase, 'KnowledgeBasePage');
