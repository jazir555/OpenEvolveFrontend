import React, { useState } from 'react';
import { 
  BubbleCard, 
  BubbleField, 
  BubbleInput, 
  BubbleSelect, 
  BubbleButton, 
  BubbleBadge,
  BubbleToggle
} from '../bubblelab';
import { useKnowledgeEngine } from '@/hooks/useKnowledgeEngine';
import { useEnhancedOpenEvolveConfig } from '@/hooks/useEnhancedOpenEvolveConfig';
import { KnowledgeConfig } from '@/types/knowledge-types';
import { toast } from 'react-toastify';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

const KnowledgeConfigPanelBase: React.FC = () => {
  const { indexProject, ingestDocument, getStatistics, loading } = useKnowledgeEngine();
  const { config, updateConfig } = useEnhancedOpenEvolveConfig();
  
  const knowledgeConfig = config?.knowledgeConfig || {
    projectPath: '.',
    targetStructure: 'Analyze code for concepts and relationships.',
    defaultMethod: 'hybrid',
    confidenceThreshold: 0.6,
    autoUpdateKnowledgeBase: true,
    enableKnowledgeGraphs: true,
  } as KnowledgeConfig;

  const [analyzing, setAnalyzing] = useState(false);
  const [healing, setHealing] = useState(false);
  const [synthesizing, setSynthesizing] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [documentPath, setDocumentPath] = useState('');
  const [stats, setStats] = useState<{ entity_count: number; relationship_count: number } | null>(null);

  const { selfHeal, synthesize, analyzeGraph } = useKnowledgeEngine();

  const handleSelfHeal = async () => {
    setHealing(true);
    try {
      const result = await selfHeal();
      if (result) {
        toast.success(`Self-healing complete: ${result.healed_count} items repaired`);
        handleRefreshStats();
      }
    } catch (error) {
      toast.error('Self-healing failed');
    } finally {
      setHealing(false);
    }
  };

  const handleSynthesize = async () => {
    setSynthesizing(true);
    try {
      const result = await synthesize();
      if (result && result.success) {
        toast.success(`Synthesis complete: ${result.meta_nodes_created} Meta-Nodes created`);
        handleRefreshStats();
      }
    } catch (error) {
      toast.error('Synthesis failed');
    } finally {
      setSynthesizing(false);
    }
  };

  const handleConfigChange = (updates: Partial<KnowledgeConfig>) => {
    updateConfig({
      knowledgeConfig: {
        ...knowledgeConfig,
        ...updates
      }
    });
  };

  const handleIndexProject = async () => {
    setIndexing(true);
    try {
      const result = await indexProject({
        projectPath: knowledgeConfig.projectPath,
        targetStructure: knowledgeConfig.targetStructure
      });
      if (result) {
        toast.success('Project indexing completed successfully');
        handleRefreshStats();
      }
    } catch (error) {
      toast.error(`Indexing failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIndexing(false);
    }
  };

  const handleIngestDocument = async () => {
    setIngesting(true);
    try {
      const result = await ingestDocument(documentPath);
      if (result) {
        toast.success(`Document ingested: ${result.content_length} characters extracted`);
        setDocumentPath('');
        handleRefreshStats();
      }
    } catch (error) {
      toast.error(`Ingestion failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIngesting(false);
    }
  };

  const handleRefreshStats = async () => {
    try {
      const newStats = await getStatistics();
      if (newStats) setStats(newStats);
    } catch (error) {
      errorLogger.logError(, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'error' } });
    }
  };

  React.useEffect(() => {
    handleRefreshStats();
  }, []);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <BubbleCard 
          title="Project Indexing" 
          description="Index a local directory to populate the knowledge engine with project-specific context."
        >
          <div className="space-y-4">
            <BubbleField label="Project Root Path">
              <BubbleInput 
                value={knowledgeConfig.projectPath}
                onChange={(e) => handleConfigChange({ projectPath: e.target.value })}
                placeholder="./"
              />
            </BubbleField>
            <BubbleField label="Target Analysis Structure">
              <BubbleInput 
                value={knowledgeConfig.targetStructure}
                onChange={(e) => handleConfigChange({ targetStructure: e.target.value })}
                placeholder="What concepts should the indexer focus on?"
              />
            </BubbleField>
            <BubbleButton 
              onClick={handleIndexProject} 
              disabled={indexing || !knowledgeConfig.projectPath}
              className="w-full"
            >
              {indexing ? 'Indexing Project...' : 'Start Indexing'}
            </BubbleButton>
          </div>
        </BubbleCard>

        <BubbleCard 
          title="Document Ingestion" 
          description="Ingest a single document (PDF, TXT) or a URL into the knowledge base."
        >
          <div className="space-y-4">
            <BubbleField label="File Path or URL">
              <BubbleInput 
                value={documentPath}
                onChange={(e) => setDocumentPath(e.target.value)}
                placeholder="e.g. https://example.com/doc.pdf"
              />
            </BubbleField>
            <BubbleButton 
              onClick={handleIngestDocument} 
              disabled={ingesting || !documentPath}
              variant="primary"
              className="w-full"
            >
              {ingesting ? 'Ingesting Document...' : 'Ingest Document'}
            </BubbleButton>
          </div>
        </BubbleCard>

        <BubbleCard 
          title="Knowledge Engine Status" 
          description="Current health and metrics of the backend knowledge engine."
        >
          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-sm font-medium">Entities</span>
              <BubbleBadge tone="info">{stats?.entity_count || 0}</BubbleBadge>
            </div>
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-sm font-medium">Relationships</span>
              <BubbleBadge tone="info">{stats?.relationship_count || 0}</BubbleBadge>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-sm font-medium">Persistence</span>
              <span className="text-xs text-green-600 font-bold">ACTIVE (knowledge_graph.json)</span>
            </div>
            <BubbleButton 
              variant="secondary" 
              onClick={handleRefreshStats}
              className="w-full"
            >
              Refresh Metrics
            </BubbleButton>
          </div>
        </BubbleCard>

        <BubbleCard 
          title="Autonomous Operations" 
          description="Proactive maintenance and self-improvement loops."
        >
          <div className="space-y-4">
            <p className="text-xs text-slate-500">Trigger the self-healing engine to identify and repair low-performing knowledge artifacts using Research Quests.</p>
            <BubbleButton 
              variant="primary" 
              onClick={handleSelfHeal}
              disabled={healing}
              className="w-full"
            >
              {healing ? 'Healing Knowledge Base...' : 'Run Self-Healing'}
            </BubbleButton>
            <BubbleButton 
              variant="primary" 
              onClick={handleSynthesize}
              disabled={synthesizing}
              className="w-full"
            >
              {synthesizing ? 'Synthesizing Architecture...' : 'Run Recursive Synthesis'}
            </BubbleButton>
            <BubbleButton 
              variant="secondary" 
              onClick={async () => {
                setAnalyzing(true);
                await analyzeGraph();
                setAnalyzing(false);
                toast.success('Advanced AI analysis triggered');
              }}
              disabled={analyzing}
              className="w-full"
            >
              {analyzing ? 'Analyzing Graph...' : 'Run Karate Club Analysis'}
            </BubbleButton>
          </div>
        </BubbleCard>
      </div>

      <BubbleCard 
        title="Extraction Settings" 
        description="Configure default behavior for knowledge extraction nodes."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <BubbleField label="Default Extraction Method">
              <BubbleSelect 
                value={knowledgeConfig.defaultMethod}
                onChange={(e) => handleConfigChange({ defaultMethod: e.target.value as any })}
              >
                <option value="pattern">Pattern Matching (Fast)</option>
                <option value="llm">LLM (High Fidelity)</option>
                <option value="hybrid">Hybrid (Optimized)</option>
              </BubbleSelect>
            </BubbleField>
            <BubbleField label="Confidence Threshold">
              <BubbleInput 
                type="number" 
                step="0.1" 
                min="0" 
                max="1" 
                value={knowledgeConfig.confidenceThreshold}
                onChange={(e) => handleConfigChange({ confidenceThreshold: parseFloat(e.target.value) })}
              />
            </BubbleField>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Auto-Update Knowledge Base</p>
                <p className="text-xs text-gray-500">Automatically save extracted entities to the graph.</p>
              </div>
              <BubbleToggle 
                checked={knowledgeConfig.autoUpdateKnowledgeBase}
                onChange={(checked) => handleConfigChange({ autoUpdateKnowledgeBase: checked })}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Enable Knowledge Graphs</p>
                <p className="text-xs text-gray-500">Perform graph-based enrichment during extraction.</p>
              </div>
              <BubbleToggle 
                checked={knowledgeConfig.enableKnowledgeGraphs}
                onChange={(checked) => handleConfigChange({ enableKnowledgeGraphs: checked })}
              />
            </div>
          </div>
        </div>
      </BubbleCard>
    </div>
  );
};

export const KnowledgeConfigPanel = withComponentBoundary(KnowledgeConfigPanelBase, 'KnowledgeConfigPanel');