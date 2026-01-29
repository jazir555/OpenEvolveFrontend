/**
 * Components Module - React Components Export
 *
 * Exports all React components for the OpenEvolve BubbleLab plugin.
 * These components integrate with BubbleLab's UI system.
 *
 * @module components
 * @version 1.0.0
 */

// Configuration Panels
import { EnhancedOpenEvolveConfigPanel } from './EnhancedOpenEvolveConfigPanel';
import { OpenEvolveConfigPanel } from './OpenEvolveConfigPanel';
import { OpenEvolveConfigPanel as ConfigPanel } from './config/OpenEvolveConfigPanel';
import { EnhancedOpenEvolveConfigPanel as EnhancedConfigPanel } from './config/EnhancedOpenEvolveConfigPanel';
import { EvolutionConfigPanel } from './config/EvolutionConfigPanel';
import { AdversarialConfigPanel } from './config/AdversarialConfigPanel';
import { DecompositionConfigPanel } from './config/DecompositionConfigPanel';
import { IntegrationConfigPanel } from './config/IntegrationConfigPanel';
import { KnowledgeConfigPanel } from './config/KnowledgeConfigPanel';
import { ProviderSettingsPanel } from './config/ProviderSettingsPanel';
import { ResearchQuestConfigPanel } from './config/ResearchQuestConfigPanel';
import { PyGraphistryConfigPanel } from './config/PyGraphistryConfigPanel';

// Pages
import { OpenEvolveDashboard } from './pages/OpenEvolveDashboard';
import { AnalyticsDashboard } from './pages/AnalyticsDashboard';
import { WorkflowBuilder } from './pages/WorkflowBuilder';
import { EvolutionPage } from './pages/EvolutionPage';
import { AdversarialPage } from './pages/AdversarialPage';
import { LeanAidePage } from './pages/LeanAidePage';
import { KnowledgeBasePage } from './pages/KnowledgeBasePage';

// Workflow Components
import { ConfigPanel as WorkflowConfigPanel } from './workflow/ConfigPanel';
import { ExecutionMonitor } from './workflow/ExecutionMonitor';
import { WorkflowCard } from './workflow/WorkflowCard';
import { WorkflowList } from './workflow/WorkflowList';
import { WorkflowTabs } from './workflow/WorkflowTabs';

// Analytics Components
import { ArtifactTable } from './analytics/ArtifactTable';
import { MetricCard } from './analytics/MetricCard';
import { PerformanceChart } from './analytics/PerformanceChart';
import { StatGrid } from './analytics/StatGrid';

// Knowledge Components
import { ArtifactDetail } from './knowledge/ArtifactDetail';
import { ArtifactEditor } from './knowledge/ArtifactEditor';
import { ArtifactList } from './knowledge/ArtifactList';
import { KnowledgeSearch } from './knowledge/KnowledgeSearch';

// LeanAide Components
import { ModelSelector } from './leanaide/ModelSelector';
import { ProgressTracker } from './leanaide/ProgressTracker';
import { ProofEditor } from './leanaide/ProofEditor';
import { VerificationDisplay } from './leanaide/VerificationDisplay';

// Shared Components
import { ProgressBar } from './shared/ProgressBar';
import { LiveLogViewer } from './shared/LiveLogViewer';
import { StatusBadge } from './shared/StatusBadge';
import { FormWrapper } from './shared/FormWrapper';
import { WelcomeBanner } from './shared/WelcomeBanner';
import { SystemHealthPanel } from './shared/SystemHealthPanel';

// BubbleLab UI Components
import {
  BubbleCard,
  BubbleField,
  BubbleInput,
  BubbleTextArea,
  BubbleSelect,
  BubbleButton,
  BubbleBadge,
  BubbleToggle,
  BubbleCheckbox,
} from './bubblelab';

export { ComponentErrorBoundary, withComponentBoundary } from './shared/ComponentErrorBoundary';
export { PageErrorBoundary } from './shared/PageErrorBoundary';
export { VizErrorBoundary } from './shared/VizErrorBoundary';
export { ApplicationErrorBoundary, withApplicationErrorBoundary } from './shared/ApplicationErrorBoundary';
export { default as ErrorReportingDashboard } from './shared/ErrorReportingDashboard';

export {
  EnhancedOpenEvolveConfigPanel,
  OpenEvolveConfigPanel,
  ConfigPanel,
  EnhancedConfigPanel,
  EvolutionConfigPanel,
  AdversarialConfigPanel,
  DecompositionConfigPanel,
  IntegrationConfigPanel,
  KnowledgeConfigPanel,
  ProviderSettingsPanel,
  ResearchQuestConfigPanel,
  PyGraphistryConfigPanel,
  OpenEvolveDashboard,
  AnalyticsDashboard,
  WorkflowBuilder,
  EvolutionPage,
  AdversarialPage,
  LeanAidePage,
  KnowledgeBasePage,
  WorkflowConfigPanel,
  ExecutionMonitor,
  WorkflowCard,
  WorkflowList,
  WorkflowTabs,
  ArtifactTable,
  MetricCard,
  PerformanceChart,
  StatGrid,
  ArtifactDetail,
  ArtifactEditor,
  ArtifactList,
  KnowledgeSearch,
  ModelSelector,
  ProgressTracker,
  ProofEditor,
  VerificationDisplay,
  ProgressBar,
  LiveLogViewer,
  StatusBadge,
  FormWrapper,
  WelcomeBanner,
  SystemHealthPanel,
  BubbleCard,
  BubbleField,
  BubbleInput,
  BubbleTextArea,
  BubbleSelect,
  BubbleButton,
  BubbleBadge,
  BubbleToggle,
  BubbleCheckbox,
};
