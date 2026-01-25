/**
 * OpenEvolve BubbleLab Plugin - Main Export Index
 * 
 * This file exports all components and functions needed for the BubbleLab plugin system.
 * It consolidates all the converted Streamlit UI components into a single import point.
 */

// Export main application components
export { default as OpenEvolveDashboard } from './pages/OpenEvolveDashboard';
export { default as WorkflowOrchestrator } from './pages/WorkflowOrchestrator';
export { default as EvolutionPage } from './pages/EvolutionPage';
export { default as AdversarialPage } from './pages/AdversarialPage';
export { default as KnowledgeBasePage } from './pages/KnowledgeBasePage';
export { default as WorkflowBuilder } from './pages/WorkflowBuilder';
export { default as AnalyticsDashboard } from './pages/AnalyticsDashboard';
export { default as AdvancedMonitoringDashboard } from './pages/AdvancedMonitoringDashboard';
export { default as UIComponents } from './pages/UIComponents';
export { default as MainApplication } from './pages/MainApplication';

// Export BubbleLab-specific UI components
export { default as BubbleButton } from './components/bubblelab/BubbleButton';
export { default as BubbleCard } from './components/bubblelab/BubbleCard';
export { default as BubbleInput } from './components/bubblelab/BubbleInput';
export { default as BubbleSelect } from './components/bubblelab/BubbleSelect';
export { default as BubbleTabs, BubbleTab } from './components/bubblelab/BubbleTabs';

// Export layout components
export { default as MainLayout } from './components/MainLayout';
export { default as Sidebar } from './components/Sidebar';

// Export utility functions
export { gracefulErrorHandler } from './utils/gracefulErrorHandler';

// Export plugin definition
export { OpenEvolvePlugin } from './plugin';
export { OpenEvolvePlugin as default } from './plugin';

// Export types and interfaces
export type { 
  PluginDefinition,
  WorkflowState,
  SubProblem,
  SolutionAttempt,
  CritiqueReport,
  VerificationReport
} from './types';