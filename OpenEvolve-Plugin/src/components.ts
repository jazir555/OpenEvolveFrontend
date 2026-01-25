/**
 * OpenEvolve BubbleLab Components Index
 * 
 * Exports all React components that are compatible with the BubbleLab plugin system
 */

export { default as OpenEvolveDashboard } from './pages/OpenEvolveDashboard';
export { default as WorkflowOrchestrator } from './pages/WorkflowOrchestrator';
export { default as EvolutionPage } from './pages/EvolutionPage';
export { default as AdversarialPage } from './pages/AdversarialPage';
export { default as KnowledgeBasePage } from './pages/KnowledgeBasePage';
export { default as WorkflowBuilder } from './pages/WorkflowBuilder';

// Export BubbleLab-specific UI components
export { default as BubbleButton } from './components/bubblelab/BubbleButton';
export { default as BubbleCard } from './components/bubblelab/BubbleCard';
export { default as BubbleInput } from './components/bubblelab/BubbleInput';
export { default as BubbleSelect } from './components/bubblelab/BubbleSelect';
export { default as BubbleTabs, BubbleTab } from './components/bubblelab/BubbleTabs';

// Export utility functions
export { gracefulErrorHandler } from './utils/gracefulErrorHandler';