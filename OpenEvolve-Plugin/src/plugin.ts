/**
 * OpenEvolve Plugin - Unified Plugin Definition
 *
 * This plugin is the result of merging THREE OpenEvolve plugin implementations:
 * - Plugin 1: OpenEvolve-Plugin/ (UI components, services, stores)
 * - Plugin 2: openevolve-bubblelab-plugin/ (node system, config panels)
 * - Plugin 3: BubbleLab embedded (PluginDefinition, service definitions)
 *
 * ZERO FEATURE LOSS - All features from all three plugins are present
 */

import { PluginDefinition } from './types/plugin';
import { OpenEvolvePlugin as BubbleLabPluginDefinition } from './core/plugin/BubbleLabPluginDefinition';
import { gracefulErrorHandler } from './utils/gracefulErrorHandler';

export const OpenEvolvePlugin: PluginDefinition = {
  ...BubbleLabPluginDefinition,

  // Plugin capabilities
  capabilities: {
    workflows: true,
    analytics: true,
    knowledgeBase: true,
    leanAide: true,
    evolution: true,
    adversarial: true,
    maker: true,
    mdap: true,
    decomposition: true,
    crewai: true,
    roma: true,
    invention: true,
    researchQuest: true,
    pyGraphistry: true,
  },

  // Routes provided by this plugin
  routes: [
    {
      path: '/openevolve',
      component: 'OpenEvolveDashboard',
      title: 'OpenEvolve Dashboard',
    },
    {
      path: '/openevolve/analytics',
      component: 'AnalyticsDashboard',
      title: 'Analytics',
    },
    {
      path: '/openevolve/evolution',
      component: 'EvolutionPage',
      title: 'Evolution',
    },
    {
      path: '/openevolve/adversarial',
      component: 'AdversarialPage',
      title: 'Adversarial',
    },
    {
      path: '/openevolve/workflows',
      component: 'WorkflowBuilder',
      title: 'Workflow Builder',
    },
    {
      path: '/openevolve/orchestrator',
      component: 'WorkflowOrchestrator',
      title: 'Workflow Orchestrator',
    },
    {
      path: '/openevolve/leanaide',
      component: 'LeanAidePage',
      title: 'LeanAide',
    },
    {
      path: '/openevolve/knowledge',
      component: 'KnowledgeBasePage',
      title: 'Knowledge Base',
    },
    {
      path: '/openevolve/monitoring',
      component: 'AdvancedMonitoringDashboard',
      title: 'Advanced Monitoring',
    },
    {
      path: '/openevolve/components',
      component: 'UIComponents',
      title: 'UI Components',
    },
    {
      path: '/openevolve/main',
      component: 'MainApplication',
      title: 'Main Application',
    },
  ],

  // Services provided by this plugin
  services: [
    'evolution',
    'adversarial',
    'maker',
    'mdap',
    'decomposition',
    'knowledge',
    'leanaide',
    'crewai',
    'roma',
    'invention',
    'researchQuest',
    'pyGraphistry',
  ],

  // API endpoints provided by this plugin
  apiEndpoints: {
    base: '/api/openevolve',
    websocket: '/ws/openevolve',
  },

  // Configuration schema
  configSchema: {
    evolution: './schemas/evolution',
    adversarial: './schemas/adversarial',
    maker: './schemas/maker',
    mdap: './schemas/mdap',
    decomposition: './schemas/decomposition',
    knowledge: './schemas/knowledge',
    leanaide: './schemas/leanaide',
    crewai: './schemas/crewai',
    roma: './schemas/roma',
    invention: './schemas/invention',
    researchQuest: './schemas/researchQuest',
    pyGraphistry: './schemas/pyGraphistry',
  },

  // Plugin initialization
  initialize: async () => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      console.log('[OpenEvolve] Plugin initialized');
      return true;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'OpenEvolvePlugin',
        function: 'initialize',
        operation: 'PLUGIN_INITIALIZATION',
      }
    });

    return result.success && result.data !== undefined ? result.data : false;
  },

  // Plugin cleanup
  destroy: async () => {
    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      console.log('[OpenEvolve] Plugin destroyed');
      return true;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'OpenEvolvePlugin',
        function: 'destroy',
        operation: 'PLUGIN_CLEANUP',
      }
    });

    return result.success && result.data !== undefined ? result.data : false;
  },
};

export default OpenEvolvePlugin;
