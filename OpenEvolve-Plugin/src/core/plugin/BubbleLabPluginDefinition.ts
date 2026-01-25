/**
 * OpenEvolve Plugin for BubbleLab
 *
 * Integrates the OpenEvolve AI evolution and optimization platform
 * as a plugin within BubbleLab's workflow system.
 */

import { PluginDefinition } from '@/types/plugin';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';

export const OpenEvolvePlugin: PluginDefinition = {
  id: 'openevolve',
  name: 'OpenEvolve',
  version: '1.0.0',
  description: 'OpenEvolve AI evolution and optimization platform',
  icon: '/integrations/openevolve.svg',
  author: 'OpenEvolve',
  website: 'https://openevolve.ai',
  documentation: 'https://docs.openevolve.ai',

  // Sub-engines as services
  services: [
    'evolution',
    'adversarial',
    'decomposition',
    'mdap',
    'maker',
    'knowledge',
    'leanaide',
    'crewai',
    'roma',
    'invention',
    'researchQuest',
    'pyGraphistry',
  ],

  // Configuration schemas for each service
  schemas: {
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

  // UI components
  components: {
    config: '@/components/openevolve/workflow/ConfigPanel',
    monitor: '@/components/openevolve/workflow/ExecutionMonitor',
    editor: '@/components/openevolve/workflow/WorkflowBuilder',
    dashboard: '@/pages/OpenEvolveDashboard',
    workflowOrchestrator: '@/pages/WorkflowOrchestrator',
    evolutionPage: '@/pages/EvolutionPage',
    adversarialPage: '@/pages/AdversarialPage',
    knowledgeBasePage: '@/pages/KnowledgeBasePage',
    workflowBuilder: '@/pages/WorkflowBuilder',
  },

  // API integration
  api: {
    baseUrl: process.env.VITE_OPENEVOLVE_API_URL || 'http://localhost:8000',
    endpoints: {
      evolution: '/api/v1/evolution',
      adversarial: '/api/v1/adversarial',
      maker: '/api/v1/maker',
      mdap: '/api/v1/mdap',
      decomposition: '/api/v1/decomposition',
      knowledge: '/api/v1/knowledge',
      leanaide: '/api/v1/leanaide',
      crewai: '/api/v1/crewai',
      roma: '/api/v1/roma',
      invention: '/api/v1/invention',
      researchQuest: '/api/v1/research-quest',
      pyGraphistry: '/api/v1/pygraphistry',
    },
    headers: {
      'Content-Type': 'application/json',
    },
  },

  // Lifecycle hooks
  hooks: {
    onBeforeExecute: async (serviceId: string, params: any) => {
      await gracefulErrorHandler.executeWithErrorHandling(async () => {
        console.log(`[OpenEvolve] Starting ${serviceId} execution`, params);
      }, {
        strategy: 'retry',
        maxRetries: 2,
        retryDelay: 500,
        showUserNotification: false,
        logError: true,
        context: {
          component: 'OpenEvolvePlugin',
          function: 'onBeforeExecute',
          operation: `BEFORE_EXECUTE_${serviceId}`,
          additionalData: { serviceId, params }
        }
      }).catch(console.error);
    },
    onAfterExecute: async (serviceId: string, result: any) => {
      await gracefulErrorHandler.executeWithErrorHandling(async () => {
        console.log(`[OpenEvolve] Completed ${serviceId} execution`, result);
      }, {
        strategy: 'retry',
        maxRetries: 2,
        retryDelay: 500,
        showUserNotification: false,
        logError: true,
        context: {
          component: 'OpenEvolvePlugin',
          function: 'onAfterExecute',
          operation: `AFTER_EXECUTE_${serviceId}`,
          additionalData: { serviceId, result }
        }
      }).catch(console.error);
    },
    onError: async (serviceId: string, error: Error) => {
      await gracefulErrorHandler.executeWithErrorHandling(async () => {
        console.error(`[OpenEvolve] Error in ${serviceId}`, error);
      }, {
        strategy: 'retry',
        maxRetries: 2,
        retryDelay: 500,
        showUserNotification: true,
        logError: true,
        context: {
          component: 'OpenEvolvePlugin',
          function: 'onError',
          operation: `ERROR_${serviceId}`,
          additionalData: { serviceId, error: error.message }
        }
      }).catch(console.error);
    },
  },
};

export default OpenEvolvePlugin;
