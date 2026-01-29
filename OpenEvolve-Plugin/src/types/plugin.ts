import { ReactNode } from 'react';

export interface PluginDefinition {
  id: string;
  name: string;
  version: string;
  description: string;
  author: string;
  icon: string;
  capabilities: PluginCapabilities;
  routes: PluginRoute[];
  services: string[];
  apiEndpoints: {
    base: string;
    websocket: string;
  };
  configSchema: Record<string, string>;
  init?: () => Promise<boolean>;
  destroy?: () => Promise<boolean>;
  initialize?: () => Promise<boolean>;
}

export interface PluginCapabilities {
  workflows?: boolean;
  analytics?: boolean;
  knowledgeBase?: boolean;
  leanAide?: boolean;
  evolution?: boolean;
  adversarial?: boolean;
  maker?: boolean;
  mdap?: boolean;
  decomposition?: boolean;
  crewai?: boolean;
  roma?: boolean;
  invention?: boolean;
}

// ParameterSchema type for export
export interface ParameterSchema {
  type: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'enum' | 'select' | 'textarea' | 'slider' | 'text' | 'multiselect';
  description?: string;
  required?: boolean;
  default?: any;
  defaultValue?: any;
  enum?: any[];
  min?: number;
  max?: number;
  pattern?: string;
  properties?: Record<string, ParameterSchema>;
  items?: ParameterSchema | any[];
  name?: string;
  label?: string;
  options?: any[];
  multiline?: boolean;
  placeholder?: string;
  step?: number;
  condition?: (params: any) => boolean;
}

export interface PluginRoute {
  path: string;
  component: string;
  title: string;
  icon?: string;
  exact?: boolean;
}

export interface PluginContext {
  plugin: PluginDefinition;
  enabled: boolean;
  config?: Record<string, unknown>;
}
