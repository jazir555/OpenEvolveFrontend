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
