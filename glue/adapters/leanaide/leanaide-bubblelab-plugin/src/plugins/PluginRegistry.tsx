import {
  pluginRegistry,
  PluginManager,
  type LeanAidePluginInterface,
  type LeanAidePluginLifecycle,
} from '../PluginInterface';

export interface PluginConfiguration {
  [pluginId: string]: Record<string, unknown>;
}

export class PluginLifecycleManager {
  private readonly states = new Map<string, 'active' | 'inactive' | 'error'>();

  async startPlugin(pluginId: string): Promise<boolean> {
    const success = await pluginRegistry.activate(pluginId);
    this.states.set(pluginId, success ? 'active' : 'error');
    return success;
  }

  async stopPlugin(pluginId: string): Promise<boolean> {
    const success = await pluginRegistry.deactivate(pluginId);
    this.states.set(pluginId, success ? 'inactive' : 'error');
    return success;
  }

  getPluginState(pluginId: string): 'active' | 'inactive' | 'error' | undefined {
    return this.states.get(pluginId);
  }
}

export class PluginInstaller {
  static async installFromUrl(_url: string): Promise<boolean> {
    return true;
  }

  static async installFromPackage(_name: string): Promise<boolean> {
    return true;
  }
}

export class PluginConfigurationManager {
  private configurations: PluginConfiguration = {};

  getConfiguration(pluginId: string): Record<string, unknown> {
    return this.configurations[pluginId] ?? {};
  }

  setConfiguration(pluginId: string, config: Record<string, unknown>): void {
    this.configurations[pluginId] = { ...config };
  }

  getAllConfigurations(): PluginConfiguration {
    return { ...this.configurations };
  }
}

export const pluginLifecycleManager = new PluginLifecycleManager();
export const pluginConfigurationManager = new PluginConfigurationManager();

export type { LeanAidePluginInterface, LeanAidePluginLifecycle };
export { pluginRegistry, PluginManager };
