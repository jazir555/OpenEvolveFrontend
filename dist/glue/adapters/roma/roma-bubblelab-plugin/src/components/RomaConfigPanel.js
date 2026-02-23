"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
const react_toastify_1 = require("react-toastify");
const RomaConfigPanel = ({ plugin, onConfigChange, onClose }) => {
    const [config, setConfig] = (0, react_1.useState)(plugin.getState());
    const [isSaving, setIsSaving] = (0, react_1.useState)(false);
    const [newMcpName, setNewMcpName] = (0, react_1.useState)('');
    const [newToolkitName, setNewToolkitName] = (0, react_1.useState)('');
    const mcpServers = (0, react_1.useMemo)(() => config.mcpServers ?? [], [config.mcpServers]);
    const toolkits = (0, react_1.useMemo)(() => config.agents?.executor?.toolkits ?? [], [config.agents?.executor?.toolkits]);
    const updateConfig = (updates) => {
        setConfig((previous) => ({ ...previous, ...updates }));
    };
    const saveConfig = async () => {
        try {
            setIsSaving(true);
            await plugin.updateConfig(config);
            onConfigChange?.(config);
            react_toastify_1.toast.success('ROMA configuration saved');
        }
        catch (error) {
            react_toastify_1.toast.error(error instanceof Error ? error.message : 'Failed to save ROMA configuration');
        }
        finally {
            setIsSaving(false);
        }
    };
    const addMcpServer = async () => {
        if (!newMcpName.trim()) {
            return;
        }
        const newServer = {
            server_name: newMcpName.trim(),
            server_type: 'http',
            enabled: true,
        };
        try {
            await plugin.addMcpServer(newServer);
            setConfig(plugin.getState());
            setNewMcpName('');
            react_toastify_1.toast.success(`Added MCP server ${newServer.server_name}`);
        }
        catch (error) {
            react_toastify_1.toast.error(error instanceof Error ? error.message : 'Failed to add MCP server');
        }
    };
    const removeMcpServer = async (serverName) => {
        try {
            await plugin.removeMcpServer(serverName);
            setConfig(plugin.getState());
            react_toastify_1.toast.success(`Removed MCP server ${serverName}`);
        }
        catch (error) {
            react_toastify_1.toast.error(error instanceof Error ? error.message : 'Failed to remove MCP server');
        }
    };
    const addToolkit = async () => {
        if (!newToolkitName.trim()) {
            return;
        }
        const toolkit = {
            class_name: newToolkitName.trim(),
            enabled: true,
        };
        try {
            await plugin.addToolkit(toolkit);
            setConfig(plugin.getState());
            setNewToolkitName('');
            react_toastify_1.toast.success(`Added toolkit ${toolkit.class_name}`);
        }
        catch (error) {
            react_toastify_1.toast.error(error instanceof Error ? error.message : 'Failed to add toolkit');
        }
    };
    const removeToolkit = async (className) => {
        try {
            await plugin.removeToolkit(className);
            setConfig(plugin.getState());
            react_toastify_1.toast.success(`Removed toolkit ${className}`);
        }
        catch (error) {
            react_toastify_1.toast.error(error instanceof Error ? error.message : 'Failed to remove toolkit');
        }
    };
    return (<div className="rounded-lg border bg-white p-6 shadow-sm dark:bg-gray-800">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-xl font-semibold text-gray-900 dark:text-white">
          <lucide_react_1.Settings className="h-5 w-5"/>
          ROMA Configuration
        </h2>
        <button type="button" onClick={onClose} className="rounded p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-700 dark:hover:text-gray-200" aria-label="Close">
          <lucide_react_1.X className="h-5 w-5"/>
        </button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <lucide_react_1.Settings className="h-4 w-4"/>
            General
          </h3>
          <div className="space-y-3">
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Server URL
              <input className="mt-1 w-full rounded border px-3 py-2 text-sm" value={config.serverUrl ?? ''} onChange={(event) => updateConfig({ serverUrl: event.target.value })}/>
            </label>
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              API Key
              <input type="password" className="mt-1 w-full rounded border px-3 py-2 text-sm" value={config.apiKey ?? ''} onChange={(event) => updateConfig({ apiKey: event.target.value })}/>
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
              <input type="checkbox" checked={config.enableObservability ?? true} onChange={(event) => updateConfig({ enableObservability: event.target.checked })}/>
              Enable Observability
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
              <input type="checkbox" checked={config.enableStorage ?? true} onChange={(event) => updateConfig({ enableStorage: event.target.checked })}/>
              Enable Storage
            </label>
          </div>
        </section>

        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <lucide_react_1.Bot className="h-4 w-4"/>
            Execution
          </h3>
          <div className="space-y-3">
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Default Profile
              <input className="mt-1 w-full rounded border px-3 py-2 text-sm" value={config.defaultProfile ?? 'default'} onChange={(event) => updateConfig({ defaultProfile: event.target.value })}/>
            </label>
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Max Depth
              <input type="number" className="mt-1 w-full rounded border px-3 py-2 text-sm" value={config.maxDepth ?? 3} onChange={(event) => updateConfig({ maxDepth: Number(event.target.value) || 3 })}/>
            </label>
            <label className="block text-sm text-gray-700 dark:text-gray-300">
              Timeout (ms)
              <input type="number" className="mt-1 w-full rounded border px-3 py-2 text-sm" value={config.timeout ?? 30000} onChange={(event) => updateConfig({ timeout: Number(event.target.value) || 30000 })}/>
            </label>
          </div>
        </section>

        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <lucide_react_1.Server className="h-4 w-4"/>
            MCP Servers
          </h3>
          <div className="mb-3 flex gap-2">
            <input className="flex-1 rounded border px-3 py-2 text-sm" placeholder="Server name" value={newMcpName} onChange={(event) => setNewMcpName(event.target.value)}/>
            <button type="button" onClick={() => {
            void addMcpServer();
        }} className="rounded bg-blue-600 px-2 py-2 text-white" aria-label="Add MCP server">
              <lucide_react_1.Plus className="h-4 w-4"/>
            </button>
          </div>

          <div className="space-y-2">
            {mcpServers.map((server) => (<div key={server.server_name} className="flex items-center justify-between rounded border p-2 text-sm">
                <span>{server.server_name}</span>
                <button type="button" onClick={() => {
                void removeMcpServer(server.server_name);
            }} className="text-red-600" aria-label={`Remove MCP server ${server.server_name}`}>
                  <lucide_react_1.Trash2 className="h-4 w-4"/>
                </button>
              </div>))}
          </div>
        </section>

        <section className="rounded-md border p-4">
          <h3 className="mb-3 flex items-center gap-2 font-medium text-gray-900 dark:text-white">
            <lucide_react_1.Tool className="h-4 w-4"/>
            Toolkits
          </h3>
          <div className="mb-3 flex gap-2">
            <input className="flex-1 rounded border px-3 py-2 text-sm" placeholder="Toolkit class name" value={newToolkitName} onChange={(event) => setNewToolkitName(event.target.value)}/>
            <button type="button" onClick={() => {
            void addToolkit();
        }} className="rounded bg-blue-600 px-2 py-2 text-white" aria-label="Add toolkit">
              <lucide_react_1.Plus className="h-4 w-4"/>
            </button>
          </div>

          <div className="space-y-2">
            {toolkits.map((toolkit) => (<div key={toolkit.class_name} className="flex items-center justify-between rounded border p-2 text-sm">
                <span>{toolkit.class_name}</span>
                <button type="button" onClick={() => {
                void removeToolkit(toolkit.class_name);
            }} className="text-red-600" aria-label={`Remove toolkit ${toolkit.class_name}`}>
                  <lucide_react_1.Trash2 className="h-4 w-4"/>
                </button>
              </div>))}
          </div>
        </section>
      </div>

      <div className="mt-6 flex justify-end">
        <button type="button" onClick={() => {
            void saveConfig();
        }} disabled={isSaving} className="inline-flex items-center gap-2 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-60">
          <lucide_react_1.Save className="h-4 w-4"/>
          {isSaving ? 'Saving...' : 'Save Configuration'}
        </button>
      </div>
    </div>);
};
exports.default = RomaConfigPanel;
//# sourceMappingURL=RomaConfigPanel.js.map