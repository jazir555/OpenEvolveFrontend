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
exports.VersionControlTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readStorage = (key, fallback = "") => {
    try {
        return globalThis.localStorage?.getItem(key) ?? fallback;
    }
    catch {
        return fallback;
    }
};
const writeStorage = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, value);
    }
    catch {
        // ignore storage errors
    }
};
const readProtocolFromState = () => {
    try {
        const raw = globalThis.localStorage?.getItem("openevolve-state");
        if (!raw)
            return "";
        const parsed = JSON.parse(raw);
        return parsed.protocolText ?? "";
    }
    catch {
        return "";
    }
};
const updateProtocolState = (protocolText) => {
    try {
        const raw = globalThis.localStorage?.getItem("openevolve-state");
        const parsed = raw ? JSON.parse(raw) : {};
        const next = { ...parsed, protocolText };
        globalThis.localStorage?.setItem("openevolve-state", JSON.stringify(next));
    }
    catch {
        // ignore storage errors
    }
};
const VersionControlTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => readStorage("openevolve_api_key"));
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [versions, setVersions] = (0, react_1.useState)([]);
    const [currentVersionId, setCurrentVersionId] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [form, setForm] = (0, react_1.useState)({
        protocol_text: readProtocolFromState(),
        version_name: "",
        comment: "",
        author: readStorage("openevolve_user", ""),
    });
    const [branchNames, setBranchNames] = (0, react_1.useState)({});
    const [compareSelection, setCompareSelection] = (0, react_1.useState)({
        version_id_1: "",
        version_id_2: "",
    });
    const [compareResult, setCompareResult] = (0, react_1.useState)(null);
    const loadVersions = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listVersions(apiConfig);
            setVersions(response.versions ?? []);
            setCurrentVersionId(response.current_version_id ?? null);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load versions.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        loadVersions();
    }, [apiConfig.apiKey]);
    const handleCreate = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!form.protocol_text.trim()) {
            setErrorMessage("Protocol text is required.");
            return;
        }
        try {
            const response = await openevolveApi_1.openevolveApi.createVersion(form, apiConfig);
            setStatusMessage(`Version created: ${response.version_id}`);
            setForm((prev) => ({ ...prev, version_name: "", comment: "" }));
            updateProtocolState(form.protocol_text);
            await loadVersions();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to create version.");
        }
    };
    const handleLoad = async (version) => {
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.loadVersion(version.id, apiConfig);
            setCurrentVersionId(response.current?.id ?? version.id);
            updateProtocolState(version.protocol_text);
            setForm((prev) => ({ ...prev, protocol_text: version.protocol_text }));
            setStatusMessage(`Loaded version ${version.name}`);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load version.");
        }
    };
    const handleDelete = async (versionId) => {
        setErrorMessage(null);
        try {
            await openevolveApi_1.openevolveApi.deleteVersion(versionId, apiConfig);
            await loadVersions();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete version.");
        }
    };
    const handleBranch = async (versionId) => {
        const branchName = branchNames[versionId] || "";
        if (!branchName.trim()) {
            setErrorMessage("Branch name is required.");
            return;
        }
        setErrorMessage(null);
        try {
            await openevolveApi_1.openevolveApi.branchVersion(versionId, { new_version_name: branchName }, apiConfig);
            setBranchNames((prev) => ({ ...prev, [versionId]: "" }));
            await loadVersions();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to branch version.");
        }
    };
    const handleCompare = async () => {
        if (!compareSelection.version_id_1 || !compareSelection.version_id_2) {
            setErrorMessage("Select two versions to compare.");
            return;
        }
        setErrorMessage(null);
        setCompareResult(null);
        try {
            const result = await openevolveApi_1.openevolveApi.compareVersions(compareSelection, apiConfig);
            setCompareResult(result);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to compare versions.");
        }
    };
    const handleRefreshProtocol = () => {
        setForm((prev) => ({ ...prev, protocol_text: readProtocolFromState() }));
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Version Control</card_1.CardTitle>
          <card_1.CardDescription>Track protocol versions, branches, and comparisons.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            writeStorage("openevolve_api_key", value);
        }}/>
            </div>
            <div className="flex items-end gap-2">
              <button_1.Button variant="outline" onClick={loadVersions} disabled={loading}>
                Refresh Versions
              </button_1.Button>
              <button_1.Button variant="outline" onClick={handleRefreshProtocol}>
                Pull Protocol Text
              </button_1.Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
        </card_1.CardContent>
      </card_1.Card>

      <tabs_1.Tabs defaultValue="create" className="w-full">
        <tabs_1.TabsList className="grid w-full grid-cols-3">
          <tabs_1.TabsTrigger value="create">Create Version</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="history">History</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="compare">Compare</tabs_1.TabsTrigger>
        </tabs_1.TabsList>

        <tabs_1.TabsContent value="create" className="mt-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Save Current Protocol</card_1.CardTitle>
              <card_1.CardDescription>Create a new version snapshot.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Version Name</label_1.Label>
                  <input_1.Input value={form.version_name} onChange={(event) => setForm((prev) => ({ ...prev, version_name: event.target.value }))}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Author</label_1.Label>
                  <input_1.Input value={form.author} onChange={(event) => {
            const value = event.target.value;
            setForm((prev) => ({ ...prev, author: value }));
            writeStorage("openevolve_user", value);
        }}/>
                </div>
              </div>
              <div className="space-y-2">
                <label_1.Label>Comment</label_1.Label>
                <input_1.Input value={form.comment} onChange={(event) => setForm((prev) => ({ ...prev, comment: event.target.value }))}/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Protocol Text</label_1.Label>
                <textarea_1.Textarea value={form.protocol_text} onChange={(event) => setForm((prev) => ({ ...prev, protocol_text: event.target.value }))} className="min-h-[200px]"/>
              </div>
              <button_1.Button onClick={handleCreate}>Save Version</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="history" className="mt-4 space-y-4">
          {versions.length === 0 ? (<card_1.Card>
              <card_1.CardContent className="py-6 text-sm text-muted-foreground">
                No versions available yet.
              </card_1.CardContent>
            </card_1.Card>) : (versions.map((version) => {
            const isCurrent = version.id === currentVersionId;
            return (<card_1.Card key={version.id}>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-base flex items-center gap-2">
                      {version.name}
                      {isCurrent ? <badge_1.Badge>Current</badge_1.Badge> : null}
                    </card_1.CardTitle>
                    <card_1.CardDescription>
                      {version.timestamp?.replace("T", " ")?.slice(0, 19) ?? ""}
                    </card_1.CardDescription>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-3">
                    <div className="text-sm text-muted-foreground">ID: {version.id}</div>
                    {version.comment ? <div className="text-sm">{version.comment}</div> : null}
                    <div className="text-xs text-muted-foreground">Author: {version.author ?? "Unknown"}</div>
                    <div className="flex flex-wrap gap-2">
                      <button_1.Button size="sm" variant="outline" onClick={() => handleLoad(version)}>
                        Load
                      </button_1.Button>
                      <button_1.Button size="sm" variant="outline" onClick={() => handleDelete(version.id)}>
                        Delete
                      </button_1.Button>
                    </div>
                    <separator_1.Separator />
                    <div className="grid gap-2 md:grid-cols-2">
                      <div className="space-y-2">
                        <label_1.Label>Branch Name</label_1.Label>
                        <input_1.Input value={branchNames[version.id] ?? ""} onChange={(event) => setBranchNames((prev) => ({ ...prev, [version.id]: event.target.value }))} placeholder={`Branch of ${version.name}`}/>
                      </div>
                      <div className="flex items-end">
                        <button_1.Button size="sm" onClick={() => handleBranch(version.id)}>
                          Create Branch
                        </button_1.Button>
                      </div>
                    </div>
                  </card_1.CardContent>
                </card_1.Card>);
        }))}
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="compare" className="mt-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Compare Versions</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Version A</label_1.Label>
                  <select className="w-full rounded border px-3 py-2 text-sm" value={compareSelection.version_id_1} onChange={(event) => setCompareSelection((prev) => ({ ...prev, version_id_1: event.target.value }))}>
                    <option value="">Select version</option>
                    {versions.map((version) => (<option key={version.id} value={version.id}>
                        {version.name}
                      </option>))}
                  </select>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Version B</label_1.Label>
                  <select className="w-full rounded border px-3 py-2 text-sm" value={compareSelection.version_id_2} onChange={(event) => setCompareSelection((prev) => ({ ...prev, version_id_2: event.target.value }))}>
                    <option value="">Select version</option>
                    {versions.map((version) => (<option key={version.id} value={version.id}>
                        {version.name}
                      </option>))}
                  </select>
                </div>
              </div>
              <button_1.Button onClick={handleCompare}>Compare</button_1.Button>
              {compareResult ? (<div className="rounded border p-3 text-sm space-y-2">
                  {compareResult.error ? (<div className="text-red-500">{compareResult.error}</div>) : (<>
                      <div>
                        {compareResult.version1} vs {compareResult.version2}
                      </div>
                      <div>Characters added: {compareResult.chars_added}</div>
                      <div>Characters removed: {compareResult.chars_removed}</div>
                      <div>Total change: {compareResult.total_chars_change}</div>
                    </>)}
                </div>) : null}
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>
      </tabs_1.Tabs>
    </div>);
};
exports.VersionControlTab = VersionControlTab;
//# sourceMappingURL=VersionControlTab.js.map