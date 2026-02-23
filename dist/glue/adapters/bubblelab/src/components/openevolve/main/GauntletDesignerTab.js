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
exports.GauntletDesignerTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const select_1 = require("@/components/ui/select");
const tabs_1 = require("@/components/ui/tabs");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const types_1 = require("../../../lib/types");
const GauntletDesignerTab = () => {
    const [gauntlets, setGauntlets] = (0, react_1.useState)([]);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [editingGauntlet, setEditingGauntlet] = (0, react_1.useState)(null);
    const [formGauntlet, setFormGauntlet] = (0, react_1.useState)((0, types_1.createDefaultGauntlet)());
    const [roundJson, setRoundJson] = (0, react_1.useState)([]);
    const [roundErrors, setRoundErrors] = (0, react_1.useState)([]);
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const syncRoundsFromGauntlet = (gauntlet) => {
        const jsonValues = gauntlet.rounds.map((round) => JSON.stringify(round, null, 2));
        setRoundJson(jsonValues);
        setRoundErrors(jsonValues.map(() => ""));
    };
    const resetForm = () => {
        const next = (0, types_1.createDefaultGauntlet)();
        setFormGauntlet(next);
        setEditingGauntlet(null);
        syncRoundsFromGauntlet(next);
    };
    const loadGauntlets = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.listGauntlets(apiConfig);
            setGauntlets(result.gauntlets ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load gauntlets.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        syncRoundsFromGauntlet(formGauntlet);
    }, []);
    (0, react_1.useEffect)(() => {
        loadGauntlets();
    }, [apiConfig.apiKey]);
    const handleRoundJsonChange = (index, value) => {
        const updatedJson = [...roundJson];
        updatedJson[index] = value;
        setRoundJson(updatedJson);
        try {
            const parsed = JSON.parse(value);
            const updatedRounds = [...formGauntlet.rounds];
            updatedRounds[index] = parsed;
            setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
            const errors = [...roundErrors];
            errors[index] = "";
            setRoundErrors(errors);
        }
        catch (error) {
            const errors = [...roundErrors];
            errors[index] = error?.message ?? "Invalid JSON";
            setRoundErrors(errors);
        }
    };
    const handleAddRound = () => {
        const nextRoundNumber = formGauntlet.rounds.length + 1;
        const nextRound = (0, types_1.createDefaultGauntletRound)(nextRoundNumber);
        const updatedRounds = [...formGauntlet.rounds, nextRound];
        setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
        setRoundJson([...roundJson, JSON.stringify(nextRound, null, 2)]);
        setRoundErrors([...roundErrors, ""]);
    };
    const handleRemoveRound = (index) => {
        const updatedRounds = formGauntlet.rounds.filter((_, idx) => idx !== index);
        setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
        setRoundJson(roundJson.filter((_, idx) => idx !== index));
        setRoundErrors(roundErrors.filter((_, idx) => idx !== index));
    };
    const handleEditGauntlet = async (gauntlet) => {
        setErrorMessage(null);
        try {
            const detailed = await openevolveApi_1.openevolveApi.getGauntlet(gauntlet.name, apiConfig);
            setEditingGauntlet(gauntlet.name);
            setFormGauntlet(detailed);
            syncRoundsFromGauntlet(detailed);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load gauntlet details.");
        }
    };
    const handleDeleteGauntlet = async (name) => {
        if (!confirm(`Delete gauntlet "${name}"? This cannot be undone.`)) {
            return;
        }
        try {
            await openevolveApi_1.openevolveApi.deleteGauntlet(name, apiConfig);
            setStatusMessage(`Deleted gauntlet ${name}.`);
            await loadGauntlets();
            if (editingGauntlet === name) {
                resetForm();
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete gauntlet.");
        }
    };
    const handleSaveGauntlet = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!formGauntlet.name.trim()) {
            setErrorMessage("Gauntlet name is required.");
            return;
        }
        if (!formGauntlet.team_name.trim()) {
            setErrorMessage("Team name is required.");
            return;
        }
        if (roundErrors.some((err) => err)) {
            setErrorMessage("Fix invalid round JSON before saving.");
            return;
        }
        try {
            if (editingGauntlet) {
                await openevolveApi_1.openevolveApi.updateGauntlet(editingGauntlet, formGauntlet, apiConfig);
                setStatusMessage(`Updated gauntlet ${formGauntlet.name}.`);
            }
            else {
                await openevolveApi_1.openevolveApi.createGauntlet(formGauntlet, apiConfig);
                setStatusMessage(`Created gauntlet ${formGauntlet.name}.`);
            }
            await loadGauntlets();
            resetForm();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to save gauntlet.");
        }
    };
    const updateGauntletField = (field, value) => {
        setFormGauntlet({ ...formGauntlet, [field]: value });
    };
    const attackModesValue = (formGauntlet.attack_modes ?? []).join(", ");
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Gauntlet Designer</card_1.CardTitle>
          <card_1.CardDescription>Configure multi-round gauntlets for Red/Gold reviews.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (X-API-Key)</label>
              <input_1.Input value={apiKey} type="password" placeholder="Paste API key for /gauntlets endpoints" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore storage errors
            }
        }}/>
            </div>
            <div className="flex gap-2">
              <button_1.Button variant="outline" onClick={loadGauntlets} disabled={loading}>
                Refresh Gauntlets
              </button_1.Button>
              <button_1.Button variant="secondary" onClick={resetForm}>
                New Gauntlet
              </button_1.Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
            <card_1.Card className="h-full">
              <card_1.CardHeader>
                <card_1.CardTitle className="text-base">Existing Gauntlets</card_1.CardTitle>
                <card_1.CardDescription>Click a gauntlet to edit.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3">
                {gauntlets.length === 0 && (<div className="text-sm text-muted-foreground">No gauntlets yet.</div>)}
                {gauntlets.map((gauntlet) => (<div key={gauntlet.name} className={`rounded border p-3 ${editingGauntlet === gauntlet.name ? "border-primary" : "border-border"}`}>
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <div className="text-sm font-semibold">{gauntlet.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {gauntlet.team_name} · {gauntlet.round_count} round(s)
                        </div>
                      </div>
                      <badge_1.Badge variant="outline">standard</badge_1.Badge>
                    </div>
                    <div className="mt-3 flex gap-2">
                      <button_1.Button size="sm" variant="secondary" onClick={() => handleEditGauntlet(gauntlet)}>
                        Edit
                      </button_1.Button>
                      <button_1.Button size="sm" variant="destructive" onClick={() => handleDeleteGauntlet(gauntlet.name)}>
                        Delete
                      </button_1.Button>
                    </div>
                  </div>))}
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-base">
                  {editingGauntlet ? `Edit Gauntlet: ${editingGauntlet}` : "Create New Gauntlet"}
                </card_1.CardTitle>
                <card_1.CardDescription>Define gauntlet metadata, rounds, and verification rules.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Gauntlet Name</label>
                    <input_1.Input value={formGauntlet.name} onChange={(event) => updateGauntletField("name", event.target.value)} placeholder="Gauntlet name"/>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Team Name</label>
                    <input_1.Input value={formGauntlet.team_name} onChange={(event) => updateGauntletField("team_name", event.target.value)} placeholder="Team assigned to this gauntlet"/>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Generation Mode</label>
                    <select_1.Select value={formGauntlet.generation_mode ?? "single_candidate"} onValueChange={(value) => updateGauntletField("generation_mode", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        <select_1.SelectItem value="single_candidate">Single Candidate</select_1.SelectItem>
                        <select_1.SelectItem value="multi_candidate_peer_review">Multi-candidate Peer Review</select_1.SelectItem>
                        <select_1.SelectItem value="evolutionary">Evolutionary</select_1.SelectItem>
                        <select_1.SelectItem value="hybrid">Hybrid</select_1.SelectItem>
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Gauntlet Type</label>
                    <select_1.Select value={formGauntlet.gauntlet_type ?? "standard"} onValueChange={(value) => updateGauntletField("gauntlet_type", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        <select_1.SelectItem value="standard">Standard</select_1.SelectItem>
                        <select_1.SelectItem value="adaptive">Adaptive</select_1.SelectItem>
                        <select_1.SelectItem value="hierarchical">Hierarchical</select_1.SelectItem>
                        <select_1.SelectItem value="competitive">Competitive</select_1.SelectItem>
                        <select_1.SelectItem value="collaborative">Collaborative</select_1.SelectItem>
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Description</label>
                  <textarea_1.Textarea value={formGauntlet.description ?? ""} onChange={(event) => updateGauntletField("description", event.target.value)} placeholder="Describe this gauntlet"/>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Attack Modes (comma-separated)</label>
                  <input_1.Input value={attackModesValue} onChange={(event) => updateGauntletField("attack_modes", event.target.value
            .split(",")
            .map((entry) => entry.trim())
            .filter(Boolean))}/>
                </div>

                <tabs_1.Tabs defaultValue="rounds">
                  <tabs_1.TabsList className="grid w-full grid-cols-3">
                    <tabs_1.TabsTrigger value="rounds">Rounds</tabs_1.TabsTrigger>
                    <tabs_1.TabsTrigger value="redflags">Red Flags</tabs_1.TabsTrigger>
                    <tabs_1.TabsTrigger value="formal">Formal Verification</tabs_1.TabsTrigger>
                  </tabs_1.TabsList>
                  <tabs_1.TabsContent value="rounds" className="space-y-4 pt-4">
                    {formGauntlet.rounds.map((round, index) => (<card_1.Card key={index}>
                        <card_1.CardHeader className="flex flex-row items-center justify-between">
                          <div>
                            <card_1.CardTitle className="text-sm">Round {round.round_number}</card_1.CardTitle>
                            <card_1.CardDescription>Quorum {round.quorum_required_approvals}</card_1.CardDescription>
                          </div>
                          <button_1.Button variant="destructive" size="sm" onClick={() => handleRemoveRound(index)} disabled={formGauntlet.rounds.length === 1}>
                            Remove
                          </button_1.Button>
                        </card_1.CardHeader>
                        <card_1.CardContent className="space-y-4">
                          <div className="grid gap-3 md:grid-cols-3">
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Quorum Approvals</label>
                              <input_1.Input type="number" value={round.quorum_required_approvals} onChange={(event) => {
                const updatedRounds = [...formGauntlet.rounds];
                updatedRounds[index] = {
                    ...updatedRounds[index],
                    quorum_required_approvals: Number(event.target.value) || 0,
                };
                setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
                setRoundJson((prev) => {
                    const next = [...prev];
                    next[index] = JSON.stringify(updatedRounds[index], null, 2);
                    return next;
                });
            }}/>
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Panel Size</label>
                              <input_1.Input type="number" value={round.quorum_from_panel_size} onChange={(event) => {
                const updatedRounds = [...formGauntlet.rounds];
                updatedRounds[index] = {
                    ...updatedRounds[index],
                    quorum_from_panel_size: Number(event.target.value) || 0,
                };
                setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
                setRoundJson((prev) => {
                    const next = [...prev];
                    next[index] = JSON.stringify(updatedRounds[index], null, 2);
                    return next;
                });
            }}/>
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Min Confidence</label>
                              <input_1.Input type="number" step="0.01" value={round.min_overall_confidence ?? 0} onChange={(event) => {
                const updatedRounds = [...formGauntlet.rounds];
                updatedRounds[index] = {
                    ...updatedRounds[index],
                    min_overall_confidence: Number(event.target.value) || 0,
                };
                setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
                setRoundJson((prev) => {
                    const next = [...prev];
                    next[index] = JSON.stringify(updatedRounds[index], null, 2);
                    return next;
                });
            }}/>
                            </div>
                          </div>
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Round Config (JSON)</label>
                            <textarea_1.Textarea value={roundJson[index] ?? ""} onChange={(event) => handleRoundJsonChange(index, event.target.value)} rows={10}/>
                            {roundErrors[index] ? (<div className="text-xs text-red-500">{roundErrors[index]}</div>) : null}
                          </div>
                        </card_1.CardContent>
                      </card_1.Card>))}

                    <button_1.Button variant="secondary" onClick={handleAddRound}>
                      Add Round
                    </button_1.Button>
                  </tabs_1.TabsContent>
                  <tabs_1.TabsContent value="redflags" className="space-y-4 pt-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Red Flags (JSON)</label>
                      <textarea_1.Textarea value={JSON.stringify(formGauntlet.red_flags ?? {}, null, 2)} onChange={(event) => {
            try {
                updateGauntletField("red_flags", JSON.parse(event.target.value));
            }
            catch {
                // ignore invalid JSON, allow user to keep typing
            }
        }} rows={8}/>
                    </div>
                  </tabs_1.TabsContent>
                  <tabs_1.TabsContent value="formal" className="space-y-4 pt-4">
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Formal Verification</label>
                        <select_1.Select value={formGauntlet.formal_verification_enabled ? "enabled" : "disabled"} onValueChange={(value) => updateGauntletField("formal_verification_enabled", value === "enabled")}>
                          <select_1.SelectTrigger>
                            <select_1.SelectValue />
                          </select_1.SelectTrigger>
                          <select_1.SelectContent>
                            <select_1.SelectItem value="enabled">Enabled</select_1.SelectItem>
                            <select_1.SelectItem value="disabled">Disabled</select_1.SelectItem>
                          </select_1.SelectContent>
                        </select_1.Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Threshold</label>
                        <input_1.Input type="number" step="0.01" value={formGauntlet.formal_verification_threshold ?? 0.9} onChange={(event) => updateGauntletField("formal_verification_threshold", Number(event.target.value) || 0)}/>
                      </div>
                    </div>
                    <separator_1.Separator />
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Verification Methods (comma-separated)</label>
                      <input_1.Input value={(formGauntlet.verification_methods ?? []).join(", ")} onChange={(event) => updateGauntletField("verification_methods", event.target.value
            .split(",")
            .map((entry) => entry.trim())
            .filter(Boolean))}/>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Lean Verification Config (JSON)</label>
                      <textarea_1.Textarea value={JSON.stringify(formGauntlet.lean_verification_config ?? {}, null, 2)} onChange={(event) => {
            try {
                updateGauntletField("lean_verification_config", JSON.parse(event.target.value));
            }
            catch {
                // ignore invalid JSON for now
            }
        }} rows={8}/>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Mathematical Requirements (JSON)</label>
                      <textarea_1.Textarea value={JSON.stringify(formGauntlet.mathematical_requirements ?? {}, null, 2)} onChange={(event) => {
            try {
                updateGauntletField("mathematical_requirements", JSON.parse(event.target.value));
            }
            catch {
                // ignore invalid JSON for now
            }
        }} rows={8}/>
                    </div>
                  </tabs_1.TabsContent>
                </tabs_1.Tabs>

                <div className="flex justify-end gap-2">
                  <button_1.Button variant="outline" onClick={resetForm}>
                    Reset
                  </button_1.Button>
                  <button_1.Button onClick={handleSaveGauntlet}>
                    {editingGauntlet ? "Update Gauntlet" : "Create Gauntlet"}
                  </button_1.Button>
                </div>
              </card_1.CardContent>
            </card_1.Card>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.GauntletDesignerTab = GauntletDesignerTab;
//# sourceMappingURL=GauntletDesignerTab.js.map