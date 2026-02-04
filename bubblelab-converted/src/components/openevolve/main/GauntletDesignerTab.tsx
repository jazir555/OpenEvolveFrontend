import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "@/lib/openevolveApi";
import {
  GauntletDefinition,
  GauntletRoundRule,
  GauntletSummary,
  createDefaultGauntlet,
  createDefaultGauntletRound,
} from "@/lib/types";

export const GauntletDesignerTab: React.FC = () => {
  const [gauntlets, setGauntlets] = useState<GauntletSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [editingGauntlet, setEditingGauntlet] = useState<string | null>(null);
  const [formGauntlet, setFormGauntlet] = useState<GauntletDefinition>(createDefaultGauntlet());
  const [roundJson, setRoundJson] = useState<string[]>([]);
  const [roundErrors, setRoundErrors] = useState<string[]>([]);
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });

  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const syncRoundsFromGauntlet = (gauntlet: GauntletDefinition) => {
    const jsonValues = gauntlet.rounds.map((round) => JSON.stringify(round, null, 2));
    setRoundJson(jsonValues);
    setRoundErrors(jsonValues.map(() => ""));
  };

  const resetForm = () => {
    const next = createDefaultGauntlet();
    setFormGauntlet(next);
    setEditingGauntlet(null);
    syncRoundsFromGauntlet(next);
  };

  const loadGauntlets = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const result = await openevolveApi.listGauntlets(apiConfig);
      setGauntlets(result.gauntlets ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load gauntlets.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    syncRoundsFromGauntlet(formGauntlet);
  }, []);

  useEffect(() => {
    loadGauntlets();
  }, [apiConfig.apiKey]);

  const handleRoundJsonChange = (index: number, value: string) => {
    const updatedJson = [...roundJson];
    updatedJson[index] = value;
    setRoundJson(updatedJson);

    try {
      const parsed = JSON.parse(value) as GauntletRoundRule;
      const updatedRounds = [...formGauntlet.rounds];
      updatedRounds[index] = parsed;
      setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
      const errors = [...roundErrors];
      errors[index] = "";
      setRoundErrors(errors);
    } catch (error: any) {
      const errors = [...roundErrors];
      errors[index] = error?.message ?? "Invalid JSON";
      setRoundErrors(errors);
    }
  };

  const handleAddRound = () => {
    const nextRoundNumber = formGauntlet.rounds.length + 1;
    const nextRound = createDefaultGauntletRound(nextRoundNumber);
    const updatedRounds = [...formGauntlet.rounds, nextRound];
    setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
    setRoundJson([...roundJson, JSON.stringify(nextRound, null, 2)]);
    setRoundErrors([...roundErrors, ""]);
  };

  const handleRemoveRound = (index: number) => {
    const updatedRounds = formGauntlet.rounds.filter((_, idx) => idx !== index);
    setFormGauntlet({ ...formGauntlet, rounds: updatedRounds });
    setRoundJson(roundJson.filter((_, idx) => idx !== index));
    setRoundErrors(roundErrors.filter((_, idx) => idx !== index));
  };

  const handleEditGauntlet = async (gauntlet: GauntletSummary) => {
    setErrorMessage(null);
    try {
      const detailed = await openevolveApi.getGauntlet(gauntlet.name, apiConfig);
      setEditingGauntlet(gauntlet.name);
      setFormGauntlet(detailed);
      syncRoundsFromGauntlet(detailed);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load gauntlet details.");
    }
  };

  const handleDeleteGauntlet = async (name: string) => {
    if (!confirm(`Delete gauntlet "${name}"? This cannot be undone.`)) {
      return;
    }
    try {
      await openevolveApi.deleteGauntlet(name, apiConfig);
      setStatusMessage(`Deleted gauntlet ${name}.`);
      await loadGauntlets();
      if (editingGauntlet === name) {
        resetForm();
      }
    } catch (error: any) {
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
        await openevolveApi.updateGauntlet(editingGauntlet, formGauntlet, apiConfig);
        setStatusMessage(`Updated gauntlet ${formGauntlet.name}.`);
      } else {
        await openevolveApi.createGauntlet(formGauntlet, apiConfig);
        setStatusMessage(`Created gauntlet ${formGauntlet.name}.`);
      }
      await loadGauntlets();
      resetForm();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to save gauntlet.");
    }
  };

  const updateGauntletField = (field: keyof GauntletDefinition, value: any) => {
    setFormGauntlet({ ...formGauntlet, [field]: value });
  };

  const attackModesValue = (formGauntlet.attack_modes ?? []).join(", ");

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Gauntlet Designer</CardTitle>
          <CardDescription>Configure multi-round gauntlets for Red/Gold reviews.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (X-API-Key)</label>
              <Input
                value={apiKey}
                type="password"
                placeholder="Paste API key for /gauntlets endpoints"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  try {
                    globalThis.localStorage?.setItem("openevolve_api_key", value);
                  } catch {
                    // ignore storage errors
                  }
                }}
              />
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={loadGauntlets} disabled={loading}>
                Refresh Gauntlets
              </Button>
              <Button variant="secondary" onClick={resetForm}>
                New Gauntlet
              </Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="text-base">Existing Gauntlets</CardTitle>
                <CardDescription>Click a gauntlet to edit.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {gauntlets.length === 0 && (
                  <div className="text-sm text-muted-foreground">No gauntlets yet.</div>
                )}
                {gauntlets.map((gauntlet) => (
                  <div
                    key={gauntlet.name}
                    className={`rounded border p-3 ${editingGauntlet === gauntlet.name ? "border-primary" : "border-border"}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <div className="text-sm font-semibold">{gauntlet.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {gauntlet.team_name} · {gauntlet.round_count} round(s)
                        </div>
                      </div>
                      <Badge variant="outline">standard</Badge>
                    </div>
                    <div className="mt-3 flex gap-2">
                      <Button size="sm" variant="secondary" onClick={() => handleEditGauntlet(gauntlet)}>
                        Edit
                      </Button>
                      <Button size="sm" variant="destructive" onClick={() => handleDeleteGauntlet(gauntlet.name)}>
                        Delete
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  {editingGauntlet ? `Edit Gauntlet: ${editingGauntlet}` : "Create New Gauntlet"}
                </CardTitle>
                <CardDescription>Define gauntlet metadata, rounds, and verification rules.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Gauntlet Name</label>
                    <Input
                      value={formGauntlet.name}
                      onChange={(event) => updateGauntletField("name", event.target.value)}
                      placeholder="Gauntlet name"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Team Name</label>
                    <Input
                      value={formGauntlet.team_name}
                      onChange={(event) => updateGauntletField("team_name", event.target.value)}
                      placeholder="Team assigned to this gauntlet"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Generation Mode</label>
                    <Select
                      value={formGauntlet.generation_mode ?? "single_candidate"}
                      onValueChange={(value) => updateGauntletField("generation_mode", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="single_candidate">Single Candidate</SelectItem>
                        <SelectItem value="multi_candidate_peer_review">Multi-candidate Peer Review</SelectItem>
                        <SelectItem value="evolutionary">Evolutionary</SelectItem>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Gauntlet Type</label>
                    <Select
                      value={formGauntlet.gauntlet_type ?? "standard"}
                      onValueChange={(value) => updateGauntletField("gauntlet_type", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="standard">Standard</SelectItem>
                        <SelectItem value="adaptive">Adaptive</SelectItem>
                        <SelectItem value="hierarchical">Hierarchical</SelectItem>
                        <SelectItem value="competitive">Competitive</SelectItem>
                        <SelectItem value="collaborative">Collaborative</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Description</label>
                  <Textarea
                    value={formGauntlet.description ?? ""}
                    onChange={(event) => updateGauntletField("description", event.target.value)}
                    placeholder="Describe this gauntlet"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Attack Modes (comma-separated)</label>
                  <Input
                    value={attackModesValue}
                    onChange={(event) =>
                      updateGauntletField(
                        "attack_modes",
                        event.target.value
                          .split(",")
                          .map((entry) => entry.trim())
                          .filter(Boolean),
                      )
                    }
                  />
                </div>

                <Tabs defaultValue="rounds">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="rounds">Rounds</TabsTrigger>
                    <TabsTrigger value="redflags">Red Flags</TabsTrigger>
                    <TabsTrigger value="formal">Formal Verification</TabsTrigger>
                  </TabsList>
                  <TabsContent value="rounds" className="space-y-4 pt-4">
                    {formGauntlet.rounds.map((round, index) => (
                      <Card key={index}>
                        <CardHeader className="flex flex-row items-center justify-between">
                          <div>
                            <CardTitle className="text-sm">Round {round.round_number}</CardTitle>
                            <CardDescription>Quorum {round.quorum_required_approvals}</CardDescription>
                          </div>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleRemoveRound(index)}
                            disabled={formGauntlet.rounds.length === 1}
                          >
                            Remove
                          </Button>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <div className="grid gap-3 md:grid-cols-3">
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Quorum Approvals</label>
                              <Input
                                type="number"
                                value={round.quorum_required_approvals}
                                onChange={(event) => {
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
                                }}
                              />
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Panel Size</label>
                              <Input
                                type="number"
                                value={round.quorum_from_panel_size}
                                onChange={(event) => {
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
                                }}
                              />
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Min Confidence</label>
                              <Input
                                type="number"
                                step="0.01"
                                value={round.min_overall_confidence ?? 0}
                                onChange={(event) => {
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
                                }}
                              />
                            </div>
                          </div>
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Round Config (JSON)</label>
                            <Textarea
                              value={roundJson[index] ?? ""}
                              onChange={(event) => handleRoundJsonChange(index, event.target.value)}
                              rows={10}
                            />
                            {roundErrors[index] ? (
                              <div className="text-xs text-red-500">{roundErrors[index]}</div>
                            ) : null}
                          </div>
                        </CardContent>
                      </Card>
                    ))}

                    <Button variant="secondary" onClick={handleAddRound}>
                      Add Round
                    </Button>
                  </TabsContent>
                  <TabsContent value="redflags" className="space-y-4 pt-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Red Flags (JSON)</label>
                      <Textarea
                        value={JSON.stringify(formGauntlet.red_flags ?? {}, null, 2)}
                        onChange={(event) => {
                          try {
                            updateGauntletField("red_flags", JSON.parse(event.target.value));
                          } catch {
                            // ignore invalid JSON, allow user to keep typing
                          }
                        }}
                        rows={8}
                      />
                    </div>
                  </TabsContent>
                  <TabsContent value="formal" className="space-y-4 pt-4">
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Formal Verification</label>
                        <Select
                          value={formGauntlet.formal_verification_enabled ? "enabled" : "disabled"}
                          onValueChange={(value) =>
                            updateGauntletField("formal_verification_enabled", value === "enabled")
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="enabled">Enabled</SelectItem>
                            <SelectItem value="disabled">Disabled</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Threshold</label>
                        <Input
                          type="number"
                          step="0.01"
                          value={formGauntlet.formal_verification_threshold ?? 0.9}
                          onChange={(event) =>
                            updateGauntletField("formal_verification_threshold", Number(event.target.value) || 0)
                          }
                        />
                      </div>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Verification Methods (comma-separated)</label>
                      <Input
                        value={(formGauntlet.verification_methods ?? []).join(", ")}
                        onChange={(event) =>
                          updateGauntletField(
                            "verification_methods",
                            event.target.value
                              .split(",")
                              .map((entry) => entry.trim())
                              .filter(Boolean),
                          )
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Lean Verification Config (JSON)</label>
                      <Textarea
                        value={JSON.stringify(formGauntlet.lean_verification_config ?? {}, null, 2)}
                        onChange={(event) => {
                          try {
                            updateGauntletField("lean_verification_config", JSON.parse(event.target.value));
                          } catch {
                            // ignore invalid JSON for now
                          }
                        }}
                        rows={8}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Mathematical Requirements (JSON)</label>
                      <Textarea
                        value={JSON.stringify(formGauntlet.mathematical_requirements ?? {}, null, 2)}
                        onChange={(event) => {
                          try {
                            updateGauntletField("mathematical_requirements", JSON.parse(event.target.value));
                          } catch {
                            // ignore invalid JSON for now
                          }
                        }}
                        rows={8}
                      />
                    </div>
                  </TabsContent>
                </Tabs>

                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={resetForm}>
                    Reset
                  </Button>
                  <Button onClick={handleSaveGauntlet}>
                    {editingGauntlet ? "Update Gauntlet" : "Create Gauntlet"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
