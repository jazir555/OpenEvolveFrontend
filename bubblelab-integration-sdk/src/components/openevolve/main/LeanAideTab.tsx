import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "@/lib/openevolveApi";
import type { LeanAideProofResponse, LeanAideStatusResponse, LeanAideTreeResponse } from "@/lib/types";

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

export const LeanAideTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [status, setStatus] = useState<LeanAideStatusResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<Record<string, unknown> | null>(null);

  const [theoremText, setTheoremText] = useState("");
  const [theoremName, setTheoremName] = useState("");
  const [theoremCode, setTheoremCode] = useState("");
  const [mathQuery, setMathQuery] = useState("");
  const [mctsConfig, setMctsConfig] = useState({
    max_iterations: 1000,
    time_budget: 300,
    c_param: 1.414,
    expansion_agents: 3,
    simulation_voters: 5,
  });

  const [treeIds, setTreeIds] = useState<string[]>([]);
  const [selectedTreeId, setSelectedTreeId] = useState<string>("");
  const [treeDetail, setTreeDetail] = useState<LeanAideTreeResponse | null>(null);

  const [proofIds, setProofIds] = useState<string[]>([]);
  const [selectedProofId, setSelectedProofId] = useState<string>("");
  const [proofDetail, setProofDetail] = useState<LeanAideProofResponse | null>(null);

  const loadStatus = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.bubblelabsLeanAideStatus(apiConfig);
      setStatus(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load LeanAide status.");
    } finally {
      setLoading(false);
    }
  };

  const loadTrees = async () => {
    try {
      const response = await openevolveApi.bubblelabsLeanAideTrees(apiConfig);
      setTreeIds(response.tree_ids ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load trees.");
    }
  };

  const loadProofs = async () => {
    try {
      const response = await openevolveApi.bubblelabsLeanAideProofs(apiConfig);
      setProofIds(response.proof_ids ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load proofs.");
    }
  };

  useEffect(() => {
    loadStatus();
    loadTrees();
    loadProofs();
  }, [apiConfig.apiKey]);

  const executeTask = async (taskType: string, payload: Record<string, unknown>) => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.bubblelabsLeanAideExecute(
        { task_type: taskType, payload },
        apiConfig,
      );
      setLastResult(response.result ?? null);
      await loadTrees();
      await loadProofs();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "LeanAide task failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectTree = async (treeId: string) => {
    setSelectedTreeId(treeId);
    if (!treeId) {
      setTreeDetail(null);
      return;
    }
    try {
      const response = await openevolveApi.bubblelabsLeanAideTree(treeId, apiConfig);
      setTreeDetail(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load tree.");
    }
  };

  const handleSelectProof = async (proofId: string) => {
    setSelectedProofId(proofId);
    if (!proofId) {
      setProofDetail(null);
      return;
    }
    try {
      const response = await openevolveApi.bubblelabsLeanAideProof(proofId, apiConfig);
      setProofDetail(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load proof.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>LeanAide Formal Verification</CardTitle>
          <CardDescription>Run theorem proving, MCTS search, and Lean4 verification.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                type="password"
                value={apiKey}
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
            <Button variant="outline" onClick={loadStatus} disabled={loading}>
              Refresh Status
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          {status ? (
            <div className="grid gap-4 md:grid-cols-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">MCTS</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-semibold">
                  {status.mcts_available ? "✅" : "❌"}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">MDAP</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-semibold">
                  {status.mdap_available ? "✅" : "❌"}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Lean4</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-semibold">
                  {status.lean4_available ? "✅" : "❌"}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Active Proofs</CardTitle>
                </CardHeader>
                <CardContent className="text-2xl font-semibold">{status.active_proofs}</CardContent>
              </Card>
            </div>
          ) : null}

          <Tabs defaultValue="theorem" className="w-full">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="theorem">Theorem Proving</TabsTrigger>
              <TabsTrigger value="mcts">MCTS Visualization</TabsTrigger>
              <TabsTrigger value="lean4">Lean4 Verification</TabsTrigger>
              <TabsTrigger value="math">Math Queries</TabsTrigger>
            </TabsList>

            <TabsContent value="theorem" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Theorem Proving</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Theorem Statement</Label>
                    <Textarea
                      value={theoremText}
                      onChange={(event) => setTheoremText(event.target.value)}
                      className="min-h-[120px]"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Theorem Name</Label>
                    <Input
                      value={theoremName}
                      onChange={(event) => setTheoremName(event.target.value)}
                      placeholder="Optional name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Theorem Code (optional Lean)</Label>
                    <Textarea
                      value={theoremCode}
                      onChange={(event) => setTheoremCode(event.target.value)}
                      className="min-h-[120px]"
                    />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      onClick={() =>
                        executeTask("translate_theorem", {
                          theorem_text: theoremText,
                          theorem_name: theoremName || undefined,
                        })
                      }
                      disabled={loading}
                    >
                      Translate
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() =>
                        executeTask("generate_proof", {
                          theorem_text: theoremText,
                          theorem_code: theoremCode || undefined,
                        })
                      }
                      disabled={loading}
                    >
                      Generate Proof
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() =>
                        executeTask("verify_solution", {
                          code: theoremCode,
                        })
                      }
                      disabled={loading}
                    >
                      Verify Code
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() =>
                        executeTask("mcts_search", {
                          theorem: theoremText,
                          theorem_name: theoremName || undefined,
                          ...mctsConfig,
                        })
                      }
                      disabled={loading}
                    >
                      MCTS Search
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="mcts" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">MCTS Trees</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Select Tree</Label>
                      <Select value={selectedTreeId} onValueChange={handleSelectTree}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select tree" />
                        </SelectTrigger>
                        <SelectContent>
                          {treeIds.map((treeId) => (
                            <SelectItem key={treeId} value={treeId}>
                              {treeId.slice(0, 8)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Iterations</Label>
                      <Input
                        type="number"
                        value={mctsConfig.max_iterations}
                        onChange={(event) =>
                          setMctsConfig((prev) => ({
                            ...prev,
                            max_iterations: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Time Budget (s)</Label>
                      <Input
                        type="number"
                        value={mctsConfig.time_budget}
                        onChange={(event) =>
                          setMctsConfig((prev) => ({
                            ...prev,
                            time_budget: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Exploration Constant</Label>
                      <Input
                        type="number"
                        value={mctsConfig.c_param}
                        onChange={(event) =>
                          setMctsConfig((prev) => ({
                            ...prev,
                            c_param: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                  </div>

                  {treeDetail ? (
                    <div className="rounded border p-3 text-xs whitespace-pre-wrap">
                      {JSON.stringify(treeDetail.tree, null, 2)}
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">Select a tree to view details.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="lean4" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Lean4 Proofs</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Select Proof</Label>
                    <Select value={selectedProofId} onValueChange={handleSelectProof}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select proof" />
                      </SelectTrigger>
                      <SelectContent>
                        {proofIds.map((proofId) => (
                          <SelectItem key={proofId} value={proofId}>
                            {proofId.slice(0, 8)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  {proofDetail ? (
                    <div className="rounded border p-3 text-xs whitespace-pre-wrap">
                      {JSON.stringify(proofDetail.proof, null, 2)}
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">Select a proof to view details.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="math" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Math Query</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Query</Label>
                    <Textarea
                      value={mathQuery}
                      onChange={(event) => setMathQuery(event.target.value)}
                      className="min-h-[120px]"
                    />
                  </div>
                  <Button
                    variant="outline"
                    onClick={() => executeTask("math_query", { query: mathQuery })}
                    disabled={loading}
                  >
                    Run Query
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {lastResult ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Last Result</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(lastResult, null, 2)}</pre>
              </CardContent>
            </Card>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
};
