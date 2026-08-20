import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "@/lib/openevolveApi";
import type { VersionEntry, VersionCompareResult } from "@/lib/types";

const readStorage = (key: string, fallback = "") => {
  try {
    return globalThis.localStorage?.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
};

const writeStorage = (key: string, value: string) => {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // ignore storage errors
  }
};

const readProtocolFromState = () => {
  try {
    const raw = globalThis.localStorage?.getItem("openevolve-state");
    if (!raw) return "";
    const parsed = JSON.parse(raw) as { protocolText?: string };
    return parsed.protocolText ?? "";
  } catch {
    return "";
  }
};

const updateProtocolState = (protocolText: string) => {
  try {
    const raw = globalThis.localStorage?.getItem("openevolve-state");
    const parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    const next = { ...parsed, protocolText };
    globalThis.localStorage?.setItem("openevolve-state", JSON.stringify(next));
  } catch {
    // ignore storage errors
  }
};

export const VersionControlTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => readStorage("openevolve_api_key"));
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [versions, setVersions] = useState<VersionEntry[]>([]);
  const [currentVersionId, setCurrentVersionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [form, setForm] = useState({
    protocol_text: readProtocolFromState(),
    version_name: "",
    comment: "",
    author: readStorage("openevolve_user", ""),
  });

  const [branchNames, setBranchNames] = useState<Record<string, string>>({});
  const [compareSelection, setCompareSelection] = useState({
    version_id_1: "",
    version_id_2: "",
  });
  const [compareResult, setCompareResult] = useState<VersionCompareResult | null>(null);

  const loadVersions = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listVersions(apiConfig);
      setVersions(response.versions ?? []);
      setCurrentVersionId(response.current_version_id ?? null);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load versions.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
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
      const response = await openevolveApi.createVersion(form, apiConfig);
      setStatusMessage(`Version created: ${response.version_id}`);
      setForm((prev) => ({ ...prev, version_name: "", comment: "" }));
      updateProtocolState(form.protocol_text);
      await loadVersions();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create version.");
    }
  };

  const handleLoad = async (version: VersionEntry) => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.loadVersion(version.id, apiConfig);
      setCurrentVersionId(response.current?.id ?? version.id);
      updateProtocolState(version.protocol_text);
      setForm((prev) => ({ ...prev, protocol_text: version.protocol_text }));
      setStatusMessage(`Loaded version ${version.name}`);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load version.");
    }
  };

  const handleDelete = async (versionId: string) => {
    setErrorMessage(null);
    try {
      await openevolveApi.deleteVersion(versionId, apiConfig);
      await loadVersions();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to delete version.");
    }
  };

  const handleBranch = async (versionId: string) => {
    const branchName = branchNames[versionId] || "";
    if (!branchName.trim()) {
      setErrorMessage("Branch name is required.");
      return;
    }
    setErrorMessage(null);
    try {
      await openevolveApi.branchVersion(versionId, { new_version_name: branchName }, apiConfig);
      setBranchNames((prev) => ({ ...prev, [versionId]: "" }));
      await loadVersions();
    } catch (error: any) {
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
      const result = await openevolveApi.compareVersions(compareSelection, apiConfig);
      setCompareResult(result);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to compare versions.");
    }
  };

  const handleRefreshProtocol = () => {
    setForm((prev) => ({ ...prev, protocol_text: readProtocolFromState() }));
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Version Control</CardTitle>
          <CardDescription>Track protocol versions, branches, and comparisons.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                value={apiKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  writeStorage("openevolve_api_key", value);
                }}
              />
            </div>
            <div className="flex items-end gap-2">
              <Button variant="outline" onClick={loadVersions} disabled={loading}>
                Refresh Versions
              </Button>
              <Button variant="outline" onClick={handleRefreshProtocol}>
                Pull Protocol Text
              </Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
        </CardContent>
      </Card>

      <Tabs defaultValue="create" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="create">Create Version</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="compare">Compare</TabsTrigger>
        </TabsList>

        <TabsContent value="create" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Save Current Protocol</CardTitle>
              <CardDescription>Create a new version snapshot.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Version Name</Label>
                  <Input
                    value={form.version_name}
                    onChange={(event) => setForm((prev) => ({ ...prev, version_name: event.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Author</Label>
                  <Input
                    value={form.author}
                    onChange={(event) => {
                      const value = event.target.value;
                      setForm((prev) => ({ ...prev, author: value }));
                      writeStorage("openevolve_user", value);
                    }}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label>Comment</Label>
                <Input
                  value={form.comment}
                  onChange={(event) => setForm((prev) => ({ ...prev, comment: event.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label>Protocol Text</Label>
                <Textarea
                  value={form.protocol_text}
                  onChange={(event) => setForm((prev) => ({ ...prev, protocol_text: event.target.value }))}
                  className="min-h-[200px]"
                />
              </div>
              <Button onClick={handleCreate}>Save Version</Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="mt-4 space-y-4">
          {versions.length === 0 ? (
            <Card>
              <CardContent className="py-6 text-sm text-muted-foreground">
                No versions available yet.
              </CardContent>
            </Card>
          ) : (
            versions.map((version) => {
              const isCurrent = version.id === currentVersionId;
              return (
                <Card key={version.id}>
                  <CardHeader>
                    <CardTitle className="text-base flex items-center gap-2">
                      {version.name}
                      {isCurrent ? <Badge>Current</Badge> : null}
                    </CardTitle>
                    <CardDescription>
                      {version.timestamp?.replace("T", " ")?.slice(0, 19) ?? ""}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="text-sm text-muted-foreground">ID: {version.id}</div>
                    {version.comment ? <div className="text-sm">{version.comment}</div> : null}
                    <div className="text-xs text-muted-foreground">Author: {version.author ?? "Unknown"}</div>
                    <div className="flex flex-wrap gap-2">
                      <Button size="sm" variant="outline" onClick={() => handleLoad(version)}>
                        Load
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => handleDelete(version.id)}>
                        Delete
                      </Button>
                    </div>
                    <Separator />
                    <div className="grid gap-2 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>Branch Name</Label>
                        <Input
                          value={branchNames[version.id] ?? ""}
                          onChange={(event) =>
                            setBranchNames((prev) => ({ ...prev, [version.id]: event.target.value }))
                          }
                          placeholder={`Branch of ${version.name}`}
                        />
                      </div>
                      <div className="flex items-end">
                        <Button size="sm" onClick={() => handleBranch(version.id)}>
                          Create Branch
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })
          )}
        </TabsContent>

        <TabsContent value="compare" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Compare Versions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Version A</Label>
                  <select
                    className="w-full rounded border px-3 py-2 text-sm"
                    value={compareSelection.version_id_1}
                    onChange={(event) =>
                      setCompareSelection((prev) => ({ ...prev, version_id_1: event.target.value }))
                    }
                  >
                    <option value="">Select version</option>
                    {versions.map((version) => (
                      <option key={version.id} value={version.id}>
                        {version.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <Label>Version B</Label>
                  <select
                    className="w-full rounded border px-3 py-2 text-sm"
                    value={compareSelection.version_id_2}
                    onChange={(event) =>
                      setCompareSelection((prev) => ({ ...prev, version_id_2: event.target.value }))
                    }
                  >
                    <option value="">Select version</option>
                    {versions.map((version) => (
                      <option key={version.id} value={version.id}>
                        {version.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <Button onClick={handleCompare}>Compare</Button>
              {compareResult ? (
                <div className="rounded border p-3 text-sm space-y-2">
                  {compareResult.error ? (
                    <div className="text-red-500">{compareResult.error}</div>
                  ) : (
                    <>
                      <div>
                        {compareResult.version1} vs {compareResult.version2}
                      </div>
                      <div>Characters added: {compareResult.chars_added}</div>
                      <div>Characters removed: {compareResult.chars_removed}</div>
                      <div>Total change: {compareResult.total_chars_change}</div>
                    </>
                  )}
                </div>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
