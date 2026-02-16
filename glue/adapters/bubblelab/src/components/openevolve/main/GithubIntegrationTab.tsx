import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { apiLogger } from '../../../../../../lib/structuredLogger';

type GithubUser = {
  login: string;
  id: number;
  avatar_url?: string;
  html_url?: string;
};

type GithubRepo = {
  id: number;
  full_name: string;
  html_url: string;
  default_branch: string;
};

type CommitEntry = {
  id: string;
  repo: string;
  branch: string;
  file_path: string;
  message: string;
  timestamp: string;
  url?: string;
};

// Law of Configuration Explicitness: API base from environment
const GITHUB_API_BASE = typeof process !== 'undefined' && process.env?.GITHUB_API_BASE
  ? process.env.GITHUB_API_BASE
  : "https://api.github.com";

const readStorage = <T,>(key: string, fallback: T): T => {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const writeStorage = (key: string, value: unknown) => {
  try {
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  } catch (error) {
    apiLogger.warn('Failed to write to localStorage', {
      key,
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

const encodeBase64 = (value: string) => {
  if (typeof btoa === "function") {
    return btoa(unescape(encodeURIComponent(value)));
  }
  const encoder = new TextEncoder();
  const bytes = encoder.encode(value);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary);
};

const GithubIntegrationTab: React.FC = () => {
  const [token, setToken] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_github_token") ?? "";
    } catch {
      return "";
    }
  });
  const [user, setUser] = useState<GithubUser | null>(null);
  const [repos, setRepos] = useState<GithubRepo[]>([]);
  const [linkedRepos, setLinkedRepos] = useState<GithubRepo[]>(() =>
    readStorage<GithubRepo[]>("openevolve_github_linked", []),
  );
  const [commitHistory, setCommitHistory] = useState<CommitEntry[]>(() =>
    readStorage<CommitEntry[]>("openevolve_github_commits", []),
  );
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [branchForm, setBranchForm] = useState({
    repo: "",
    baseBranch: "main",
    newBranch: "",
  });

  const [commitForm, setCommitForm] = useState({
    repo: "",
    branch: "main",
    filePath: "",
    content: "",
    message: "",
  });

  const linkedRepoOptions = useMemo(
    () => linkedRepos.map((repo) => repo.full_name),
    [linkedRepos],
  );

  useEffect(() => {
    const storedState = readStorage<Record<string, any> | null>("openevolve-state", null);
    const suggested = storedState?.evolutionCurrentBest ?? storedState?.protocolText ?? "";
    setCommitForm((prev) => ({
      ...prev,
      content: prev.content || suggested,
    }));
  }, []);

  const persistLinkedRepos = (next: GithubRepo[]) => {
    setLinkedRepos(next);
    writeStorage("openevolve_github_linked", next);
  };

  const persistCommitHistory = (next: CommitEntry[]) => {
    setCommitHistory(next);
    writeStorage("openevolve_github_commits", next);
  };

  const fetchGithub = async (path: string, options: RequestInit = {}) => {
    if (!token) {
      throw new Error("GitHub token is required.");
    }

    // MANDATORY: All HTTP requests must have a timeout (Law 3.2)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

    try {
      const response = await fetch(`${GITHUB_API_BASE}${path}`, {
        ...options,
        headers: {
          Accept: "application/vnd.github+json",
          Authorization: `Bearer ${token}`,
          "X-GitHub-Api-Version": "2022-11-28",
          ...(options.headers ?? {}),
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const text = await response.text();
      let data: any = text;
      try {
        data = text ? JSON.parse(text) : {};
      } catch {
        data = text;
      }
      if (!response.ok) {
        const error = new Error(
          data?.message || `GitHub request failed (${response.status})`,
        ) as Error & { status?: number };
        error.status = response.status;
        throw error;
      }
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('GitHub API request timeout after 30 seconds');
      }
      throw error;
    }
  };

  const authenticate = async () => {
    setLoading(true);
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const data = await fetchGithub("/user");
      setUser(data);
      setStatusMessage(`Authenticated as ${data.login}.`);
      writeStorage("openevolve_github_user", data);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to authenticate with GitHub.");
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const loadRepos = async () => {
    setLoading(true);
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const data = await fetchGithub("/user/repos?sort=updated&per_page=100");
      const repoList = (data ?? []).map((repo: any) => ({
        id: repo.id,
        full_name: repo.full_name,
        html_url: repo.html_url,
        default_branch: repo.default_branch,
      }));
      setRepos(repoList);
      setStatusMessage(`Loaded ${repoList.length} repositories.`);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load repositories.");
    } finally {
      setLoading(false);
    }
  };

  const linkRepo = (repoName: string) => {
    const repo = repos.find((item) => item.full_name === repoName);
    if (!repo) {
      setErrorMessage("Repository not found.");
      return;
    }
    if (linkedRepos.some((item) => item.full_name === repoName)) {
      setStatusMessage("Repository already linked.");
      return;
    }
    const next = [...linkedRepos, repo];
    persistLinkedRepos(next);
    setStatusMessage(`Linked ${repoName}.`);
  };

  const unlinkRepo = (repoName: string) => {
    const next = linkedRepos.filter((repo) => repo.full_name !== repoName);
    persistLinkedRepos(next);
    setStatusMessage(`Unlinked ${repoName}.`);
  };

  const handleCreateBranch = async () => {
    setLoading(true);
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      if (!branchForm.repo || !branchForm.newBranch || !branchForm.baseBranch) {
        throw new Error("Repo, base branch, and new branch are required.");
      }
      const baseRef = await fetchGithub(
        `/repos/${branchForm.repo}/git/refs/heads/${branchForm.baseBranch}`,
      );
      const baseSha = baseRef?.object?.sha;
      if (!baseSha) {
        throw new Error("Base branch SHA not found.");
      }
      await fetchGithub(`/repos/${branchForm.repo}/git/refs`, {
        method: "POST",
        body: JSON.stringify({
          ref: `refs/heads/${branchForm.newBranch}`,
          sha: baseSha,
        }),
      });
      setStatusMessage(`Created branch ${branchForm.newBranch}.`);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create branch.");
    } finally {
      setLoading(false);
    }
  };

  const handleCommit = async () => {
    setLoading(true);
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const { repo, branch, filePath, content, message } = commitForm;
      if (!repo || !branch || !filePath || !content || !message) {
        throw new Error("Repository, branch, file path, content, and message are required.");
      }

      let sha: string | undefined;
      try {
        const existing = await fetchGithub(
          `/repos/${repo}/contents/${encodeURIComponent(filePath)}?ref=${branch}`,
        );
        sha = existing?.sha;
      } catch (error: any) {
        if (error?.status !== 404) {
          throw error;
        }
      }

      const payload: Record<string, unknown> = {
        message,
        content: encodeBase64(content),
        branch,
      };
      if (sha) {
        payload.sha = sha;
      }

      const result = await fetchGithub(`/repos/${repo}/contents/${encodeURIComponent(filePath)}`, {
        method: "PUT",
        body: JSON.stringify(payload),
      });

      const entry: CommitEntry = {
        id: crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
        repo,
        branch,
        file_path: filePath,
        message,
        timestamp: new Date().toISOString(),
        url: result?.commit?.html_url,
      };
      persistCommitHistory([entry, ...commitHistory].slice(0, 50));
      setStatusMessage(`Committed ${filePath} to ${repo}@${branch}.`);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to commit changes.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const storedUser = readStorage<GithubUser | null>("openevolve_github_user", null);
    if (storedUser && !user) {
      setUser(storedUser);
    }
  }, [user]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>GitHub Integration</CardTitle>
          <CardDescription>Manage GitHub repositories, branches, and commits.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
            <div className="space-y-2">
              <Label htmlFor="github-token">GitHub Personal Access Token</Label>
              <Input
                id="github-token"
                type="password"
                value={token}
                onChange={(event) => {
                  const value = event.target.value;
                  setToken(value);
                  try {
                    globalThis.localStorage?.setItem("openevolve_github_token", value);
                  } catch (error) {
                    apiLogger.warn('Failed to persist GitHub token to localStorage', {
                      error: error instanceof Error ? error.message : String(error)
                    });
                  }
                }}
                placeholder="ghp_..."
              />
            </div>
            <div className="flex flex-col gap-2 md:justify-end">
              <Button onClick={authenticate} disabled={loading || !token}>
                Authenticate
              </Button>
              <Button variant="outline" onClick={loadRepos} disabled={loading || !token}>
                Load Repositories
              </Button>
            </div>
          </div>

          {user ? (
            <div className="flex items-center gap-2 text-sm">
              <Badge variant="secondary">Signed in</Badge>
              <span>{user.login}</span>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">Not authenticated.</div>
          )}

          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Linked Repositories</CardTitle>
          <CardDescription>Link repositories you want to sync with OpenEvolve.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Available Repositories</Label>
              <Select onValueChange={linkRepo} disabled={repos.length === 0}>
                <SelectTrigger>
                  <SelectValue placeholder="Select repository to link" />
                </SelectTrigger>
                <SelectContent>
                  {repos.map((repo) => (
                    <SelectItem key={repo.id} value={repo.full_name}>
                      {repo.full_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Linked</Label>
              <div className="space-y-2">
                {linkedRepos.length === 0 && (
                  <div className="text-sm text-muted-foreground">No linked repositories.</div>
                )}
                {linkedRepos.map((repo) => (
                  <div
                    key={repo.id}
                    className="flex items-center justify-between rounded border p-2 text-sm"
                  >
                    <span>{repo.full_name}</span>
                    <Button variant="ghost" onClick={() => unlinkRepo(repo.full_name)}>
                      Unlink
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Branch Management</CardTitle>
          <CardDescription>Create branches for evolution runs and reviews.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Repository</Label>
              <Select
                value={branchForm.repo}
                onValueChange={(value) => setBranchForm((prev) => ({ ...prev, repo: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select repository" />
                </SelectTrigger>
                <SelectContent>
                  {linkedRepoOptions.map((repo) => (
                    <SelectItem key={repo} value={repo}>
                      {repo}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Base Branch</Label>
              <Input
                value={branchForm.baseBranch}
                onChange={(event) =>
                  setBranchForm((prev) => ({ ...prev, baseBranch: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>New Branch Name</Label>
              <Input
                value={branchForm.newBranch}
                onChange={(event) =>
                  setBranchForm((prev) => ({ ...prev, newBranch: event.target.value }))
                }
              />
            </div>
          </div>
          <Button onClick={handleCreateBranch} disabled={loading}>
            Create Branch
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Commit Changes</CardTitle>
          <CardDescription>Push evolved content to GitHub.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Repository</Label>
              <Select
                value={commitForm.repo}
                onValueChange={(value) => setCommitForm((prev) => ({ ...prev, repo: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select repository" />
                </SelectTrigger>
                <SelectContent>
                  {linkedRepoOptions.map((repo) => (
                    <SelectItem key={repo} value={repo}>
                      {repo}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Branch</Label>
              <Input
                value={commitForm.branch}
                onChange={(event) =>
                  setCommitForm((prev) => ({ ...prev, branch: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>File Path</Label>
              <Input
                value={commitForm.filePath}
                onChange={(event) =>
                  setCommitForm((prev) => ({ ...prev, filePath: event.target.value }))
                }
                placeholder="e.g., protocols/evolution.md"
              />
            </div>
            <div className="space-y-2">
              <Label>Commit Message</Label>
              <Input
                value={commitForm.message}
                onChange={(event) =>
                  setCommitForm((prev) => ({ ...prev, message: event.target.value }))
                }
                placeholder="Add evolved protocol"
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label>File Content</Label>
            <Textarea
              value={commitForm.content}
              onChange={(event) =>
                setCommitForm((prev) => ({ ...prev, content: event.target.value }))
              }
              className="min-h-[180px]"
            />
          </div>
          <Button onClick={handleCommit} disabled={loading}>
            Commit to GitHub
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Commit History</CardTitle>
          <CardDescription>Recent GitHub sync activity.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {commitHistory.length === 0 && (
            <div className="text-sm text-muted-foreground">No commits yet.</div>
          )}
          {commitHistory.map((entry) => (
            <div key={entry.id} className="rounded border p-3 text-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold">{entry.message}</div>
                <Badge variant="secondary">{entry.branch}</Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {entry.repo} · {entry.file_path}
              </div>
              <div className="text-xs text-muted-foreground">
                {new Date(entry.timestamp).toLocaleString()}
              </div>
              {entry.url ? (
                <div className="text-xs text-muted-foreground break-all">{entry.url}</div>
              ) : null}
            </div>
          ))}
        </CardContent>
      </Card>

      <Separator />
    </div>
  );
};

export { GithubIntegrationTab };
