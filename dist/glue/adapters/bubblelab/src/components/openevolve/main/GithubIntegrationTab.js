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
exports.GithubIntegrationTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const select_1 = require("@/components/ui/select");
const structuredLogger_1 = require("../../../../../../lib/structuredLogger");
// Law of Configuration Explicitness: API base from environment
const GITHUB_API_BASE = typeof process !== 'undefined' && process.env?.GITHUB_API_BASE
    ? process.env.GITHUB_API_BASE
    : "https://api.github.com";
const readStorage = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw) {
            return fallback;
        }
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const writeStorage = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, JSON.stringify(value));
    }
    catch (error) {
        structuredLogger_1.apiLogger.warn('Failed to write to localStorage', {
            key,
            error: error instanceof Error ? error.message : String(error)
        });
    }
};
const encodeBase64 = (value) => {
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
const GithubIntegrationTab = () => {
    const [token, setToken] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_github_token") ?? "";
        }
        catch {
            return "";
        }
    });
    const [user, setUser] = (0, react_1.useState)(null);
    const [repos, setRepos] = (0, react_1.useState)([]);
    const [linkedRepos, setLinkedRepos] = (0, react_1.useState)(() => readStorage("openevolve_github_linked", []));
    const [commitHistory, setCommitHistory] = (0, react_1.useState)(() => readStorage("openevolve_github_commits", []));
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [branchForm, setBranchForm] = (0, react_1.useState)({
        repo: "",
        baseBranch: "main",
        newBranch: "",
    });
    const [commitForm, setCommitForm] = (0, react_1.useState)({
        repo: "",
        branch: "main",
        filePath: "",
        content: "",
        message: "",
    });
    const linkedRepoOptions = (0, react_1.useMemo)(() => linkedRepos.map((repo) => repo.full_name), [linkedRepos]);
    (0, react_1.useEffect)(() => {
        const storedState = readStorage("openevolve-state", null);
        const suggested = storedState?.evolutionCurrentBest ?? storedState?.protocolText ?? "";
        setCommitForm((prev) => ({
            ...prev,
            content: prev.content || suggested,
        }));
    }, []);
    const persistLinkedRepos = (next) => {
        setLinkedRepos(next);
        writeStorage("openevolve_github_linked", next);
    };
    const persistCommitHistory = (next) => {
        setCommitHistory(next);
        writeStorage("openevolve_github_commits", next);
    };
    const fetchGithub = async (path, options = {}) => {
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
            let data = text;
            try {
                data = text ? JSON.parse(text) : {};
            }
            catch {
                data = text;
            }
            if (!response.ok) {
                const error = new Error(data?.message || `GitHub request failed (${response.status})`);
                error.status = response.status;
                throw error;
            }
            return data;
        }
        catch (error) {
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
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to authenticate with GitHub.");
            setUser(null);
        }
        finally {
            setLoading(false);
        }
    };
    const loadRepos = async () => {
        setLoading(true);
        setStatusMessage(null);
        setErrorMessage(null);
        try {
            const data = await fetchGithub("/user/repos?sort=updated&per_page=100");
            const repoList = (data ?? []).map((repo) => ({
                id: repo.id,
                full_name: repo.full_name,
                html_url: repo.html_url,
                default_branch: repo.default_branch,
            }));
            setRepos(repoList);
            setStatusMessage(`Loaded ${repoList.length} repositories.`);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load repositories.");
        }
        finally {
            setLoading(false);
        }
    };
    const linkRepo = (repoName) => {
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
    const unlinkRepo = (repoName) => {
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
            const baseRef = await fetchGithub(`/repos/${branchForm.repo}/git/refs/heads/${branchForm.baseBranch}`);
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
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to create branch.");
        }
        finally {
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
            let sha;
            try {
                const existing = await fetchGithub(`/repos/${repo}/contents/${encodeURIComponent(filePath)}?ref=${branch}`);
                sha = existing?.sha;
            }
            catch (error) {
                if (error?.status !== 404) {
                    throw error;
                }
            }
            const payload = {
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
            const entry = {
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
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to commit changes.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        const storedUser = readStorage("openevolve_github_user", null);
        if (storedUser && !user) {
            setUser(storedUser);
        }
    }, [user]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>GitHub Integration</card_1.CardTitle>
          <card_1.CardDescription>Manage GitHub repositories, branches, and commits.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
            <div className="space-y-2">
              <label_1.Label htmlFor="github-token">GitHub Personal Access Token</label_1.Label>
              <input_1.Input id="github-token" type="password" value={token} onChange={(event) => {
            const value = event.target.value;
            setToken(value);
            try {
                globalThis.localStorage?.setItem("openevolve_github_token", value);
            }
            catch (error) {
                structuredLogger_1.apiLogger.warn('Failed to persist GitHub token to localStorage', {
                    error: error instanceof Error ? error.message : String(error)
                });
            }
        }} placeholder="ghp_..."/>
            </div>
            <div className="flex flex-col gap-2 md:justify-end">
              <button_1.Button onClick={authenticate} disabled={loading || !token}>
                Authenticate
              </button_1.Button>
              <button_1.Button variant="outline" onClick={loadRepos} disabled={loading || !token}>
                Load Repositories
              </button_1.Button>
            </div>
          </div>

          {user ? (<div className="flex items-center gap-2 text-sm">
              <badge_1.Badge variant="secondary">Signed in</badge_1.Badge>
              <span>{user.login}</span>
            </div>) : (<div className="text-sm text-muted-foreground">Not authenticated.</div>)}

          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Linked Repositories</card_1.CardTitle>
          <card_1.CardDescription>Link repositories you want to sync with OpenEvolve.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Available Repositories</label_1.Label>
              <select_1.Select onValueChange={linkRepo} disabled={repos.length === 0}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select repository to link"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {repos.map((repo) => (<select_1.SelectItem key={repo.id} value={repo.full_name}>
                      {repo.full_name}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Linked</label_1.Label>
              <div className="space-y-2">
                {linkedRepos.length === 0 && (<div className="text-sm text-muted-foreground">No linked repositories.</div>)}
                {linkedRepos.map((repo) => (<div key={repo.id} className="flex items-center justify-between rounded border p-2 text-sm">
                    <span>{repo.full_name}</span>
                    <button_1.Button variant="ghost" onClick={() => unlinkRepo(repo.full_name)}>
                      Unlink
                    </button_1.Button>
                  </div>))}
              </div>
            </div>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Branch Management</card_1.CardTitle>
          <card_1.CardDescription>Create branches for evolution runs and reviews.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <label_1.Label>Repository</label_1.Label>
              <select_1.Select value={branchForm.repo} onValueChange={(value) => setBranchForm((prev) => ({ ...prev, repo: value }))}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select repository"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {linkedRepoOptions.map((repo) => (<select_1.SelectItem key={repo} value={repo}>
                      {repo}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Base Branch</label_1.Label>
              <input_1.Input value={branchForm.baseBranch} onChange={(event) => setBranchForm((prev) => ({ ...prev, baseBranch: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>New Branch Name</label_1.Label>
              <input_1.Input value={branchForm.newBranch} onChange={(event) => setBranchForm((prev) => ({ ...prev, newBranch: event.target.value }))}/>
            </div>
          </div>
          <button_1.Button onClick={handleCreateBranch} disabled={loading}>
            Create Branch
          </button_1.Button>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Commit Changes</card_1.CardTitle>
          <card_1.CardDescription>Push evolved content to GitHub.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Repository</label_1.Label>
              <select_1.Select value={commitForm.repo} onValueChange={(value) => setCommitForm((prev) => ({ ...prev, repo: value }))}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select repository"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {linkedRepoOptions.map((repo) => (<select_1.SelectItem key={repo} value={repo}>
                      {repo}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Branch</label_1.Label>
              <input_1.Input value={commitForm.branch} onChange={(event) => setCommitForm((prev) => ({ ...prev, branch: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>File Path</label_1.Label>
              <input_1.Input value={commitForm.filePath} onChange={(event) => setCommitForm((prev) => ({ ...prev, filePath: event.target.value }))} placeholder="e.g., protocols/evolution.md"/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Commit Message</label_1.Label>
              <input_1.Input value={commitForm.message} onChange={(event) => setCommitForm((prev) => ({ ...prev, message: event.target.value }))} placeholder="Add evolved protocol"/>
            </div>
          </div>
          <div className="space-y-2">
            <label_1.Label>File Content</label_1.Label>
            <textarea_1.Textarea value={commitForm.content} onChange={(event) => setCommitForm((prev) => ({ ...prev, content: event.target.value }))} className="min-h-[180px]"/>
          </div>
          <button_1.Button onClick={handleCommit} disabled={loading}>
            Commit to GitHub
          </button_1.Button>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Commit History</card_1.CardTitle>
          <card_1.CardDescription>Recent GitHub sync activity.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          {commitHistory.length === 0 && (<div className="text-sm text-muted-foreground">No commits yet.</div>)}
          {commitHistory.map((entry) => (<div key={entry.id} className="rounded border p-3 text-sm">
              <div className="flex items-center justify-between">
                <div className="font-semibold">{entry.message}</div>
                <badge_1.Badge variant="secondary">{entry.branch}</badge_1.Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {entry.repo} · {entry.file_path}
              </div>
              <div className="text-xs text-muted-foreground">
                {new Date(entry.timestamp).toLocaleString()}
              </div>
              {entry.url ? (<div className="text-xs text-muted-foreground break-all">{entry.url}</div>) : null}
            </div>))}
        </card_1.CardContent>
      </card_1.Card>

      <separator_1.Separator />
    </div>);
};
exports.GithubIntegrationTab = GithubIntegrationTab;
//# sourceMappingURL=GithubIntegrationTab.js.map