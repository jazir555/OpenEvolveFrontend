import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
/**
 * GitHub Bubble - Version Control and Development Platform Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. createIssue - Create a new issue in a repository
 * 2. updateIssue - Update an existing issue
 * 3. listIssues - List issues in a repository
 * 4. createPullRequest - Create a pull request
 * 5. mergePullRequest - Merge a pull request
 * 6. listPullRequests - List pull requests in a repository
 * 7. createBranch - Create a new branch
 * 8. deleteBranch - Delete a branch
 * 9. getRepository - Get repository information
 * 10. createCommit - Create a commit
 */
// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================
const CreateIssueParamsSchema = z.object({
    operation: z.literal('createIssue'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    title: z.string().min(1, 'Issue title is required'),
    body: z.string().optional().describe('Issue description in markdown'),
    labels: z.array(z.string()).optional().describe('Issue labels'),
    assignees: z.array(z.string()).optional().describe('User login to assign'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const UpdateIssueParamsSchema = z.object({
    operation: z.literal('updateIssue'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    issueNumber: z.number().int().positive().describe('Issue number'),
    title: z.string().optional().describe('Updated issue title'),
    body: z.string().optional().describe('Updated issue description'),
    state: z.enum(['open', 'closed']).optional().describe('Issue state'),
    labels: z.array(z.string()).optional().describe('Issue labels'),
    assignees: z.array(z.string()).optional().describe('User login to assign'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const ListIssuesParamsSchema = z.object({
    operation: z.literal('listIssues'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    state: z.enum(['open', 'closed', 'all']).optional().default('open'),
    labels: z.array(z.string()).optional().describe('Filter by labels'),
    creator: z.string().optional().describe('Filter by creator'),
    assignee: z.string().optional().describe('Filter by assignee'),
    limit: z.number().int().positive().optional().default(30),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const CreatePullRequestParamsSchema = z.object({
    operation: z.literal('createPullRequest'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    title: z.string().min(1, 'Pull request title is required'),
    head: z.string().min(1, 'Branch name containing changes'),
    base: z.string().min(1, 'Branch name to merge into'),
    body: z.string().optional().describe('Pull request description in markdown'),
    draft: z.boolean().optional().default(false).describe('Create as draft PR'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const MergePullRequestParamsSchema = z.object({
    operation: z.literal('mergePullRequest'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    pullNumber: z.number().int().positive().describe('Pull request number'),
    commitTitle: z.string().optional().describe('Merge commit title'),
    commitMessage: z.string().optional().describe('Merge commit message'),
    mergeMethod: z.enum(['merge', 'squash', 'rebase']).optional().default('merge'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const ListPullRequestsParamsSchema = z.object({
    operation: z.literal('listPullRequests'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    state: z.enum(['open', 'closed', 'all']).optional().default('open'),
    head: z.string().optional().describe('Filter by head branch'),
    base: z.string().optional().describe('Filter by base branch'),
    limit: z.number().int().positive().optional().default(30),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const CreateBranchParamsSchema = z.object({
    operation: z.literal('createBranch'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    branchName: z.string().min(1, 'New branch name'),
    fromBranch: z.string().optional().default('main').describe('Branch to create from'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const DeleteBranchParamsSchema = z.object({
    operation: z.literal('deleteBranch'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    branchName: z.string().min(1, 'Branch name to delete'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const GetRepositoryParamsSchema = z.object({
    operation: z.literal('getRepository'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const CreateCommitParamsSchema = z.object({
    operation: z.literal('createCommit'),
    owner: z.string().min(1, 'Repository owner is required'),
    repo: z.string().min(1, 'Repository name is required'),
    branch: z.string().min(1, 'Branch to commit to'),
    message: z.string().min(1, 'Commit message'),
    files: z.array(z.object({
        path: z.string().describe('File path in repository'),
        content: z.string().describe('File content'),
        mode: z.enum(['100644', '100755', '040000', '160000', '120000']).optional().default('100644'),
    })).min(1, 'At least one file is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
// Union of all parameter schemas
const GithubBubbleParamsSchema = z.discriminatedUnion('operation', [
    CreateIssueParamsSchema,
    UpdateIssueParamsSchema,
    ListIssuesParamsSchema,
    CreatePullRequestParamsSchema,
    MergePullRequestParamsSchema,
    ListPullRequestsParamsSchema,
    CreateBranchParamsSchema,
    DeleteBranchParamsSchema,
    GetRepositoryParamsSchema,
    CreateCommitParamsSchema,
]);
// Result schema
const GithubBubbleResultSchema = z.object({
    success: z.boolean(),
    data: z.unknown().describe('Operation result data'),
    error: z.string(),
    meta: z.object({
        operation: z.string(),
        repository: z.string().optional(),
        owner: z.string().optional(),
    }),
});
// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================
export class GithubBubble extends ServiceBubble {
    static service = 'github';
    static authType = 'token';
    static bubbleName = 'github';
    static type = 'service';
    static schema = GithubBubbleParamsSchema;
    static resultSchema = GithubBubbleResultSchema;
    static shortDescription = 'Version control and collaborative development platform';
    static longDescription = `
    GitHub Bubble for repository management and development operations.

    Features:
    - Issue and pull request management
    - Branch operations (create, delete)
    - Repository information retrieval
    - Commit creation and file management
    - Team collaboration tools
    - CI/CD integration capabilities

    Use cases:
    - Automated issue tracking
    - Pull request automation
    - Release management
    - Code review workflows
    - Repository monitoring
    - Development analytics
  `;
    static alias = 'git';
    authToken = null;
    baseUrl = 'https://api.github.com';
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    getCredentialType() {
        return CredentialType.GITHUB_CRED;
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('GitHub credentials are required');
        }
        return credentials[CredentialType.GITHUB_CRED];
    }
    async testCredential() {
        try {
            const token = this.getToken();
            const response = await fetch(`${this.baseUrl}/user`, {
                method: 'GET',
                headers: {
                    'Authorization': `token ${token}`,
                    'Accept': 'application/vnd.github.v3+json',
                },
            });
            return response.ok;
        }
        catch (error) {
            console.error('[GitHub] Credential test failed:', error);
            return false;
        }
    }
    getToken() {
        if (!this.authToken) {
            const credential = this.chooseCredential();
            if (!credential) {
                throw new Error('GitHub credentials not found');
            }
            // Parse credential (expected format: JSON string with token or accessToken)
            let config;
            try {
                config = typeof credential === 'string' ? JSON.parse(credential) : credential;
            }
            catch {
                throw new Error('Invalid GitHub credentials format. Expected JSON string.');
            }
            if (!config.token && !config.accessToken && !config.personalAccessToken) {
                throw new Error('GitHub token is required in credentials');
            }
            this.authToken = config.token || config.accessToken || config.personalAccessToken;
            console.log('[GitHub] Token initialized successfully');
        }
        if (!this.authToken) {
            throw new Error('GitHub token initialization failed');
        }
        return this.authToken;
    }
    async performAction(context) {
        void context;
        try {
            const operation = this.params.operation;
            let result;
            console.log(`[GitHub] Executing operation: ${operation}`);
            switch (operation) {
                case 'createIssue':
                    result = await this.createIssue();
                    break;
                case 'updateIssue':
                    result = await this.updateIssue();
                    break;
                case 'listIssues':
                    result = await this.listIssues();
                    break;
                case 'createPullRequest':
                    result = await this.createPullRequest();
                    break;
                case 'mergePullRequest':
                    result = await this.mergePullRequest();
                    break;
                case 'listPullRequests':
                    result = await this.listPullRequests();
                    break;
                case 'createBranch':
                    result = await this.createBranch();
                    break;
                case 'deleteBranch':
                    result = await this.deleteBranch();
                    break;
                case 'getRepository':
                    result = await this.getRepository();
                    break;
                case 'createCommit':
                    result = await this.createCommit();
                    break;
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
            return {
                success: true,
                data: result,
                error: '', // Empty string for successful operations
                meta: {
                    operation,
                    repository: this.extractRepo(),
                    owner: this.extractOwner(),
                },
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[GitHub] Operation failed:`, errorMessage);
            return {
                success: false,
                data: null,
                error: errorMessage,
                meta: {
                    operation: this.params.operation,
                    repository: this.extractRepo(),
                    owner: this.extractOwner(),
                },
            };
        }
    }
    async makeRequest(method, endpoint, body) {
        const token = this.getToken();
        const headers = {
            'Authorization': `token ${token}`,
            'Accept': 'application/vnd.github.v3+json',
        };
        if (body) {
            headers['Content-Type'] = 'application/json';
        }
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method,
            headers,
            body: body ? JSON.stringify(body) : undefined,
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `GitHub API error: ${response.statusText}`);
        }
        return response.json();
    }
    async createIssue() {
        const params = this.params;
        const body = {
            title: params.title,
        };
        if (params.body) {
            body.body = params.body;
        }
        if (params.labels && params.labels.length > 0) {
            body.labels = params.labels;
        }
        if (params.assignees && params.assignees.length > 0) {
            body.assignees = params.assignees;
        }
        const result = await this.makeRequest('POST', `/repos/${params.owner}/${params.repo}/issues`, body);
        console.log(`[GitHub] Issue created: #${result.number} in ${params.owner}/${params.repo}`);
        return {
            issueNumber: result.number,
            title: result.title,
            state: result.state,
            url: result.html_url,
            status: 'created',
        };
    }
    async updateIssue() {
        const params = this.params;
        const body = {};
        if (params.title !== undefined) {
            body.title = params.title;
        }
        if (params.body !== undefined) {
            body.body = params.body;
        }
        if (params.state) {
            body.state = params.state;
        }
        if (params.labels) {
            body.labels = params.labels;
        }
        if (params.assignees) {
            body.assignees = params.assignees;
        }
        const result = await this.makeRequest('PATCH', `/repos/${params.owner}/${params.repo}/issues/${params.issueNumber}`, body);
        console.log(`[GitHub] Issue updated: #${params.issueNumber} in ${params.owner}/${params.repo}`);
        return {
            issueNumber: result.number,
            title: result.title,
            state: result.state,
            url: result.html_url,
            status: 'updated',
        };
    }
    async listIssues() {
        const params = this.params;
        const queryParams = new URLSearchParams({
            state: params.state,
            per_page: String(params.limit),
        });
        if (params.labels && params.labels.length > 0) {
            queryParams.append('labels', params.labels.join(','));
        }
        if (params.creator) {
            queryParams.append('creator', params.creator);
        }
        if (params.assignee) {
            queryParams.append('assignee', params.assignee);
        }
        const result = await this.makeRequest('GET', `/repos/${params.owner}/${params.repo}/issues?${queryParams.toString()}`);
        console.log(`[GitHub] Listed ${result.length} issues in ${params.owner}/${params.repo}`);
        return {
            issues: result.map((issue) => ({
                number: issue.number,
                title: issue.title,
                state: issue.state,
                author: issue.user?.login,
                assignees: issue.assignees?.map((a) => a.login) || [],
                labels: issue.labels?.map((l) => l.name) || [],
                createdAt: issue.created_at,
                url: issue.html_url,
            })),
            count: result.length,
        };
    }
    async createPullRequest() {
        const params = this.params;
        const body = {
            title: params.title,
            head: params.head,
            base: params.base,
            draft: params.draft,
        };
        if (params.body) {
            body.body = params.body;
        }
        const result = await this.makeRequest('POST', `/repos/${params.owner}/${params.repo}/pulls`, body);
        console.log(`[GitHub] PR created: #${result.number} in ${params.owner}/${params.repo}`);
        return {
            pullNumber: result.number,
            title: result.title,
            state: result.state,
            url: result.html_url,
            draft: result.draft,
            status: 'created',
        };
    }
    async mergePullRequest() {
        const params = this.params;
        const body = {
            merge_method: params.mergeMethod,
        };
        if (params.commitTitle) {
            body.commit_title = params.commitTitle;
        }
        if (params.commitMessage) {
            body.commit_message = params.commitMessage;
        }
        const result = await this.makeRequest('PUT', `/repos/${params.owner}/${params.repo}/pulls/${params.pullNumber}/merge`, body);
        console.log(`[GitHub] PR merged: #${params.pullNumber} in ${params.owner}/${params.repo}`);
        return {
            pullNumber: params.pullNumber,
            merged: result.merged,
            message: result.message,
            sha: result.sha,
            status: 'merged',
        };
    }
    async listPullRequests() {
        const params = this.params;
        const queryParams = new URLSearchParams({
            state: params.state,
            per_page: String(params.limit),
        });
        if (params.head) {
            queryParams.append('head', params.head);
        }
        if (params.base) {
            queryParams.append('base', params.base);
        }
        const result = await this.makeRequest('GET', `/repos/${params.owner}/${params.repo}/pulls?${queryParams.toString()}`);
        console.log(`[GitHub] Listed ${result.length} PRs in ${params.owner}/${params.repo}`);
        return {
            pullRequests: result.map((pr) => ({
                number: pr.number,
                title: pr.title,
                state: pr.state,
                author: pr.user?.login,
                head: pr.head?.ref,
                base: pr.base?.ref,
                draft: pr.draft,
                createdAt: pr.created_at,
                url: pr.html_url,
            })),
            count: result.length,
        };
    }
    async createBranch() {
        const params = this.params;
        // First, get the SHA of the base branch
        const baseBranch = await this.makeRequest('GET', `/repos/${params.owner}/${params.repo}/git/refs/heads/${params.fromBranch}`);
        const sha = baseBranch.object.sha;
        // Create the new branch
        const body = {
            ref: `refs/heads/${params.branchName}`,
            sha,
        };
        const result = await this.makeRequest('POST', `/repos/${params.owner}/${params.repo}/git/refs`, body);
        console.log(`[GitHub] Branch created: ${params.branchName} from ${params.fromBranch}`);
        return {
            branchName: params.branchName,
            fromBranch: params.fromBranch,
            sha: result.object.sha,
            status: 'created',
        };
    }
    async deleteBranch() {
        const params = this.params;
        await this.makeRequest('DELETE', `/repos/${params.owner}/${params.repo}/git/refs/heads/${params.branchName}`);
        console.log(`[GitHub] Branch deleted: ${params.branchName}`);
        return {
            branchName: params.branchName,
            status: 'deleted',
        };
    }
    async getRepository() {
        const params = this.params;
        const result = await this.makeRequest('GET', `/repos/${params.owner}/${params.repo}`);
        console.log(`[GitHub] Retrieved repository info: ${params.owner}/${params.repo}`);
        return {
            repository: {
                name: result.name,
                fullName: result.full_name,
                description: result.description,
                private: result.private,
                language: result.language,
                stars: result.stargazers_count,
                forks: result.forks_count,
                openIssues: result.open_issues_count,
                createdAt: result.created_at,
                updatedAt: result.updated_at,
                url: result.html_url,
            },
        };
    }
    async createCommit() {
        const params = this.params;
        // Get the current commit SHA for the branch
        const branchRef = await this.makeRequest('GET', `/repos/${params.owner}/${params.repo}/git/refs/heads/${params.branch}`);
        const treeItems = await Promise.all(params.files.map(async (file) => {
            // Create blob for each file
            const blob = await this.makeRequest('POST', `/repos/${params.owner}/${params.repo}/git/blobs`, {
                content: file.content,
                encoding: 'utf-8',
            });
            return {
                path: file.path,
                mode: file.mode,
                type: 'blob',
                sha: blob.sha,
            };
        }));
        // Create tree
        const tree = await this.makeRequest('POST', `/repos/${params.owner}/${params.repo}/git/trees`, {
            base_tree: branchRef.object.sha,
            tree: treeItems,
        });
        // Create commit
        const commit = await this.makeRequest('POST', `/repos/${params.owner}/${params.repo}/git/commits`, {
            message: params.message,
            tree: tree.sha,
            parents: [branchRef.object.sha],
        });
        // Update branch reference
        await this.makeRequest('PATCH', `/repos/${params.owner}/${params.repo}/git/refs/heads/${params.branch}`, {
            sha: commit.sha,
        });
        console.log(`[GitHub] Commit created: ${commit.sha.substring(0, 7)} on ${params.branch}`);
        return {
            sha: commit.sha,
            message: params.message,
            branch: params.branch,
            files: params.files.map((f) => f.path),
            status: 'committed',
        };
    }
    extractRepo() {
        const params = this.params;
        return params.repo;
    }
    extractOwner() {
        const params = this.params;
        return params.owner;
    }
}
//# sourceMappingURL=github-bubble.js.map