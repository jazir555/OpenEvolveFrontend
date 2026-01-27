import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// GitHub API base URL
const GITHUB_API_BASE = 'https://api.github.com';
// GitHub file content schema
const GithubFileContentSchema = z.object({
    name: z.string().describe('File name'),
    path: z.string().describe('Full path to the file in the repository'),
    sha: z.string().describe('Git SHA hash of the file'),
    size: z.number().describe('File size in bytes'),
    url: z.string().describe('API URL for this file'),
    html_url: z.string().describe('Web URL to view the file on GitHub'),
    git_url: z.string().describe('Git URL for the file object'),
    download_url: z
        .string()
        .nullable()
        .describe('Direct download URL for the file'),
    type: z
        .enum(['file', 'dir', 'symlink', 'submodule'])
        .describe('Type of the content'),
    content: z.string().optional().describe('Base64 encoded content (for files)'),
    encoding: z.string().optional().describe('Encoding type (usually base64)'),
});
// GitHub repository schema
const GithubRepositorySchema = z.object({
    id: z.number().describe('Repository ID'),
    node_id: z.string().describe('GraphQL node ID'),
    name: z.string().describe('Repository name'),
    full_name: z.string().describe('Full repository name (owner/repo)'),
    private: z.boolean().describe('Whether the repository is private'),
    owner: z
        .object({
        login: z.string().describe('Owner username'),
        id: z.number().describe('Owner ID'),
        avatar_url: z.string().describe('Owner avatar URL'),
        html_url: z.string().describe('Owner profile URL'),
    })
        .describe('Repository owner information'),
    html_url: z.string().describe('Repository web URL'),
    description: z.string().nullable().describe('Repository description'),
    fork: z.boolean().describe('Whether this is a fork'),
    created_at: z.string().describe('ISO datetime when repository was created'),
    updated_at: z
        .string()
        .describe('ISO datetime when repository was last updated'),
    pushed_at: z.string().describe('ISO datetime of last push'),
    size: z.number().describe('Repository size in KB'),
    stargazers_count: z.number().describe('Number of stars'),
    watchers_count: z.number().describe('Number of watchers'),
    language: z.string().nullable().describe('Primary programming language'),
    forks_count: z.number().describe('Number of forks'),
    open_issues_count: z.number().describe('Number of open issues'),
    default_branch: z.string().describe('Default branch name'),
    visibility: z
        .string()
        .optional()
        .describe('Repository visibility (public, private, internal)'),
});
// GitHub pull request schema
const GithubPullRequestSchema = z.object({
    id: z.number().describe('Pull request ID'),
    node_id: z.string().describe('GraphQL node ID'),
    number: z.number().describe('Pull request number'),
    state: z.enum(['open', 'closed']).describe('Pull request state'),
    title: z.string().describe('Pull request title'),
    body: z.string().nullable().describe('Pull request description'),
    created_at: z.string().describe('ISO datetime when PR was created'),
    updated_at: z.string().describe('ISO datetime when PR was last updated'),
    closed_at: z.string().nullable().describe('ISO datetime when PR was closed'),
    merged_at: z.string().nullable().describe('ISO datetime when PR was merged'),
    user: z
        .object({
        login: z.string().describe('Author username'),
        id: z.number().describe('Author ID'),
        avatar_url: z.string().describe('Author avatar URL'),
    })
        .describe('Pull request author'),
    html_url: z.string().describe('Web URL to view the PR'),
    draft: z.boolean().describe('Whether this is a draft PR'),
    head: z
        .object({
        ref: z.string().describe('Source branch name'),
        sha: z.string().describe('Source commit SHA'),
    })
        .describe('Source branch information'),
    base: z
        .object({
        ref: z.string().describe('Target branch name'),
        sha: z.string().describe('Target commit SHA'),
    })
        .describe('Target branch information'),
    merged: z
        .boolean()
        .optional()
        .describe('Whether the PR has been merged (may not be present in list responses)'),
    mergeable: z
        .boolean()
        .nullable()
        .optional()
        .describe('Whether the PR can be merged (may not be present in list responses)'),
    mergeable_state: z
        .string()
        .optional()
        .describe('Mergeable state (clean, unstable, dirty, etc.)'),
    comments: z.number().optional().describe('Number of comments'),
    review_comments: z.number().optional().describe('Number of review comments'),
    commits: z.number().optional().describe('Number of commits'),
    additions: z.number().optional().describe('Lines added'),
    deletions: z.number().optional().describe('Lines deleted'),
    changed_files: z.number().optional().describe('Number of files changed'),
});
// GitHub issue/PR comment schema
const GithubCommentSchema = z.object({
    id: z.number().describe('Comment ID'),
    node_id: z.string().describe('GraphQL node ID'),
    body: z.string().describe('Comment text content'),
    user: z
        .object({
        login: z.string().describe('Comment author username'),
        id: z.number().describe('Comment author ID'),
    })
        .describe('Comment author information'),
    created_at: z.string().describe('ISO datetime when comment was created'),
    updated_at: z.string().describe('ISO datetime when comment was last updated'),
    html_url: z.string().describe('Web URL to view the comment'),
});
// GitHub issue schema
const GithubIssueSchema = z.object({
    id: z.number().describe('Issue ID'),
    node_id: z.string().describe('GraphQL node ID'),
    number: z.number().describe('Issue number'),
    state: z.enum(['open', 'closed']).describe('Issue state'),
    title: z.string().describe('Issue title'),
    body: z.string().nullable().describe('Issue description'),
    user: z
        .object({
        login: z.string().describe('Issue creator username'),
        id: z.number().describe('Issue creator ID'),
    })
        .describe('Issue creator information'),
    labels: z
        .array(z.object({
        id: z.number().describe('Label ID'),
        name: z.string().describe('Label name'),
        color: z.string().describe('Label color (hex)'),
        description: z.string().nullable().describe('Label description'),
    }))
        .describe('Issue labels'),
    created_at: z.string().describe('ISO datetime when issue was created'),
    updated_at: z.string().describe('ISO datetime when issue was last updated'),
    closed_at: z
        .string()
        .nullable()
        .describe('ISO datetime when issue was closed'),
    html_url: z.string().describe('Web URL to view the issue'),
    comments: z.number().describe('Number of comments'),
});
// Define the parameters schema for different GitHub operations
const GithubParamsSchema = z.discriminatedUnion('operation', [
    // Get file content operation
    z.object({
        operation: z
            .literal('get_file')
            .describe('Get the contents of a file from a GitHub repository'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        path: z
            .string()
            .min(1, 'File path is required')
            .describe('Path to the file in the repository (e.g., src/index.ts)'),
        ref: z
            .string()
            .optional()
            .describe('Git reference (branch, tag, or commit SHA). Defaults to the default branch'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get directory contents operation
    z.object({
        operation: z
            .literal('get_directory')
            .describe('Get the contents of a directory from a GitHub repository'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        path: z
            .string()
            .optional()
            .default('')
            .describe('Path to the directory in the repository (empty string for root)'),
        ref: z
            .string()
            .optional()
            .describe('Git reference (branch, tag, or commit SHA). Defaults to the default branch'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // List pull requests operation
    z.object({
        operation: z
            .literal('list_pull_requests')
            .describe('List pull requests in a GitHub repository'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        state: z
            .enum(['open', 'closed', 'all'])
            .optional()
            .default('open')
            .describe('Filter by PR state'),
        sort: z
            .enum(['created', 'updated', 'popularity', 'long-running'])
            .optional()
            .default('created')
            .describe('Sort order for results'),
        direction: z
            .enum(['asc', 'desc'])
            .optional()
            .default('desc')
            .describe('Sort direction'),
        per_page: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(30)
            .describe('Number of results per page (1-100)'),
        page: z
            .number()
            .min(1)
            .optional()
            .default(1)
            .describe('Page number for pagination'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get pull request details operation
    z.object({
        operation: z
            .literal('get_pull_request')
            .describe('Get detailed information about a specific pull request'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        pull_number: z
            .number()
            .min(1, 'Pull request number is required')
            .describe('Pull request number'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Create PR comment operation
    z.object({
        operation: z
            .literal('create_pr_comment')
            .describe('Add a comment to a pull request'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        pull_number: z
            .number()
            .min(1, 'Pull request number is required')
            .describe('Pull request number'),
        body: z
            .string()
            .min(1, 'Comment text is required')
            .describe('Comment text content (supports GitHub Markdown)'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // List repositories operation
    z.object({
        operation: z
            .literal('list_repositories')
            .describe('List repositories for the authenticated user'),
        visibility: z
            .enum(['all', 'public', 'private'])
            .optional()
            .default('all')
            .describe('Filter by repository visibility'),
        affiliation: z
            .enum(['owner', 'collaborator', 'organization_member'])
            .optional()
            .default('owner')
            .describe('Filter by user affiliation'),
        sort: z
            .enum(['created', 'updated', 'pushed', 'full_name'])
            .optional()
            .default('updated')
            .describe('Sort order for results'),
        direction: z
            .enum(['asc', 'desc'])
            .optional()
            .default('desc')
            .describe('Sort direction'),
        per_page: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(30)
            .describe('Number of results per page (1-100)'),
        page: z
            .number()
            .min(1)
            .optional()
            .default(1)
            .describe('Page number for pagination'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get repository details operation
    z.object({
        operation: z
            .literal('get_repository')
            .describe('Get detailed information about a specific repository'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Create issue comment operation
    z.object({
        operation: z
            .literal('create_issue_comment')
            .describe('Add a comment to an issue'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        issue_number: z
            .number()
            .min(1, 'Issue number is required')
            .describe('Issue number'),
        body: z
            .string()
            .min(1, 'Comment text is required')
            .describe('Comment text content (supports GitHub Markdown)'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // List issues operation
    z.object({
        operation: z
            .literal('list_issues')
            .describe('List issues in a GitHub repository'),
        owner: z
            .string()
            .min(1, 'Repository owner is required')
            .describe('Repository owner (username or organization name)'),
        repo: z
            .string()
            .min(1, 'Repository name is required')
            .describe('Repository name'),
        state: z
            .enum(['open', 'closed', 'all'])
            .optional()
            .default('open')
            .describe('Filter by issue state'),
        labels: z
            .string()
            .optional()
            .describe('Filter by labels (comma-separated list)'),
        sort: z
            .enum(['created', 'updated', 'comments'])
            .optional()
            .default('created')
            .describe('Sort order for results'),
        direction: z
            .enum(['asc', 'desc'])
            .optional()
            .default('desc')
            .describe('Sort direction'),
        per_page: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(30)
            .describe('Number of results per page (1-100)'),
        page: z
            .number()
            .min(1)
            .optional()
            .default(1)
            .describe('Page number for pagination'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
]);
// Define the result schemas for different GitHub operations
const GithubResultSchema = z.discriminatedUnion('operation', [
    z
        .object({
        operation: z.literal('get_file'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
    })
        .merge(GithubFileContentSchema.partial()),
    z.object({
        operation: z.literal('get_directory'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
        contents: z
            .array(GithubFileContentSchema)
            .optional()
            .describe('Array of directory contents'),
    }),
    z.object({
        operation: z.literal('list_pull_requests'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
        pull_requests: z
            .array(GithubPullRequestSchema)
            .optional()
            .describe('Array of pull requests'),
    }),
    z
        .object({
        operation: z.literal('get_pull_request'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
    })
        .merge(GithubPullRequestSchema.partial()),
    z
        .object({
        operation: z.literal('create_pr_comment'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
    })
        .merge(GithubCommentSchema.partial()),
    z.object({
        operation: z.literal('list_repositories'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
        repositories: z
            .array(GithubRepositorySchema)
            .optional()
            .describe('Array of repositories'),
    }),
    z
        .object({
        operation: z.literal('get_repository'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
    })
        .merge(GithubRepositorySchema.partial()),
    z
        .object({
        operation: z.literal('create_issue_comment'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
    })
        .merge(GithubCommentSchema.partial()),
    z.object({
        operation: z.literal('list_issues'),
        success: z.boolean().describe('Whether the operation succeeded'),
        error: z.string().describe('Error message if operation failed'),
        issues: z.array(GithubIssueSchema).optional().describe('Array of issues'),
    }),
]);
export class GithubBubble extends ServiceBubble {
    static type = 'service';
    static service = 'github';
    static authType = 'apikey';
    static bubbleName = 'github';
    static schema = GithubParamsSchema;
    static resultSchema = GithubResultSchema;
    static shortDescription = 'GitHub API integration for repository operations';
    static longDescription = `
    GitHub API integration for accessing repositories, pull requests, issues, and files.
    
    Features:
    - Get file contents from repositories
    - List and browse directory contents
    - Manage pull requests (list, get details, comment)
    - Manage issues (list, comment)
    - List and get repository information
    - Non-sensitive read and comment operations only
    
    Use cases:
    - Code review automation and PR management
    - Repository file access and content retrieval
    - Issue and PR comment automation
    - Repository exploration and documentation
    - CI/CD integration and status checks
    
    Security Features:
    - Personal access token authentication (GitHub PAT)
    - Read-only operations with safe comment capabilities
    - No file deletion or destructive operations
    - Respects repository permissions
  `;
    static alias = 'gh';
    constructor(params = {
        operation: 'get_repository',
        owner: 'octocat',
        repo: 'Hello-World',
    }, context) {
        super(params, context);
    }
    async testCredential() {
        try {
            // Test the credential by fetching the authenticated user
            const token = this.chooseCredential();
            if (!token) {
                return false;
            }
            const response = await fetch(`${GITHUB_API_BASE}/user`, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            return response.ok;
        }
        catch (error) {
            console.error('GitHub credential test failed:', error);
            return false;
        }
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            return undefined;
        }
        return credentials[CredentialType.GITHUB_TOKEN];
    }
    async performAction(context) {
        void context;
        const { operation } = this.params;
        switch (operation) {
            case 'get_file':
                return this.handleGetFile(this.params);
            case 'get_directory':
                return this.handleGetDirectory(this.params);
            case 'list_pull_requests':
                return this.handleListPullRequests(this.params);
            case 'get_pull_request':
                return this.handleGetPullRequest(this.params);
            case 'create_pr_comment':
                return this.handleCreatePrComment(this.params);
            case 'list_repositories':
                return this.handleListRepositories(this.params);
            case 'get_repository':
                return this.handleGetRepository(this.params);
            case 'create_issue_comment':
                return this.handleCreateIssueComment(this.params);
            case 'list_issues':
                return this.handleListIssues(this.params);
            default:
                return {
                    operation: operation,
                    success: false,
                    error: `Unknown operation: ${operation}`,
                };
        }
    }
    async handleGetFile(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo, path, ref } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'get_file',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            let url = `${GITHUB_API_BASE}/repos/${owner}/${repo}/contents/${path}`;
            if (ref) {
                url += `?ref=${encodeURIComponent(ref)}`;
            }
            const response = await fetch(url, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'get_file',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = GithubFileContentSchema.parse(data);
            return {
                operation: 'get_file',
                success: true,
                error: '',
                ...validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'get_file',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleGetDirectory(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo, path, ref } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'get_directory',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            let url = `${GITHUB_API_BASE}/repos/${owner}/${repo}/contents/${path}`;
            if (ref) {
                url += `?ref=${encodeURIComponent(ref)}`;
            }
            const response = await fetch(url, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'get_directory',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = z.array(GithubFileContentSchema).parse(data);
            return {
                operation: 'get_directory',
                success: true,
                error: '',
                contents: validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'get_directory',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleListPullRequests(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo, state, sort, direction, per_page, page } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'list_pull_requests',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            const url = new URL(`${GITHUB_API_BASE}/repos/${owner}/${repo}/pulls`);
            url.searchParams.set('state', state);
            url.searchParams.set('sort', sort);
            url.searchParams.set('direction', direction);
            url.searchParams.set('per_page', per_page.toString());
            url.searchParams.set('page', page.toString());
            const response = await fetch(url.toString(), {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'list_pull_requests',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = z.array(GithubPullRequestSchema).parse(data);
            return {
                operation: 'list_pull_requests',
                success: true,
                error: '',
                pull_requests: validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'list_pull_requests',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleGetPullRequest(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo, pull_number } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'get_pull_request',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            const url = `${GITHUB_API_BASE}/repos/${owner}/${repo}/pulls/${pull_number}`;
            const response = await fetch(url, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'get_pull_request',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = GithubPullRequestSchema.parse(data);
            return {
                operation: 'get_pull_request',
                success: true,
                error: '',
                ...validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'get_pull_request',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleCreatePrComment(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo, pull_number, body } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'create_pr_comment',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            const url = `${GITHUB_API_BASE}/repos/${owner}/${repo}/issues/${pull_number}/comments`;
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ body }),
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'create_pr_comment',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = GithubCommentSchema.parse(data);
            return {
                operation: 'create_pr_comment',
                success: true,
                error: '',
                ...validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'create_pr_comment',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleListRepositories(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { visibility, affiliation, sort, direction, per_page, page } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'list_repositories',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            const url = new URL(`${GITHUB_API_BASE}/user/repos`);
            url.searchParams.set('visibility', visibility);
            url.searchParams.set('affiliation', affiliation);
            url.searchParams.set('sort', sort);
            url.searchParams.set('direction', direction);
            url.searchParams.set('per_page', per_page.toString());
            url.searchParams.set('page', page.toString());
            const response = await fetch(url.toString(), {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'list_repositories',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = z.array(GithubRepositorySchema).parse(data);
            return {
                operation: 'list_repositories',
                success: true,
                error: '',
                repositories: validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'list_repositories',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleGetRepository(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'get_repository',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            const url = `${GITHUB_API_BASE}/repos/${owner}/${repo}`;
            const response = await fetch(url, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'get_repository',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = GithubRepositorySchema.parse(data);
            return {
                operation: 'get_repository',
                success: true,
                error: '',
                ...validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'get_repository',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleCreateIssueComment(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo, issue_number, body } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'create_issue_comment',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            const url = `${GITHUB_API_BASE}/repos/${owner}/${repo}/issues/${issue_number}/comments`;
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ body }),
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'create_issue_comment',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = GithubCommentSchema.parse(data);
            return {
                operation: 'create_issue_comment',
                success: true,
                error: '',
                ...validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'create_issue_comment',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    async handleListIssues(params) {
        const parsed = GithubParamsSchema.parse(params);
        const { owner, repo, state, labels, sort, direction, per_page, page } = parsed;
        try {
            const token = this.chooseCredential();
            if (!token) {
                return {
                    operation: 'list_issues',
                    success: false,
                    error: 'GitHub token credential not found',
                };
            }
            const url = new URL(`${GITHUB_API_BASE}/repos/${owner}/${repo}/issues`);
            url.searchParams.set('state', state);
            if (labels) {
                url.searchParams.set('labels', labels);
            }
            url.searchParams.set('sort', sort);
            url.searchParams.set('direction', direction);
            url.searchParams.set('per_page', per_page.toString());
            url.searchParams.set('page', page.toString());
            const response = await fetch(url.toString(), {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            });
            if (!response.ok) {
                const error = await response.text();
                return {
                    operation: 'list_issues',
                    success: false,
                    error: `GitHub API error: ${response.status} ${error}`,
                };
            }
            const data = await response.json();
            const validatedData = z.array(GithubIssueSchema).parse(data);
            return {
                operation: 'list_issues',
                success: true,
                error: '',
                issues: validatedData,
            };
        }
        catch (error) {
            return {
                operation: 'list_issues',
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
}
//# sourceMappingURL=github.js.map