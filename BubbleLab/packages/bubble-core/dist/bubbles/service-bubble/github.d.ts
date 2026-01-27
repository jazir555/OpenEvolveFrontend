import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const GithubParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"get_file">;
    owner: z.ZodString;
    repo: z.ZodString;
    path: z.ZodString;
    ref: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    path: string;
    operation: "get_file";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ref?: string | undefined;
}, {
    path: string;
    operation: "get_file";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ref?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_directory">;
    owner: z.ZodString;
    repo: z.ZodString;
    path: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    ref: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    path: string;
    operation: "get_directory";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ref?: string | undefined;
}, {
    operation: "get_directory";
    owner: string;
    repo: string;
    path?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    ref?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_pull_requests">;
    owner: z.ZodString;
    repo: z.ZodString;
    state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
    sort: z.ZodDefault<z.ZodOptional<z.ZodEnum<["created", "updated", "popularity", "long-running"]>>>;
    direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
    per_page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    sort: "created" | "updated" | "popularity" | "long-running";
    operation: "list_pull_requests";
    owner: string;
    page: number;
    state: "closed" | "open" | "all";
    repo: string;
    direction: "asc" | "desc";
    per_page: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "list_pull_requests";
    owner: string;
    repo: string;
    sort?: "created" | "updated" | "popularity" | "long-running" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    page?: number | undefined;
    state?: "closed" | "open" | "all" | undefined;
    direction?: "asc" | "desc" | undefined;
    per_page?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_pull_request">;
    owner: z.ZodString;
    repo: z.ZodString;
    pull_number: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_pull_request";
    owner: string;
    repo: string;
    pull_number: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_pull_request";
    owner: string;
    repo: string;
    pull_number: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_pr_comment">;
    owner: z.ZodString;
    repo: z.ZodString;
    pull_number: z.ZodNumber;
    body: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_pr_comment";
    body: string;
    owner: string;
    repo: string;
    pull_number: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "create_pr_comment";
    body: string;
    owner: string;
    repo: string;
    pull_number: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_repositories">;
    visibility: z.ZodDefault<z.ZodOptional<z.ZodEnum<["all", "public", "private"]>>>;
    affiliation: z.ZodDefault<z.ZodOptional<z.ZodEnum<["owner", "collaborator", "organization_member"]>>>;
    sort: z.ZodDefault<z.ZodOptional<z.ZodEnum<["created", "updated", "pushed", "full_name"]>>>;
    direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
    per_page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    sort: "created" | "updated" | "full_name" | "pushed";
    operation: "list_repositories";
    page: number;
    direction: "asc" | "desc";
    per_page: number;
    visibility: "public" | "private" | "all";
    affiliation: "owner" | "collaborator" | "organization_member";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "list_repositories";
    sort?: "created" | "updated" | "full_name" | "pushed" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    page?: number | undefined;
    direction?: "asc" | "desc" | undefined;
    per_page?: number | undefined;
    visibility?: "public" | "private" | "all" | undefined;
    affiliation?: "owner" | "collaborator" | "organization_member" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_repository">;
    owner: z.ZodString;
    repo: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_repository";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_repository";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_issue_comment">;
    owner: z.ZodString;
    repo: z.ZodString;
    issue_number: z.ZodNumber;
    body: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_issue_comment";
    body: string;
    owner: string;
    repo: string;
    issue_number: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "create_issue_comment";
    body: string;
    owner: string;
    repo: string;
    issue_number: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_issues">;
    owner: z.ZodString;
    repo: z.ZodString;
    state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
    labels: z.ZodOptional<z.ZodString>;
    sort: z.ZodDefault<z.ZodOptional<z.ZodEnum<["created", "updated", "comments"]>>>;
    direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
    per_page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    sort: "created" | "updated" | "comments";
    operation: "list_issues";
    owner: string;
    page: number;
    state: "closed" | "open" | "all";
    repo: string;
    direction: "asc" | "desc";
    per_page: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    labels?: string | undefined;
}, {
    operation: "list_issues";
    owner: string;
    repo: string;
    sort?: "created" | "updated" | "comments" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    labels?: string | undefined;
    page?: number | undefined;
    state?: "closed" | "open" | "all" | undefined;
    direction?: "asc" | "desc" | undefined;
    per_page?: number | undefined;
}>]>;
declare const GithubResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"get_file">;
    success: z.ZodBoolean;
    error: z.ZodString;
} & {
    name: z.ZodOptional<z.ZodString>;
    path: z.ZodOptional<z.ZodString>;
    sha: z.ZodOptional<z.ZodString>;
    size: z.ZodOptional<z.ZodNumber>;
    url: z.ZodOptional<z.ZodString>;
    html_url: z.ZodOptional<z.ZodString>;
    git_url: z.ZodOptional<z.ZodString>;
    download_url: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    type: z.ZodOptional<z.ZodEnum<["file", "dir", "symlink", "submodule"]>>;
    content: z.ZodOptional<z.ZodOptional<z.ZodString>>;
    encoding: z.ZodOptional<z.ZodOptional<z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_file";
    path?: string | undefined;
    type?: "file" | "dir" | "symlink" | "submodule" | undefined;
    name?: string | undefined;
    encoding?: string | undefined;
    content?: string | undefined;
    url?: string | undefined;
    size?: number | undefined;
    sha?: string | undefined;
    html_url?: string | undefined;
    git_url?: string | undefined;
    download_url?: string | null | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_file";
    path?: string | undefined;
    type?: "file" | "dir" | "symlink" | "submodule" | undefined;
    name?: string | undefined;
    encoding?: string | undefined;
    content?: string | undefined;
    url?: string | undefined;
    size?: number | undefined;
    sha?: string | undefined;
    html_url?: string | undefined;
    git_url?: string | undefined;
    download_url?: string | null | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_directory">;
    success: z.ZodBoolean;
    error: z.ZodString;
    contents: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        path: z.ZodString;
        sha: z.ZodString;
        size: z.ZodNumber;
        url: z.ZodString;
        html_url: z.ZodString;
        git_url: z.ZodString;
        download_url: z.ZodNullable<z.ZodString>;
        type: z.ZodEnum<["file", "dir", "symlink", "submodule"]>;
        content: z.ZodOptional<z.ZodString>;
        encoding: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        type: "file" | "dir" | "symlink" | "submodule";
        name: string;
        url: string;
        size: number;
        sha: string;
        html_url: string;
        git_url: string;
        download_url: string | null;
        encoding?: string | undefined;
        content?: string | undefined;
    }, {
        path: string;
        type: "file" | "dir" | "symlink" | "submodule";
        name: string;
        url: string;
        size: number;
        sha: string;
        html_url: string;
        git_url: string;
        download_url: string | null;
        encoding?: string | undefined;
        content?: string | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_directory";
    contents?: {
        path: string;
        type: "file" | "dir" | "symlink" | "submodule";
        name: string;
        url: string;
        size: number;
        sha: string;
        html_url: string;
        git_url: string;
        download_url: string | null;
        encoding?: string | undefined;
        content?: string | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_directory";
    contents?: {
        path: string;
        type: "file" | "dir" | "symlink" | "submodule";
        name: string;
        url: string;
        size: number;
        sha: string;
        html_url: string;
        git_url: string;
        download_url: string | null;
        encoding?: string | undefined;
        content?: string | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_pull_requests">;
    success: z.ZodBoolean;
    error: z.ZodString;
    pull_requests: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        node_id: z.ZodString;
        number: z.ZodNumber;
        state: z.ZodEnum<["open", "closed"]>;
        title: z.ZodString;
        body: z.ZodNullable<z.ZodString>;
        created_at: z.ZodString;
        updated_at: z.ZodString;
        closed_at: z.ZodNullable<z.ZodString>;
        merged_at: z.ZodNullable<z.ZodString>;
        user: z.ZodObject<{
            login: z.ZodString;
            id: z.ZodNumber;
            avatar_url: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: number;
            login: string;
            avatar_url: string;
        }, {
            id: number;
            login: string;
            avatar_url: string;
        }>;
        html_url: z.ZodString;
        draft: z.ZodBoolean;
        head: z.ZodObject<{
            ref: z.ZodString;
            sha: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            ref: string;
            sha: string;
        }, {
            ref: string;
            sha: string;
        }>;
        base: z.ZodObject<{
            ref: z.ZodString;
            sha: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            ref: string;
            sha: string;
        }, {
            ref: string;
            sha: string;
        }>;
        merged: z.ZodOptional<z.ZodBoolean>;
        mergeable: z.ZodOptional<z.ZodNullable<z.ZodBoolean>>;
        mergeable_state: z.ZodOptional<z.ZodString>;
        comments: z.ZodOptional<z.ZodNumber>;
        review_comments: z.ZodOptional<z.ZodNumber>;
        commits: z.ZodOptional<z.ZodNumber>;
        additions: z.ZodOptional<z.ZodNumber>;
        deletions: z.ZodOptional<z.ZodNumber>;
        changed_files: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
            avatar_url: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        draft: boolean;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
        merged_at: string | null;
        head: {
            ref: string;
            sha: string;
        };
        base: {
            ref: string;
            sha: string;
        };
        comments?: number | undefined;
        merged?: boolean | undefined;
        mergeable?: boolean | null | undefined;
        mergeable_state?: string | undefined;
        review_comments?: number | undefined;
        commits?: number | undefined;
        additions?: number | undefined;
        deletions?: number | undefined;
        changed_files?: number | undefined;
    }, {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
            avatar_url: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        draft: boolean;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
        merged_at: string | null;
        head: {
            ref: string;
            sha: string;
        };
        base: {
            ref: string;
            sha: string;
        };
        comments?: number | undefined;
        merged?: boolean | undefined;
        mergeable?: boolean | null | undefined;
        mergeable_state?: string | undefined;
        review_comments?: number | undefined;
        commits?: number | undefined;
        additions?: number | undefined;
        deletions?: number | undefined;
        changed_files?: number | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_pull_requests";
    pull_requests?: {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
            avatar_url: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        draft: boolean;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
        merged_at: string | null;
        head: {
            ref: string;
            sha: string;
        };
        base: {
            ref: string;
            sha: string;
        };
        comments?: number | undefined;
        merged?: boolean | undefined;
        mergeable?: boolean | null | undefined;
        mergeable_state?: string | undefined;
        review_comments?: number | undefined;
        commits?: number | undefined;
        additions?: number | undefined;
        deletions?: number | undefined;
        changed_files?: number | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_pull_requests";
    pull_requests?: {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
            avatar_url: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        draft: boolean;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
        merged_at: string | null;
        head: {
            ref: string;
            sha: string;
        };
        base: {
            ref: string;
            sha: string;
        };
        comments?: number | undefined;
        merged?: boolean | undefined;
        mergeable?: boolean | null | undefined;
        mergeable_state?: string | undefined;
        review_comments?: number | undefined;
        commits?: number | undefined;
        additions?: number | undefined;
        deletions?: number | undefined;
        changed_files?: number | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_pull_request">;
    success: z.ZodBoolean;
    error: z.ZodString;
} & {
    id: z.ZodOptional<z.ZodNumber>;
    node_id: z.ZodOptional<z.ZodString>;
    number: z.ZodOptional<z.ZodNumber>;
    state: z.ZodOptional<z.ZodEnum<["open", "closed"]>>;
    title: z.ZodOptional<z.ZodString>;
    body: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    created_at: z.ZodOptional<z.ZodString>;
    updated_at: z.ZodOptional<z.ZodString>;
    closed_at: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    merged_at: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    user: z.ZodOptional<z.ZodObject<{
        login: z.ZodString;
        id: z.ZodNumber;
        avatar_url: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        id: number;
        login: string;
        avatar_url: string;
    }, {
        id: number;
        login: string;
        avatar_url: string;
    }>>;
    html_url: z.ZodOptional<z.ZodString>;
    draft: z.ZodOptional<z.ZodBoolean>;
    head: z.ZodOptional<z.ZodObject<{
        ref: z.ZodString;
        sha: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        ref: string;
        sha: string;
    }, {
        ref: string;
        sha: string;
    }>>;
    base: z.ZodOptional<z.ZodObject<{
        ref: z.ZodString;
        sha: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        ref: string;
        sha: string;
    }, {
        ref: string;
        sha: string;
    }>>;
    merged: z.ZodOptional<z.ZodOptional<z.ZodBoolean>>;
    mergeable: z.ZodOptional<z.ZodOptional<z.ZodNullable<z.ZodBoolean>>>;
    mergeable_state: z.ZodOptional<z.ZodOptional<z.ZodString>>;
    comments: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
    review_comments: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
    commits: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
    additions: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
    deletions: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
    changed_files: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_pull_request";
    number?: number | undefined;
    title?: string | undefined;
    user?: {
        id: number;
        login: string;
        avatar_url: string;
    } | undefined;
    id?: number | undefined;
    created_at?: string | undefined;
    body?: string | null | undefined;
    draft?: boolean | undefined;
    comments?: number | undefined;
    state?: "closed" | "open" | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
    closed_at?: string | null | undefined;
    merged_at?: string | null | undefined;
    head?: {
        ref: string;
        sha: string;
    } | undefined;
    base?: {
        ref: string;
        sha: string;
    } | undefined;
    merged?: boolean | undefined;
    mergeable?: boolean | null | undefined;
    mergeable_state?: string | undefined;
    review_comments?: number | undefined;
    commits?: number | undefined;
    additions?: number | undefined;
    deletions?: number | undefined;
    changed_files?: number | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_pull_request";
    number?: number | undefined;
    title?: string | undefined;
    user?: {
        id: number;
        login: string;
        avatar_url: string;
    } | undefined;
    id?: number | undefined;
    created_at?: string | undefined;
    body?: string | null | undefined;
    draft?: boolean | undefined;
    comments?: number | undefined;
    state?: "closed" | "open" | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
    closed_at?: string | null | undefined;
    merged_at?: string | null | undefined;
    head?: {
        ref: string;
        sha: string;
    } | undefined;
    base?: {
        ref: string;
        sha: string;
    } | undefined;
    merged?: boolean | undefined;
    mergeable?: boolean | null | undefined;
    mergeable_state?: string | undefined;
    review_comments?: number | undefined;
    commits?: number | undefined;
    additions?: number | undefined;
    deletions?: number | undefined;
    changed_files?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_pr_comment">;
    success: z.ZodBoolean;
    error: z.ZodString;
} & {
    id: z.ZodOptional<z.ZodNumber>;
    node_id: z.ZodOptional<z.ZodString>;
    body: z.ZodOptional<z.ZodString>;
    user: z.ZodOptional<z.ZodObject<{
        login: z.ZodString;
        id: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        id: number;
        login: string;
    }, {
        id: number;
        login: string;
    }>>;
    created_at: z.ZodOptional<z.ZodString>;
    updated_at: z.ZodOptional<z.ZodString>;
    html_url: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_pr_comment";
    user?: {
        id: number;
        login: string;
    } | undefined;
    id?: number | undefined;
    created_at?: string | undefined;
    body?: string | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_pr_comment";
    user?: {
        id: number;
        login: string;
    } | undefined;
    id?: number | undefined;
    created_at?: string | undefined;
    body?: string | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_repositories">;
    success: z.ZodBoolean;
    error: z.ZodString;
    repositories: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        node_id: z.ZodString;
        name: z.ZodString;
        full_name: z.ZodString;
        private: z.ZodBoolean;
        owner: z.ZodObject<{
            login: z.ZodString;
            id: z.ZodNumber;
            avatar_url: z.ZodString;
            html_url: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        }, {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        }>;
        html_url: z.ZodString;
        description: z.ZodNullable<z.ZodString>;
        fork: z.ZodBoolean;
        created_at: z.ZodString;
        updated_at: z.ZodString;
        pushed_at: z.ZodString;
        size: z.ZodNumber;
        stargazers_count: z.ZodNumber;
        watchers_count: z.ZodNumber;
        language: z.ZodNullable<z.ZodString>;
        forks_count: z.ZodNumber;
        open_issues_count: z.ZodNumber;
        default_branch: z.ZodString;
        visibility: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        description: string | null;
        name: string;
        id: number;
        size: number;
        private: boolean;
        created_at: string;
        owner: {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        };
        full_name: string;
        language: string | null;
        html_url: string;
        node_id: string;
        updated_at: string;
        fork: boolean;
        pushed_at: string;
        stargazers_count: number;
        watchers_count: number;
        forks_count: number;
        open_issues_count: number;
        default_branch: string;
        visibility?: string | undefined;
    }, {
        description: string | null;
        name: string;
        id: number;
        size: number;
        private: boolean;
        created_at: string;
        owner: {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        };
        full_name: string;
        language: string | null;
        html_url: string;
        node_id: string;
        updated_at: string;
        fork: boolean;
        pushed_at: string;
        stargazers_count: number;
        watchers_count: number;
        forks_count: number;
        open_issues_count: number;
        default_branch: string;
        visibility?: string | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_repositories";
    repositories?: {
        description: string | null;
        name: string;
        id: number;
        size: number;
        private: boolean;
        created_at: string;
        owner: {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        };
        full_name: string;
        language: string | null;
        html_url: string;
        node_id: string;
        updated_at: string;
        fork: boolean;
        pushed_at: string;
        stargazers_count: number;
        watchers_count: number;
        forks_count: number;
        open_issues_count: number;
        default_branch: string;
        visibility?: string | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_repositories";
    repositories?: {
        description: string | null;
        name: string;
        id: number;
        size: number;
        private: boolean;
        created_at: string;
        owner: {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        };
        full_name: string;
        language: string | null;
        html_url: string;
        node_id: string;
        updated_at: string;
        fork: boolean;
        pushed_at: string;
        stargazers_count: number;
        watchers_count: number;
        forks_count: number;
        open_issues_count: number;
        default_branch: string;
        visibility?: string | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_repository">;
    success: z.ZodBoolean;
    error: z.ZodString;
} & {
    id: z.ZodOptional<z.ZodNumber>;
    node_id: z.ZodOptional<z.ZodString>;
    name: z.ZodOptional<z.ZodString>;
    full_name: z.ZodOptional<z.ZodString>;
    private: z.ZodOptional<z.ZodBoolean>;
    owner: z.ZodOptional<z.ZodObject<{
        login: z.ZodString;
        id: z.ZodNumber;
        avatar_url: z.ZodString;
        html_url: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        id: number;
        html_url: string;
        login: string;
        avatar_url: string;
    }, {
        id: number;
        html_url: string;
        login: string;
        avatar_url: string;
    }>>;
    html_url: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    fork: z.ZodOptional<z.ZodBoolean>;
    created_at: z.ZodOptional<z.ZodString>;
    updated_at: z.ZodOptional<z.ZodString>;
    pushed_at: z.ZodOptional<z.ZodString>;
    size: z.ZodOptional<z.ZodNumber>;
    stargazers_count: z.ZodOptional<z.ZodNumber>;
    watchers_count: z.ZodOptional<z.ZodNumber>;
    language: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    forks_count: z.ZodOptional<z.ZodNumber>;
    open_issues_count: z.ZodOptional<z.ZodNumber>;
    default_branch: z.ZodOptional<z.ZodString>;
    visibility: z.ZodOptional<z.ZodOptional<z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_repository";
    description?: string | null | undefined;
    name?: string | undefined;
    id?: number | undefined;
    size?: number | undefined;
    private?: boolean | undefined;
    created_at?: string | undefined;
    owner?: {
        id: number;
        html_url: string;
        login: string;
        avatar_url: string;
    } | undefined;
    full_name?: string | undefined;
    language?: string | null | undefined;
    visibility?: string | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
    fork?: boolean | undefined;
    pushed_at?: string | undefined;
    stargazers_count?: number | undefined;
    watchers_count?: number | undefined;
    forks_count?: number | undefined;
    open_issues_count?: number | undefined;
    default_branch?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_repository";
    description?: string | null | undefined;
    name?: string | undefined;
    id?: number | undefined;
    size?: number | undefined;
    private?: boolean | undefined;
    created_at?: string | undefined;
    owner?: {
        id: number;
        html_url: string;
        login: string;
        avatar_url: string;
    } | undefined;
    full_name?: string | undefined;
    language?: string | null | undefined;
    visibility?: string | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
    fork?: boolean | undefined;
    pushed_at?: string | undefined;
    stargazers_count?: number | undefined;
    watchers_count?: number | undefined;
    forks_count?: number | undefined;
    open_issues_count?: number | undefined;
    default_branch?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_issue_comment">;
    success: z.ZodBoolean;
    error: z.ZodString;
} & {
    id: z.ZodOptional<z.ZodNumber>;
    node_id: z.ZodOptional<z.ZodString>;
    body: z.ZodOptional<z.ZodString>;
    user: z.ZodOptional<z.ZodObject<{
        login: z.ZodString;
        id: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        id: number;
        login: string;
    }, {
        id: number;
        login: string;
    }>>;
    created_at: z.ZodOptional<z.ZodString>;
    updated_at: z.ZodOptional<z.ZodString>;
    html_url: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_issue_comment";
    user?: {
        id: number;
        login: string;
    } | undefined;
    id?: number | undefined;
    created_at?: string | undefined;
    body?: string | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_issue_comment";
    user?: {
        id: number;
        login: string;
    } | undefined;
    id?: number | undefined;
    created_at?: string | undefined;
    body?: string | undefined;
    html_url?: string | undefined;
    node_id?: string | undefined;
    updated_at?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_issues">;
    success: z.ZodBoolean;
    error: z.ZodString;
    issues: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        node_id: z.ZodString;
        number: z.ZodNumber;
        state: z.ZodEnum<["open", "closed"]>;
        title: z.ZodString;
        body: z.ZodNullable<z.ZodString>;
        user: z.ZodObject<{
            login: z.ZodString;
            id: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            id: number;
            login: string;
        }, {
            id: number;
            login: string;
        }>;
        labels: z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            name: z.ZodString;
            color: z.ZodString;
            description: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            description: string | null;
            name: string;
            color: string;
            id: number;
        }, {
            description: string | null;
            name: string;
            color: string;
            id: number;
        }>, "many">;
        created_at: z.ZodString;
        updated_at: z.ZodString;
        closed_at: z.ZodNullable<z.ZodString>;
        html_url: z.ZodString;
        comments: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        labels: {
            description: string | null;
            name: string;
            color: string;
            id: number;
        }[];
        comments: number;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
    }, {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        labels: {
            description: string | null;
            name: string;
            color: string;
            id: number;
        }[];
        comments: number;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_issues";
    issues?: {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        labels: {
            description: string | null;
            name: string;
            color: string;
            id: number;
        }[];
        comments: number;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_issues";
    issues?: {
        number: number;
        title: string;
        user: {
            id: number;
            login: string;
        };
        id: number;
        created_at: string;
        body: string | null;
        labels: {
            description: string | null;
            name: string;
            color: string;
            id: number;
        }[];
        comments: number;
        state: "closed" | "open";
        html_url: string;
        node_id: string;
        updated_at: string;
        closed_at: string | null;
    }[] | undefined;
}>]>;
export type GithubParamsInput = z.input<typeof GithubParamsSchema>;
type GithubParams = z.input<typeof GithubParamsSchema>;
type GithubResult = z.output<typeof GithubResultSchema>;
export type GithubGetFileParams = Extract<GithubParams, {
    operation: 'get_file';
}>;
export type GithubGetDirectoryParams = Extract<GithubParams, {
    operation: 'get_directory';
}>;
export type GithubListPullRequestsParams = Extract<GithubParams, {
    operation: 'list_pull_requests';
}>;
export type GithubGetPullRequestParams = Extract<GithubParams, {
    operation: 'get_pull_request';
}>;
export type GithubCreatePrCommentParams = Extract<GithubParams, {
    operation: 'create_pr_comment';
}>;
export type GithubListRepositoriesParams = Extract<GithubParams, {
    operation: 'list_repositories';
}>;
export type GithubGetRepositoryParams = Extract<GithubParams, {
    operation: 'get_repository';
}>;
export type GithubCreateIssueCommentParams = Extract<GithubParams, {
    operation: 'create_issue_comment';
}>;
export type GithubListIssuesParams = Extract<GithubParams, {
    operation: 'list_issues';
}>;
export type GithubOperationResult<T extends GithubParams['operation']> = Extract<GithubResult, {
    operation: T;
}>;
export declare class GithubBubble<T extends GithubParams = GithubParams> extends ServiceBubble<T, Extract<GithubResult, {
    operation: T['operation'];
}>> {
    static readonly type: "service";
    static readonly service = "github";
    static readonly authType: "apikey";
    static readonly bubbleName = "github";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"get_file">;
        owner: z.ZodString;
        repo: z.ZodString;
        path: z.ZodString;
        ref: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        operation: "get_file";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ref?: string | undefined;
    }, {
        path: string;
        operation: "get_file";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ref?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_directory">;
        owner: z.ZodString;
        repo: z.ZodString;
        path: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        ref: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        operation: "get_directory";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ref?: string | undefined;
    }, {
        operation: "get_directory";
        owner: string;
        repo: string;
        path?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        ref?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_pull_requests">;
        owner: z.ZodString;
        repo: z.ZodString;
        state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
        sort: z.ZodDefault<z.ZodOptional<z.ZodEnum<["created", "updated", "popularity", "long-running"]>>>;
        direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
        per_page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        sort: "created" | "updated" | "popularity" | "long-running";
        operation: "list_pull_requests";
        owner: string;
        page: number;
        state: "closed" | "open" | "all";
        repo: string;
        direction: "asc" | "desc";
        per_page: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "list_pull_requests";
        owner: string;
        repo: string;
        sort?: "created" | "updated" | "popularity" | "long-running" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        page?: number | undefined;
        state?: "closed" | "open" | "all" | undefined;
        direction?: "asc" | "desc" | undefined;
        per_page?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_pull_request">;
        owner: z.ZodString;
        repo: z.ZodString;
        pull_number: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_pull_request";
        owner: string;
        repo: string;
        pull_number: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_pull_request";
        owner: string;
        repo: string;
        pull_number: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_pr_comment">;
        owner: z.ZodString;
        repo: z.ZodString;
        pull_number: z.ZodNumber;
        body: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_pr_comment";
        body: string;
        owner: string;
        repo: string;
        pull_number: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "create_pr_comment";
        body: string;
        owner: string;
        repo: string;
        pull_number: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_repositories">;
        visibility: z.ZodDefault<z.ZodOptional<z.ZodEnum<["all", "public", "private"]>>>;
        affiliation: z.ZodDefault<z.ZodOptional<z.ZodEnum<["owner", "collaborator", "organization_member"]>>>;
        sort: z.ZodDefault<z.ZodOptional<z.ZodEnum<["created", "updated", "pushed", "full_name"]>>>;
        direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
        per_page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        sort: "created" | "updated" | "full_name" | "pushed";
        operation: "list_repositories";
        page: number;
        direction: "asc" | "desc";
        per_page: number;
        visibility: "public" | "private" | "all";
        affiliation: "owner" | "collaborator" | "organization_member";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "list_repositories";
        sort?: "created" | "updated" | "full_name" | "pushed" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        page?: number | undefined;
        direction?: "asc" | "desc" | undefined;
        per_page?: number | undefined;
        visibility?: "public" | "private" | "all" | undefined;
        affiliation?: "owner" | "collaborator" | "organization_member" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_repository">;
        owner: z.ZodString;
        repo: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_repository";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_repository";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_issue_comment">;
        owner: z.ZodString;
        repo: z.ZodString;
        issue_number: z.ZodNumber;
        body: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_issue_comment";
        body: string;
        owner: string;
        repo: string;
        issue_number: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "create_issue_comment";
        body: string;
        owner: string;
        repo: string;
        issue_number: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_issues">;
        owner: z.ZodString;
        repo: z.ZodString;
        state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
        labels: z.ZodOptional<z.ZodString>;
        sort: z.ZodDefault<z.ZodOptional<z.ZodEnum<["created", "updated", "comments"]>>>;
        direction: z.ZodDefault<z.ZodOptional<z.ZodEnum<["asc", "desc"]>>>;
        per_page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        page: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        sort: "created" | "updated" | "comments";
        operation: "list_issues";
        owner: string;
        page: number;
        state: "closed" | "open" | "all";
        repo: string;
        direction: "asc" | "desc";
        per_page: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        labels?: string | undefined;
    }, {
        operation: "list_issues";
        owner: string;
        repo: string;
        sort?: "created" | "updated" | "comments" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        labels?: string | undefined;
        page?: number | undefined;
        state?: "closed" | "open" | "all" | undefined;
        direction?: "asc" | "desc" | undefined;
        per_page?: number | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"get_file">;
        success: z.ZodBoolean;
        error: z.ZodString;
    } & {
        name: z.ZodOptional<z.ZodString>;
        path: z.ZodOptional<z.ZodString>;
        sha: z.ZodOptional<z.ZodString>;
        size: z.ZodOptional<z.ZodNumber>;
        url: z.ZodOptional<z.ZodString>;
        html_url: z.ZodOptional<z.ZodString>;
        git_url: z.ZodOptional<z.ZodString>;
        download_url: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        type: z.ZodOptional<z.ZodEnum<["file", "dir", "symlink", "submodule"]>>;
        content: z.ZodOptional<z.ZodOptional<z.ZodString>>;
        encoding: z.ZodOptional<z.ZodOptional<z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_file";
        path?: string | undefined;
        type?: "file" | "dir" | "symlink" | "submodule" | undefined;
        name?: string | undefined;
        encoding?: string | undefined;
        content?: string | undefined;
        url?: string | undefined;
        size?: number | undefined;
        sha?: string | undefined;
        html_url?: string | undefined;
        git_url?: string | undefined;
        download_url?: string | null | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_file";
        path?: string | undefined;
        type?: "file" | "dir" | "symlink" | "submodule" | undefined;
        name?: string | undefined;
        encoding?: string | undefined;
        content?: string | undefined;
        url?: string | undefined;
        size?: number | undefined;
        sha?: string | undefined;
        html_url?: string | undefined;
        git_url?: string | undefined;
        download_url?: string | null | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_directory">;
        success: z.ZodBoolean;
        error: z.ZodString;
        contents: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            path: z.ZodString;
            sha: z.ZodString;
            size: z.ZodNumber;
            url: z.ZodString;
            html_url: z.ZodString;
            git_url: z.ZodString;
            download_url: z.ZodNullable<z.ZodString>;
            type: z.ZodEnum<["file", "dir", "symlink", "submodule"]>;
            content: z.ZodOptional<z.ZodString>;
            encoding: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            path: string;
            type: "file" | "dir" | "symlink" | "submodule";
            name: string;
            url: string;
            size: number;
            sha: string;
            html_url: string;
            git_url: string;
            download_url: string | null;
            encoding?: string | undefined;
            content?: string | undefined;
        }, {
            path: string;
            type: "file" | "dir" | "symlink" | "submodule";
            name: string;
            url: string;
            size: number;
            sha: string;
            html_url: string;
            git_url: string;
            download_url: string | null;
            encoding?: string | undefined;
            content?: string | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_directory";
        contents?: {
            path: string;
            type: "file" | "dir" | "symlink" | "submodule";
            name: string;
            url: string;
            size: number;
            sha: string;
            html_url: string;
            git_url: string;
            download_url: string | null;
            encoding?: string | undefined;
            content?: string | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_directory";
        contents?: {
            path: string;
            type: "file" | "dir" | "symlink" | "submodule";
            name: string;
            url: string;
            size: number;
            sha: string;
            html_url: string;
            git_url: string;
            download_url: string | null;
            encoding?: string | undefined;
            content?: string | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_pull_requests">;
        success: z.ZodBoolean;
        error: z.ZodString;
        pull_requests: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            node_id: z.ZodString;
            number: z.ZodNumber;
            state: z.ZodEnum<["open", "closed"]>;
            title: z.ZodString;
            body: z.ZodNullable<z.ZodString>;
            created_at: z.ZodString;
            updated_at: z.ZodString;
            closed_at: z.ZodNullable<z.ZodString>;
            merged_at: z.ZodNullable<z.ZodString>;
            user: z.ZodObject<{
                login: z.ZodString;
                id: z.ZodNumber;
                avatar_url: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                id: number;
                login: string;
                avatar_url: string;
            }, {
                id: number;
                login: string;
                avatar_url: string;
            }>;
            html_url: z.ZodString;
            draft: z.ZodBoolean;
            head: z.ZodObject<{
                ref: z.ZodString;
                sha: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                ref: string;
                sha: string;
            }, {
                ref: string;
                sha: string;
            }>;
            base: z.ZodObject<{
                ref: z.ZodString;
                sha: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                ref: string;
                sha: string;
            }, {
                ref: string;
                sha: string;
            }>;
            merged: z.ZodOptional<z.ZodBoolean>;
            mergeable: z.ZodOptional<z.ZodNullable<z.ZodBoolean>>;
            mergeable_state: z.ZodOptional<z.ZodString>;
            comments: z.ZodOptional<z.ZodNumber>;
            review_comments: z.ZodOptional<z.ZodNumber>;
            commits: z.ZodOptional<z.ZodNumber>;
            additions: z.ZodOptional<z.ZodNumber>;
            deletions: z.ZodOptional<z.ZodNumber>;
            changed_files: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
                avatar_url: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            draft: boolean;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
            merged_at: string | null;
            head: {
                ref: string;
                sha: string;
            };
            base: {
                ref: string;
                sha: string;
            };
            comments?: number | undefined;
            merged?: boolean | undefined;
            mergeable?: boolean | null | undefined;
            mergeable_state?: string | undefined;
            review_comments?: number | undefined;
            commits?: number | undefined;
            additions?: number | undefined;
            deletions?: number | undefined;
            changed_files?: number | undefined;
        }, {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
                avatar_url: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            draft: boolean;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
            merged_at: string | null;
            head: {
                ref: string;
                sha: string;
            };
            base: {
                ref: string;
                sha: string;
            };
            comments?: number | undefined;
            merged?: boolean | undefined;
            mergeable?: boolean | null | undefined;
            mergeable_state?: string | undefined;
            review_comments?: number | undefined;
            commits?: number | undefined;
            additions?: number | undefined;
            deletions?: number | undefined;
            changed_files?: number | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_pull_requests";
        pull_requests?: {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
                avatar_url: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            draft: boolean;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
            merged_at: string | null;
            head: {
                ref: string;
                sha: string;
            };
            base: {
                ref: string;
                sha: string;
            };
            comments?: number | undefined;
            merged?: boolean | undefined;
            mergeable?: boolean | null | undefined;
            mergeable_state?: string | undefined;
            review_comments?: number | undefined;
            commits?: number | undefined;
            additions?: number | undefined;
            deletions?: number | undefined;
            changed_files?: number | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_pull_requests";
        pull_requests?: {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
                avatar_url: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            draft: boolean;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
            merged_at: string | null;
            head: {
                ref: string;
                sha: string;
            };
            base: {
                ref: string;
                sha: string;
            };
            comments?: number | undefined;
            merged?: boolean | undefined;
            mergeable?: boolean | null | undefined;
            mergeable_state?: string | undefined;
            review_comments?: number | undefined;
            commits?: number | undefined;
            additions?: number | undefined;
            deletions?: number | undefined;
            changed_files?: number | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_pull_request">;
        success: z.ZodBoolean;
        error: z.ZodString;
    } & {
        id: z.ZodOptional<z.ZodNumber>;
        node_id: z.ZodOptional<z.ZodString>;
        number: z.ZodOptional<z.ZodNumber>;
        state: z.ZodOptional<z.ZodEnum<["open", "closed"]>>;
        title: z.ZodOptional<z.ZodString>;
        body: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        closed_at: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        merged_at: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        user: z.ZodOptional<z.ZodObject<{
            login: z.ZodString;
            id: z.ZodNumber;
            avatar_url: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: number;
            login: string;
            avatar_url: string;
        }, {
            id: number;
            login: string;
            avatar_url: string;
        }>>;
        html_url: z.ZodOptional<z.ZodString>;
        draft: z.ZodOptional<z.ZodBoolean>;
        head: z.ZodOptional<z.ZodObject<{
            ref: z.ZodString;
            sha: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            ref: string;
            sha: string;
        }, {
            ref: string;
            sha: string;
        }>>;
        base: z.ZodOptional<z.ZodObject<{
            ref: z.ZodString;
            sha: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            ref: string;
            sha: string;
        }, {
            ref: string;
            sha: string;
        }>>;
        merged: z.ZodOptional<z.ZodOptional<z.ZodBoolean>>;
        mergeable: z.ZodOptional<z.ZodOptional<z.ZodNullable<z.ZodBoolean>>>;
        mergeable_state: z.ZodOptional<z.ZodOptional<z.ZodString>>;
        comments: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
        review_comments: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
        commits: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
        additions: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
        deletions: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
        changed_files: z.ZodOptional<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_pull_request";
        number?: number | undefined;
        title?: string | undefined;
        user?: {
            id: number;
            login: string;
            avatar_url: string;
        } | undefined;
        id?: number | undefined;
        created_at?: string | undefined;
        body?: string | null | undefined;
        draft?: boolean | undefined;
        comments?: number | undefined;
        state?: "closed" | "open" | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
        closed_at?: string | null | undefined;
        merged_at?: string | null | undefined;
        head?: {
            ref: string;
            sha: string;
        } | undefined;
        base?: {
            ref: string;
            sha: string;
        } | undefined;
        merged?: boolean | undefined;
        mergeable?: boolean | null | undefined;
        mergeable_state?: string | undefined;
        review_comments?: number | undefined;
        commits?: number | undefined;
        additions?: number | undefined;
        deletions?: number | undefined;
        changed_files?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_pull_request";
        number?: number | undefined;
        title?: string | undefined;
        user?: {
            id: number;
            login: string;
            avatar_url: string;
        } | undefined;
        id?: number | undefined;
        created_at?: string | undefined;
        body?: string | null | undefined;
        draft?: boolean | undefined;
        comments?: number | undefined;
        state?: "closed" | "open" | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
        closed_at?: string | null | undefined;
        merged_at?: string | null | undefined;
        head?: {
            ref: string;
            sha: string;
        } | undefined;
        base?: {
            ref: string;
            sha: string;
        } | undefined;
        merged?: boolean | undefined;
        mergeable?: boolean | null | undefined;
        mergeable_state?: string | undefined;
        review_comments?: number | undefined;
        commits?: number | undefined;
        additions?: number | undefined;
        deletions?: number | undefined;
        changed_files?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_pr_comment">;
        success: z.ZodBoolean;
        error: z.ZodString;
    } & {
        id: z.ZodOptional<z.ZodNumber>;
        node_id: z.ZodOptional<z.ZodString>;
        body: z.ZodOptional<z.ZodString>;
        user: z.ZodOptional<z.ZodObject<{
            login: z.ZodString;
            id: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            id: number;
            login: string;
        }, {
            id: number;
            login: string;
        }>>;
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        html_url: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_pr_comment";
        user?: {
            id: number;
            login: string;
        } | undefined;
        id?: number | undefined;
        created_at?: string | undefined;
        body?: string | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_pr_comment";
        user?: {
            id: number;
            login: string;
        } | undefined;
        id?: number | undefined;
        created_at?: string | undefined;
        body?: string | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_repositories">;
        success: z.ZodBoolean;
        error: z.ZodString;
        repositories: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            node_id: z.ZodString;
            name: z.ZodString;
            full_name: z.ZodString;
            private: z.ZodBoolean;
            owner: z.ZodObject<{
                login: z.ZodString;
                id: z.ZodNumber;
                avatar_url: z.ZodString;
                html_url: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                id: number;
                html_url: string;
                login: string;
                avatar_url: string;
            }, {
                id: number;
                html_url: string;
                login: string;
                avatar_url: string;
            }>;
            html_url: z.ZodString;
            description: z.ZodNullable<z.ZodString>;
            fork: z.ZodBoolean;
            created_at: z.ZodString;
            updated_at: z.ZodString;
            pushed_at: z.ZodString;
            size: z.ZodNumber;
            stargazers_count: z.ZodNumber;
            watchers_count: z.ZodNumber;
            language: z.ZodNullable<z.ZodString>;
            forks_count: z.ZodNumber;
            open_issues_count: z.ZodNumber;
            default_branch: z.ZodString;
            visibility: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            description: string | null;
            name: string;
            id: number;
            size: number;
            private: boolean;
            created_at: string;
            owner: {
                id: number;
                html_url: string;
                login: string;
                avatar_url: string;
            };
            full_name: string;
            language: string | null;
            html_url: string;
            node_id: string;
            updated_at: string;
            fork: boolean;
            pushed_at: string;
            stargazers_count: number;
            watchers_count: number;
            forks_count: number;
            open_issues_count: number;
            default_branch: string;
            visibility?: string | undefined;
        }, {
            description: string | null;
            name: string;
            id: number;
            size: number;
            private: boolean;
            created_at: string;
            owner: {
                id: number;
                html_url: string;
                login: string;
                avatar_url: string;
            };
            full_name: string;
            language: string | null;
            html_url: string;
            node_id: string;
            updated_at: string;
            fork: boolean;
            pushed_at: string;
            stargazers_count: number;
            watchers_count: number;
            forks_count: number;
            open_issues_count: number;
            default_branch: string;
            visibility?: string | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_repositories";
        repositories?: {
            description: string | null;
            name: string;
            id: number;
            size: number;
            private: boolean;
            created_at: string;
            owner: {
                id: number;
                html_url: string;
                login: string;
                avatar_url: string;
            };
            full_name: string;
            language: string | null;
            html_url: string;
            node_id: string;
            updated_at: string;
            fork: boolean;
            pushed_at: string;
            stargazers_count: number;
            watchers_count: number;
            forks_count: number;
            open_issues_count: number;
            default_branch: string;
            visibility?: string | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_repositories";
        repositories?: {
            description: string | null;
            name: string;
            id: number;
            size: number;
            private: boolean;
            created_at: string;
            owner: {
                id: number;
                html_url: string;
                login: string;
                avatar_url: string;
            };
            full_name: string;
            language: string | null;
            html_url: string;
            node_id: string;
            updated_at: string;
            fork: boolean;
            pushed_at: string;
            stargazers_count: number;
            watchers_count: number;
            forks_count: number;
            open_issues_count: number;
            default_branch: string;
            visibility?: string | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_repository">;
        success: z.ZodBoolean;
        error: z.ZodString;
    } & {
        id: z.ZodOptional<z.ZodNumber>;
        node_id: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        full_name: z.ZodOptional<z.ZodString>;
        private: z.ZodOptional<z.ZodBoolean>;
        owner: z.ZodOptional<z.ZodObject<{
            login: z.ZodString;
            id: z.ZodNumber;
            avatar_url: z.ZodString;
            html_url: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        }, {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        }>>;
        html_url: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        fork: z.ZodOptional<z.ZodBoolean>;
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        pushed_at: z.ZodOptional<z.ZodString>;
        size: z.ZodOptional<z.ZodNumber>;
        stargazers_count: z.ZodOptional<z.ZodNumber>;
        watchers_count: z.ZodOptional<z.ZodNumber>;
        language: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        forks_count: z.ZodOptional<z.ZodNumber>;
        open_issues_count: z.ZodOptional<z.ZodNumber>;
        default_branch: z.ZodOptional<z.ZodString>;
        visibility: z.ZodOptional<z.ZodOptional<z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_repository";
        description?: string | null | undefined;
        name?: string | undefined;
        id?: number | undefined;
        size?: number | undefined;
        private?: boolean | undefined;
        created_at?: string | undefined;
        owner?: {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        } | undefined;
        full_name?: string | undefined;
        language?: string | null | undefined;
        visibility?: string | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
        fork?: boolean | undefined;
        pushed_at?: string | undefined;
        stargazers_count?: number | undefined;
        watchers_count?: number | undefined;
        forks_count?: number | undefined;
        open_issues_count?: number | undefined;
        default_branch?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_repository";
        description?: string | null | undefined;
        name?: string | undefined;
        id?: number | undefined;
        size?: number | undefined;
        private?: boolean | undefined;
        created_at?: string | undefined;
        owner?: {
            id: number;
            html_url: string;
            login: string;
            avatar_url: string;
        } | undefined;
        full_name?: string | undefined;
        language?: string | null | undefined;
        visibility?: string | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
        fork?: boolean | undefined;
        pushed_at?: string | undefined;
        stargazers_count?: number | undefined;
        watchers_count?: number | undefined;
        forks_count?: number | undefined;
        open_issues_count?: number | undefined;
        default_branch?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_issue_comment">;
        success: z.ZodBoolean;
        error: z.ZodString;
    } & {
        id: z.ZodOptional<z.ZodNumber>;
        node_id: z.ZodOptional<z.ZodString>;
        body: z.ZodOptional<z.ZodString>;
        user: z.ZodOptional<z.ZodObject<{
            login: z.ZodString;
            id: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            id: number;
            login: string;
        }, {
            id: number;
            login: string;
        }>>;
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        html_url: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_issue_comment";
        user?: {
            id: number;
            login: string;
        } | undefined;
        id?: number | undefined;
        created_at?: string | undefined;
        body?: string | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_issue_comment";
        user?: {
            id: number;
            login: string;
        } | undefined;
        id?: number | undefined;
        created_at?: string | undefined;
        body?: string | undefined;
        html_url?: string | undefined;
        node_id?: string | undefined;
        updated_at?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_issues">;
        success: z.ZodBoolean;
        error: z.ZodString;
        issues: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            node_id: z.ZodString;
            number: z.ZodNumber;
            state: z.ZodEnum<["open", "closed"]>;
            title: z.ZodString;
            body: z.ZodNullable<z.ZodString>;
            user: z.ZodObject<{
                login: z.ZodString;
                id: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                id: number;
                login: string;
            }, {
                id: number;
                login: string;
            }>;
            labels: z.ZodArray<z.ZodObject<{
                id: z.ZodNumber;
                name: z.ZodString;
                color: z.ZodString;
                description: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                description: string | null;
                name: string;
                color: string;
                id: number;
            }, {
                description: string | null;
                name: string;
                color: string;
                id: number;
            }>, "many">;
            created_at: z.ZodString;
            updated_at: z.ZodString;
            closed_at: z.ZodNullable<z.ZodString>;
            html_url: z.ZodString;
            comments: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            labels: {
                description: string | null;
                name: string;
                color: string;
                id: number;
            }[];
            comments: number;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
        }, {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            labels: {
                description: string | null;
                name: string;
                color: string;
                id: number;
            }[];
            comments: number;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_issues";
        issues?: {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            labels: {
                description: string | null;
                name: string;
                color: string;
                id: number;
            }[];
            comments: number;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_issues";
        issues?: {
            number: number;
            title: string;
            user: {
                id: number;
                login: string;
            };
            id: number;
            created_at: string;
            body: string | null;
            labels: {
                description: string | null;
                name: string;
                color: string;
                id: number;
            }[];
            comments: number;
            state: "closed" | "open";
            html_url: string;
            node_id: string;
            updated_at: string;
            closed_at: string | null;
        }[] | undefined;
    }>]>;
    static readonly shortDescription = "GitHub API integration for repository operations";
    static readonly longDescription = "\n    GitHub API integration for accessing repositories, pull requests, issues, and files.\n    \n    Features:\n    - Get file contents from repositories\n    - List and browse directory contents\n    - Manage pull requests (list, get details, comment)\n    - Manage issues (list, comment)\n    - List and get repository information\n    - Non-sensitive read and comment operations only\n    \n    Use cases:\n    - Code review automation and PR management\n    - Repository file access and content retrieval\n    - Issue and PR comment automation\n    - Repository exploration and documentation\n    - CI/CD integration and status checks\n    \n    Security Features:\n    - Personal access token authentication (GitHub PAT)\n    - Read-only operations with safe comment capabilities\n    - No file deletion or destructive operations\n    - Respects repository permissions\n  ";
    static readonly alias = "gh";
    constructor(params?: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<GithubResult, {
        operation: T['operation'];
    }>>;
    private handleGetFile;
    private handleGetDirectory;
    private handleListPullRequests;
    private handleGetPullRequest;
    private handleCreatePrComment;
    private handleListRepositories;
    private handleGetRepository;
    private handleCreateIssueComment;
    private handleListIssues;
}
export {};
//# sourceMappingURL=github.d.ts.map