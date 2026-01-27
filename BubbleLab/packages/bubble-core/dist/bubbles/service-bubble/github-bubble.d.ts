import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const GithubBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createIssue">;
    owner: z.ZodString;
    repo: z.ZodString;
    title: z.ZodString;
    body: z.ZodOptional<z.ZodString>;
    labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    assignees: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    operation: "createIssue";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    body?: string | undefined;
    labels?: string[] | undefined;
    assignees?: string[] | undefined;
}, {
    title: string;
    operation: "createIssue";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    body?: string | undefined;
    labels?: string[] | undefined;
    assignees?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateIssue">;
    owner: z.ZodString;
    repo: z.ZodString;
    issueNumber: z.ZodNumber;
    title: z.ZodOptional<z.ZodString>;
    body: z.ZodOptional<z.ZodString>;
    state: z.ZodOptional<z.ZodEnum<["open", "closed"]>>;
    labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    assignees: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateIssue";
    owner: string;
    repo: string;
    issueNumber: number;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    body?: string | undefined;
    labels?: string[] | undefined;
    state?: "closed" | "open" | undefined;
    assignees?: string[] | undefined;
}, {
    operation: "updateIssue";
    owner: string;
    repo: string;
    issueNumber: number;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    body?: string | undefined;
    labels?: string[] | undefined;
    state?: "closed" | "open" | undefined;
    assignees?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listIssues">;
    owner: z.ZodString;
    repo: z.ZodString;
    state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
    labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    creator: z.ZodOptional<z.ZodString>;
    assignee: z.ZodOptional<z.ZodString>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listIssues";
    limit: number;
    owner: string;
    state: "closed" | "open" | "all";
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    creator?: string | undefined;
    labels?: string[] | undefined;
    assignee?: string | undefined;
}, {
    operation: "listIssues";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    creator?: string | undefined;
    labels?: string[] | undefined;
    state?: "closed" | "open" | "all" | undefined;
    assignee?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createPullRequest">;
    owner: z.ZodString;
    repo: z.ZodString;
    title: z.ZodString;
    head: z.ZodString;
    base: z.ZodString;
    body: z.ZodOptional<z.ZodString>;
    draft: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    operation: "createPullRequest";
    owner: string;
    draft: boolean;
    repo: string;
    head: string;
    base: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    body?: string | undefined;
}, {
    title: string;
    operation: "createPullRequest";
    owner: string;
    repo: string;
    head: string;
    base: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    body?: string | undefined;
    draft?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"mergePullRequest">;
    owner: z.ZodString;
    repo: z.ZodString;
    pullNumber: z.ZodNumber;
    commitTitle: z.ZodOptional<z.ZodString>;
    commitMessage: z.ZodOptional<z.ZodString>;
    mergeMethod: z.ZodDefault<z.ZodOptional<z.ZodEnum<["merge", "squash", "rebase"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "mergePullRequest";
    owner: string;
    repo: string;
    pullNumber: number;
    mergeMethod: "merge" | "squash" | "rebase";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    commitTitle?: string | undefined;
    commitMessage?: string | undefined;
}, {
    operation: "mergePullRequest";
    owner: string;
    repo: string;
    pullNumber: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    commitTitle?: string | undefined;
    commitMessage?: string | undefined;
    mergeMethod?: "merge" | "squash" | "rebase" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listPullRequests">;
    owner: z.ZodString;
    repo: z.ZodString;
    state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
    head: z.ZodOptional<z.ZodString>;
    base: z.ZodOptional<z.ZodString>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listPullRequests";
    limit: number;
    owner: string;
    state: "closed" | "open" | "all";
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    head?: string | undefined;
    base?: string | undefined;
}, {
    operation: "listPullRequests";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    state?: "closed" | "open" | "all" | undefined;
    head?: string | undefined;
    base?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createBranch">;
    owner: z.ZodString;
    repo: z.ZodString;
    branchName: z.ZodString;
    fromBranch: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createBranch";
    owner: string;
    repo: string;
    branchName: string;
    fromBranch: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "createBranch";
    owner: string;
    repo: string;
    branchName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fromBranch?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteBranch">;
    owner: z.ZodString;
    repo: z.ZodString;
    branchName: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteBranch";
    owner: string;
    repo: string;
    branchName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteBranch";
    owner: string;
    repo: string;
    branchName: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRepository">;
    owner: z.ZodString;
    repo: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getRepository";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getRepository";
    owner: string;
    repo: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createCommit">;
    owner: z.ZodString;
    repo: z.ZodString;
    branch: z.ZodString;
    message: z.ZodString;
    files: z.ZodArray<z.ZodObject<{
        path: z.ZodString;
        content: z.ZodString;
        mode: z.ZodDefault<z.ZodOptional<z.ZodEnum<["100644", "100755", "040000", "160000", "120000"]>>>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        content: string;
        mode: "100644" | "100755" | "040000" | "160000" | "120000";
    }, {
        path: string;
        content: string;
        mode?: "100644" | "100755" | "040000" | "160000" | "120000" | undefined;
    }>, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    message: string;
    operation: "createCommit";
    owner: string;
    files: {
        path: string;
        content: string;
        mode: "100644" | "100755" | "040000" | "160000" | "120000";
    }[];
    repo: string;
    branch: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    message: string;
    operation: "createCommit";
    owner: string;
    files: {
        path: string;
        content: string;
        mode?: "100644" | "100755" | "040000" | "160000" | "120000" | undefined;
    }[];
    repo: string;
    branch: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type GithubBubbleParams = z.input<typeof GithubBubbleParamsSchema>;
declare const GithubBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        repository: z.ZodOptional<z.ZodString>;
        owner: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        owner?: string | undefined;
        repository?: string | undefined;
    }, {
        operation: string;
        owner?: string | undefined;
        repository?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        owner?: string | undefined;
        repository?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        owner?: string | undefined;
        repository?: string | undefined;
    };
    data?: unknown;
}>;
type GithubBubbleResult = z.output<typeof GithubBubbleResultSchema>;
export declare class GithubBubble extends ServiceBubble<GithubBubbleParams, GithubBubbleResult> {
    static readonly service = "github";
    static readonly authType: "token";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createIssue">;
        owner: z.ZodString;
        repo: z.ZodString;
        title: z.ZodString;
        body: z.ZodOptional<z.ZodString>;
        labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        assignees: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        operation: "createIssue";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        body?: string | undefined;
        labels?: string[] | undefined;
        assignees?: string[] | undefined;
    }, {
        title: string;
        operation: "createIssue";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        body?: string | undefined;
        labels?: string[] | undefined;
        assignees?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateIssue">;
        owner: z.ZodString;
        repo: z.ZodString;
        issueNumber: z.ZodNumber;
        title: z.ZodOptional<z.ZodString>;
        body: z.ZodOptional<z.ZodString>;
        state: z.ZodOptional<z.ZodEnum<["open", "closed"]>>;
        labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        assignees: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateIssue";
        owner: string;
        repo: string;
        issueNumber: number;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        body?: string | undefined;
        labels?: string[] | undefined;
        state?: "closed" | "open" | undefined;
        assignees?: string[] | undefined;
    }, {
        operation: "updateIssue";
        owner: string;
        repo: string;
        issueNumber: number;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        body?: string | undefined;
        labels?: string[] | undefined;
        state?: "closed" | "open" | undefined;
        assignees?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listIssues">;
        owner: z.ZodString;
        repo: z.ZodString;
        state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
        labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        creator: z.ZodOptional<z.ZodString>;
        assignee: z.ZodOptional<z.ZodString>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listIssues";
        limit: number;
        owner: string;
        state: "closed" | "open" | "all";
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        creator?: string | undefined;
        labels?: string[] | undefined;
        assignee?: string | undefined;
    }, {
        operation: "listIssues";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        creator?: string | undefined;
        labels?: string[] | undefined;
        state?: "closed" | "open" | "all" | undefined;
        assignee?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createPullRequest">;
        owner: z.ZodString;
        repo: z.ZodString;
        title: z.ZodString;
        head: z.ZodString;
        base: z.ZodString;
        body: z.ZodOptional<z.ZodString>;
        draft: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        operation: "createPullRequest";
        owner: string;
        draft: boolean;
        repo: string;
        head: string;
        base: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        body?: string | undefined;
    }, {
        title: string;
        operation: "createPullRequest";
        owner: string;
        repo: string;
        head: string;
        base: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        body?: string | undefined;
        draft?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"mergePullRequest">;
        owner: z.ZodString;
        repo: z.ZodString;
        pullNumber: z.ZodNumber;
        commitTitle: z.ZodOptional<z.ZodString>;
        commitMessage: z.ZodOptional<z.ZodString>;
        mergeMethod: z.ZodDefault<z.ZodOptional<z.ZodEnum<["merge", "squash", "rebase"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "mergePullRequest";
        owner: string;
        repo: string;
        pullNumber: number;
        mergeMethod: "merge" | "squash" | "rebase";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        commitTitle?: string | undefined;
        commitMessage?: string | undefined;
    }, {
        operation: "mergePullRequest";
        owner: string;
        repo: string;
        pullNumber: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        commitTitle?: string | undefined;
        commitMessage?: string | undefined;
        mergeMethod?: "merge" | "squash" | "rebase" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listPullRequests">;
        owner: z.ZodString;
        repo: z.ZodString;
        state: z.ZodDefault<z.ZodOptional<z.ZodEnum<["open", "closed", "all"]>>>;
        head: z.ZodOptional<z.ZodString>;
        base: z.ZodOptional<z.ZodString>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listPullRequests";
        limit: number;
        owner: string;
        state: "closed" | "open" | "all";
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        head?: string | undefined;
        base?: string | undefined;
    }, {
        operation: "listPullRequests";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        state?: "closed" | "open" | "all" | undefined;
        head?: string | undefined;
        base?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createBranch">;
        owner: z.ZodString;
        repo: z.ZodString;
        branchName: z.ZodString;
        fromBranch: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createBranch";
        owner: string;
        repo: string;
        branchName: string;
        fromBranch: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "createBranch";
        owner: string;
        repo: string;
        branchName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fromBranch?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteBranch">;
        owner: z.ZodString;
        repo: z.ZodString;
        branchName: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteBranch";
        owner: string;
        repo: string;
        branchName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteBranch";
        owner: string;
        repo: string;
        branchName: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRepository">;
        owner: z.ZodString;
        repo: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRepository";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getRepository";
        owner: string;
        repo: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createCommit">;
        owner: z.ZodString;
        repo: z.ZodString;
        branch: z.ZodString;
        message: z.ZodString;
        files: z.ZodArray<z.ZodObject<{
            path: z.ZodString;
            content: z.ZodString;
            mode: z.ZodDefault<z.ZodOptional<z.ZodEnum<["100644", "100755", "040000", "160000", "120000"]>>>;
        }, "strip", z.ZodTypeAny, {
            path: string;
            content: string;
            mode: "100644" | "100755" | "040000" | "160000" | "120000";
        }, {
            path: string;
            content: string;
            mode?: "100644" | "100755" | "040000" | "160000" | "120000" | undefined;
        }>, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        operation: "createCommit";
        owner: string;
        files: {
            path: string;
            content: string;
            mode: "100644" | "100755" | "040000" | "160000" | "120000";
        }[];
        repo: string;
        branch: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        message: string;
        operation: "createCommit";
        owner: string;
        files: {
            path: string;
            content: string;
            mode?: "100644" | "100755" | "040000" | "160000" | "120000" | undefined;
        }[];
        repo: string;
        branch: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            repository: z.ZodOptional<z.ZodString>;
            owner: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            owner?: string | undefined;
            repository?: string | undefined;
        }, {
            operation: string;
            owner?: string | undefined;
            repository?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            owner?: string | undefined;
            repository?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            owner?: string | undefined;
            repository?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Version control and collaborative development platform";
    static readonly longDescription = "\n    GitHub Bubble for repository management and development operations.\n\n    Features:\n    - Issue and pull request management\n    - Branch operations (create, delete)\n    - Repository information retrieval\n    - Commit creation and file management\n    - Team collaboration tools\n    - CI/CD integration capabilities\n\n    Use cases:\n    - Automated issue tracking\n    - Pull request automation\n    - Release management\n    - Code review workflows\n    - Repository monitoring\n    - Development analytics\n  ";
    static readonly alias = "git";
    private authToken;
    private baseUrl;
    constructor(params: GithubBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getToken;
    protected performAction(context?: BubbleContext): Promise<GithubBubbleResult>;
    private makeRequest;
    private createIssue;
    private updateIssue;
    private listIssues;
    private createPullRequest;
    private mergePullRequest;
    private listPullRequests;
    private createBranch;
    private deleteBranch;
    private getRepository;
    private createCommit;
    private extractRepo;
    private extractOwner;
}
export {};
//# sourceMappingURL=github-bubble.d.ts.map