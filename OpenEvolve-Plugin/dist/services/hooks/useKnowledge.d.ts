/**
 * Knowledge base operations hook
 */
export declare function useKnowledge(): {
    artifacts: import('../../stores/knowledgeStore').KnowledgeArtifact[];
    isLoading: boolean;
    error: Error;
    createArtifact: import('@tanstack/react-query').UseMutateAsyncFunction<import('../../stores/knowledgeStore').KnowledgeArtifact, Error, {
        title: string;
        content: string;
        language?: string;
        tags?: string[];
    }, unknown>;
    updateArtifact: import('@tanstack/react-query').UseMutateAsyncFunction<import('../../stores/knowledgeStore').KnowledgeArtifact, Error, {
        id: string;
        data: any;
    }, unknown>;
    deleteArtifact: import('@tanstack/react-query').UseMutateAsyncFunction<string, Error, string, unknown>;
    isCreating: boolean;
    isUpdating: boolean;
    isDeleting: boolean;
};
/**
 * Single artifact hook
 */
export declare function useArtifact(artifactId?: string): {
    artifact: import('../../stores/knowledgeStore').KnowledgeArtifact;
    isLoading: boolean;
    error: Error;
    refetch: () => any;
};
/**
 * Artifact versions hook
 */
export declare function useArtifactVersions(artifactId?: string): import('@tanstack/react-query').UseQueryResult<{
    version: number;
    created_at: string;
    created_by: string;
    comment: string;
}[], Error>;
/**
 * Artifact diff hook
 */
export declare function useArtifactDiff(artifactId?: string, version1?: number, version2?: number): import('@tanstack/react-query').UseQueryResult<{
    version1: number;
    version2: number;
    diff: string;
}, Error>;
/**
 * Knowledge search hook
 */
export declare function useKnowledgeSearch(query: string, filters?: {
    tags?: string[];
    language?: string;
}): {
    results: import('../../stores/knowledgeStore').KnowledgeArtifact[];
    isLoading: boolean;
    total: number;
};
/**
 * Knowledge graph hook
 */
export declare function useKnowledgeGraph(): {
    graphData: import('../../stores/knowledgeStore').KnowledgeGraph;
    isLoading: boolean;
    buildGraph: () => void;
};
/**
 * Artifact comments hook
 */
export declare function useArtifactComments(artifactId?: string): {
    comments: {
        comment_id: string;
        user_id: string;
        username: string;
        comment: string;
        line_start?: number;
        line_end?: number;
        created_at: string;
        replies: any[];
    }[];
    isLoading: boolean;
    error: Error;
    addComment: import('@tanstack/react-query').UseMutateAsyncFunction<{
        comment_id: string;
        comment: string;
        created_at: string;
    }, Error, {
        comment: string;
        line_start?: number;
        line_end?: number;
        parent_comment_id?: string;
    }, unknown>;
    isAdding: boolean;
};
/**
 * Collaboration hook
 */
export declare function useCollaboration(contentId?: string): {
    createRoom: import('@tanstack/react-query').UseMutateAsyncFunction<{
        room_id: string;
        room_name: string;
        websocket_url: string;
        created_at: string;
    }, Error, {
        room_name?: string;
    }, unknown>;
    users: any[];
    isCreatingRoom: boolean;
    isLoadingUsers: boolean;
};
