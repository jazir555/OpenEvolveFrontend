/**
 * useKnowledge Hook
 * Stub implementation - TODO: Implement actual knowledge base functionality
 */
export declare function useKnowledge(): {
    artifacts: any[];
    isLoading: boolean;
    createArtifact: (data: any) => Promise<any>;
    updateArtifact: (id: string, data: any) => Promise<any>;
    deleteArtifact: (id: string) => Promise<boolean>;
    searchArtifacts: (query: string) => Promise<any[]>;
};
