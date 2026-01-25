/**
 * Generic API call hook
 */
export declare function useApi(): {
    useGet: <T>(queryKey: string[], fn: () => Promise<T>, options?: {
        enabled?: boolean;
        refetchInterval?: number;
        staleTime?: number;
    }) => import('@tanstack/react-query').UseQueryResult<import('@tanstack/query-core').NoInfer<T>, Error>;
    usePost: <T, V>(fn: (data: V) => Promise<T>, options?: {
        onSuccess?: (data: T, variables: V) => void;
        onError?: (error: any) => void;
    }) => import('@tanstack/react-query').UseMutationResult<T, any, V, unknown>;
    queryClient: import('@tanstack/query-core').QueryClient;
    isAuthenticated: boolean;
};
/**
 * Auth hooks
 */
export declare function useAuth(): {
    user: import('../../stores/authStore').User;
    token: string;
    isAuthenticated: boolean;
    isLoading: boolean;
    error: string;
    login: (email: string, password: string) => Promise<void>;
    logout: () => Promise<void>;
    register: (email: string, password: string, username: string, full_name?: string) => Promise<void>;
    updateUser: (updates: Partial<import('../../stores/authStore').User>) => void;
    clearError: () => void;
};
/**
 * Evolution hooks
 */
export declare function useEvolution(evolutionId?: string): {
    evolution: import('../../stores').WorkflowExecution;
    isLoading: boolean;
    error: Error;
    refetch: (options?: import('@tanstack/query-core').RefetchOptions) => Promise<import('@tanstack/query-core').QueryObserverResult<import('../../stores').WorkflowExecution, Error>>;
    startEvolution: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    pauseEvolution: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    resumeEvolution: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    stopEvolution: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    isStarting: boolean;
    isPausing: boolean;
    isResuming: boolean;
    isStopping: boolean;
};
/**
 * Evolution list hook
 */
export declare function useEvolutions(params?: {
    status?: string;
    limit?: number;
    offset?: number;
}): import('@tanstack/react-query').UseQueryResult<{
    evolutions: import('../../stores').WorkflowExecution[];
    total: number;
    limit: number;
    offset: number;
}, Error>;
/**
 * Adversarial testing hooks
 */
export declare function useAdversarialTest(testId?: string): {
    test: import('../../stores').AdversarialTest;
    isLoading: boolean;
    error: Error;
    refetch: (options?: import('@tanstack/query-core').RefetchOptions) => Promise<import('@tanstack/query-core').QueryObserverResult<import('../../stores').AdversarialTest, Error>>;
    startTest: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    approvePatch: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    stopTest: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    isStarting: boolean;
    isApproving: boolean;
    isStopping: boolean;
};
/**
 * Adversarial tests list hook
 */
export declare function useAdversarialTests(params?: {
    status?: string;
    limit?: number;
    offset?: number;
}): import('@tanstack/react-query').UseQueryResult<{
    tests: import('../../stores').AdversarialTest[];
    total: number;
}, Error>;
/**
 * Analytics hooks
 */
export declare function useAnalytics(dateRange?: {
    start: string;
    end: string;
    granularity?: 'hour' | 'day' | 'week' | 'month';
}): {
    metrics: import('../../stores').AnalyticsData;
    performance: import('../../stores').PerformanceAnalytics;
    isLoading: boolean;
    error: Error;
    refetch: () => void;
};
/**
 * Content hooks
 */
export declare function useContent(contentId?: string): {
    content: import('../../stores').KnowledgeArtifact;
    isLoading: boolean;
    error: Error;
    updateContent: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    deleteContent: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    isUpdating: boolean;
    isDeleting: boolean;
};
/**
 * Content list hook
 */
export declare function useContentList(params?: {
    tag?: string;
    language?: string;
    limit?: number;
    offset?: number;
}): import('@tanstack/react-query').UseQueryResult<{
    content: import('../../stores').KnowledgeArtifact[];
    total: number;
}, Error>;
/**
 * Knowledge graph hook
 */
export declare function useKnowledgeGraph(): {
    graphData: {
        nodes: {
            id: string;
            label: string;
            type: string;
            data: import('../../stores').KnowledgeArtifact;
        }[];
        edges: {
            id: string;
            source: string;
            target: string;
            type: string;
        }[];
    };
    isLoading: boolean;
};
/**
 * Monitoring hooks
 */
export declare function useMonitoring(refreshInterval?: number): {
    health: {
        status: string;
        services: Record<string, string>;
        resource_usage: {
            cpu_percent: number;
            memory_percent: number;
            disk_percent: number;
        };
        active_operations: {
            evolutions_running: number;
            adversarial_tests_running: number;
        };
    };
    isLoading: boolean;
    error: Error;
};
/**
 * Configuration hooks
 */
export declare function useConfig(): {
    providers: {
        provider: string;
        name: string;
        models: string[];
        requires_api_key: boolean;
    }[];
    parameters: {
        generation: {
            temperature: number;
            top_p: number;
            max_tokens: number;
        };
        evolution: {
            max_iterations: number;
            population_size: number;
        };
    };
    isLoading: boolean;
    saveApiKey: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    updateParameters: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    isSavingKey: boolean;
    isUpdatingParams: boolean;
};
/**
 * LeanAide hooks
 */
export declare function useLeanAide(): {
    models: {
        provider: string;
        models: string[];
    }[];
    generateProof: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    verifyProof: import('@tanstack/react-query').UseMutateAsyncFunction<unknown, Error, void, unknown>;
    isGenerating: boolean;
    isVerifying: boolean;
};
