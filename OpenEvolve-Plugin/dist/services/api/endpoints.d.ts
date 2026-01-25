import { User, WorkflowExecution, AnalyticsData, PerformanceAnalytics, KnowledgeArtifact, AdversarialTest, LeanCodeOutput, VerificationResult } from '../../stores/index';
/**
 * Authentication Endpoints
 */
export declare const authApi: {
    /**
     * Login user
     */
    login: (email: string, password: string) => Promise<{
        access_token: string;
        refresh_token: string;
        user?: User;
    }>;
    /**
     * Register new user
     */
    register: (email: string, password: string, username: string, full_name?: string) => Promise<User>;
    /**
     * Refresh access token
     */
    refreshToken: (refreshToken: string) => Promise<{
        access_token: string;
    }>;
    /**
     * Logout user
     */
    logout: () => Promise<unknown>;
};
/**
 * User Management Endpoints
 */
export declare const userApi: {
    /**
     * Get current user profile
     */
    getProfile: () => Promise<User>;
    /**
     * Update current user profile
     */
    updateProfile: (updates: Partial<User>) => Promise<User>;
};
/**
 * Evolution Engine Endpoints
 */
export declare const evolutionApi: {
    /**
     * Start evolution
     */
    start: (data: {
        content: string;
        mode: "standard" | "quality_diversity" | "island_model";
        parameters: {
            max_iterations: number;
            population_size: number;
            temperature: number;
            top_p: number;
        };
        models: Array<{
            provider: string;
            model: string;
            api_key: string;
        }>;
    }) => Promise<{
        evolution_id: string;
        status: string;
        created_at: string;
        websocket_url: string;
    }>;
    /**
     * Get evolution status
     */
    getStatus: (evolutionId: string) => Promise<WorkflowExecution>;
    /**
     * Pause evolution
     */
    pause: (evolutionId: string) => Promise<{
        evolution_id: string;
        status: string;
        paused_at: string;
    }>;
    /**
     * Resume evolution
     */
    resume: (evolutionId: string) => Promise<{
        evolution_id: string;
        status: string;
        resumed_at: string;
    }>;
    /**
     * Stop evolution
     */
    stop: (evolutionId: string) => Promise<{
        evolution_id: string;
        status: string;
        stopped_at: string;
        final_results: any;
    }>;
    /**
     * Delete evolution
     */
    delete: (evolutionId: string) => Promise<unknown>;
    /**
     * List evolutions
     */
    list: (params?: {
        status?: string;
        limit?: number;
        offset?: number;
        sort?: string;
        order?: "asc" | "desc";
    }) => Promise<{
        evolutions: WorkflowExecution[];
        total: number;
        limit: number;
        offset: number;
    }>;
};
/**
 * Adversarial Testing Endpoints
 */
export declare const adversarialApi: {
    /**
     * Start adversarial test
     */
    start: (data: {
        content: string;
        attack_modes: string[];
        parameters: {
            num_rounds: number;
            red_team_models: Array<{
                provider: string;
                model: string;
            }>;
            blue_team_models: Array<{
                provider: string;
                model: string;
            }>;
        };
    }) => Promise<{
        test_id: string;
        status: string;
        created_at: string;
        websocket_url: string;
    }>;
    /**
     * Get adversarial test status
     */
    getStatus: (testId: string) => Promise<AdversarialTest>;
    /**
     * Approve or reject patch
     */
    approvePatch: (testId: string, data: {
        round: number;
        approved: boolean;
        feedback?: string;
    }) => Promise<{
        test_id: string;
        round: number;
        patch_approved: boolean;
    }>;
    /**
     * Stop adversarial test
     */
    stop: (testId: string) => Promise<{
        test_id: string;
        status: string;
        stopped_at: string;
    }>;
    /**
     * List adversarial tests
     */
    list: (params?: {
        status?: string;
        limit?: number;
        offset?: number;
    }) => Promise<{
        tests: AdversarialTest[];
        total: number;
    }>;
};
/**
 * Analytics Endpoints
 */
export declare const analyticsApi: {
    /**
     * Get metrics
     */
    getMetrics: (params: {
        start_date: string;
        end_date: string;
        granularity: "hour" | "day" | "week" | "month";
    }) => Promise<AnalyticsData>;
    /**
     * Get performance analytics
     */
    getPerformance: () => Promise<PerformanceAnalytics>;
};
/**
 * Monitoring Endpoints
 */
export declare const monitoringApi: {
    /**
     * Get system health
     */
    getHealth: () => Promise<{
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
    }>;
    /**
     * Get application logs
     */
    getLogs: (params?: {
        level?: "INFO" | "WARNING" | "ERROR";
        limit?: number;
        offset?: number;
    }) => Promise<{
        logs: Array<{
            timestamp: string;
            level: string;
            message: string;
            context?: any;
        }>;
        total: number;
    }>;
};
/**
 * Content Management Endpoints
 */
export declare const contentApi: {
    /**
     * Create content
     */
    create: (data: {
        title: string;
        content: string;
        language?: string;
        tags?: string[];
    }) => Promise<KnowledgeArtifact>;
    /**
     * Get content by ID
     */
    getById: (contentId: string) => Promise<KnowledgeArtifact>;
    /**
     * Update content
     */
    update: (contentId: string, data: Partial<KnowledgeArtifact>) => Promise<KnowledgeArtifact>;
    /**
     * Delete content
     */
    delete: (contentId: string) => Promise<unknown>;
    /**
     * List content
     */
    list: (params?: {
        tag?: string;
        language?: string;
        limit?: number;
        offset?: number;
    }) => Promise<{
        content: KnowledgeArtifact[];
        total: number;
    }>;
};
/**
 * Version Control Endpoints
 */
export declare const versionApi: {
    /**
     * Get version history
     */
    getHistory: (contentId: string) => Promise<{
        version: number;
        created_at: string;
        created_by: string;
        comment: string;
    }[]>;
    /**
     * Revert to version
     */
    revert: (contentId: string, version: number) => Promise<{
        content_id: string;
        reverted_to_version: number;
        new_version: number;
        reverted_at: string;
    }>;
    /**
     * Get diff between versions
     */
    getDiff: (contentId: string, version1: number, version2: number) => Promise<{
        version1: number;
        version2: number;
        diff: string;
    }>;
    /**
     * Create branch
     */
    createBranch: (contentId: string, data: {
        branch_name: string;
        from_version: number;
    }) => Promise<{
        branch_id: string;
        branch_name: string;
        created_at: string;
    }>;
    /**
     * List branches
     */
    listBranches: (contentId: string) => Promise<{
        branch_id: string;
        branch_name: string;
        version: number;
        created_at: string;
    }[]>;
};
/**
 * Collaboration Endpoints
 */
export declare const collaborationApi: {
    /**
     * Create collaboration room
     */
    createRoom: (data: {
        content_id: string;
        room_name?: string;
    }) => Promise<{
        room_id: string;
        room_name: string;
        websocket_url: string;
        created_at: string;
    }>;
    /**
     * Get active users in room
     */
    getRoomUsers: (roomId: string) => Promise<{
        user_id: string;
        username: string;
        joined_at: string;
        cursor_position?: {
            line: number;
            column: number;
        };
    }[]>;
};
/**
 * Comments Endpoints
 */
export declare const commentsApi: {
    /**
     * Add comment to content
     */
    add: (contentId: string, data: {
        comment: string;
        line_start?: number;
        line_end?: number;
        parent_comment_id?: string;
    }) => Promise<{
        comment_id: string;
        comment: string;
        created_at: string;
    }>;
    /**
     * Get comments for content
     */
    get: (contentId: string) => Promise<{
        comment_id: string;
        user_id: string;
        username: string;
        comment: string;
        line_start?: number;
        line_end?: number;
        created_at: string;
        replies: any[];
    }[]>;
};
/**
 * Configuration Endpoints
 */
export declare const configApi: {
    /**
     * Get available providers
     */
    getProviders: () => Promise<{
        provider: string;
        name: string;
        models: string[];
        requires_api_key: boolean;
    }[]>;
    /**
     * Save API key for provider
     */
    saveApiKey: (provider: string, apiKey: string) => Promise<{
        provider: string;
        api_key_last_four: string;
        saved_at: string;
    }>;
    /**
     * Get user parameters
     */
    getParameters: () => Promise<{
        generation: {
            temperature: number;
            top_p: number;
            max_tokens: number;
        };
        evolution: {
            max_iterations: number;
            population_size: number;
        };
    }>;
    /**
     * Update user parameters
     */
    updateParameters: (params: {
        generation?: {
            temperature?: number;
            top_p?: number;
            max_tokens?: number;
        };
        evolution?: {
            max_iterations?: number;
            population_size?: number;
        };
    }) => Promise<unknown>;
};
/**
 * Workflow Endpoints
 */
export declare const workflowApi: {
    /**
     * Start integrated workflow
     */
    start: (data: {
        problem_statement: string;
        workflow_template?: string;
        parameters?: Record<string, any>;
    }) => Promise<{
        workflow_id: string;
        status: string;
        current_stage: string;
        websocket_url: string;
    }>;
    /**
     * Get workflow status
     */
    getStatus: (workflowId: string) => Promise<{
        workflow_id: string;
        status: string;
        current_stage: string;
        stages: Array<{
            stage: string;
            status: string;
            result?: any;
            progress?: number;
        }>;
    }>;
};
/**
 * File Operations Endpoints
 */
export declare const filesApi: {
    /**
     * Upload file
     */
    upload: (file: File, onProgress?: (progress: number) => void) => Promise<{
        file_id: string;
        filename: string;
        size: number;
        mime_type: string;
        uploaded_at: string;
    }>;
    /**
     * Download file
     */
    download: (fileId: string, filename?: string) => Promise<void>;
    /**
     * Get file metadata
     */
    getMetadata: (fileId: string) => Promise<{
        file_id: string;
        filename: string;
        size: number;
        mime_type: string;
        uploaded_at: string;
    }>;
};
/**
 * LeanAide Endpoints
 */
export declare const leanaideApi: {
    /**
     * Generate Lean 4 proof
     */
    generateProof: (data: {
        theorem: string;
        proof_attempt?: string;
        model: string;
        temperature: number;
    }) => Promise<LeanCodeOutput>;
    /**
     * Verify Lean 4 proof
     */
    verifyProof: (code: string) => Promise<VerificationResult>;
    /**
     * Get supported models
     */
    getModels: () => Promise<{
        provider: string;
        models: string[];
    }[]>;
    /**
     * Run benchmark
     */
    runBenchmark: (data: {
        dataset: any[];
        model: string;
        evaluator: string;
    }) => Promise<{
        benchmark_id: string;
        status: string;
    }>;
    /**
     * Get benchmark results
     */
    getBenchmarkResults: (benchmarkId: string) => Promise<any[]>;
};
/**
 * Export all APIs
 */
export declare const api: {
    auth: {
        /**
         * Login user
         */
        login: (email: string, password: string) => Promise<{
            access_token: string;
            refresh_token: string;
            user?: User;
        }>;
        /**
         * Register new user
         */
        register: (email: string, password: string, username: string, full_name?: string) => Promise<User>;
        /**
         * Refresh access token
         */
        refreshToken: (refreshToken: string) => Promise<{
            access_token: string;
        }>;
        /**
         * Logout user
         */
        logout: () => Promise<unknown>;
    };
    user: {
        /**
         * Get current user profile
         */
        getProfile: () => Promise<User>;
        /**
         * Update current user profile
         */
        updateProfile: (updates: Partial<User>) => Promise<User>;
    };
    evolution: {
        /**
         * Start evolution
         */
        start: (data: {
            content: string;
            mode: "standard" | "quality_diversity" | "island_model";
            parameters: {
                max_iterations: number;
                population_size: number;
                temperature: number;
                top_p: number;
            };
            models: Array<{
                provider: string;
                model: string;
                api_key: string;
            }>;
        }) => Promise<{
            evolution_id: string;
            status: string;
            created_at: string;
            websocket_url: string;
        }>;
        /**
         * Get evolution status
         */
        getStatus: (evolutionId: string) => Promise<WorkflowExecution>;
        /**
         * Pause evolution
         */
        pause: (evolutionId: string) => Promise<{
            evolution_id: string;
            status: string;
            paused_at: string;
        }>;
        /**
         * Resume evolution
         */
        resume: (evolutionId: string) => Promise<{
            evolution_id: string;
            status: string;
            resumed_at: string;
        }>;
        /**
         * Stop evolution
         */
        stop: (evolutionId: string) => Promise<{
            evolution_id: string;
            status: string;
            stopped_at: string;
            final_results: any;
        }>;
        /**
         * Delete evolution
         */
        delete: (evolutionId: string) => Promise<unknown>;
        /**
         * List evolutions
         */
        list: (params?: {
            status?: string;
            limit?: number;
            offset?: number;
            sort?: string;
            order?: "asc" | "desc";
        }) => Promise<{
            evolutions: WorkflowExecution[];
            total: number;
            limit: number;
            offset: number;
        }>;
    };
    adversarial: {
        /**
         * Start adversarial test
         */
        start: (data: {
            content: string;
            attack_modes: string[];
            parameters: {
                num_rounds: number;
                red_team_models: Array<{
                    provider: string;
                    model: string;
                }>;
                blue_team_models: Array<{
                    provider: string;
                    model: string;
                }>;
            };
        }) => Promise<{
            test_id: string;
            status: string;
            created_at: string;
            websocket_url: string;
        }>;
        /**
         * Get adversarial test status
         */
        getStatus: (testId: string) => Promise<AdversarialTest>;
        /**
         * Approve or reject patch
         */
        approvePatch: (testId: string, data: {
            round: number;
            approved: boolean;
            feedback?: string;
        }) => Promise<{
            test_id: string;
            round: number;
            patch_approved: boolean;
        }>;
        /**
         * Stop adversarial test
         */
        stop: (testId: string) => Promise<{
            test_id: string;
            status: string;
            stopped_at: string;
        }>;
        /**
         * List adversarial tests
         */
        list: (params?: {
            status?: string;
            limit?: number;
            offset?: number;
        }) => Promise<{
            tests: AdversarialTest[];
            total: number;
        }>;
    };
    analytics: {
        /**
         * Get metrics
         */
        getMetrics: (params: {
            start_date: string;
            end_date: string;
            granularity: "hour" | "day" | "week" | "month";
        }) => Promise<AnalyticsData>;
        /**
         * Get performance analytics
         */
        getPerformance: () => Promise<PerformanceAnalytics>;
    };
    monitoring: {
        /**
         * Get system health
         */
        getHealth: () => Promise<{
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
        }>;
        /**
         * Get application logs
         */
        getLogs: (params?: {
            level?: "INFO" | "WARNING" | "ERROR";
            limit?: number;
            offset?: number;
        }) => Promise<{
            logs: Array<{
                timestamp: string;
                level: string;
                message: string;
                context?: any;
            }>;
            total: number;
        }>;
    };
    content: {
        /**
         * Create content
         */
        create: (data: {
            title: string;
            content: string;
            language?: string;
            tags?: string[];
        }) => Promise<KnowledgeArtifact>;
        /**
         * Get content by ID
         */
        getById: (contentId: string) => Promise<KnowledgeArtifact>;
        /**
         * Update content
         */
        update: (contentId: string, data: Partial<KnowledgeArtifact>) => Promise<KnowledgeArtifact>;
        /**
         * Delete content
         */
        delete: (contentId: string) => Promise<unknown>;
        /**
         * List content
         */
        list: (params?: {
            tag?: string;
            language?: string;
            limit?: number;
            offset?: number;
        }) => Promise<{
            content: KnowledgeArtifact[];
            total: number;
        }>;
    };
    version: {
        /**
         * Get version history
         */
        getHistory: (contentId: string) => Promise<{
            version: number;
            created_at: string;
            created_by: string;
            comment: string;
        }[]>;
        /**
         * Revert to version
         */
        revert: (contentId: string, version: number) => Promise<{
            content_id: string;
            reverted_to_version: number;
            new_version: number;
            reverted_at: string;
        }>;
        /**
         * Get diff between versions
         */
        getDiff: (contentId: string, version1: number, version2: number) => Promise<{
            version1: number;
            version2: number;
            diff: string;
        }>;
        /**
         * Create branch
         */
        createBranch: (contentId: string, data: {
            branch_name: string;
            from_version: number;
        }) => Promise<{
            branch_id: string;
            branch_name: string;
            created_at: string;
        }>;
        /**
         * List branches
         */
        listBranches: (contentId: string) => Promise<{
            branch_id: string;
            branch_name: string;
            version: number;
            created_at: string;
        }[]>;
    };
    collaboration: {
        /**
         * Create collaboration room
         */
        createRoom: (data: {
            content_id: string;
            room_name?: string;
        }) => Promise<{
            room_id: string;
            room_name: string;
            websocket_url: string;
            created_at: string;
        }>;
        /**
         * Get active users in room
         */
        getRoomUsers: (roomId: string) => Promise<{
            user_id: string;
            username: string;
            joined_at: string;
            cursor_position?: {
                line: number;
                column: number;
            };
        }[]>;
    };
    comments: {
        /**
         * Add comment to content
         */
        add: (contentId: string, data: {
            comment: string;
            line_start?: number;
            line_end?: number;
            parent_comment_id?: string;
        }) => Promise<{
            comment_id: string;
            comment: string;
            created_at: string;
        }>;
        /**
         * Get comments for content
         */
        get: (contentId: string) => Promise<{
            comment_id: string;
            user_id: string;
            username: string;
            comment: string;
            line_start?: number;
            line_end?: number;
            created_at: string;
            replies: any[];
        }[]>;
    };
    config: {
        /**
         * Get available providers
         */
        getProviders: () => Promise<{
            provider: string;
            name: string;
            models: string[];
            requires_api_key: boolean;
        }[]>;
        /**
         * Save API key for provider
         */
        saveApiKey: (provider: string, apiKey: string) => Promise<{
            provider: string;
            api_key_last_four: string;
            saved_at: string;
        }>;
        /**
         * Get user parameters
         */
        getParameters: () => Promise<{
            generation: {
                temperature: number;
                top_p: number;
                max_tokens: number;
            };
            evolution: {
                max_iterations: number;
                population_size: number;
            };
        }>;
        /**
         * Update user parameters
         */
        updateParameters: (params: {
            generation?: {
                temperature?: number;
                top_p?: number;
                max_tokens?: number;
            };
            evolution?: {
                max_iterations?: number;
                population_size?: number;
            };
        }) => Promise<unknown>;
    };
    workflow: {
        /**
         * Start integrated workflow
         */
        start: (data: {
            problem_statement: string;
            workflow_template?: string;
            parameters?: Record<string, any>;
        }) => Promise<{
            workflow_id: string;
            status: string;
            current_stage: string;
            websocket_url: string;
        }>;
        /**
         * Get workflow status
         */
        getStatus: (workflowId: string) => Promise<{
            workflow_id: string;
            status: string;
            current_stage: string;
            stages: Array<{
                stage: string;
                status: string;
                result?: any;
                progress?: number;
            }>;
        }>;
    };
    files: {
        /**
         * Upload file
         */
        upload: (file: File, onProgress?: (progress: number) => void) => Promise<{
            file_id: string;
            filename: string;
            size: number;
            mime_type: string;
            uploaded_at: string;
        }>;
        /**
         * Download file
         */
        download: (fileId: string, filename?: string) => Promise<void>;
        /**
         * Get file metadata
         */
        getMetadata: (fileId: string) => Promise<{
            file_id: string;
            filename: string;
            size: number;
            mime_type: string;
            uploaded_at: string;
        }>;
    };
    leanaide: {
        /**
         * Generate Lean 4 proof
         */
        generateProof: (data: {
            theorem: string;
            proof_attempt?: string;
            model: string;
            temperature: number;
        }) => Promise<LeanCodeOutput>;
        /**
         * Verify Lean 4 proof
         */
        verifyProof: (code: string) => Promise<VerificationResult>;
        /**
         * Get supported models
         */
        getModels: () => Promise<{
            provider: string;
            models: string[];
        }[]>;
        /**
         * Run benchmark
         */
        runBenchmark: (data: {
            dataset: any[];
            model: string;
            evaluator: string;
        }) => Promise<{
            benchmark_id: string;
            status: string;
        }>;
        /**
         * Get benchmark results
         */
        getBenchmarkResults: (benchmarkId: string) => Promise<any[]>;
    };
};
export { authApi as auth };
export { userApi as user };
export { evolutionApi as evolution };
export { adversarialApi as adversarial };
export { analyticsApi as analytics };
export { monitoringApi as monitoring };
export { contentApi as content };
export { versionApi as version };
export { collaborationApi as collaboration };
export { commentsApi as comments };
export { configApi as config };
export { workflowApi as workflow };
export { filesApi as files };
export { leanaideApi as leanaide };
