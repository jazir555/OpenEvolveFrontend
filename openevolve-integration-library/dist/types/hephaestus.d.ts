import { ExecutionConfig } from './common';
export interface HephaestusInputs {
    operation: 'delegate' | 'status' | 'create' | 'list';
    input: Task | Ticket | StatusInput;
    config?: ExecutionConfig;
}
export interface Task {
    title: string;
    description: string;
    type: 'development' | 'research' | 'testing' | 'documentation' | 'custom';
    priority: 'low' | 'medium' | 'high' | 'critical';
    skills?: string[];
    requirements?: TaskRequirement[];
    estimatedEffort?: {
        hours?: number;
        complexity?: 'low' | 'medium' | 'high';
    };
    dependencies?: string[];
    dueDate?: string;
    tags?: string[];
    metadata?: Record<string, any>;
}
export interface TaskRequirement {
    type: string;
    description: string;
    required: boolean;
    value?: any;
}
export interface Ticket {
    ticketId?: string;
    task: Task;
    status?: TicketStatus;
    assignment?: TicketAssignment;
    progress?: TicketProgress;
    created?: string;
    updated?: string;
}
export type TicketStatus = 'open' | 'assigned' | 'in-progress' | 'review' | 'completed' | 'cancelled' | 'blocked';
export interface TicketAssignment {
    agentId?: string;
    userId?: string;
    assignedAt?: string;
    reason?: string;
}
export interface TicketProgress {
    percentage: number;
    message?: string;
    milestones: string[];
}
export interface StatusInput {
    ticketId: string;
    includeProgress?: boolean;
    includeHistory?: boolean;
}
export interface DelegationResult {
    ticketId: string;
    status: 'submitted' | 'assigned' | 'queued' | 'failed';
    assignedAgent?: string;
    estimatedCompletion?: string;
    metadata: DelegationMetadata;
}
export interface DelegationMetadata {
    delegatedAt: string;
    strategy: string;
    matchingScore?: number;
    queuePosition?: number;
}
export interface TicketStatusResult {
    ticketId: string;
    status: TicketStatus;
    progress: TicketProgress;
}
export interface TicketStatusHistory {
    status: TicketStatus;
    timestamp: string;
    changedBy: string;
    comments?: string;
}
export interface TicketListResult {
    tickets: TicketSummary[];
    total: number;
    pagination?: {
        page: number;
        pageSize: number;
        totalPages: number;
    };
}
export interface TicketSummary {
    ticketId: string;
    title: string;
    type: string;
    priority: string;
    status: TicketStatus;
    assignedTo?: string;
    created: string;
    dueDate?: string;
}
export interface HephaestusResult {
    type: 'delegate' | 'status' | 'create' | 'list';
    result: DelegationResult | TicketStatusResult | Ticket | TicketListResult;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
//# sourceMappingURL=hephaestus.d.ts.map