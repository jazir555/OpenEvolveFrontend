export interface Workflow {
    evolution_id: string;
    name?: string;
    description?: string;
    status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';
    created_at: string;
    updated_at: string;
    progress?: {
        current_iteration: number;
        max_iterations: number;
        percentage: number;
    };
}
interface WorkflowCardProps {
    workflow: Workflow;
    onClick?: (workflow: Workflow) => void;
    onExecute?: (evolutionId: string) => void;
    onPause?: (evolutionId: string) => void;
    onResume?: (evolutionId: string) => void;
    onStop?: (evolutionId: string) => void;
    onDelete?: (evolutionId: string) => void;
    className?: string;
}
export declare function WorkflowCard({ workflow, onClick, onExecute, onPause, onResume, onStop, onDelete, className, }: WorkflowCardProps): import("react/jsx-runtime").JSX.Element;
export {};
