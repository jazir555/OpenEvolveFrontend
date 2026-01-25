import { Workflow } from './WorkflowCard';
interface WorkflowListProps {
    workflows: Workflow[];
    onWorkflowSelect?: (workflow: Workflow) => void;
    onExecute?: (evolutionId: string) => void;
    onPause?: (evolutionId: string) => void;
    onResume?: (evolutionId: string) => void;
    onStop?: (evolutionId: string) => void;
    onDelete?: (evolutionId: string) => void;
    className?: string;
}
export declare function WorkflowList({ workflows, onWorkflowSelect, onExecute, onPause, onResume, onStop, onDelete, className, }: WorkflowListProps): import("react/jsx-runtime").JSX.Element;
export {};
