export interface WorkflowTab {
    id: string;
    label: string;
    icon?: string;
    content: React.ReactNode;
}
interface WorkflowTabsProps {
    tabs: WorkflowTab[];
    defaultTab?: string;
    className?: string;
}
export declare function WorkflowTabs({ tabs, defaultTab, className }: WorkflowTabsProps): import("react/jsx-runtime").JSX.Element;
export {};
