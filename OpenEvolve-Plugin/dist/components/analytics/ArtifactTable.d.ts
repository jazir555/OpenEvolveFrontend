export interface Artifact {
    id: string;
    name: string;
    type: string;
    created: string;
    status: string;
    fitness?: number;
    generation?: number;
}
interface ArtifactTableProps {
    artifacts: Artifact[];
    onRowClick?: (artifact: Artifact) => void;
    className?: string;
}
export declare function ArtifactTable({ artifacts, onRowClick, className }: ArtifactTableProps): import("react/jsx-runtime").JSX.Element;
export {};
