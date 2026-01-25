export interface ArtifactVersion {
    version: number;
    content: string;
    created: string;
    created_by: string;
    comment?: string;
}
export interface ArtifactDetail {
    id: string;
    name: string;
    type: string;
    description: string;
    content: string;
    tags: string[];
    versions: ArtifactVersion[];
    current_version: number;
    created: string;
    updated: string;
    created_by: string;
}
interface ArtifactDetailProps {
    artifact: ArtifactDetail;
    onEdit?: () => void;
    onDelete?: () => void;
    onVersionRestore?: (version: number) => void;
    className?: string;
}
export declare function ArtifactDetail({ artifact, onEdit, onDelete, onVersionRestore, className, }: ArtifactDetailProps): import("react/jsx-runtime").JSX.Element;
export {};
