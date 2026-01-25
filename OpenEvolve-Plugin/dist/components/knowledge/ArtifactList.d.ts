export interface Artifact {
    id: string;
    name: string;
    type: string;
    description?: string;
    created: string;
    updated: string;
    version: number;
    tags: string[];
}
interface ArtifactListProps {
    artifacts: Artifact[];
    onArtifactSelect?: (artifact: Artifact) => void;
    onArtifactDelete?: (artifactId: string) => void;
    className?: string;
}
export declare function ArtifactList({ artifacts, onArtifactSelect, onArtifactDelete, className, }: ArtifactListProps): import("react/jsx-runtime").JSX.Element;
export {};
