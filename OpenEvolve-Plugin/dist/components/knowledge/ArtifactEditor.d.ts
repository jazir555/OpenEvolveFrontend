export interface ArtifactData {
    id?: string;
    name: string;
    type: string;
    description: string;
    content: string;
    tags: string[];
    metadata?: Record<string, any>;
}
interface ArtifactEditorProps {
    artifact?: ArtifactData;
    onSave: (artifact: ArtifactData) => Promise<void>;
    onCancel?: () => void;
    types: string[];
    className?: string;
}
export declare function ArtifactEditor({ artifact, onSave, onCancel, types, className, }: ArtifactEditorProps): import("react/jsx-runtime").JSX.Element;
export {};
