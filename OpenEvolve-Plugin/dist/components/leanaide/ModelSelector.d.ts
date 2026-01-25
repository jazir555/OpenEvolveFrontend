export interface LeanModel {
    id: string;
    name: string;
    provider: string;
    description?: string;
    capabilities: string[];
}
interface ModelSelectorProps {
    models: LeanModel[];
    selectedModel: string;
    onModelChange: (modelId: string) => void;
    className?: string;
}
export declare function ModelSelector({ models, selectedModel, onModelChange, className, }: ModelSelectorProps): import("react/jsx-runtime").JSX.Element;
export {};
