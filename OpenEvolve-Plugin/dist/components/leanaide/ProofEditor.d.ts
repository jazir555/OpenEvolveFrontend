interface ProofEditorProps {
    value: string;
    onChange: (value: string) => void;
    language?: string;
    readOnly?: boolean;
    className?: string;
}
export declare function ProofEditor({ value, onChange, language, readOnly, className, }: ProofEditorProps): import("react/jsx-runtime").JSX.Element;
export {};
