import { ReactNode } from 'react';
export declare function BubbleCard({ title, description, actions, children, className, }: {
    title?: string;
    description?: string;
    actions?: ReactNode;
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function BubbleField({ label, hint, children, className, }: {
    label: string;
    hint?: string;
    children: ReactNode;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function BubbleInput({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>): import("react/jsx-runtime").JSX.Element;
export declare function BubbleTextArea({ className, ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement>): import("react/jsx-runtime").JSX.Element;
export declare function BubbleSelect({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>): import("react/jsx-runtime").JSX.Element;
export declare function BubbleButton({ className, variant, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'primary' | 'secondary' | 'ghost';
}): import("react/jsx-runtime").JSX.Element;
export declare function BubbleBadge({ children, tone, className, }: {
    children: ReactNode;
    tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'info';
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function BubbleToggle({ checked, onChange, label, className, }: {
    checked: boolean;
    onChange: (checked: boolean) => void;
    label?: string;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare function BubbleCheckbox({ label, className, ...props }: React.InputHTMLAttributes<HTMLInputElement> & {
    label?: string;
}): import("react/jsx-runtime").JSX.Element;
