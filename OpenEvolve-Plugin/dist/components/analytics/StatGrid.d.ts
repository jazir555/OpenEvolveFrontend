export interface Stat {
    title: string;
    value: number | string;
    change?: number;
    unit?: string;
    icon?: React.ReactNode;
    trend?: 'up' | 'down' | 'neutral';
}
interface StatGridProps {
    stats: Stat[];
    columns?: 2 | 3 | 4;
    className?: string;
}
export declare function StatGrid({ stats, columns, className }: StatGridProps): import("react/jsx-runtime").JSX.Element;
export {};
