interface MetricCardProps {
    title: string;
    value: number | string;
    change?: number;
    unit?: string;
    icon?: React.ReactNode;
    trend?: 'up' | 'down' | 'neutral';
    className?: string;
}
export declare function MetricCard({ title, value, change, unit, icon, trend, className, }: MetricCardProps): import("react/jsx-runtime").JSX.Element;
export {};
