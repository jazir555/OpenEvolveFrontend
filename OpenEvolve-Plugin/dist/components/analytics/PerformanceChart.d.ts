export interface PerformanceDataPoint {
    timestamp: string;
    value: number;
    label?: string;
}
interface PerformanceChartProps {
    data: PerformanceDataPoint[];
    type?: 'line' | 'bar';
    title?: string;
    color?: string;
    className?: string;
    height?: number;
}
export declare function PerformanceChart({ data, type, title, color, className, height, }: PerformanceChartProps): import("react/jsx-runtime").JSX.Element;
export {};
