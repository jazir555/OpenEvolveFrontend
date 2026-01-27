import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const ChartJSToolParamsSchema: z.ZodObject<{
    data: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
    chartType: z.ZodEnum<["line", "bar", "pie", "doughnut", "radar", "scatter", "bubble", "polarArea"]>;
    xColumn: z.ZodOptional<z.ZodString>;
    yColumn: z.ZodOptional<z.ZodString>;
    groupByColumn: z.ZodOptional<z.ZodString>;
    options: z.ZodOptional<z.ZodObject<{
        title: z.ZodOptional<z.ZodString>;
        xAxisLabel: z.ZodOptional<z.ZodString>;
        yAxisLabel: z.ZodOptional<z.ZodString>;
        colorScheme: z.ZodDefault<z.ZodEnum<["default", "viridis", "plasma", "inferno", "magma", "blues", "greens", "reds", "oranges", "categorical"]>>;
        responsive: z.ZodDefault<z.ZodBoolean>;
        maintainAspectRatio: z.ZodDefault<z.ZodBoolean>;
        showLegend: z.ZodDefault<z.ZodBoolean>;
        showTooltips: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        colorScheme: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical";
        responsive: boolean;
        maintainAspectRatio: boolean;
        showLegend: boolean;
        showTooltips: boolean;
        title?: string | undefined;
        xAxisLabel?: string | undefined;
        yAxisLabel?: string | undefined;
    }, {
        title?: string | undefined;
        xAxisLabel?: string | undefined;
        yAxisLabel?: string | undefined;
        colorScheme?: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical" | undefined;
        responsive?: boolean | undefined;
        maintainAspectRatio?: boolean | undefined;
        showLegend?: boolean | undefined;
        showTooltips?: boolean | undefined;
    }>>;
    advancedConfig: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    reasoning: z.ZodString;
    generateFile: z.ZodDefault<z.ZodBoolean>;
    filePath: z.ZodOptional<z.ZodString>;
    fileName: z.ZodOptional<z.ZodString>;
    width: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    height: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    data: Record<string, unknown>[];
    width: number;
    height: number;
    reasoning: string;
    chartType: "line" | "bar" | "pie" | "doughnut" | "radar" | "scatter" | "bubble" | "polarArea";
    generateFile: boolean;
    options?: {
        colorScheme: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical";
        responsive: boolean;
        maintainAspectRatio: boolean;
        showLegend: boolean;
        showTooltips: boolean;
        title?: string | undefined;
        xAxisLabel?: string | undefined;
        yAxisLabel?: string | undefined;
    } | undefined;
    filePath?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
    fileName?: string | undefined;
    xColumn?: string | undefined;
    yColumn?: string | undefined;
    groupByColumn?: string | undefined;
    advancedConfig?: Record<string, unknown> | undefined;
}, {
    data: Record<string, unknown>[];
    reasoning: string;
    chartType: "line" | "bar" | "pie" | "doughnut" | "radar" | "scatter" | "bubble" | "polarArea";
    options?: {
        title?: string | undefined;
        xAxisLabel?: string | undefined;
        yAxisLabel?: string | undefined;
        colorScheme?: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical" | undefined;
        responsive?: boolean | undefined;
        maintainAspectRatio?: boolean | undefined;
        showLegend?: boolean | undefined;
        showTooltips?: boolean | undefined;
    } | undefined;
    filePath?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
    fileName?: string | undefined;
    width?: number | undefined;
    height?: number | undefined;
    xColumn?: string | undefined;
    yColumn?: string | undefined;
    groupByColumn?: string | undefined;
    advancedConfig?: Record<string, unknown> | undefined;
    generateFile?: boolean | undefined;
}>;
type ChartJSToolParamsInput = z.input<typeof ChartJSToolParamsSchema>;
type ChartJSToolParams = z.output<typeof ChartJSToolParamsSchema>;
type ChartJSToolResult = z.output<typeof ChartJSToolResultSchema>;
declare const ChartJSToolResultSchema: z.ZodObject<{
    chartConfig: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    chartType: z.ZodString;
    datasetCount: z.ZodNumber;
    dataPointCount: z.ZodNumber;
    suggestedSize: z.ZodObject<{
        width: z.ZodNumber;
        height: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        width: number;
        height: number;
    }, {
        width: number;
        height: number;
    }>;
    metadata: z.ZodObject<{
        xColumn: z.ZodOptional<z.ZodString>;
        yColumn: z.ZodOptional<z.ZodString>;
        groupByColumn: z.ZodOptional<z.ZodString>;
        colorScheme: z.ZodString;
        generatedAt: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        colorScheme: string;
        generatedAt: string;
        xColumn?: string | undefined;
        yColumn?: string | undefined;
        groupByColumn?: string | undefined;
    }, {
        colorScheme: string;
        generatedAt: string;
        xColumn?: string | undefined;
        yColumn?: string | undefined;
        groupByColumn?: string | undefined;
    }>;
    filePath: z.ZodOptional<z.ZodString>;
    fileExists: z.ZodOptional<z.ZodBoolean>;
    fileSize: z.ZodOptional<z.ZodNumber>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    metadata: {
        colorScheme: string;
        generatedAt: string;
        xColumn?: string | undefined;
        yColumn?: string | undefined;
        groupByColumn?: string | undefined;
    };
    chartType: string;
    chartConfig: Record<string, unknown>;
    datasetCount: number;
    dataPointCount: number;
    suggestedSize: {
        width: number;
        height: number;
    };
    filePath?: string | undefined;
    fileSize?: number | undefined;
    fileExists?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    metadata: {
        colorScheme: string;
        generatedAt: string;
        xColumn?: string | undefined;
        yColumn?: string | undefined;
        groupByColumn?: string | undefined;
    };
    chartType: string;
    chartConfig: Record<string, unknown>;
    datasetCount: number;
    dataPointCount: number;
    suggestedSize: {
        width: number;
        height: number;
    };
    filePath?: string | undefined;
    fileSize?: number | undefined;
    fileExists?: boolean | undefined;
}>;
/**
 * ChartJSTool - Generate Chart.js configurations from data
 *
 * This tool bubble converts data (typically from SQL queries) into Chart.js
 * configuration objects that can be used to render interactive charts.
 */
export declare class ChartJSTool extends ToolBubble<ChartJSToolParams, ChartJSToolResult> {
    static readonly type: "tool";
    static readonly bubbleName = "chart-js-tool";
    static readonly schema: z.ZodObject<{
        data: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">;
        chartType: z.ZodEnum<["line", "bar", "pie", "doughnut", "radar", "scatter", "bubble", "polarArea"]>;
        xColumn: z.ZodOptional<z.ZodString>;
        yColumn: z.ZodOptional<z.ZodString>;
        groupByColumn: z.ZodOptional<z.ZodString>;
        options: z.ZodOptional<z.ZodObject<{
            title: z.ZodOptional<z.ZodString>;
            xAxisLabel: z.ZodOptional<z.ZodString>;
            yAxisLabel: z.ZodOptional<z.ZodString>;
            colorScheme: z.ZodDefault<z.ZodEnum<["default", "viridis", "plasma", "inferno", "magma", "blues", "greens", "reds", "oranges", "categorical"]>>;
            responsive: z.ZodDefault<z.ZodBoolean>;
            maintainAspectRatio: z.ZodDefault<z.ZodBoolean>;
            showLegend: z.ZodDefault<z.ZodBoolean>;
            showTooltips: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            colorScheme: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical";
            responsive: boolean;
            maintainAspectRatio: boolean;
            showLegend: boolean;
            showTooltips: boolean;
            title?: string | undefined;
            xAxisLabel?: string | undefined;
            yAxisLabel?: string | undefined;
        }, {
            title?: string | undefined;
            xAxisLabel?: string | undefined;
            yAxisLabel?: string | undefined;
            colorScheme?: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical" | undefined;
            responsive?: boolean | undefined;
            maintainAspectRatio?: boolean | undefined;
            showLegend?: boolean | undefined;
            showTooltips?: boolean | undefined;
        }>>;
        advancedConfig: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        reasoning: z.ZodString;
        generateFile: z.ZodDefault<z.ZodBoolean>;
        filePath: z.ZodOptional<z.ZodString>;
        fileName: z.ZodOptional<z.ZodString>;
        width: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        height: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        data: Record<string, unknown>[];
        width: number;
        height: number;
        reasoning: string;
        chartType: "line" | "bar" | "pie" | "doughnut" | "radar" | "scatter" | "bubble" | "polarArea";
        generateFile: boolean;
        options?: {
            colorScheme: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical";
            responsive: boolean;
            maintainAspectRatio: boolean;
            showLegend: boolean;
            showTooltips: boolean;
            title?: string | undefined;
            xAxisLabel?: string | undefined;
            yAxisLabel?: string | undefined;
        } | undefined;
        filePath?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
        fileName?: string | undefined;
        xColumn?: string | undefined;
        yColumn?: string | undefined;
        groupByColumn?: string | undefined;
        advancedConfig?: Record<string, unknown> | undefined;
    }, {
        data: Record<string, unknown>[];
        reasoning: string;
        chartType: "line" | "bar" | "pie" | "doughnut" | "radar" | "scatter" | "bubble" | "polarArea";
        options?: {
            title?: string | undefined;
            xAxisLabel?: string | undefined;
            yAxisLabel?: string | undefined;
            colorScheme?: "default" | "viridis" | "plasma" | "inferno" | "magma" | "blues" | "greens" | "reds" | "oranges" | "categorical" | undefined;
            responsive?: boolean | undefined;
            maintainAspectRatio?: boolean | undefined;
            showLegend?: boolean | undefined;
            showTooltips?: boolean | undefined;
        } | undefined;
        filePath?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
        fileName?: string | undefined;
        width?: number | undefined;
        height?: number | undefined;
        xColumn?: string | undefined;
        yColumn?: string | undefined;
        groupByColumn?: string | undefined;
        advancedConfig?: Record<string, unknown> | undefined;
        generateFile?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        chartConfig: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        chartType: z.ZodString;
        datasetCount: z.ZodNumber;
        dataPointCount: z.ZodNumber;
        suggestedSize: z.ZodObject<{
            width: z.ZodNumber;
            height: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            width: number;
            height: number;
        }, {
            width: number;
            height: number;
        }>;
        metadata: z.ZodObject<{
            xColumn: z.ZodOptional<z.ZodString>;
            yColumn: z.ZodOptional<z.ZodString>;
            groupByColumn: z.ZodOptional<z.ZodString>;
            colorScheme: z.ZodString;
            generatedAt: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            colorScheme: string;
            generatedAt: string;
            xColumn?: string | undefined;
            yColumn?: string | undefined;
            groupByColumn?: string | undefined;
        }, {
            colorScheme: string;
            generatedAt: string;
            xColumn?: string | undefined;
            yColumn?: string | undefined;
            groupByColumn?: string | undefined;
        }>;
        filePath: z.ZodOptional<z.ZodString>;
        fileExists: z.ZodOptional<z.ZodBoolean>;
        fileSize: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        metadata: {
            colorScheme: string;
            generatedAt: string;
            xColumn?: string | undefined;
            yColumn?: string | undefined;
            groupByColumn?: string | undefined;
        };
        chartType: string;
        chartConfig: Record<string, unknown>;
        datasetCount: number;
        dataPointCount: number;
        suggestedSize: {
            width: number;
            height: number;
        };
        filePath?: string | undefined;
        fileSize?: number | undefined;
        fileExists?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        metadata: {
            colorScheme: string;
            generatedAt: string;
            xColumn?: string | undefined;
            yColumn?: string | undefined;
            groupByColumn?: string | undefined;
        };
        chartType: string;
        chartConfig: Record<string, unknown>;
        datasetCount: number;
        dataPointCount: number;
        suggestedSize: {
            width: number;
            height: number;
        };
        filePath?: string | undefined;
        fileSize?: number | undefined;
        fileExists?: boolean | undefined;
    }>;
    static readonly shortDescription = "Generate Chart.js configurations from data for interactive visualizations";
    static readonly longDescription = "\n    A tool bubble that converts data into Chart.js configuration objects for creating\n    interactive charts and visualizations.\n\n    Features:\n    - Support for multiple chart types (line, bar, pie, scatter, etc.)\n    - Automatic data column detection and mapping\n    - Smart color scheme selection\n    - Responsive chart configurations\n    - Support for grouped data and multiple series\n    - Advanced customization through Chart.js config\n\n    Chart Types:\n    - Line charts: Time series, trends, continuous data\n    - Bar charts: Categorical comparisons, counts\n    - Pie/Doughnut: Parts of a whole, percentages\n    - Scatter: Correlation analysis, x-y relationships\n    - Radar: Multi-dimensional comparisons\n    - Bubble: Three-dimensional data visualization\n\n    Use cases:\n    - Converting SQL query results into visual charts\n    - Creating dashboards and reports\n    - Data analysis and presentation\n    - Interactive data exploration\n  ";
    static readonly alias = "chart";
    private readonly logger;
    constructor(params: ChartJSToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<ChartJSToolResult>;
    /**
     * Auto-detect appropriate columns for X and Y axes
     */
    private detectColumns;
    /**
     * Generate complete Chart.js configuration
     */
    private generateChartConfig;
    /**
     * Prepare data in Chart.js format
     */
    private prepareChartData;
    /**
     * Prepare single series data
     */
    private prepareSingleSeriesData;
    /**
     * Prepare grouped data (multiple series)
     */
    private prepareGroupedData;
    /**
     * Generate Chart.js options
     */
    private generateChartOptions;
    /**
     * Get color palette based on scheme
     */
    private getColorPalette;
    /**
     * Calculate total data point count
     */
    private calculateDataPointCount;
    /**
     * Get suggested canvas size based on chart type and data
     */
    private getSuggestedSize;
    /**
     * Generate actual chart file using chartjs-node-canvas
     */
    private generateChartFile;
}
export {};
//# sourceMappingURL=chart-js-tool.d.ts.map