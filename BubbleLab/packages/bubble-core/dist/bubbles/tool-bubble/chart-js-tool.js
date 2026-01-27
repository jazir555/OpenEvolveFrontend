import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import * as fs from 'fs/promises';
import * as path from 'path';
import { createLogger } from '../../utils/logger.js';
import { CHART, CHART_COLORS, CHART_COLOR_SCHEMES, CHART_TYPES, DEFAULTS, } from '../../utils/constants.js';
// Define supported chart types
const ChartType = z.enum([
    'line',
    'bar',
    'pie',
    'doughnut',
    'radar',
    'scatter',
    'bubble',
    'polarArea',
]);
// Define color schemes
const ColorScheme = z.enum([
    'default',
    'viridis',
    'plasma',
    'inferno',
    'magma',
    'blues',
    'greens',
    'reds',
    'oranges',
    'categorical',
]);
// Define chart options for common customizations
const ChartOptionsSchema = z.object({
    title: z.string().optional().describe('Chart title'),
    xAxisLabel: z.string().optional().describe('X-axis label'),
    yAxisLabel: z.string().optional().describe('Y-axis label'),
    colorScheme: ColorScheme.default('default').describe('Color scheme for the chart'),
    responsive: z.boolean().default(true).describe('Make chart responsive'),
    maintainAspectRatio: z
        .boolean()
        .default(true)
        .describe('Maintain aspect ratio'),
    showLegend: z.boolean().default(true).describe('Show chart legend'),
    showTooltips: z.boolean().default(true).describe('Show tooltips on hover'),
});
// Define the parameters schema
const ChartJSToolParamsSchema = z.object({
    data: z
        .array(z.record(z.unknown()))
        .min(1, 'Data array cannot be empty')
        .describe('Array of data objects (typically from SQL query results)'),
    chartType: ChartType.describe('Type of chart to generate'),
    xColumn: z
        .string()
        .optional()
        .describe('Column name to use for X-axis (auto-detected if not provided)'),
    yColumn: z
        .string()
        .optional()
        .describe('Column name to use for Y-axis (auto-detected if not provided)'),
    groupByColumn: z
        .string()
        .optional()
        .describe('Column to group data by for multiple series'),
    options: ChartOptionsSchema.optional().describe('Chart customization options'),
    advancedConfig: z
        .record(z.unknown())
        .optional()
        .describe('Advanced Chart.js configuration object (overrides simple options)'),
    reasoning: z
        .string()
        .describe('Explain why this chart type and configuration was chosen'),
    generateFile: z
        .boolean()
        .default(false)
        .describe('Generate an actual chart image file (PNG format)'),
    filePath: z
        .string()
        .optional()
        .describe('Custom file path for generated chart (defaults to temp directory)'),
    fileName: z
        .string()
        .optional()
        .describe('Custom file name for generated chart (defaults to auto-generated name)'),
    width: z
        .number()
        .optional()
        .default(DEFAULTS.CHART_WIDTH)
        .describe('Chart width in pixels (default: 800)'),
    height: z
        .number()
        .optional()
        .default(DEFAULTS.CHART_HEIGHT)
        .describe('Chart height in pixels (default: 600)'),
    // Hidden from AI agents - injected at runtime
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Credentials (HIDDEN from AI - injected at runtime)'),
    config: z
        .record(z.string(), z.unknown())
        .optional()
        .describe('Configuration for the tool bubble (HIDDEN from AI - injected at runtime)'),
});
// Result schema
const ChartJSToolResultSchema = z.object({
    chartConfig: z
        .record(z.unknown())
        .describe('Complete Chart.js configuration object'),
    chartType: z.string().describe('Chart type that was generated'),
    datasetCount: z.number().describe('Number of datasets in the chart'),
    dataPointCount: z.number().describe('Total number of data points'),
    suggestedSize: z
        .object({
        width: z.number(),
        height: z.number(),
    })
        .describe('Suggested canvas size for the chart'),
    metadata: z
        .object({
        xColumn: z.string().optional(),
        yColumn: z.string().optional(),
        groupByColumn: z.string().optional(),
        colorScheme: z.string(),
        generatedAt: z.string(),
    })
        .describe('Metadata about chart generation'),
    filePath: z
        .string()
        .optional()
        .describe('Path to generated chart file (if generateFile was true)'),
    fileExists: z
        .boolean()
        .optional()
        .describe('Whether the generated file exists on disk'),
    fileSize: z.number().optional().describe('Size of generated file in bytes'),
    // Standard result fields
    success: z.boolean(),
    error: z.string(),
});
/**
 * ChartJSTool - Generate Chart.js configurations from data
 *
 * This tool bubble converts data (typically from SQL queries) into Chart.js
 * configuration objects that can be used to render interactive charts.
 */
export class ChartJSTool extends ToolBubble {
    static type = 'tool';
    static bubbleName = 'chart-js-tool';
    static schema = ChartJSToolParamsSchema;
    static resultSchema = ChartJSToolResultSchema;
    static shortDescription = 'Generate Chart.js configurations from data for interactive visualizations';
    static longDescription = `
    A tool bubble that converts data into Chart.js configuration objects for creating
    interactive charts and visualizations.

    Features:
    - Support for multiple chart types (line, bar, pie, scatter, etc.)
    - Automatic data column detection and mapping
    - Smart color scheme selection
    - Responsive chart configurations
    - Support for grouped data and multiple series
    - Advanced customization through Chart.js config

    Chart Types:
    - Line charts: Time series, trends, continuous data
    - Bar charts: Categorical comparisons, counts
    - Pie/Doughnut: Parts of a whole, percentages
    - Scatter: Correlation analysis, x-y relationships
    - Radar: Multi-dimensional comparisons
    - Bubble: Three-dimensional data visualization

    Use cases:
    - Converting SQL query results into visual charts
    - Creating dashboards and reports
    - Data analysis and presentation
    - Interactive data exploration
  `;
    static alias = 'chart';
    logger;
    constructor(params, context) {
        super(params, context);
        this.logger = createLogger('ChartJSTool');
    }
    async performAction(context) {
        void context;
        try {
            this.logger.debug('Generating chart', {
                chart_type: this.params.chartType,
                reasoning: this.params.reasoning,
                data_points: this.params.data?.length,
            });
            const { data, chartType, xColumn, yColumn, groupByColumn, options, advancedConfig, } = this.params;
            // Auto-detect columns if not provided
            const detectedColumns = this.detectColumns(data, xColumn, yColumn);
            const finalXColumn = xColumn || detectedColumns.xColumn;
            const finalYColumn = yColumn || detectedColumns.yColumn;
            if (!finalYColumn) {
                throw new Error('Could not detect Y-axis column. Please specify yColumn parameter.');
            }
            // Generate chart configuration
            const chartConfig = await this.generateChartConfig(data, chartType, finalXColumn, finalYColumn, groupByColumn, options, advancedConfig);
            // Calculate metadata
            const configData = chartConfig.data;
            const datasetCount = Array.isArray(configData?.datasets)
                ? configData.datasets.length
                : 1;
            const dataPointCount = this.calculateDataPointCount(chartConfig);
            const suggestedSize = this.getSuggestedSize(chartType, dataPointCount);
            // Generate file if requested
            let filePath;
            let fileExists;
            let fileSize;
            if (this.params.generateFile) {
                const parsedParams = ChartJSToolParamsSchema.parse(this.params);
                const dimensions = {
                    width: parsedParams.width,
                    height: parsedParams.height,
                };
                const fileResult = await this.generateChartFile(chartConfig, dimensions);
                filePath = fileResult.filePath;
                fileExists = fileResult.fileExists;
                fileSize = fileResult.fileSize;
            }
            this.logger.info('Chart generated successfully', {
                chart_type: chartType,
                dataset_count: datasetCount,
                data_point_count: dataPointCount,
                file_path: filePath,
                file_size: fileSize,
                file_exists: fileExists,
            });
            return {
                chartConfig,
                chartType,
                datasetCount,
                dataPointCount,
                suggestedSize,
                metadata: {
                    xColumn: finalXColumn,
                    yColumn: finalYColumn,
                    groupByColumn,
                    colorScheme: options?.colorScheme || CHART_COLOR_SCHEMES.DEFAULT,
                    generatedAt: new Date().toISOString(),
                },
                filePath,
                fileExists,
                fileSize,
                success: true,
                error: '',
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
            this.logger.error('Chart generation failed', error, {
                chart_type: this.params.chartType,
            });
            return {
                chartConfig: {},
                chartType: this.params.chartType,
                datasetCount: 0,
                dataPointCount: 0,
                suggestedSize: CHART.SIZE_SMALL,
                metadata: {
                    colorScheme: CHART_COLOR_SCHEMES.DEFAULT,
                    generatedAt: new Date().toISOString(),
                },
                filePath: undefined,
                fileExists: false,
                fileSize: undefined,
                success: false,
                error: errorMessage,
            };
        }
    }
    /**
     * Auto-detect appropriate columns for X and Y axes
     */
    detectColumns(data, xColumn, yColumn) {
        if (!data.length)
            return {};
        const firstRow = data[0];
        const columns = Object.keys(firstRow);
        // If both are provided, return them
        if (xColumn && yColumn) {
            return { xColumn, yColumn };
        }
        // Detect numeric columns for Y-axis
        const numericColumns = columns.filter((col) => {
            const values = data
                .slice(0, CHART.COLUMN_DETECTION_SAMPLE_SIZE)
                .map((row) => row[col]);
            return values.every((val) => val !== null &&
                val !== undefined &&
                (typeof val === 'number' || !isNaN(Number(val))));
        });
        // Detect categorical/date columns for X-axis
        const categoricalColumns = columns.filter((col) => !numericColumns.includes(col));
        return {
            xColumn: xColumn || categoricalColumns[0],
            yColumn: yColumn || numericColumns[0],
        };
    }
    /**
     * Generate complete Chart.js configuration
     */
    async generateChartConfig(data, chartType, xColumn, yColumn, groupByColumn, options, advancedConfig) {
        // If advanced config is provided, use it as base
        if (advancedConfig) {
            return {
                type: chartType,
                ...advancedConfig,
            };
        }
        // Generate basic configuration
        const chartData = this.prepareChartData(data, chartType, xColumn, yColumn, groupByColumn, options);
        const chartOptions = this.generateChartOptions(options);
        const config = {
            type: chartType,
            data: chartData,
            options: chartOptions,
        };
        return config;
    }
    /**
     * Prepare data in Chart.js format
     */
    prepareChartData(data, chartType, xColumn, yColumn, groupByColumn, options) {
        const colors = this.getColorPalette(options?.colorScheme || 'default');
        if (groupByColumn) {
            return this.prepareGroupedData(data, chartType, xColumn, yColumn, groupByColumn, colors);
        }
        else {
            return this.prepareSingleSeriesData(data, chartType, xColumn, yColumn, colors);
        }
    }
    /**
     * Prepare single series data
     */
    prepareSingleSeriesData(data, chartType, xColumn, yColumn, colors) {
        const labels = xColumn ? data.map((row) => String(row[xColumn])) : [];
        const values = yColumn ? data.map((row) => Number(row[yColumn])) : [];
        // For pie/doughnut charts, use categories as labels
        if (chartType === CHART_TYPES.PIE || chartType === CHART_TYPES.DOUGHNUT) {
            return {
                labels: labels.length ? labels : values.map((_, i) => `Item ${i + 1}`),
                datasets: [
                    {
                        data: values,
                        backgroundColor: colors,
                        borderColor: colors.map((c) => c.replace(CHART.OPACITY_DEFAULT, CHART.OPACITY_SOLID)),
                        borderWidth: CHART.BORDER_WIDTH_THIN,
                    },
                ],
            };
        }
        // For scatter/bubble charts
        if (chartType === CHART_TYPES.SCATTER || chartType === CHART_TYPES.BUBBLE) {
            const scatterData = data.map((row, i) => ({
                x: xColumn ? Number(row[xColumn]) : i,
                y: yColumn ? Number(row[yColumn]) : 0,
                ...(chartType === CHART_TYPES.BUBBLE && {
                    r: CHART.BUBBLE_RADIUS_DEFAULT,
                }),
            }));
            return {
                datasets: [
                    {
                        label: yColumn || 'Data',
                        data: scatterData,
                        backgroundColor: colors[0],
                        borderColor: colors[0].replace(CHART.OPACITY_DEFAULT, CHART.OPACITY_SOLID),
                    },
                ],
            };
        }
        // For line/bar charts
        return {
            labels: labels.length ? labels : data.map((_, i) => `Point ${i + 1}`),
            datasets: [
                {
                    label: yColumn || 'Data',
                    data: values,
                    backgroundColor: colors[0],
                    borderColor: colors[0].replace(CHART.OPACITY_DEFAULT, CHART.OPACITY_SOLID),
                    borderWidth: chartType === CHART_TYPES.LINE
                        ? CHART.BORDER_WIDTH_THICK
                        : CHART.BORDER_WIDTH_THIN,
                    fill: chartType === CHART_TYPES.LINE ? false : true,
                },
            ],
        };
    }
    /**
     * Prepare grouped data (multiple series)
     */
    prepareGroupedData(data, chartType, xColumn, yColumn, groupByColumn, colors) {
        // Group data by groupByColumn
        const groups = new Map();
        data.forEach((row) => {
            const groupKey = String(row[groupByColumn]);
            if (!groups.has(groupKey)) {
                groups.set(groupKey, []);
            }
            groups.get(groupKey).push(row);
        });
        const labels = xColumn
            ? [...new Set(data.map((row) => String(row[xColumn])))]
            : [];
        const datasets = Array.from(groups.entries()).map(([groupName, groupData], index) => {
            const values = yColumn
                ? groupData.map((row) => Number(row[yColumn]))
                : [];
            const color = colors[index % colors.length];
            return {
                label: groupName,
                data: values,
                backgroundColor: color,
                borderColor: color.replace(CHART.OPACITY_DEFAULT, CHART.OPACITY_SOLID),
                borderWidth: chartType === CHART_TYPES.LINE
                    ? CHART.BORDER_WIDTH_THICK
                    : CHART.BORDER_WIDTH_THIN,
                fill: chartType === CHART_TYPES.LINE ? false : true,
            };
        });
        return {
            labels: labels.length ? labels : [],
            datasets,
        };
    }
    /**
     * Generate Chart.js options
     */
    generateChartOptions(options) {
        const chartOptions = {
            responsive: options?.responsive ?? true,
            maintainAspectRatio: options?.maintainAspectRatio ?? true,
        };
        if (options?.title) {
            chartOptions.plugins = {
                title: {
                    display: true,
                    text: options.title,
                },
                legend: {
                    display: options.showLegend ?? true,
                },
                tooltip: {
                    enabled: options.showTooltips ?? true,
                },
            };
        }
        if (options?.xAxisLabel || options?.yAxisLabel) {
            chartOptions.scales = {
                x: {
                    display: true,
                    title: {
                        display: !!options?.xAxisLabel,
                        text: options?.xAxisLabel || '',
                    },
                },
                y: {
                    display: true,
                    title: {
                        display: !!options?.yAxisLabel,
                        text: options?.yAxisLabel || '',
                    },
                },
            };
        }
        return chartOptions;
    }
    /**
     * Get color palette based on scheme
     */
    getColorPalette(scheme) {
        const schemeKey = scheme.toUpperCase();
        if (schemeKey in CHART_COLORS) {
            return [...CHART_COLORS[schemeKey]];
        }
        this.logger.warn('Unknown color scheme, using default', {
            scheme,
            available_schemes: Object.keys(CHART_COLORS),
        });
        return [...CHART_COLORS.DEFAULT];
    }
    /**
     * Calculate total data point count
     */
    calculateDataPointCount(chartConfig) {
        const data = chartConfig.data;
        if (!data?.datasets)
            return 0;
        return data.datasets.reduce((total, dataset) => {
            return total + (Array.isArray(dataset.data) ? dataset.data.length : 0);
        }, 0);
    }
    /**
     * Get suggested canvas size based on chart type and data
     */
    getSuggestedSize(chartType, dataPointCount) {
        const baseSize = CHART.SIZE_SMALL;
        // Adjust for chart type
        if (chartType === CHART_TYPES.PIE ||
            chartType === CHART_TYPES.DOUGHNUT ||
            chartType === CHART_TYPES.POLAR_AREA) {
            return CHART.SIZE_SQUARE;
        }
        if (chartType === CHART_TYPES.RADAR) {
            return CHART.SIZE_RADAR;
        }
        // Adjust for data density
        if (dataPointCount > CHART.DATA_DENSITY_THRESHOLD) {
            return CHART.SIZE_MEDIUM;
        }
        return baseSize;
    }
    /**
     * Generate actual chart file using chartjs-node-canvas
     */
    async generateChartFile(chartConfig, dimensions) {
        const { width, height } = dimensions;
        // Create chartjs-node-canvas instance
        const chartJSNodeCanvas = new ChartJSNodeCanvas({
            width,
            height,
            backgroundColour: 'white',
        });
        try {
            // Generate the chart buffer
            this.logger.debug('Rendering chart to buffer', { width, height });
            const buffer = await chartJSNodeCanvas.renderToBuffer(chartConfig);
            // Determine file path
            const defaultDir = '/tmp/charts';
            let outputDir = this.params.filePath || defaultDir;
            let fileName = this.params.fileName ||
                `chart-${this.params.chartType}-${Date.now()}.png`;
            // SECURITY FIX: Prevent path traversal attacks
            // Normalize and validate output directory
            outputDir = path.normalize(outputDir);
            // Remove any parent directory references
            if (outputDir.includes('..')) {
                throw new Error(`Invalid file path: path traversal detected in directory`);
            }
            // Ensure directory is within allowed paths
            if (!outputDir.startsWith(defaultDir) && !outputDir.startsWith('/tmp/') && !path.isAbsolute(outputDir)) {
                outputDir = path.join(defaultDir, path.basename(outputDir));
            }
            // SECURITY FIX: Validate filename doesn't contain path traversal
            fileName = path.basename(fileName); // Remove any directory components
            if (fileName !== this.params.fileName && this.params.fileName?.includes('..')) {
                throw new Error(`Invalid file name: path traversal detected`);
            }
            const fullPath = path.join(outputDir, fileName);
            // Ensure directory exists
            await fs.mkdir(outputDir, { recursive: true });
            // Write file
            await fs.writeFile(fullPath, buffer);
            // Verify file exists and get size
            const stats = await fs.stat(fullPath);
            this.logger.info('Chart file generated', {
                file_path: fullPath,
                file_size: stats.size,
                dimensions: { width, height },
            });
            return {
                filePath: fullPath,
                fileExists: true,
                fileSize: stats.size,
            };
        }
        catch (error) {
            this.logger.error('File generation failed', error, { dimensions });
            throw new Error(`Chart file generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
}
//# sourceMappingURL=chart-js-tool.js.map