/**
 * Edge Case and Boundary Tests for Chart.js Tool
 *
 * Comprehensive edge case coverage including:
 * - Input boundaries (empty, null, max length, unicode, special characters)
 * - Data boundaries (numeric limits, array sizes, nested data)
 * - Chart type edge cases
 * - Color and styling edge cases
 * - Performance edge cases (large datasets, many series)
 * - Error paths (invalid data, missing fields)
 */

import { describe, it, expect } from 'vitest';
import { ChartJSTool } from './chart-js-tool.js';

describe('ChartJSTool - Edge Cases and Boundary Tests', () => {
  describe('Input Boundary Tests', () => {
    describe('Data Array Boundaries', () => {
      it('should handle empty data array', async () => {
        const tool = new ChartJSTool({
          data: [],
          chartType: 'line',
          reasoning: 'Testing empty data array',
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
        expect(result.error).toContain('at least one data point');
      });

      it('should handle single data point', async () => {
        const tool = new ChartJSTool({
          data: [{ x: 1, y: 2 }],
          chartType: 'scatter',
          reasoning: 'Testing single data point',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
        expect(result.data?.dataPointCount).toBe(1);
      });

      it('should handle maximum practical data size', async () => {
        const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
          x: i,
          y: Math.random() * 100,
        }));

        const tool = new ChartJSTool({
          data: largeDataset,
          chartType: 'scatter',
          reasoning: 'Testing large dataset',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
        expect(result.data?.dataPointCount).toBe(10000);
      });

      it('should handle data with all null values', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: null, y: null },
            { x: null, y: null },
          ],
          chartType: 'scatter',
          reasoning: 'Testing all null values',
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle data with mixed null values', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: 1, y: 2 },
            { x: null, y: 3 },
            { x: 3, y: null },
            { x: 4, y: 5 },
          ],
          chartType: 'scatter',
          reasoning: 'Testing mixed null values',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });
    });

    describe('Numeric Boundaries', () => {
      it('should handle maximum numeric values (Number.MAX_SAFE_INTEGER)', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: Number.MAX_SAFE_INTEGER, y: Number.MAX_SAFE_INTEGER },
          ],
          chartType: 'scatter',
          reasoning: 'Testing max safe integer',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle minimum numeric values (Number.MIN_SAFE_INTEGER)', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: Number.MIN_SAFE_INTEGER, y: Number.MIN_SAFE_INTEGER },
          ],
          chartType: 'scatter',
          reasoning: 'Testing min safe integer',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle zero values', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 0, y: 1 },
          ],
          chartType: 'scatter',
          reasoning: 'Testing zero values',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle negative values', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: -100, y: -200 },
            { x: -50, y: -100 },
            { x: -25, y: -50 },
          ],
          chartType: 'scatter',
          reasoning: 'Testing negative values',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle decimal precision', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: 0.123456789, y: 0.987654321 },
            { x: 1.1, y: 2.2 },
            { x: 3.14159, y: 2.71828 },
          ],
          chartType: 'scatter',
          reasoning: 'Testing decimal precision',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle scientific notation values', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: 1e10, y: 1e-10 },
            { x: 1.5e5, y: 2.5e-5 },
          ],
          chartType: 'scatter',
          reasoning: 'Testing scientific notation',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle Infinity values', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: 1, y: 2 },
            { x: Infinity, y: 3 },
            { x: 4, y: -Infinity },
          ],
          chartType: 'scatter',
          reasoning: 'Testing infinity values',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle NaN values', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: 1, y: 2 },
            { x: NaN, y: 3 },
            { x: 4, y: NaN },
          ],
          chartType: 'scatter',
          reasoning: 'Testing NaN values',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });
    });

    describe('String Boundaries', () => {
      it('should handle empty string labels', async () => {
        const tool = new ChartJSTool({
          data: [
            { label: '', value: 10 },
            { label: 'test', value: 20 },
          ],
          chartType: 'bar',
          xColumn: 'label',
          yColumn: 'value',
          reasoning: 'Testing empty string labels',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle maximum length string labels (1000 chars)', async () => {
        const longLabel = 'x'.repeat(1000);

        const tool = new ChartJSTool({
          data: [
            { label: longLabel, value: 10 },
            { label: 'test', value: 20 },
          ],
          chartType: 'bar',
          xColumn: 'label',
          yColumn: 'value',
          reasoning: 'Testing long string labels',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle unicode characters in labels', async () => {
        const tool = new ChartJSTool({
          data: [
            { label: 'Hello 世界 🌍 Привет', value: 10 },
            { label: 'مرحبا 😀', value: 20 },
          ],
          chartType: 'bar',
          xColumn: 'label',
          yColumn: 'value',
          reasoning: 'Testing unicode labels',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle special characters in labels', async () => {
        const tool = new ChartJSTool({
          data: [
            { label: 'Test\n\t\r\\\"\'', value: 10 },
            { label: '<tag>&amp;</tag>', value: 20 },
            { label: 'emoji 😀 🎉', value: 30 },
          ],
          chartType: 'bar',
          xColumn: 'label',
          yColumn: 'value',
          reasoning: 'Testing special characters',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle whitespace-only labels', async () => {
        const tool = new ChartJSTool({
          data: [
            { label: '   ', value: 10 },
            { label: '\t\t', value: 20 },
            { label: '\n\n', value: 30 },
          ],
          chartType: 'bar',
          xColumn: 'label',
          yColumn: 'value',
          reasoning: 'Testing whitespace labels',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
      });

      it('should handle case-sensitive labels', async () => {
        const tool = new ChartJSTool({
          data: [
            { label: 'Test', value: 10 },
            { label: 'test', value: 20 },
            { label: 'TEST', value: 30 },
          ],
          chartType: 'bar',
          xColumn: 'label',
          yColumn: 'value',
          reasoning: 'Testing case sensitivity',
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
        expect(result.data?.datasetCount).toBe(1);
      });
    });

    describe('Column Name Boundaries', () => {
      it('should handle non-existent column names', async () => {
        const tool = new ChartJSTool({
          data: [
            { actualColumn: 10 },
            { actualColumn: 20 },
          ],
          chartType: 'bar',
          xColumn: 'nonexistentColumn',
          yColumn: 'anotherNonexistent',
          reasoning: 'Testing non-existent columns',
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle null/undefined column names', async () => {
        const tool = new ChartJSTool({
          data: [
            { x: 1, y: 2 },
          ],
          chartType: 'scatter',
          xColumn: null as any,
          yColumn: undefined as any,
          reasoning: 'Testing null column names',
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });

      it('should handle column names with special characters', async () => {
        const tool = new ChartJSTool({
          data: [
            { 'column-with-dash': 10 },
            { 'column_with_underscore': 20 },
            { 'column.with.dot': 30 },
          ],
          chartType: 'bar',
          xColumn: 'column-with-dash',
          yColumn: 'column_with_underscore',
          reasoning: 'Testing special characters in column names',
        });

        const result = await tool.action();

        expect(result.success).toBe(false);
      });
    });
  });

  describe('Chart Type Edge Cases', () => {
    it('should handle all supported chart types', async () => {
      const chartTypes = ['line', 'bar', 'pie', 'doughnut', 'radar', 'polarArea', 'scatter', 'bubble'] as const;

      for (const chartType of chartTypes) {
        const tool = new ChartJSTool({
          data: [{ x: 1, y: 2 }],
          chartType,
          reasoning: `Testing ${chartType} chart type`,
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
        expect(result.data?.chartType).toBe(chartType);
      }
    });

    it('should handle invalid chart type', async () => {
      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: 'invalid_type' as any,
        reasoning: 'Testing invalid chart type',
      });

      const result = await tool.action();

      expect(result.success).toBe(false);
    });

    it('should handle time series data with dates', async () => {
      const tool = new ChartJSTool({
        data: [
          { date: '2024-01-01', value: 10 },
          { date: '2024-02-01', value: 20 },
          { date: '2024-03-01', value: 30 },
        ],
        chartType: 'line',
        xColumn: 'date',
        yColumn: 'value',
        reasoning: 'Testing date data',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('Color and Styling Edge Cases', () => {
    it('should handle all named color schemes', async () => {
      const colorSchemes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'rainbow'];

      for (const scheme of colorSchemes) {
        const tool = new ChartJSTool({
          data: [
            { x: 1, y: 2 },
            { x: 2, y: 4 },
          ],
          chartType: 'bar',
          options: {
            colorScheme: scheme,
          },
          reasoning: `Testing ${scheme} color scheme`,
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
        expect(result.data?.metadata?.colorScheme).toBe(scheme);
      }
    });

    it('should handle custom color arrays', async () => {
      const tool = new ChartJSTool({
        data: [
          { category: 'A', value: 10 },
          { category: 'B', value: 20 },
          { category: 'C', value: 30 },
        ],
        chartType: 'pie',
        xColumn: 'category',
        yColumn: 'value',
        advancedConfig: {
          data: {
            datasets: [
              {
                data: [10, 20, 30],
                backgroundColor: ['#FF0000', '#00FF00', '#0000FF'],
              },
            ],
          },
        },
        reasoning: 'Testing custom colors',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle transparent colors', async () => {
      const tool = new ChartJSTool({
        data: [
          { x: 1, y: 2 },
          { x: 2, y: 4 },
        ],
        chartType: 'bar',
        advancedConfig: {
          data: {
            datasets: [
              {
                data: [2, 4],
                backgroundColor: 'rgba(255, 0, 0, 0.5)',
              },
            ],
          },
        },
        reasoning: 'Testing transparent colors',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle invalid color values', async () => {
      const tool = new ChartJSTool({
        data: [
          { x: 1, y: 2 },
        ],
        chartType: 'bar',
        advancedConfig: {
          data: {
            datasets: [
              {
                data: [2],
                backgroundColor: 'not-a-color',
              },
            ],
          },
        },
        reasoning: 'Testing invalid color',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('Grouping and Aggregation Edge Cases', () => {
    it('should handle single group', async () => {
      const tool = new ChartJSTool({
        data: [
          { x: 1, y: 2, group: 'A' },
          { x: 2, y: 4, group: 'A' },
        ],
        chartType: 'bar',
        xColumn: 'x',
        yColumn: 'y',
        groupByColumn: 'group',
        reasoning: 'Testing single group',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
      expect(result.data?.datasetCount).toBe(1);
    });

    it('should handle many groups (100)', async () => {
      const data = Array.from({ length: 100 }, (_, i) => ({
        x: i,
        y: i * 2,
        group: `Group ${i}`,
      }));

      const tool = new ChartJSTool({
        data,
        chartType: 'bar',
        xColumn: 'x',
        yColumn: 'y',
        groupByColumn: 'group',
        reasoning: 'Testing many groups',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
      expect(result.data?.datasetCount).toBe(100);
    });

    it('should handle groups with single item', async () => {
      const tool = new ChartJSTool({
        data: [
          { x: 1, y: 2, group: 'A' },
          { x: 2, y: 4, group: 'B' },
          { x: 3, y: 6, group: 'B' },
        ],
        chartType: 'bar',
        xColumn: 'x',
        yColumn: 'y',
        groupByColumn: 'group',
        reasoning: 'Testing single-item group',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
      expect(result.data?.datasetCount).toBe(2);
    });

    it('should handle groups with same name (case insensitive)', async () => {
      const tool = new ChartJSTool({
        data: [
          { x: 1, y: 2, group: 'group' },
          { x: 2, y: 4, group: 'Group' },
          { x: 3, y: 6, group: 'GROUP' },
        ],
        chartType: 'bar',
        xColumn: 'x',
        yColumn: 'y',
        groupByColumn: 'group',
        reasoning: 'Testing case-insensitive groups',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('Performance Edge Cases', () => {
    it('should handle dataset with 10000 points', async () => {
      const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
        x: i,
        y: Math.sin(i / 100) * 100,
      }));

      const tool = new ChartJSTool({
        data: largeDataset,
        chartType: 'line',
        reasoning: 'Testing performance with large dataset',
      });

      const startTime = Date.now();
      const result = await tool.action();
      const duration = Date.now() - startTime;

      expect(result.success).toBe(true);
      expect(duration).toBeLessThan(5000); // Should complete in under 5 seconds
    });

    it('should handle 100 datasets', async () => {
      const data = [];
      for (let i = 0; i < 100; i++) {
        data.push({ x: i, y: i, series: `Series ${i}` });
      }

      const tool = new ChartJSTool({
        data,
        chartType: 'line',
        xColumn: 'x',
        yColumn: 'y',
        groupByColumn: 'series',
        reasoning: 'Testing many datasets',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
      expect(result.data?.datasetCount).toBe(100);
    });

    it('should handle complex nested data structures', async () => {
      const tool = new ChartJSTool({
        data: [
          {
            nested: { value: { deep: { x: 1, y: 2 } } },
          },
          {
            nested: { value: { deep: { x: 2, y: 4 } } },
          },
        ],
        chartType: 'scatter',
        reasoning: 'Testing nested data',
      });

      const result = await tool.action();

      expect(result.success).toBe(false);
    });
  });

  describe('Options and Configuration Edge Cases', () => {
    it('should handle empty options object', async () => {
      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: 'scatter',
        options: {},
        reasoning: 'Testing empty options',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle all boolean option combinations', async () => {
      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: 'scatter',
        options: {
          showLegend: true,
        },
        reasoning: 'Testing boolean options',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle extremely long titles', async () => {
      const longTitle = 'x'.repeat(1000);

      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: 'scatter',
        options: {
          title: longTitle,
        },
        reasoning: 'Testing long title',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle unicode in titles and labels', async () => {
      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: 'scatter',
        options: {
          title: '标题 Título 📊',
          xAxisLabel: 'محور X',
          yAxisLabel: 'Ось Y',
        },
        reasoning: 'Testing unicode in options',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle advanced config override', async () => {
      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: 'scatter',
        advancedConfig: {
          type: 'scatter',
          data: {
            datasets: [
              {
                label: 'Custom Dataset',
                data: [{ x: 1, y: 2 }],
                backgroundColor: 'red',
                borderColor: 'blue',
                borderWidth: 5,
                pointRadius: 10,
                pointHoverRadius: 15,
              },
            ],
          },
          options: {
            responsive: false,
            animation: false,
            plugins: {
              legend: {
                display: true,
                position: 'bottom',
              },
            },
            scales: {
              x: {
                display: true,
                title: {
                  display: true,
                  text: 'Custom X Axis',
                },
              },
              y: {
                display: true,
                title: {
                  display: true,
                  text: 'Custom Y Axis',
                },
              },
            },
          },
        },
        reasoning: 'Testing advanced config',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });

    it('should handle invalid advanced config', async () => {
      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: 'scatter',
        advancedConfig: {
          type: 'scatter',
          data: 'invalid data',
        } as any,
        reasoning: 'Testing invalid advanced config',
      });

      const result = await tool.action();

      expect(result.success).toBe(true);
    });
  });

  describe('Size Suggestion Edge Cases', () => {
    it('should suggest appropriate sizes for all chart types', async () => {
      const chartTypes = [
        { type: 'line' as const, width: 400, height: 300 },
        { type: 'bar' as const, width: 400, height: 300 },
        { type: 'pie' as const, width: 400, height: 400 },
        { type: 'doughnut' as const, width: 400, height: 400 },
        { type: 'radar' as const, width: 450, height: 450 },
        { type: 'polarArea' as const, width: 400, height: 400 },
        { type: 'scatter' as const, width: 400, height: 300 },
        { type: 'bubble' as const, width: 400, height: 300 },
      ];

      for (const { type, width, height } of chartTypes) {
        const tool = new ChartJSTool({
          data: [{ x: 1, y: 2 }],
          chartType: type,
          reasoning: `Testing size for ${type}`,
        });

        const result = await tool.action();

        expect(result.success).toBe(true);
        expect(result.data?.suggestedSize?.width).toBe(width);
        expect(result.data?.suggestedSize?.height).toBe(height);
      }
    });
  });

  describe('Error Path Coverage', () => {
    it('should handle missing required field (data)', async () => {
      const tool = new ChartJSTool({
        data: null as any,
        chartType: 'line',
        reasoning: 'Testing missing data',
      });

      const result = await tool.action();

      expect(result.success).toBe(false);
    });

    it('should handle missing required field (chartType)', async () => {
      const tool = new ChartJSTool({
        data: [{ x: 1, y: 2 }],
        chartType: null as any,
        reasoning: 'Testing missing chartType',
      });

      const result = await tool.action();

      expect(result.success).toBe(false);
    });

    it('should handle non-array data', async () => {
      const tool = new ChartJSTool({
        data: 'not an array' as any,
        chartType: 'line',
        reasoning: 'Testing non-array data',
      });

      const result = await tool.action();

      expect(result.success).toBe(false);
    });

    it('should handle array of non-objects', async () => {
      const tool = new ChartJSTool({
        data: [1, 2, 3, 4, 5] as any,
        chartType: 'line',
        reasoning: 'Testing array of non-objects',
      });

      const result = await tool.action();

      expect(result.success).toBe(false);
    });
  });
});
