/**
 * MetricTable Component
 * Generic, type-safe table for rendering analytics rows.
 */

import type { ReactNode } from 'react';

export interface TableColumn<T> {
  key: string;
  header: string;
  render?: (row: T) => ReactNode;
  align?: 'left' | 'right' | 'center';
  className?: string;
}

interface MetricTableProps<T> {
  columns: TableColumn<T>[];
  data: T[];
  rowKey: (row: T, index: number) => string;
  emptyMessage?: string;
  maxHeight?: number;
  className?: string;
}

const alignClass: Record<'left' | 'right' | 'center', string> = {
  left: 'text-left',
  right: 'text-right',
  center: 'text-center',
};

export function MetricTable<T>({
  columns,
  data,
  rowKey,
  emptyMessage = 'No data available',
  maxHeight,
  className = '',
}: MetricTableProps<T>) {
  if (data.length === 0) {
    return (
      <div
        className={`rounded-lg border border-gray-200 bg-white p-6 text-center text-sm text-gray-500 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 ${className}`}
      >
        {emptyMessage}
      </div>
    );
  }

  return (
    <div
      className={`overflow-auto rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800 ${className}`}
      style={maxHeight ? { maxHeight: `${maxHeight}px` } : undefined}
    >
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead className="sticky top-0 bg-gray-50 dark:bg-gray-900">
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={`px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-600 dark:text-gray-400 ${alignClass[column.align ?? 'left']} ${column.className ?? ''}`}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
          {data.map((row, index) => (
            <tr
              key={rowKey(row, index)}
              className="hover:bg-gray-50 dark:hover:bg-gray-700/50"
            >
              {columns.map((column) => (
                <td
                  key={column.key}
                  className={`px-4 py-3 text-sm text-gray-900 dark:text-gray-100 ${alignClass[column.align ?? 'left']} ${column.className ?? ''}`}
                >
                  {column.render ? column.render(row) : (row as Record<string, ReactNode>)[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
