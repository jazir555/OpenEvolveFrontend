/**
 * DatasetUploader Component
 * Upload and parse benchmark datasets
 */

import { useState, useCallback } from 'react';
import { FileUploader } from '../file-handling/FileUploader';
import { Alert } from '../common/Alert';

interface DatasetRecord {
  [key: string]: string | number | boolean;
}

interface ParsedDataset {
  records: DatasetRecord[];
  fields: string[];
  recordCount: number;
}

export function DatasetUploader({
  onDataParsed,
  onError,
}: {
  onDataParsed: (dataset: ParsedDataset) => void;
  onError?: (error: string) => void;
}) {
  const [isParsing, setIsParsing] = useState(false);
  const [parsedData, setParsedData] = useState<ParsedDataset | null>(null);

  const handleFileUpload = useCallback(
    async (file: File) => {
      setIsParsing(true);

      try {
        const text = await file.text();
        let records: DatasetRecord[] = [];

        // Parse based on file extension
        if (file.name.endsWith('.json')) {
          records = parseJSON(text);
        } else if (file.name.endsWith('.csv')) {
          records = parseCSV(text);
        } else {
          throw new Error('Unsupported file format. Please use JSON or CSV.');
        }

        if (records.length === 0) {
          throw new Error('No records found in file');
        }

        const fields = Object.keys(records[0]);
        const dataset: ParsedDataset = {
          records,
          fields,
          recordCount: records.length,
        };

        setParsedData(dataset);
        onDataParsed(dataset);
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to parse file';
        onError?.(errorMessage);
      } finally {
        setIsParsing(false);
      }
    },
    [onDataParsed, onError]
  );

  return (
    <div className="space-y-4">
      <FileUploader
        onFileSelect={handleFileUpload}
        acceptedTypes={['.json', '.csv']}
        maxSizeMB={10}
        label="Upload Dataset"
        description="Upload a JSON or CSV file with test cases"
      />

      {isParsing && (
        <Alert variant="info" title="Parsing file...">
          Please wait while we process your dataset.
        </Alert>
      )}

      {parsedData && (
        <Alert variant="success" title="Dataset loaded successfully">
          <div className="mt-2 text-sm">
            <p>
              <strong>Records:</strong> {parsedData.recordCount}
            </p>
            <p>
              <strong>Fields:</strong> {parsedData.fields.join(', ')}
            </p>
            <p className="mt-2">
              <strong>Sample record:</strong>
            </p>
            <pre className="mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded text-xs overflow-auto max-h-32">
              {JSON.stringify(parsedData.records[0], null, 2)}
            </pre>
          </div>
        </Alert>
      )}
    </div>
  );
}

function parseJSON(text: string): DatasetRecord[] {
  try {
    const data = JSON.parse(text);
    if (Array.isArray(data)) {
      return data;
    }
    if (data.records && Array.isArray(data.records)) {
      return data.records;
    }
    throw new Error('Invalid JSON format. Expected an array of records.');
  } catch (error) {
    throw new Error('Failed to parse JSON: ' + (error as Error).message);
  }
}

function parseCSV(text: string): DatasetRecord[] {
  const lines = text.trim().split('\n');
  if (lines.length < 2) {
    throw new Error('CSV must have at least a header and one data row');
  }

  const headers = parseCSVLine(lines[0]);
  const records: DatasetRecord[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    const record: DatasetRecord = {};

    headers.forEach((header, index) => {
      record[header] = values[index] || '';
    });

    records.push(record);
  }

  return records;
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}
