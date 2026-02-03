/**
 * File Uploader Component
 * Handles file uploads for workflows (PDF, images, etc.)
 */

import { useState, useRef } from 'react';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  acceptedTypes?: string[]; // Alias for accept (as array)
  maxSize?: number; // in bytes
  maxSizeMB?: number; // Alias for maxSize (in MB)
  disabled?: boolean;
  label?: string;
  description?: string;
}

export function FileUploader({
  onFileSelect,
  accept = '.pdf,.png,.jpg,.jpeg,.txt',
  acceptedTypes,
  maxSize,
  maxSizeMB,
  disabled = false,
  label,
  description,
}: FileUploaderProps) {
  // Convert acceptedTypes array to accept string if provided
  const acceptString = acceptedTypes ? acceptedTypes.join(',') : accept;
  // Use maxSizeMB if provided, otherwise use maxSize
  const maxFileSize = maxSizeMB !== undefined ? maxSizeMB * 1024 * 1024 : (maxSize ?? 10 * 1024 * 1024);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      validateAndSelectFile(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      validateAndSelectFile(files[0]);
    }
  };

  const validateAndSelectFile = (file: File) => {
    setError(null);

    // Check file size
    if (file.size > maxFileSize) {
      setError(`File size exceeds ${Math.round(maxFileSize / 1024 / 1024)}MB limit`);
      return;
    }

    // Check file type
    const fileExtension = `.${file.name.split('.').pop()}`;
    const acceptedExtensions = acceptString.split(',');
    if (!acceptedExtensions.includes(fileExtension)) {
      setError(`File type not accepted. Accepted: ${acceptString}`);
      return;
    }

    setSelectedFile(file);
    onFileSelect(file);
  };

  const handleClick = () => {
    if (!disabled && inputRef.current) {
      inputRef.current.click();
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
        </label>
      )}
      {description && (
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {description}
        </p>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={acceptString}
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />

      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed
          ${isDragging ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-300'}
          ${disabled ? 'cursor-not-allowed opacity-50' : 'hover:border-gray-400'}
          p-12 transition-colors dark:border-gray-600
        `}
      >
        <svg
          className="h-12 w-12 text-gray-400"
          stroke="currentColor"
          fill="none"
          viewBox="0 0 48 48"
        >
          <path
            d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
          {selectedFile ? (
            <span className="font-medium text-gray-900 dark:text-white">
              {selectedFile.name}
            </span>
          ) : (
            <>
              <span className="font-semibold text-blue-600 dark:text-blue-400">
                Click to upload
              </span>{' '}
              or drag and drop
            </>
          )}
        </p>

        {selectedFile && (
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {formatFileSize(selectedFile.size)}
          </p>
        )}

        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {acceptString.toUpperCase()} up to {formatFileSize(maxFileSize)}
        </p>
      </div>

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}

      {selectedFile && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            setSelectedFile(null);
            setError(null);
            if (inputRef.current) {
              inputRef.current.value = '';
            }
          }}
          className="text-sm text-red-600 hover:text-red-700 dark:text-red-400"
        >
          Remove file
        </button>
      )}
    </div>
  );
}
