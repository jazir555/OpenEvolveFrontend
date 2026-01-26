/**
 * Execution Progress Bar Component
 * Displays workflow execution progress
 */

interface ExecutionProgressBarProps {
  progress: number;
  currentStep: string;
}

export function ExecutionProgressBar({ progress, currentStep }: ExecutionProgressBarProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Progress
        </span>
        <span className="text-sm font-semibold text-gray-900 dark:text-white">
          {Math.round(progress)}%
        </span>
      </div>

      {/* Progress Bar */}
      <div className="relative h-4 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div
          className="h-full bg-blue-600 transition-all duration-300 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Current Step */}
      {currentStep && (
        <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
          <span className="font-medium">Current step:</span> {currentStep}
        </div>
      )}
    </div>
  );
}
