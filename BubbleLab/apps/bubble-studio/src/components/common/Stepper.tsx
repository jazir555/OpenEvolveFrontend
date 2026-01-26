/**
 * Stepper Component
 * Multi-step progress indicator
 */

interface Step {
  id: string;
  label: string;
  description?: string;
  status?: 'complete' | 'current' | 'pending' | 'error';
}

interface StepperProps {
  steps: Step[];
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

export function Stepper({
  steps,
  orientation = 'horizontal',
  className = '',
}: StepperProps) {
  if (orientation === 'vertical') {
    return (
      <div className={`space-y-4 ${className}`}>
        {steps.map((step, index) => (
          <div key={step.id} className="flex">
            <div className="flex flex-col items-center mr-4">
              <div
                className={`
                  w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                  ${getStatusStyles(step.status)}
                `}
              >
                {getStepIcon(step.status, index + 1)}
              </div>
              {index < steps.length - 1 && (
                <div className="w-0.5 h-full bg-gray-300 dark:bg-gray-700 mt-2" />
              )}
            </div>
            <div className="flex-1 pb-8">
              <p
                className={`text-sm font-medium ${getLabelStyles(step.status)}`}
              >
                {step.label}
              </p>
              {step.description && (
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  {step.description}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={`flex items-center justify-between ${className}`}>
      {steps.map((step, index) => (
        <div key={step.id} className="flex items-center flex-1">
          <div className="flex flex-col items-center flex-1">
            <div
              className={`
                w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                ${getStatusStyles(step.status)}
              `}
            >
              {getStepIcon(step.status, index + 1)}
            </div>
            <p
              className={`text-xs font-medium mt-2 text-center ${getLabelStyles(
                step.status
              )}`}
            >
              {step.label}
            </p>
          </div>
          {index < steps.length - 1 && (
            <div
              className={`flex-1 h-0.5 mx-2 ${
                step.status === 'complete'
                  ? 'bg-blue-600'
                  : 'bg-gray-300 dark:bg-gray-700'
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

function getStatusStyles(
  status: 'complete' | 'current' | 'pending' | 'error' | undefined
): string {
  switch (status) {
    case 'complete':
      return 'bg-blue-600 text-white';
    case 'current':
      return 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400 ring-4 ring-blue-100 dark:ring-blue-900/30';
    case 'error':
      return 'bg-red-600 text-white';
    default:
      return 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400';
  }
}

function getLabelStyles(
  status: 'complete' | 'current' | 'pending' | 'error' | undefined
): string {
  switch (status) {
    case 'complete':
      return 'text-blue-600 dark:text-blue-400';
    case 'current':
      return 'text-blue-600 dark:text-blue-400 font-semibold';
    case 'error':
      return 'text-red-600 dark:text-red-400';
    default:
      return 'text-gray-500 dark:text-gray-400';
  }
}

function getStepIcon(
  status: 'complete' | 'current' | 'pending' | 'error' | undefined,
  stepNumber: number
): React.ReactNode {
  if (status === 'complete') {
    return (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M5 13l4 4L19 7"
        />
      </svg>
    );
  }

  if (status === 'error') {
    return (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M6 18L18 6M6 6l12 12"
        />
      </svg>
    );
  }

  return stepNumber;
}
