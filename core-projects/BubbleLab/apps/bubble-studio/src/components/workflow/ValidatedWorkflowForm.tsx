/**
 * ValidatedWorkflowForm Component
 * Workflow creation form with comprehensive validation
 */

import { useState } from 'react';
import { useNavigate } from '@tanstack/react-router';
import { useForm, useFormFields } from '../../hooks/useForm';
import { validateWorkflowName, validateProblemStatement } from '../../utils/validation';

// Wrapper functions to convert ValidationResult to string | null
const validateWorkflowNameSimple = (name: string): string | null => {
  const result = validateWorkflowName(name);
  return result.isValid ? null : result.errors.name || null;
};

const validateProblemStatementSimple = (statement: string): string | null => {
  const result = validateProblemStatement(statement);
  return result.isValid ? null : result.errors.problem_statement || null;
};
import { Button } from '../common/Button';
import { Input } from '../common/Input';
import { Textarea } from '../common/Textarea';
import { Select } from '../common/Select';
import { FormField } from '../common/FormField';
import { FormGroup } from '../common/FormGroup';
import { FormSuccess } from '../common/FormSuccess';
import { Alert } from '../common/Alert';
import { Stepper } from '../common/Stepper';
import { notify } from '../common/Notifications';

interface WorkflowFormData {
  name: string;
  description: string;
  problem_statement: string;
  content_type: string;
}

const CONTENT_TYPES = [
  { value: 'math', label: 'Mathematics' },
  { value: 'code', label: 'Coding' },
  { value: 'writing', label: 'Writing' },
  { value: 'research', label: 'Research' },
  { value: 'analysis', label: 'Data Analysis' },
];

type FormStep = 'basic' | 'problem' | 'teams' | 'review';

export function ValidatedWorkflowForm() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState<FormStep>('basic');
  const [submitSuccess, setSubmitSuccess] = useState(false);

  const { fields, updateField, setFieldError, validateField, getValues, isValid } =
    useFormFields<Record<string, string>>({
      name: '',
      description: '',
      problem_statement: '',
      content_type: 'math',
    } as Record<string, string>);

  const steps = [
    { id: 'basic', label: 'Basic Info', status: 'current' as const },
    { id: 'problem', label: 'Problem', status: 'pending' as const },
    { id: 'teams', label: 'Teams', status: 'pending' as const },
    { id: 'review', label: 'Review', status: 'pending' as const },
  ];

  const updateStepStatus = (step: FormStep) => {
    return steps.map((s) =>
      s.id === step
        ? { ...s, status: 'current' as const }
        : steps.findIndex((st) => st.id === step) > steps.findIndex((st) => st.id === step)
        ? { ...s, status: 'complete' as const }
        : { ...s, status: 'pending' as const }
    );
  };

  const handleNext = () => {
    // Validate current step before proceeding
    let valid = true;

    if (currentStep === 'basic') {
      if (!validateField('name', validateWorkflowNameSimple)) valid = false;
      if (fields.name.value.length < 3) {
        setFieldError('name', 'Workflow name must be at least 3 characters');
        valid = false;
      }
    }

    if (currentStep === 'problem') {
      if (!validateField('problem_statement', validateProblemStatementSimple)) valid = false;
    }

    if (!valid) {
      notify({
        type: 'error',
        title: 'Validation Error',
        message: 'Please fix the errors before proceeding',
      });
      return;
    }

    // Move to next step
    const stepOrder: FormStep[] = ['basic', 'problem', 'teams', 'review'];
    const currentIndex = stepOrder.indexOf(currentStep);
    if (currentIndex < stepOrder.length - 1) {
      setCurrentStep(stepOrder[currentIndex + 1]);
    }
  };

  const handleBack = () => {
    const stepOrder: FormStep[] = ['basic', 'problem', 'teams', 'review'];
    const currentIndex = stepOrder.indexOf(currentStep);
    if (currentIndex > 0) {
      setCurrentStep(stepOrder[currentIndex - 1]);
    }
  };

  const handleSubmit = async () => {
    // Final validation
    const nameValid = validateField('name', validateWorkflowNameSimple);
    const problemValid = validateField('problem_statement', validateProblemStatementSimple);

    if (!nameValid || !problemValid) {
      notify({
        type: 'error',
        title: 'Validation Error',
        message: 'Please fix all errors before submitting',
      });
      return;
    }

    // Simulate API call
    setSubmitSuccess(true);
    notify({
      type: 'success',
      title: 'Workflow Created',
      message: 'Your workflow has been created successfully',
    });

    // Navigate after delay
    setTimeout(() => {
      navigate({ to: '/oe-workflows' });
    }, 1500);
  };

  return (
    <div className="max-w-3xl mx-auto">
      <Stepper steps={updateStepStatus(currentStep)} className="mb-8" />

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        {submitSuccess ? (
          <FormSuccess
            title="Workflow Created Successfully!"
            message="You will be redirected to the workflows page shortly."
          />
        ) : (
          <>
            {/* Step 1: Basic Info */}
            {currentStep === 'basic' && (
              <FormGroup title="Basic Information" description="Enter the basic details for your workflow">
                <FormField
                  label="Workflow Name"
                  error={fields.name.error || undefined}
                  required
                  description="A descriptive name for your workflow"
                >
                  <Input
                    value={fields.name.value}
                    onChange={(e) => updateField('name', e.target.value)}
                    placeholder="e.g., Math Problem Solver"
                    onBlur={() => validateField('name', (value: string) => {
                      const result = validateWorkflowName(value);
                      return result.isValid ? null : result.errors.name || null;
                    })}
                  />
                </FormField>

                <FormField
                  label="Description"
                  description="A brief description of what this workflow does"
                >
                  <Textarea
                    value={fields.description.value}
                    onChange={(e) => updateField('description', e.target.value)}
                    placeholder="Describe the purpose of this workflow..."
                    rows={3}
                  />
                </FormField>

                <FormField
                  label="Content Type"
                  required
                  description="The type of content this workflow will process"
                >
                  <Select
                    value={fields.content_type.value}
                    onChange={(e) => updateField('content_type', (e.target as HTMLSelectElement).value)}
                    options={CONTENT_TYPES}
                  />
                </FormField>
              </FormGroup>
            )}

            {/* Step 2: Problem Statement */}
            {currentStep === 'problem' && (
              <FormGroup title="Problem Statement" description="Define the problem you want to solve">
                <FormField
                  label="Problem Statement"
                  error={fields.problem_statement.error || undefined}
                  required
                  description="Clearly describe the problem or question to be solved"
                >
                  <Textarea
                    value={fields.problem_statement.value}
                    onChange={(e) => updateField('problem_statement', e.target.value)}
                    placeholder="Enter your problem statement here..."
                    rows={8}
                    onBlur={() => validateField('problem_statement', validateProblemStatementSimple)}
                  />
                </FormField>

                <Alert variant="info" title="Tip">
                  A good problem statement is specific, clear, and well-defined. The more
                  detail you provide, the better the workflow will understand your needs.
                </Alert>
              </FormGroup>
            )}

            {/* Step 3: Teams */}
            {currentStep === 'teams' && (
              <FormGroup
                title="Select Teams"
                description="Choose which AI teams will work on this problem"
              >
                <div className="space-y-4">
                  <Alert variant="warning" title="Coming Soon">
                    Team selection will be available in the next update. For now, default
                    teams will be assigned.
                  </Alert>

                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    <p>Default teams that will be assigned:</p>
                    <ul className="list-disc list-inside mt-2 space-y-1">
                      <li>Content Analyzer - Understands the problem</li>
                      <li>Planner - Creates a solution plan</li>
                      <li>Solver - Executes the solution</li>
                      <li>Critic - Reviews and improves the result</li>
                    </ul>
                  </div>
                </div>
              </FormGroup>
            )}

            {/* Step 4: Review */}
            {currentStep === 'review' && (
              <FormGroup title="Review and Create" description="Review your workflow settings">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Workflow Details
                    </h4>
                    <dl className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <dt className="text-gray-500 dark:text-gray-400">Name</dt>
                        <dd className="text-gray-900 dark:text-white">{fields.name.value}</dd>
                      </div>
                      <div>
                        <dt className="text-gray-500 dark:text-gray-400">Content Type</dt>
                        <dd className="text-gray-900 dark:text-white">
                          {CONTENT_TYPES.find((t) => t.value === fields.content_type.value)
                            ?.label || fields.content_type.value}
                        </dd>
                      </div>
                      <div className="col-span-2">
                        <dt className="text-gray-500 dark:text-gray-400">Description</dt>
                        <dd className="text-gray-900 dark:text-white">
                          {fields.description.value || 'No description provided'}
                        </dd>
                      </div>
                      <div className="col-span-2">
                        <dt className="text-gray-500 dark:text-gray-400">Problem Statement</dt>
                        <dd className="text-gray-900 dark:text-white whitespace-pre-wrap">
                          {fields.problem_statement.value}
                        </dd>
                      </div>
                    </dl>
                  </div>
                </div>
              </FormGroup>
            )}

            {/* Actions */}
            <div className="flex justify-between mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
              <Button
                variant="secondary"
                onClick={handleBack}
                disabled={currentStep === 'basic'}
              >
                Back
              </Button>

              <div className="flex gap-3">
                <Button
                  variant="secondary"
                  onClick={() => navigate({ to: '/oe-workflows' })}
                >
                  Cancel
                </Button>

                {currentStep === 'review' ? (
                  <Button onClick={handleSubmit} disabled={!isValid}>
                    Create Workflow
                  </Button>
                ) : (
                  <Button onClick={handleNext}>Next Step</Button>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
