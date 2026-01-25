import { ReactNode } from 'react';
import { useForm, UseFormReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import { toast } from 'react-toastify';

interface FormWrapperProps<T extends z.ZodType> {
  schema: T;
  onSubmit: (data: z.infer<T>) => void | Promise<void>;
  children: (methods: UseFormReturn<z.infer<T>>) => ReactNode;
  defaultValues?: z.infer<T>;
  className?: string;
}

function FormWrapperBase<T extends z.ZodType>({
  schema,
  onSubmit,
  children,
  defaultValues,
  className,
}: FormWrapperProps<T>) {
  const methods = useForm<z.infer<T>>({
    resolver: zodResolver(schema),
    defaultValues,
  });

  const handleSubmit = async (data: z.infer<T>) => {
    try {
      // Use graceful error handling for form submission
      const result = await gracefulErrorHandler.executeWithErrorHandling(
        async () => {
          return await Promise.resolve(onSubmit(data));
        },
        {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 1000,
          showUserNotification: true,
          logError: true,
          context: {
            component: 'FormWrapper',
            function: 'handleSubmit',
            operation: 'FORM_SUBMISSION',
            additionalData: { formSchema: schema._def.typeName }
          }
        }
      );

      if (!result.success) {
        toast.error(`Form submission failed: ${result.error?.message || 'Unknown error'}`);
      }
    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Form submission error' } });
      toast.error('An error occurred while submitting the form');
    }
  };

  return (
    <form onSubmit={methods.handleSubmit(handleSubmit)} className={className}>
      {children(methods)}
    </form>
  );
}

export const FormWrapper = withComponentBoundary(
  FormWrapperBase,
  'FormWrapper'
) as typeof FormWrapperBase;
