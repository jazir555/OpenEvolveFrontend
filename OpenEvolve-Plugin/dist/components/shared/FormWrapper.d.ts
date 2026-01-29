import { ReactNode } from 'react';
import { UseFormReturn } from 'react-hook-form';
import { z } from 'zod';
interface FormWrapperProps<T extends z.ZodType> {
    schema: T;
    onSubmit: (data: z.infer<T>) => void | Promise<void>;
    children: (methods: UseFormReturn<z.infer<T>>) => ReactNode;
    defaultValues?: z.infer<T>;
    className?: string;
}
export declare function FormWrapper<T extends z.ZodType>({ schema, onSubmit, children, defaultValues, className, }: FormWrapperProps<T>): import("react/jsx-runtime").JSX.Element;
export {};
