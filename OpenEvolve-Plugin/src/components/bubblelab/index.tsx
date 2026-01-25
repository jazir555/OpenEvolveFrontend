import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

function BubbleCardBase({
  title,
  description,
  actions,
  children,
  className,
}: {
  title?: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn('rounded-xl border border-slate-200 bg-white shadow-sm', className)}>
      {(title || description || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4">
          <div>
            {title && <h3 className="text-sm font-semibold text-slate-900">{title}</h3>}
            {description && (
              <p className="mt-1 text-xs text-slate-500">{description}</p>
            )}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </header>
      )}
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}

function BubbleFieldBase({
  label,
  hint,
  children,
  className,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={cn('block space-y-2 text-sm text-slate-600', className)}>
      <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {label}
      </span>
      {children}
      {hint && <span className="block text-xs text-slate-400">{hint}</span>}
    </label>
  );
}

function BubbleInputBase({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        'w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900',
        'transition focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200',
        className
      )}
      {...props}
    />
  );
}

function BubbleTextAreaBase({
  className,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        'w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900',
        'transition focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200',
        className
      )}
      {...props}
    />
  );
}

function BubbleSelectBase({
  className,
  children,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        'w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900',
        'transition focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200',
        className
      )}
      {...props}
    >
      {children}
    </select>
  );
}

function BubbleButtonBase({
  className,
  variant = 'primary',
  onClick,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost';
}) {
  const styles = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-slate-100 text-slate-700 hover:bg-slate-200',
    ghost: 'text-slate-600 hover:bg-slate-100',
  }[variant];

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    try {
      onClick?.(e);
    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in button click handler' } });
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'BubbleButton', function: 'onClick', additionalData: { props } }
      );
    }
  };

  return (
    <button
      className={cn(
        'rounded-lg px-4 py-2 text-sm font-medium transition',
        'focus:outline-none focus:ring-2 focus:ring-blue-200',
        styles,
        className
      )}
      onClick={handleClick}
      {...props}
    />
  );
}

function BubbleBadgeBase({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode;
  tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'info';
  className?: string;
}) {
  const tones = {
    neutral: 'bg-slate-100 text-slate-700',
    success: 'bg-emerald-100 text-emerald-700',
    warning: 'bg-amber-100 text-amber-700',
    danger: 'bg-rose-100 text-rose-700',
    info: 'bg-blue-100 text-blue-700',
  }[tone];

  return (
    <span className={cn('inline-flex items-center rounded-full px-2.5 py-1 text-xs', tones, className)}>
      {children}
    </span>
  );
}

function BubbleToggleBase({
  checked,
  onChange,
  label,
  className,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  className?: string;
}) {
  const handleClick = () => {
    try {
      onChange(!checked);
    } catch (error) {
      errorLogger.logError(error, 'error', { component: 'Component', function: 'errorHandler', additionalData: { operation: 'Error in toggle change handler' } });
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { component: 'BubbleToggle', function: 'onChange', additionalData: { checked, newValue: !checked } }
      );
    }
  };

  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={handleClick}
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-2 py-1 text-xs font-medium',
        checked ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 bg-white text-slate-500',
        className
      )}
    >
      <span
        className={cn(
          'h-3 w-3 rounded-full transition',
          checked ? 'bg-blue-600' : 'bg-slate-300'
        )}
      />
      {label && <span>{label}</span>}
    </button>
  );
}

function BubbleCheckboxBase({
  label,
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & {
  label?: string;
}) {
  return (
    <label className={cn('inline-flex items-center gap-2 text-sm text-slate-600', className)}>
      <input
        type="checkbox"
        className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-200"
        {...props}
      />
      {label && <span>{label}</span>}
    </label>
  );
}

export const BubbleCard = withComponentBoundary(BubbleCardBase, 'BubbleCard');
export const BubbleField = withComponentBoundary(BubbleFieldBase, 'BubbleField');
export const BubbleInput = withComponentBoundary(BubbleInputBase, 'BubbleInput');
export const BubbleTextArea = withComponentBoundary(BubbleTextAreaBase, 'BubbleTextArea');
export const BubbleSelect = withComponentBoundary(BubbleSelectBase, 'BubbleSelect');
export const BubbleButton = withComponentBoundary(BubbleButtonBase, 'BubbleButton');
export const BubbleBadge = withComponentBoundary(BubbleBadgeBase, 'BubbleBadge');
export const BubbleToggle = withComponentBoundary(BubbleToggleBase, 'BubbleToggle');
export const BubbleCheckbox = withComponentBoundary(BubbleCheckboxBase, 'BubbleCheckbox');
