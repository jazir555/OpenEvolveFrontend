import AutoResizeTextarea from '@/components/AutoResizeTextarea';
import type { ParameterSchema, ParameterValue } from '@/types/evolution';

interface EvolutionParameterFormProps {
  parameters: ParameterSchema[];
  values: Record<string, ParameterValue>;
  onChange: (name: string, value: ParameterValue) => void;
  disabled?: boolean;
}

export function EvolutionParameterForm({
  parameters,
  values,
  onChange,
  disabled = false,
}: EvolutionParameterFormProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {parameters.map((parameter) => {
        const value = values[parameter.name];
        const isMissing =
          parameter.required &&
          (value === undefined || value === null || value === '');
        const isFullWidth =
          parameter.type === 'textarea' || parameter.multiline;

        return (
          <div
            key={parameter.name}
            className={`rounded-lg border px-4 py-3 bg-neutral-900/60 ${
              isFullWidth ? 'md:col-span-2' : ''
            } ${isMissing ? 'border-red-500/40' : 'border-neutral-800'}`}
          >
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-neutral-200">
                {parameter.label}
              </label>
              {parameter.required && (
                <span className="text-[10px] px-1.5 py-0.5 rounded border border-red-500/40 text-red-300 bg-red-500/10">
                  REQUIRED
                </span>
              )}
            </div>
            {parameter.description && (
              <p className="text-[10px] text-neutral-400 mt-1">
                {parameter.description}
              </p>
            )}
            <div className="mt-2">
              {parameter.type === 'textarea' ? (
                <AutoResizeTextarea
                  value={typeof value === 'string' ? value : ''}
                  onChange={(event) =>
                    onChange(parameter.name, event.target.value)
                  }
                  placeholder={parameter.placeholder}
                  disabled={disabled}
                  minHeight={96}
                  maxHeight={240}
                  className="w-full bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-100 placeholder-neutral-500 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              ) : parameter.type === 'text' ? (
                <input
                  type="text"
                  value={typeof value === 'string' ? value : ''}
                  onChange={(event) =>
                    onChange(parameter.name, event.target.value)
                  }
                  placeholder={parameter.placeholder}
                  disabled={disabled}
                  className="w-full bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-100 placeholder-neutral-500 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              ) : parameter.type === 'number' ? (
                <input
                  type="number"
                  value={typeof value === 'number' ? value : ''}
                  min={parameter.min}
                  max={parameter.max}
                  step={parameter.step}
                  onChange={(event) => {
                    const next =
                      event.target.value === ''
                        ? ''
                        : Number(event.target.value);
                    onChange(parameter.name, next);
                  }}
                  placeholder={parameter.placeholder}
                  disabled={disabled}
                  className="w-full bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-100 placeholder-neutral-500 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              ) : parameter.type === 'slider' ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-[11px] text-neutral-400">
                    <span>{parameter.min ?? 0}</span>
                    <span className="text-neutral-200 font-semibold">
                      {typeof value === 'number' ? value : parameter.min ?? 0}
                    </span>
                    <span>{parameter.max ?? 1}</span>
                  </div>
                  <input
                    type="range"
                    min={parameter.min}
                    max={parameter.max}
                    step={parameter.step}
                    value={
                      typeof value === 'number' ? value : parameter.min ?? 0
                    }
                    onChange={(event) =>
                      onChange(parameter.name, Number(event.target.value))
                    }
                    disabled={disabled}
                    className="w-full accent-blue-500"
                  />
                </div>
              ) : parameter.type === 'select' ? (
                <select
                  value={
                    typeof value === 'string'
                      ? value
                      : String(parameter.defaultValue ?? '')
                  }
                  onChange={(event) =>
                    onChange(parameter.name, event.target.value)
                  }
                  disabled={disabled}
                  className="w-full bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-100 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  {parameter.options?.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              ) : (
                <button
                  type="button"
                  role="switch"
                  aria-checked={Boolean(value)}
                  onClick={() =>
                    onChange(parameter.name, !Boolean(value))
                  }
                  disabled={disabled}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-neutral-900 ${
                    value ? 'bg-blue-500' : 'bg-neutral-700'
                  } ${disabled ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      value ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
