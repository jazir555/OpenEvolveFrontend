import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { BubbleButton, BubbleTextArea } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

interface ProofEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  readOnly?: boolean;
  className?: string;
}

function ProofEditorBase({
  value,
  onChange,
  language = 'lean',
  readOnly = false,
  className,
}: ProofEditorProps) {
  const [editorValue, setEditorValue] = useState(value);

  useEffect(() => {
    setEditorValue(value);
  }, [value]);

  const handleChange = (newValue: string) => {
    setEditorValue(newValue);
    onChange(newValue);
  };

  return (
    <div className={cn('proof-editor flex flex-col h-full', className)}>
      {/* Toolbar */}
      {!readOnly && (
        <div className="flex items-center gap-2 p-2 bg-gray-50 border-b border-gray-200">
          <span className="text-sm text-gray-600">Lean 4 Editor</span>
          <div className="flex-1" />
          <BubbleButton
            onClick={() => handleChange('')}
            variant="ghost"
            className="px-2 py-1"
          >
            Clear
          </BubbleButton>
          <BubbleButton
            onClick={() => {
              navigator.clipboard.readText().then((text) => handleChange(editorValue + text));
            }}
            variant="ghost"
            className="px-2 py-1"
          >
            Paste
          </BubbleButton>
        </div>
      )}

      {/* Editor Area */}
      <div className="flex-1 relative">
        <BubbleTextArea
          value={editorValue}
          onChange={(e) => handleChange(e.target.value)}
          readOnly={readOnly}
          className={cn(
            'w-full h-full p-4 font-mono text-sm resize-none',
            'bg-gray-50 border-0 focus:ring-0',
            readOnly && 'bg-gray-100'
          )}
          placeholder="-- Enter Lean 4 code here..."
          spellCheck={false}
        />
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-600">
        <div>
          Lines: {editorValue.split('\n').length} | Characters: {editorValue.length}
        </div>
        <div>{language}</div>
      </div>
    </div>
  );
}

export const ProofEditor = withComponentBoundary(ProofEditorBase, 'ProofEditor');
