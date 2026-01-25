import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { BubbleBadge, BubbleButton, BubbleField, BubbleInput, BubbleSelect, BubbleTextArea } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface ArtifactData {
  id?: string;
  name: string;
  type: string;
  description: string;
  content: string;
  tags: string[];
  metadata?: Record<string, any>;
}

interface ArtifactEditorProps {
  artifact?: ArtifactData;
  onSave: (artifact: ArtifactData) => Promise<void>;
  onCancel?: () => void;
  types: string[];
  className?: string;
}

function ArtifactEditorBase({
  artifact,
  onSave,
  onCancel,
  types,
  className,
}: ArtifactEditorProps) {
  const buildEmptyForm = (): ArtifactData => ({
    name: '',
    type: types[0] || '',
    description: '',
    content: '',
    tags: [],
  });
  const [formData, setFormData] = useState<ArtifactData>(
    artifact || buildEmptyForm()
  );
  const [tagInput, setTagInput] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (artifact) {
      setFormData({
        ...buildEmptyForm(),
        ...artifact,
        tags: artifact.tags || [],
      });
      setTagInput('');
    } else {
      setFormData(buildEmptyForm());
      setTagInput('');
    }
  }, [artifact, types]);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
      setFormData((prev) => ({
        ...prev,
        tags: [...prev.tags, tagInput.trim()],
      }));
      setTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setFormData((prev) => ({
      ...prev,
      tags: prev.tags.filter((tag) => tag !== tagToRemove),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    setError(null);

    try {
      await onSave(formData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save artifact');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={cn('artifact-editor space-y-4', className)}>
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      <BubbleField label="Name *">
        <BubbleInput
          type="text"
          name="name"
          value={formData.name}
          onChange={handleChange}
          required
        />
      </BubbleField>

      <BubbleField label="Type *">
        <BubbleSelect
          name="type"
          value={formData.type}
          onChange={handleChange}
          required
        >
          {types.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </BubbleSelect>
      </BubbleField>

      <BubbleField label="Description">
        <BubbleTextArea
          name="description"
          value={formData.description}
          onChange={handleChange}
          rows={3}
        />
      </BubbleField>

      <BubbleField label="Content *">
        <BubbleTextArea
          name="content"
          value={formData.content}
          onChange={handleChange}
          rows={10}
          required
          className="font-mono text-sm"
        />
      </BubbleField>

      <BubbleField label="Tags">
        <div className="flex gap-2 mb-2">
          <BubbleInput
            type="text"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddTag())}
            placeholder="Add a tag..."
            className="flex-1"
          />
          <BubbleButton type="button" onClick={handleAddTag}>
            Add
          </BubbleButton>
        </div>
        <div className="flex flex-wrap gap-2">
          {formData.tags.map((tag) => (
            <div key={tag} className="inline-flex items-center gap-2">
              <BubbleBadge tone="neutral">{tag}</BubbleBadge>
              <BubbleButton
                type="button"
                onClick={() => handleRemoveTag(tag)}
                variant="ghost"
                className="px-2 py-1"
              >
                Remove
              </BubbleButton>
            </div>
          ))}
        </div>
      </BubbleField>

      <div className="flex gap-2 pt-4">
        <BubbleButton type="submit" disabled={isSaving} className="flex-1">
          {isSaving ? 'Saving...' : 'Save'}
        </BubbleButton>
        {onCancel && (
          <BubbleButton type="button" onClick={onCancel} variant="secondary">
            Cancel
          </BubbleButton>
        )}
      </div>
    </form>
  );
}

export const ArtifactEditor = withComponentBoundary(ArtifactEditorBase, 'ArtifactEditor');
