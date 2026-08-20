import { createFileRoute } from '@tanstack/react-router';
import { KnowledgeBase } from '@/components/knowledge';

export const Route = createFileRoute('/openevolve/knowledge')({
  component: OpenEvolveKnowledgePage,
});

function OpenEvolveKnowledgePage() {
  return (
    <div className="p-6">
      <KnowledgeBase />
    </div>
  );
}
