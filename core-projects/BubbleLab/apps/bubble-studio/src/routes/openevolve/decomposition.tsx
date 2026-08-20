import { createFileRoute } from '@tanstack/react-router';
import { DecompositionPanel } from '@/components/decomposition';

export const Route = createFileRoute('/openevolve/decomposition')({
  component: OpenEvolveDecompositionPage,
});

function OpenEvolveDecompositionPage() {
  return (
    <div className="p-6">
      <DecompositionPanel />
    </div>
  );
}
