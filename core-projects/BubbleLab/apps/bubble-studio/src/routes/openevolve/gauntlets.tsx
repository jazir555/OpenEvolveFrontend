import { createFileRoute } from '@tanstack/react-router';
import { GauntletDesigner } from '@/components/gauntlets';

export const Route = createFileRoute('/openevolve/gauntlets')({
  component: OpenEvolveGauntletsPage,
});

function OpenEvolveGauntletsPage() {
  return (
    <div className="p-6">
      <GauntletDesigner />
    </div>
  );
}
