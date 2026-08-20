import { createFileRoute } from '@tanstack/react-router';
import { TeamManager } from '@/components/teams';

export const Route = createFileRoute('/openevolve/teams')({
  component: OpenEvolveTeamsPage,
});

function OpenEvolveTeamsPage() {
  return (
    <div className="p-6">
      <TeamManager />
    </div>
  );
}
