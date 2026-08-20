import { createFileRoute } from '@tanstack/react-router';
import { AnalyticsDashboard } from '@/components/analytics';

export const Route = createFileRoute('/openevolve/analytics')({
  component: OpenEvolveAnalyticsPage,
});

function OpenEvolveAnalyticsPage() {
  return (
    <div className="p-6">
      <AnalyticsDashboard />
    </div>
  );
}
