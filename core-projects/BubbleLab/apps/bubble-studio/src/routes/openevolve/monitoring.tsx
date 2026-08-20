import { createFileRoute } from '@tanstack/react-router';
import { MonitoringView } from '@/components/monitoring';

export const Route = createFileRoute('/openevolve/monitoring')({
  component: OpenEvolveMonitoringPage,
});

function OpenEvolveMonitoringPage() {
  return (
    <div className="p-6">
      <MonitoringView />
    </div>
  );
}
