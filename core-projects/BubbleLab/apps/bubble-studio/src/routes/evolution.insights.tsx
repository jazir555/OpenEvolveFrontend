import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { EvolutionInsightsPage } from '@/pages/EvolutionInsightsPage';
import { useAuth } from '@/hooks/useAuth';

export const Route = createFileRoute('/evolution/insights')({
  component: EvolutionInsightsRoute,
});

function EvolutionInsightsRoute() {
  const navigate = useNavigate();
  const { isSignedIn } = useAuth();

  if (!isSignedIn) {
    navigate({ to: '/home', search: { showSignIn: true }, replace: true });
    return null;
  }

  return (
    <div className="h-screen flex flex-col bg-[#1a1a1a] text-gray-100">
      <div className="flex-1 min-h-0">
        <EvolutionInsightsPage />
      </div>
    </div>
  );
}
