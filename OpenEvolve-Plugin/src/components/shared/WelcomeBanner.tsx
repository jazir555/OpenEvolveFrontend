import { BubbleButton, BubbleCard } from '@/components/bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

function WelcomeBannerBase({
  userName,
  workflowCount,
  onCreateWorkflow,
}: {
  userName?: string;
  workflowCount?: number;
  onCreateWorkflow?: () => void;
}) {
  return (
    <BubbleCard
      title="OpenEvolve Command Center"
      description="Run evolution, track system health, and coordinate MDAP/MAKER workflows."
      actions={
        onCreateWorkflow ? (
          <BubbleButton onClick={onCreateWorkflow} variant="primary">
            New Workflow
          </BubbleButton>
        ) : undefined
      }
    >
      <div className="flex flex-col gap-2 text-sm text-slate-600">
        <p>Welcome back, {userName || 'Operator'}.</p>
        <p>{workflowCount ?? 0} workflows are currently in your workspace.</p>
      </div>
    </BubbleCard>
  );
}

export const WelcomeBanner = withComponentBoundary(WelcomeBannerBase, 'WelcomeBanner');
