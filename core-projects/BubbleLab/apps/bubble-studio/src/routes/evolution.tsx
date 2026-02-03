import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { useRef, useState } from 'react';
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
  type ImperativePanelHandle,
} from 'react-resizable-panels';
import { EvolutionSettingsPage } from '@/pages/EvolutionSettingsPage';
import { EvolutionGraphView } from '@/components/evolution/EvolutionGraphView';
import { useAuth } from '@/hooks/useAuth';

export const Route = createFileRoute('/evolution')({
  component: EvolutionRoute,
});

function EvolutionRoute() {
  const navigate = useNavigate();
  const { isSignedIn } = useAuth();
  const panelRef = useRef<ImperativePanelHandle>(null);
  const [isCollapsed, setIsCollapsed] = useState(false);

  if (!isSignedIn) {
    navigate({ to: '/home', search: { showSignIn: true }, replace: true });
    return null;
  }

  const handleExpandPanel = () => {
    panelRef.current?.expand(30);
  };

  return (
    <div className="h-screen flex flex-col bg-[#1a1a1a] text-gray-100">
      <div className="flex-1 min-h-0">
        <PanelGroup direction="horizontal" autoSaveId="evolution-layout">
          <Panel
            ref={panelRef}
            defaultSize={35}
            minSize={20}
            maxSize={45}
            collapsible
            collapsedSize={0}
            onCollapse={() => setIsCollapsed(true)}
            onExpand={() => setIsCollapsed(false)}
            className="min-w-[320px]"
          >
            <EvolutionSettingsPage
              onCollapsePanel={() => panelRef.current?.collapse()}
            />
          </Panel>
          <PanelResizeHandle className="w-2 bg-[#30363d] hover:bg-white/70 transition-colors" />
          <Panel defaultSize={65} minSize={40} className="relative">
            <EvolutionGraphView />
            {isCollapsed && (
              <button
                type="button"
                onClick={handleExpandPanel}
                className="absolute top-4 left-4 z-10 px-3 py-2 rounded-lg bg-[#101826] border border-blue-500/40 text-blue-200 text-xs uppercase tracking-wide shadow-lg"
              >
                Show Settings
              </button>
            )}
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}
