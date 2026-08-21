import React from 'react';
import { Button } from '@/components/ui/button';
import { 
  Home, 
  Wrench, 
  GitBranch, 
  Activity, 
  FileText, 
  BarChart3, 
  Checklist, 
  Shield, 
  Users, 
  Database,
  Settings,
  Play,
  Square,
  RotateCcw,
  Workflow,
  Search,
  Sparkles,
  Cpu,
  Boxes,
  Coins,
  ClipboardCheck,
  LineChart,
  Bot
} from 'lucide-react';

export interface SidebarProps {
  activeTab: string;
  onNavigate: (tabId: string) => void;
}

const NAV_ITEMS: { id: string; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'openevolve', label: 'Dashboard', icon: Home },
  { id: 'evolution', label: 'Evolution Engine', icon: Wrench },
  { id: 'adversarial', label: 'Adversarial Testing', icon: Shield },
  { id: 'github', label: 'GitHub Integration', icon: GitBranch },
  { id: 'activity', label: 'Activity Feed', icon: Activity },
  { id: 'reports', label: 'Report Templates', icon: FileText },
  { id: 'models', label: 'Model Dashboard', icon: BarChart3 },
  { id: 'tasks', label: 'Tasks', icon: Checklist },
  { id: 'admin', label: 'Admin Panel', icon: Users },
  { id: 'analytics', label: 'Analytics Dashboard', icon: Database },
  { id: 'orchestrator', label: 'Orchestrator', icon: Settings },
  { id: 'workflow-visual-editor', label: 'Workflow Visual Editor', icon: Workflow },
  { id: 'ragbits', label: 'RAGBits', icon: Search },
  { id: 'dspy-graphistry', label: 'DSPy & Graphistry', icon: Sparkles },
  { id: 'determinism', label: 'Determinism', icon: Cpu },
  { id: 'bubblelabs-integrations', label: 'BubbleLabs Integrations', icon: Boxes },
  { id: 'web3', label: 'Web3', icon: Coins },
  { id: 'research-approval', label: 'Research Approval', icon: ClipboardCheck },
  { id: 'icr-dashboard', label: 'ICR Dashboard', icon: LineChart },
  { id: 'crewai', label: 'CrewAI', icon: Bot },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onNavigate }) => {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-900">OpenEvolve</h2>
        <p className="text-sm text-gray-600">AI Content Evolution Platform</p>
      </div>
      
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = activeTab === id;
          return (
            <Button
              key={id}
              variant={isActive ? 'secondary' : 'ghost'}
              className={`w-full justify-start ${isActive ? 'font-semibold' : ''}`}
              onClick={() => onNavigate(id)}
            >
              <Icon className="mr-2 h-4 w-4" />
              {label}
            </Button>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-gray-200 space-y-2">
        <Button variant="outline" className="w-full">
          <Play className="mr-2 h-4 w-4" />
          Start Services
        </Button>
        <Button variant="outline" className="w-full">
          <Square className="mr-2 h-4 w-4" />
          Stop Services
        </Button>
        <Button variant="outline" className="w-full">
          <RotateCcw className="mr-2 h-4 w-4" />
          Restart Services
        </Button>
      </div>
    </aside>
  );
};
