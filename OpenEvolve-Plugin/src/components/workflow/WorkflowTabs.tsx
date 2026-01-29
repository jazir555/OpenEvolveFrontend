import { useState } from 'react';
import { cn } from '@/lib/utils';
import { BubbleButton } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface WorkflowTab {
  id: string;
  label: string;
  icon?: string;
  content: React.ReactNode;
}

interface WorkflowTabsProps {
  tabs: WorkflowTab[];
  defaultTab?: string;
  className?: string;
}

function WorkflowTabsBase({ tabs, defaultTab, className }: WorkflowTabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

  const activeTabContent = tabs.find((tab) => tab.id === activeTab)?.content;
  const resolvedContent = activeTabContent ?? tabs[0]?.content;

  if (!tabs.length) {
    return (
      <div className={cn('workflow-tabs', className)}>
        <div className="p-6 text-sm text-gray-500">No tabs available.</div>
      </div>
    );
  }

  return (
    <div className={cn('workflow-tabs', className)}>
      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8 px-6" aria-label="Tabs">
          {tabs.map((tab) => (
            <BubbleButton
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              variant={activeTab === tab.id ? 'primary' : 'secondary'}
              className="px-3 py-2"
            >
              {tab.icon && <span className="mr-2">{tab.icon}</span>}
              {tab.label}
            </BubbleButton>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {resolvedContent}
      </div>
    </div>
  );
}

export const WorkflowTabs = withComponentBoundary(WorkflowTabsBase, 'WorkflowTabs');
