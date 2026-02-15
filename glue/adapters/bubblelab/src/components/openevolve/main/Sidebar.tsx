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
  RotateCcw
} from 'lucide-react';

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-900">OpenEvolve</h2>
        <p className="text-sm text-gray-600">AI Content Evolution Platform</p>
      </div>
      
      <nav className="flex-1 p-4 space-y-2">
        <Button variant="ghost" className="w-full justify-start">
          <Home className="mr-2 h-4 w-4" />
          Dashboard
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <Wrench className="mr-2 h-4 w-4" />
          Evolution Engine
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <Shield className="mr-2 h-4 w-4" />
          Adversarial Testing
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <GitBranch className="mr-2 h-4 w-4" />
          GitHub Integration
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <Activity className="mr-2 h-4 w-4" />
          Activity Feed
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <FileText className="mr-2 h-4 w-4" />
          Report Templates
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <BarChart3 className="mr-2 h-4 w-4" />
          Model Dashboard
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <Checklist className="mr-2 h-4 w-4" />
          Tasks
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <Users className="mr-2 h-4 w-4" />
          Admin Panel
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <Database className="mr-2 h-4 w-4" />
          Analytics Dashboard
        </Button>
        
        <Button variant="ghost" className="w-full justify-start">
          <Settings className="mr-2 h-4 w-4" />
          Orchestrator
        </Button>
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