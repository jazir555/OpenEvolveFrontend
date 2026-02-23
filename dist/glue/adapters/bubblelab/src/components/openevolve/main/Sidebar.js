"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Sidebar = void 0;
const react_1 = __importDefault(require("react"));
const button_1 = require("@/components/ui/button");
const lucide_react_1 = require("lucide-react");
const Sidebar = () => {
    return (<aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-900">OpenEvolve</h2>
        <p className="text-sm text-gray-600">AI Content Evolution Platform</p>
      </div>
      
      <nav className="flex-1 p-4 space-y-2">
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Home className="mr-2 h-4 w-4"/>
          Dashboard
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Wrench className="mr-2 h-4 w-4"/>
          Evolution Engine
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Shield className="mr-2 h-4 w-4"/>
          Adversarial Testing
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.GitBranch className="mr-2 h-4 w-4"/>
          GitHub Integration
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Activity className="mr-2 h-4 w-4"/>
          Activity Feed
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.FileText className="mr-2 h-4 w-4"/>
          Report Templates
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.BarChart3 className="mr-2 h-4 w-4"/>
          Model Dashboard
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Checklist className="mr-2 h-4 w-4"/>
          Tasks
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Users className="mr-2 h-4 w-4"/>
          Admin Panel
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Database className="mr-2 h-4 w-4"/>
          Analytics Dashboard
        </button_1.Button>
        
        <button_1.Button variant="ghost" className="w-full justify-start">
          <lucide_react_1.Settings className="mr-2 h-4 w-4"/>
          Orchestrator
        </button_1.Button>
      </nav>
      
      <div className="p-4 border-t border-gray-200 space-y-2">
        <button_1.Button variant="outline" className="w-full">
          <lucide_react_1.Play className="mr-2 h-4 w-4"/>
          Start Services
        </button_1.Button>
        <button_1.Button variant="outline" className="w-full">
          <lucide_react_1.Square className="mr-2 h-4 w-4"/>
          Stop Services
        </button_1.Button>
        <button_1.Button variant="outline" className="w-full">
          <lucide_react_1.RotateCcw className="mr-2 h-4 w-4"/>
          Restart Services
        </button_1.Button>
      </div>
    </aside>);
};
exports.Sidebar = Sidebar;
//# sourceMappingURL=Sidebar.js.map