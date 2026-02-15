/**
 * Main Integration Component for LeanAide Autoformalization with BubbleLab Analytics
 * 
 * This component provides the complete integration between LeanAide's autoformalization system
 * and BubbleLab's analytics platform, offering a comprehensive dashboard for monitoring
 * and managing the autoformalization process.
 */

import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  BarChart3, 
  Shield, 
  Database, 
  Settings, 
  Activity, 
  TrendingUp, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Zap,
  Target,
  Award,
  Flame,
  Eye,
  BarChart2,
  PieChart,
  LineChart,
  Users,
  MessageSquare
} from 'lucide-react';
import { 
  AnalyticsDashboard,
  EnhancedLeanAideVerification,
  KnowledgeGraphIntegration,
  useAutoformalizationAnalytics
} from './integration/autoformalizationAnalytics';

export interface LeanAideBubbleLabIntegrationProps {
  className?: string;
}

export function LeanAideBubbleLabIntegration({ className = '' }: LeanAideBubbleLabIntegrationProps) {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'verification' | 'knowledge' | 'settings'>('dashboard');
  const { metrics } = useAutoformalizationAnalytics();

  return (
    <div className={`bg-gray-50 min-h-screen p-6 ${className}`}>
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Brain className="w-8 h-8 text-blue-600" />
            LeanAide BubbleLab Integration
          </h1>
          <p className="text-gray-600 mt-2">
            Advanced mathematical formalization with real-time analytics and knowledge integration
          </p>
        </div>
        
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          {/* Navigation Tabs */}
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'dashboard', label: 'Analytics Dashboard', icon: BarChart3 },
                { id: 'verification', label: 'Autoformalization', icon: Shield },
                { id: 'knowledge', label: 'Knowledge Graph', icon: Database },
                { id: 'settings', label: 'Settings', icon: Settings },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                    activeTab === tab.id
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
          
          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'dashboard' && <AnalyticsDashboard />}
            
            {activeTab === 'verification' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                  <Shield className="w-6 h-6" />
                  Autoformalization Verification
                </h2>
                <EnhancedLeanAideVerification
                  problemStatement="Prove that for all natural numbers n, n + 0 = n"
                  mode="theorem"
                  enableAnalytics={true}
                  strategy="auto"
                  domain="arithmetic"
                />
              </div>
            )}
            
            {activeTab === 'knowledge' && <KnowledgeGraphIntegration />}
            
            {activeTab === 'settings' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                  <Settings className="w-6 h-6" />
                  Integration Settings
                </h2>
                <div className="bg-gray-50 p-4 rounded-lg border">
                  <h3 className="font-medium text-gray-700 mb-3">Analytics Configuration</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Real-time Tracking</p>
                        <p className="text-sm text-gray-500">Track all conversion events</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Performance Metrics</p>
                        <p className="text-sm text-gray-500">Monitor processing times</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Error Tracking</p>
                        <p className="text-sm text-gray-500">Log all conversion errors</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-white rounded border">
                      <div>
                        <p className="font-medium text-gray-800">Domain Analysis</p>
                        <p className="text-sm text-gray-500">Analyze by mathematical domain</p>
                      </div>
                      <div className="w-12 h-6 bg-green-500 rounded-full relative">
                        <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 right-0.5"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Quick Stats Footer */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Activity className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Conversions</p>
                <p className="text-xl font-bold text-gray-800">{metrics.totalAttempts}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Success Rate</p>
                <p className="text-xl font-bold text-gray-800">{Math.round(metrics.successRate * 100)}%</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Target className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Confidence</p>
                <p className="text-xl font-bold text-gray-800">{(metrics.avgConfidenceScore || 0).toFixed(2)}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <Clock className="w-5 h-5 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Time</p>
                <p className="text-xl font-bold text-gray-800">{Math.round(metrics.avgProcessingTime || 0)}ms</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Export individual components for flexible usage
export {
  AnalyticsDashboard,
  EnhancedLeanAideVerification,
  KnowledgeGraphIntegration,
  useAutoformalizationAnalytics
};

// Default export
export default LeanAideBubbleLabIntegration;