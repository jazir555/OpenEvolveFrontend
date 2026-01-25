/**
 * OpenEvolve Analytics Dashboard Component for BubbleLab
 * 
 * This component replaces the Streamlit-based analytics dashboard UI
 * with a React-based component for the BubbleLab plugin system.
 */

import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Target, 
  Users, 
  Shield, 
  GitBranch, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  TrendingUp, 
  Database, 
  Zap,
  Eye,
  EyeOff,
  Download,
  Filter,
  Search
} from 'lucide-react';
import { BubbleButton, BubbleCard, BubbleTabs, BubbleTab } from '../components/bubblelab';

// Mock data interfaces - these would come from the actual OpenEvolve API
interface WorkflowPerformance {
  workflowId: string;
  name: string;
  startDate: Date;
  endDate?: Date;
  status: 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  successRate: number;
  avgExecutionTime: number;
  totalSubProblems: number;
  solvedSubProblems: number;
}

interface TeamPerformance {
  teamId: string;
  name: string;
  role: 'red' | 'blue' | 'gold';
  totalEvaluations: number;
  approvalRate: number;
  avgScore: number;
  avgTimePerEvaluation: number;
}

interface GauntletPerformance {
  gauntletId: string;
  name: string;
  totalRuns: number;
  successRate: number;
  avgExecutionTime: number;
  avgScore: number;
  failureReasons: string[];
}

interface SolutionQuality {
  solutionId: string;
  workflowId: string;
  qualityScore: number;
  completeness: number;
  correctness: number;
  efficiency: number;
  maintainability: number;
  submittedDate: Date;
}

interface KnowledgeStats {
  totalArtifacts: number;
  totalCategories: number;
  totalTags: number;
  weeklyGrowth: number;
  mostUsedCategory: string;
  mostActiveAuthor: string;
}

const AnalyticsDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [workflowPerformance, setWorkflowPerformance] = useState<WorkflowPerformance[]>([]);
  const [teamPerformance, setTeamPerformance] = useState<TeamPerformance[]>([]);
  const [gauntletPerformance, setGauntletPerformance] = useState<GauntletPerformance[]>([]);
  const [solutionQuality, setSolutionQuality] = useState<SolutionQuality[]>([]);
  const [knowledgeStats, setKnowledgeStats] = useState<KnowledgeStats>({
    totalArtifacts: 0,
    totalCategories: 0,
    totalTags: 0,
    weeklyGrowth: 0,
    mostUsedCategory: '',
    mostActiveAuthor: ''
  });
  const [dateRange, setDateRange] = useState<{ start: Date; end: Date }>({
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
    end: new Date()
  });
  const [isLoading, setIsLoading] = useState(true);

  // Simulate loading data
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      setWorkflowPerformance([
        {
          workflowId: 'wf-001',
          name: 'Evolution Pipeline #1',
          startDate: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
          status: 'completed',
          progress: 100,
          successRate: 0.92,
          avgExecutionTime: 1245,
          totalSubProblems: 12,
          solvedSubProblems: 12
        },
        {
          workflowId: 'wf-002',
          name: 'Adversarial Test #1',
          startDate: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
          status: 'running',
          progress: 65,
          successRate: 0.85,
          avgExecutionTime: 876,
          totalSubProblems: 8,
          solvedSubProblems: 5
        },
        {
          workflowId: 'wf-003',
          name: 'Decomposition Analysis #1',
          startDate: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
          endDate: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
          status: 'completed',
          progress: 100,
          successRate: 0.95,
          avgExecutionTime: 2103,
          totalSubProblems: 15,
          solvedSubProblems: 15
        }
      ]);
      
      setTeamPerformance([
        {
          teamId: 'team-red-001',
          name: 'Red Team Alpha',
          role: 'red',
          totalEvaluations: 42,
          approvalRate: 0.23,
          avgScore: 0.65,
          avgTimePerEvaluation: 120
        },
        {
          teamId: 'team-blue-001',
          name: 'Blue Team Beta',
          role: 'blue',
          totalEvaluations: 38,
          approvalRate: 0.78,
          avgScore: 0.82,
          avgTimePerEvaluation: 95
        },
        {
          teamId: 'team-gold-001',
          name: 'Gold Team Gamma',
          role: 'gold',
          totalEvaluations: 45,
          approvalRate: 0.85,
          avgScore: 0.88,
          avgTimePerEvaluation: 110
        }
      ]);
      
      setGauntletPerformance([
        {
          gauntletId: 'gauntlet-001',
          name: 'Red Team Critique',
          totalRuns: 50,
          successRate: 0.76,
          avgExecutionTime: 180,
          avgScore: 0.72,
          failureReasons: ['Insufficient validation', 'Edge case not handled']
        },
        {
          gauntletId: 'gauntlet-002',
          name: 'Gold Team Verification',
          totalRuns: 48,
          successRate: 0.84,
          avgExecutionTime: 150,
          avgScore: 0.81,
          failureReasons: ['Minor correctness issue']
        }
      ]);
      
      setSolutionQuality([
        {
          solutionId: 'sol-001',
          workflowId: 'wf-001',
          qualityScore: 0.89,
          completeness: 0.92,
          correctness: 0.87,
          efficiency: 0.85,
          maintainability: 0.91,
          submittedDate: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000)
        },
        {
          solutionId: 'sol-002',
          workflowId: 'wf-002',
          qualityScore: 0.76,
          completeness: 0.78,
          correctness: 0.74,
          efficiency: 0.72,
          maintainability: 0.79,
          submittedDate: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000)
        }
      ]);
      
      setKnowledgeStats({
        totalArtifacts: 128,
        totalCategories: 12,
        totalTags: 45,
        weeklyGrowth: 8.5,
        mostUsedCategory: 'algorithms',
        mostActiveAuthor: 'Dr. Jane Smith'
      });
      
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  const handleExportData = () => {
    // Simulate data export
    alert('Exporting analytics data...');
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'running': return 'bg-blue-500';
      case 'failed': return 'bg-red-500';
      case 'paused': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed': return 'Completed';
      case 'running': return 'Running';
      case 'failed': return 'Failed';
      case 'paused': return 'Paused';
      default: return status;
    }
  };

  const getTeamColor = (role: string) => {
    switch (role) {
      case 'red': return 'text-red-600 dark:text-red-400';
      case 'blue': return 'text-blue-600 dark:text-blue-400';
      case 'gold': return 'text-yellow-600 dark:text-yellow-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600 dark:text-gray-300">Loading analytics dashboard...</span>
      </div>
    );
  }

  return (
    <div className="analytics-dashboard p-6 bg-white dark:bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <BarChart3 className="w-8 h-8 text-blue-600 dark:text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Decomposition Workflow Analytics
            </h1>
          </div>
          <div className="flex items-center space-x-3">
            <BubbleButton
              variant="outline"
              size="sm"
              className="flex items-center"
            >
              <Filter className="w-4 h-4 mr-2" />
              Filter
            </BubbleButton>
            <BubbleButton
              variant="outline"
              size="sm"
              className="flex items-center"
              onClick={handleExportData}
            >
              <Download className="w-4 h-4 mr-2" />
              Export
            </BubbleButton>
          </div>
        </div>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Comprehensive analytics and visualization for OpenEvolve workflows
        </p>
      </div>

      {/* Date Range Selector */}
      <div className="mb-6 flex items-center space-x-4">
        <div className="flex items-center">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mr-2">From:</label>
          <input
            type="date"
            value={dateRange.start.toISOString().split('T')[0]}
            onChange={(e) => setDateRange({...dateRange, start: new Date(e.target.value)})}
            className="border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>
        <div className="flex items-center">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mr-2">To:</label>
          <input
            type="date"
            value={dateRange.end.toISOString().split('T')[0]}
            onChange={(e) => setDateRange({...dateRange, end: new Date(e.target.value)})}
            className="border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>
      </div>

      {/* Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <BubbleCard className="p-5">
          <div className="flex items-center">
            <GitBranch className="w-8 h-8 text-blue-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Workflows</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{workflowPerformance.length}</p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <Target className="w-8 h-8 text-green-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {workflowPerformance.length > 0 
                  ? `${(workflowPerformance.filter(w => w.status === 'completed').length / workflowPerformance.length * 100).toFixed(1)}%` 
                  : '0%'}
              </p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <Zap className="w-8 h-8 text-purple-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg. Execution Time</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {workflowPerformance.length > 0 
                  ? `${Math.round(workflowPerformance.reduce((sum, wp) => sum + wp.avgExecutionTime, 0) / workflowPerformance.length)}s` 
                  : '0s'}
              </p>
            </div>
          </div>
        </BubbleCard>

        <BubbleCard className="p-5">
          <div className="flex items-center">
            <Database className="w-8 h-8 text-cyan-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Knowledge Artifacts</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{knowledgeStats.totalArtifacts}</p>
            </div>
          </div>
        </BubbleCard>
      </div>

      {/* Tabs */}
      <BubbleTabs value={activeTab} onValueChange={setActiveTab}>
        <BubbleTab value="overview" label="Overview">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Workflow Performance */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2" />
                Workflow Performance
              </h2>
              <div className="space-y-4">
                {workflowPerformance.map(workflow => (
                  <div 
                    key={workflow.workflowId} 
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">{workflow.name}</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">ID: {workflow.workflowId}</p>
                      </div>
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(workflow.status)}`}></div>
                    </div>
                    <div className="mt-3">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600 dark:text-gray-400">Progress</span>
                        <span className="font-medium text-gray-900 dark:text-white">{Math.round(workflow.progress)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full" 
                          style={{ width: `${workflow.progress}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="mt-3 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                      <span>Success: {(workflow.successRate * 100).toFixed(1)}%</span>
                      <span>Time: {workflow.avgExecutionTime}s</span>
                    </div>
                  </div>
                ))}
                
                {workflowPerformance.length === 0 && (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    No workflow performance data available
                  </div>
                )}
              </div>
            </div>

            {/* Team Analytics */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Users className="w-5 h-5 mr-2" />
                Team Performance
              </h2>
              <div className="space-y-4">
                {teamPerformance.map(team => (
                  <div 
                    key={team.teamId} 
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className={`font-medium ${getTeamColor(team.role)}`}>{team.name}</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">{team.role} team</p>
                      </div>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                        team.role === 'red' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        team.role === 'blue' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      }`}>
                        {team.role}
                      </span>
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      <div>
                        <p className="text-xs text-gray-600 dark:text-gray-400">Evaluations</p>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">{team.totalEvaluations}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600 dark:text-gray-400">Approval Rate</p>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">{(team.approvalRate * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600 dark:text-gray-400">Avg Score</p>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">{team.avgScore.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600 dark:text-gray-400">Avg Time</p>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">{team.avgTimePerEvaluation}s</p>
                      </div>
                    </div>
                  </div>
                ))}
                
                {teamPerformance.length === 0 && (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    No team performance data available
                  </div>
                )}
              </div>
            </div>
          </div>
        </BubbleTab>

        <BubbleTab value="workflow" label="Workflow Performance">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Workflow</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Progress</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Success Rate</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Time</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Sub-Problems</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {workflowPerformance.map(workflow => (
                  <tr key={workflow.workflowId} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">{workflow.name}</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">ID: {workflow.workflowId}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        workflow.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                        workflow.status === 'running' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                        workflow.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      }`}>
                        {getStatusText(workflow.status)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${workflow.progress}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-900 dark:text-white">{Math.round(workflow.progress)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(workflow.successRate * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {workflow.avgExecutionTime}s
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {workflow.solvedSubProblems}/{workflow.totalSubProblems}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="team" label="Team Analytics">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Team</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Role</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Evaluations</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Approval Rate</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Score</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Time</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {teamPerformance.map(team => (
                  <tr key={team.teamId} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">{team.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        team.role === 'red' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        team.role === 'blue' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      }`}>
                        {team.role}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {team.totalEvaluations}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(team.approvalRate * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {team.avgScore.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {team.avgTimePerEvaluation}s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="gauntlet" label="Gauntlet Analytics">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Gauntlet</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Total Runs</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Success Rate</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Time</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Score</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Failure Reasons</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {gauntletPerformance.map(gauntlet => (
                  <tr key={gauntlet.gauntletId} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">{gauntlet.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {gauntlet.totalRuns}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(gauntlet.successRate * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {gauntlet.avgExecutionTime}s
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {gauntlet.avgScore.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400 max-w-xs">
                      {gauntlet.failureReasons.join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="quality" label="Solution Quality">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Solution</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Quality Score</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Completeness</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Correctness</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Efficiency</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Maintainability</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {solutionQuality.map(solution => (
                  <tr key={solution.solutionId} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">{solution.solutionId}</div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">WF: {solution.workflowId}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                          <div 
                            className="bg-green-600 h-2 rounded-full" 
                            style={{ width: `${solution.qualityScore * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-900 dark:text-white">{(solution.qualityScore * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(solution.completeness * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(solution.correctness * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(solution.efficiency * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {(solution.maintainability * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BubbleTab>

        <BubbleTab value="knowledge" label="Knowledge Base Stats">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <BubbleCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Database className="w-5 h-5 mr-2" />
                Knowledge Base Statistics
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Total Artifacts</span>
                  <span className="font-medium text-gray-900 dark:text-white">{knowledgeStats.totalArtifacts}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Total Categories</span>
                  <span className="font-medium text-gray-900 dark:text-white">{knowledgeStats.totalCategories}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Total Tags</span>
                  <span className="font-medium text-gray-900 dark:text-white">{knowledgeStats.totalTags}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Weekly Growth</span>
                  <span className="font-medium text-gray-900 dark:text-white">{knowledgeStats.weeklyGrowth}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Most Used Category</span>
                  <span className="font-medium text-gray-900 dark:text-white">{knowledgeStats.mostUsedCategory}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Most Active Author</span>
                  <span className="font-medium text-gray-900 dark:text-white">{knowledgeStats.mostActiveAuthor}</span>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Shield className="w-5 h-5 mr-2" />
                Top Performing Solutions
              </h3>
              <div className="space-y-3">
                {solutionQuality
                  .sort((a, b) => b.qualityScore - a.qualityScore)
                  .slice(0, 5)
                  .map(solution => (
                    <div key={solution.solutionId} className="flex justify-between items-center">
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">{solution.solutionId}</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">WF: {solution.workflowId}</p>
                      </div>
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                          <div 
                            className="bg-green-600 h-2 rounded-full" 
                            style={{ width: `${solution.qualityScore * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {(solution.qualityScore * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </BubbleCard>
          </div>
        </BubbleTab>
      </BubbleTabs>

      {/* Action Buttons */}
      <div className="mt-8 flex justify-end space-x-3">
        <BubbleButton
          variant="outline"
          className="flex items-center"
          onClick={handleExportData}
        >
          <Download className="w-4 h-4 mr-2" />
          Export Data
        </BubbleButton>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;