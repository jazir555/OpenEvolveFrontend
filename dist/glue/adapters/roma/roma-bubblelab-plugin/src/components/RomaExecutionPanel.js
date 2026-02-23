"use strict";
/**
 * ROMA Execution Panel Component
 *
 * A comprehensive panel for monitoring and managing ROMA task executions.
 * Displays execution status, results, statistics, and provides controls.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.RomaExecutionPanel = void 0;
const react_1 = __importStar(require("react"));
const useRomaExecution_1 = require("../hooks/useRomaExecution");
const lucide_react_1 = require("lucide-react");
/**
 * RomaExecutionPanel Component
 *
 * Provides a full-featured execution panel with:
 * - Task input form
 * - Execution status display
 * - Real-time progress monitoring
 * - Execution history
 * - Statistics dashboard
 *
 * @example
 * ```tsx
 * function App() {
 *   return (
 *     <RomaExecutionPanel
 *       showFullHistory={true}
 *       onClose={() => console.log('Panel closed')}
 *     />
 *   );
 * }
 * ```
 */
const RomaExecutionPanel = ({ executionId, onClose, showFullHistory = false }) => {
    const { executeTask, cancelExecution, currentExecution, isExecuting, isReady, error: pluginError } = (0, useRomaExecution_1.useRomaExecution)();
    const [goal, setGoal] = (0, react_1.useState)('');
    const [maxDepth, setMaxDepth] = (0, react_1.useState)(3);
    const [useCache, setUseCache] = (0, react_1.useState)(true);
    const [selectedExecutionId, setSelectedExecutionId] = (0, react_1.useState)(executionId);
    const [showHistory, setShowHistory] = (0, react_1.useState)(showFullHistory);
    // Sync with executionId prop
    (0, react_1.useEffect)(() => {
        if (executionId) {
            setSelectedExecutionId(executionId);
        }
    }, [executionId]);
    /**
     * Handle task execution
     */
    const handleExecute = async () => {
        if (!goal.trim()) {
            return;
        }
        try {
            await executeTask(goal, {
                maxDepth,
                useCache
            });
            setGoal(''); // Clear input after execution
        }
        catch (err) {
            console.error('Execution failed:', err);
        }
    };
    /**
     * Handle execution cancellation
     */
    const handleCancel = async () => {
        try {
            await cancelExecution();
        }
        catch (err) {
            console.error('Cancellation failed:', err);
        }
    };
    /**
     * Get status icon
     */
    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return <lucide_react_1.CheckCircle className="w-5 h-5 text-green-500"/>;
            case 'failed':
                return <lucide_react_1.XCircle className="w-5 h-5 text-red-500"/>;
            case 'cancelled':
                return <lucide_react_1.XCircle className="w-5 h-5 text-yellow-500"/>;
            case 'executing':
                return <lucide_react_1.RefreshCw className="w-5 h-5 text-blue-500 animate-spin"/>;
            default:
                return <lucide_react_1.Clock className="w-5 h-5 text-gray-400"/>;
        }
    };
    /**
     * Get status color
     */
    const getStatusColor = (status) => {
        switch (status) {
            case 'completed':
                return 'text-green-600 bg-green-50 border-green-200';
            case 'failed':
                return 'text-red-600 bg-red-50 border-red-200';
            case 'cancelled':
                return 'text-yellow-600 bg-yellow-50 border-yellow-200';
            case 'executing':
                return 'text-blue-600 bg-blue-50 border-blue-200';
            default:
                return 'text-gray-600 bg-gray-50 border-gray-200';
        }
    };
    /**
     * Format execution time
     */
    const formatTime = (ms) => {
        if (ms < 1000)
            return `${ms}ms`;
        if (ms < 60000)
            return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60000).toFixed(1)}m`;
    };
    return (<div className="roma-execution-panel bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center">
          <lucide_react_1.Play className="mr-2 w-6 h-6"/> ROMA Execution Panel
        </h2>
        {onClose && (<button onClick={onClose} className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" aria-label="Close panel">
            <lucide_react_1.X className="w-5 h-5"/>
          </button>)}
      </div>

      {/* Plugin Error */}
      {pluginError && (<div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 rounded-lg">
          <div className="flex items-start">
            <lucide_react_1.AlertCircle className="w-5 h-5 text-red-500 mr-2 flex-shrink-0"/>
            <div>
              <p className="font-medium text-red-800 dark:text-red-200">Plugin Error</p>
              <p className="text-sm text-red-600 dark:text-red-300">{pluginError}</p>
            </div>
          </div>
        </div>)}

      {/* Not Ready */}
      {!isReady && !pluginError && (<div className="mb-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 rounded-lg">
          <div className="flex items-start">
            <lucide_react_1.AlertCircle className="w-5 h-5 text-yellow-500 mr-2 flex-shrink-0"/>
            <div>
              <p className="font-medium text-yellow-800 dark:text-yellow-200">Plugin Not Ready</p>
              <p className="text-sm text-yellow-600 dark:text-yellow-300">
                The ROMA plugin needs to be initialized before executing tasks.
              </p>
            </div>
          </div>
        </div>)}

      {/* Task Input */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700/50 border border-gray-200 dark:border-gray-600 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Execute New Task
        </h3>
        
        <div className="space-y-4">
          {/* Goal Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Task Goal
            </label>
            <textarea value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="Enter your task goal..." rows={3} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white resize-none" disabled={!isReady || isExecuting}/>
          </div>

          {/* Options */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Max Depth */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Max Depth: {maxDepth}
              </label>
              <input type="range" min="1" max="10" value={maxDepth} onChange={(e) => setMaxDepth(parseInt(e.target.value))} className="w-full" disabled={!isReady || isExecuting}/>
            </div>

            {/* Use Cache */}
            <div className="flex items-center">
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300">
                <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded" disabled={!isReady || isExecuting}/>
                Use Cache
              </label>
            </div>
          </div>

          {/* Execute Button */}
          <button onClick={handleExecute} disabled={!isReady || isExecuting || !goal.trim()} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed transition-colors">
            {isExecuting ? (<>
                <lucide_react_1.RefreshCw className="mr-2 w-4 h-4 animate-spin"/>
                Executing...
              </>) : (<>
                <lucide_react_1.Play className="mr-2 w-4 h-4"/>
                Execute Task
              </>)}
          </button>

          {/* Cancel Button */}
          {isExecuting && (<button onClick={handleCancel} className="w-full bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-md flex items-center justify-center transition-colors">
              <lucide_react_1.X className="mr-2 w-4 h-4"/>
              Cancel Execution
            </button>)}
        </div>
      </div>

      {/* Current Execution Status */}
      {currentExecution && (<div className="mb-6 p-4 border border-gray-200 dark:border-gray-600 rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Current Execution
            </h3>
            <div className={`flex items-center px-3 py-1 rounded-full ${getStatusColor(currentExecution.status)}`}>
              {getStatusIcon(currentExecution.status)}
              <span className="ml-2 font-medium capitalize">{currentExecution.status}</span>
            </div>
          </div>

          {/* Execution Details */}
          <div className="space-y-3">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Execution ID</p>
              <p className="font-mono text-sm text-gray-900 dark:text-white">{currentExecution.executionId}</p>
            </div>
            
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Goal</p>
              <p className="text-sm text-gray-900 dark:text-white">{currentExecution.goal}</p>
            </div>

            {/* Statistics */}
            {currentExecution.statistics && (<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Execution Time</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {formatTime(currentExecution.statistics.executionTime)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Subtasks Created</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {currentExecution.statistics.subtasksCreated}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Subtasks Completed</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {currentExecution.statistics.subtasksCompleted}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Tools Used</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {currentExecution.statistics.toolsUsed?.length || 0}
                  </p>
                </div>
              </div>)}

            {/* Result */}
            {currentExecution.result && (<div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Result</p>
                <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded border border-gray-200 dark:border-gray-600 max-h-64 overflow-y-auto">
                  <pre className="text-sm text-gray-900 dark:text-white whitespace-pre-wrap">
                    {typeof currentExecution.result === 'string'
                    ? currentExecution.result
                    : JSON.stringify(currentExecution.result, null, 2)}
                  </pre>
                </div>
              </div>)}

            {/* Error */}
            {currentExecution.error && (<div className="bg-red-50 dark:bg-red-900/20 p-3 rounded border border-red-200">
                <p className="text-sm text-red-600 dark:text-red-400 mb-1">Error</p>
                <p className="text-sm text-red-800 dark:text-red-200">{currentExecution.error}</p>
              </div>)}
          </div>
        </div>)}

      {/* Execution History Toggle */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Execution History
        </h3>
        <button onClick={() => setShowHistory(!showHistory)} className="text-blue-600 hover:text-blue-700 text-sm font-medium">
          {showHistory ? 'Hide' : 'Show'} History
        </button>
      </div>

      {/* Execution History */}
      {showHistory && (<div className="border border-gray-200 dark:border-gray-600 rounded-lg overflow-hidden">
          <div className="max-h-96 overflow-y-auto">
            {isReady && currentExecution && (<table className="min-w-full divide-y divide-gray-200 dark:divide-gray-600">
                <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Goal
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Time
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-600">
                  {/* Current execution */}
                  {currentExecution && (<tr className={`hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer ${selectedExecutionId === currentExecution.executionId ? 'bg-blue-50 dark:bg-blue-900/20' : ''}`} onClick={() => setSelectedExecutionId(currentExecution.executionId)}>
                      <td className="px-4 py-3 whitespace-nowrap">
                        {getStatusIcon(currentExecution.status)}
                      </td>
                      <td className="px-4 py-3 max-w-xs truncate" title={currentExecution.goal}>
                        {currentExecution.goal}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                        {formatTime(currentExecution.statistics?.executionTime || 0)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <button onClick={() => setSelectedExecutionId(currentExecution.executionId)} className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                          View
                        </button>
                      </td>
                    </tr>)}
                </tbody>
              </table>)}
          </div>
        </div>)}
    </div>);
};
exports.RomaExecutionPanel = RomaExecutionPanel;
exports.default = exports.RomaExecutionPanel;
//# sourceMappingURL=RomaExecutionPanel.js.map