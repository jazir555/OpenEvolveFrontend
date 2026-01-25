/**
 * React Hooks Examples for OpenEvolve Integration Library
 *
 * This file demonstrates how to use the integration library with React
 */

import React, { useState, useEffect } from 'react';
import { 
  OpenEvolveClient, 
  OpenEvolveProvider, 
  useDecomposition, 
  useLeanAide,
  useHealthCheck,
  ProgressUpdate
} from '@openevolve/integration-library';

// Initialize the client (usually outside component tree)
const client = new OpenEvolveClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// ============================================================================
// Example Component: DecompositionComponent
// ============================================================================

export function DecompositionComponent() {
  const { data, error, loading, execute } = useDecomposition();

  const handleDecompose = () => {
    execute({
      operation: 'decompose',
      input: {
        problem: 'Build a scalable microservices architecture',
        strategy: 'hierarchical',
        options: { max_depth: 3 }
      }
    });
  };

  return (
    <div>
      <h2>Problem Decomposition</h2>

      <button onClick={handleDecompose} disabled={loading}>
        {loading ? 'Decomposing...' : 'Decompose Problem'}
      </button>

      {error && (
        <div className="error" style={{ color: 'red' }}>
          Error: {error.message}
        </div>
      )}

      {data && (
        <div className="results">
          <h3>Results</h3>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Example Component: LeanAideComponent
// ============================================================================

export function LeanAideComponent() {
  const { data, error, loading, execute } = useLeanAide();

  const handleVerify = () => {
    execute({
      operation: 'prove',
      input: {
        theorem: 'forall n : Nat, n + 0 = n',
        strategy: 'mcts'
      }
    });
  };

  return (
    <div>
      <h2>LeanAide Operations</h2>

      <button onClick={handleVerify} disabled={loading}>
        {loading ? 'Verifying...' : 'Verify Proof'}
      </button>

      {error && (
        <div className="error" style={{ color: 'red' }}>
          Error: {error.message}
        </div>
      )}

      {data && (
        <div className="results">
          <h3>Results</h3>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Example Component: StreamingComponent
// ============================================================================

export function StreamingComponent() {
  const [streamData, setStreamData] = useState<any>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  const handleExecute = async () => {
    setIsStreaming(true);
    setProgress(0);
    setStatus('Starting...');

    try {
      const result = await client.executeStream(
        'decomposition',
        {
          operation: 'decompose',
          input: { problem: 'Complex problem' }
        },
        (update: ProgressUpdate) => {
          setProgress(update.progress);
          setStatus(update.message);
          if (update.data) {
            setStreamData(update.data);
          }
        }
      );
      setStreamData(result);
      setStatus('Completed');
    } catch (err: any) {
      setStatus('Error: ' + err.message);
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div>
      <h2>Streaming Execution</h2>

      <button onClick={handleExecute} disabled={isStreaming}>
        {isStreaming ? 'Executing...' : 'Execute with Streaming'}
      </button>

      {isStreaming && (
        <div className="progress">
          <p>Status: {status}</p>
          <div style={{ width: '100%', backgroundColor: '#eee' }}>
            <div style={{ 
              width: `${progress}%`, 
              height: '10px', 
              backgroundColor: 'blue',
              transition: 'width 0.3s' 
            }} />
          </div>
          <span>{progress}%</span>
        </div>
      )}

      {streamData && (
        <div className="results">
          <h3>Final Result</h3>
          <pre>{JSON.stringify(streamData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Example: Complete App Component
// ============================================================================

export function OpenEvolveApp() {
  const [activeTab, setActiveTab] = useState('decomposition');

  return (
    <OpenEvolveProvider client={client}>
      <div className="openevolve-app" style={{ padding: '20px', fontFamily: 'sans-serif' }}>
        <h1>OpenEvolve Integration Demo</h1>

        <nav style={{ marginBottom: '20px' }}>
          <button onClick={() => setActiveTab('decomposition')}>Decomposition</button>
          <button onClick={() => setActiveTab('leanaide')}>LeanAide</button>
          <button onClick={() => setActiveTab('streaming')}>Streaming</button>
        </nav>

        <main style={{ border: '1px solid #ccc', padding: '20px' }}>
          {activeTab === 'decomposition' && <DecompositionComponent />}
          {activeTab === 'leanaide' && <LeanAideComponent />}
          {activeTab === 'streaming' && <StreamingComponent />}
        </main>
      </div>
    </OpenEvolveProvider>
  );
}