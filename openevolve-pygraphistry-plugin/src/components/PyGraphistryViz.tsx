import React, { useState, useEffect } from 'react';
import { pygraphistryPlugin } from '../utils/createPyGraphistryPlugin';
import { GraphNode, GraphEdge } from '../types/plugin-types';

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  height?: string | number;
  autoGenerate?: boolean;
}

export const PyGraphistryViz: React.FC<Props> = ({ 
  nodes, 
  edges, 
  height = 600,
  autoGenerate = true 
}) => {
  const state = pygraphistryPlugin.getState();
  const isEnabled = state.features.pygraphistryEnabled;
  
  const [vizUrl, setVizUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const generateViz = async () => {
    if (!isEnabled) return;
    
    setLoading(true);
    setError(null);
    try {
      const url = await pygraphistryPlugin.generateVisualization({
        nodes,
        edges,
        layout: 'force_directed'
      });
      if (url) {
        setVizUrl(url);
      } else {
        setError('Failed to generate visualization URL. Ensure backend is running and configured.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isEnabled && autoGenerate && nodes.length > 0) {
      generateViz();
    }
  }, [nodes, edges, autoGenerate, isEnabled]);

  if (!isEnabled) {
    return (
      <div className="p-6 text-center border-2 border-dashed border-gray-200 rounded-lg bg-gray-50 text-gray-500">
        <p className="font-medium italic">PyGraphistry visualization is currently disabled in settings.</p>
      </div>
    );
  }

  if (loading) {
    return <div className="p-4 text-center">Loading PyGraphistry Visualization...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500 border border-red-200 rounded">Error: {error}</div>;
  }

  if (!vizUrl) {
    return (
      <div className="p-4 text-center">
        <button 
          onClick={generateViz}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Generate Graphistry Visualization
        </button>
      </div>
    );
  }

  return (
    <div className="relative w-full overflow-hidden rounded-lg shadow-lg border border-gray-200">
      <iframe
        src={vizUrl}
        width="100%"
        height={height}
        frameBorder="0"
        title="PyGraphistry Visualization"
        scrolling="no"
        allowFullScreen
      />
      <div className="absolute top-2 right-2 flex space-x-2">
        <a 
          href={vizUrl} 
          target="_blank" 
          rel="noopener noreferrer"
          className="p-2 bg-white/80 backdrop-blur rounded hover:bg-white text-xs font-medium"
        >
          Open External
        </a>
        <button 
          onClick={generateViz}
          className="p-2 bg-white/80 backdrop-blur rounded hover:bg-white text-xs font-medium"
        >
          Refresh
        </button>
      </div>
    </div>
  );
};
