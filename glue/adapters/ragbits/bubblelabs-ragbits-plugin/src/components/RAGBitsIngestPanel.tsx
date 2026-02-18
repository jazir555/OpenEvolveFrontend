// RAGBits Ingest Panel Component

import React, { useState } from 'react';
import { Upload, FileText } from 'lucide-react';
import type { RAGBitsIngestPanelProps } from '../types/plugin-types';
import { useRAGBitsPlugin } from '../utils/createRAGBitsPlugin';
import { ragbitsLogger } from '../../../../../lib/structuredLogger';

export const RAGBitsIngestPanel: React.FC<RAGBitsIngestPanelProps> = ({
  onSuccess,
  onClose,
  showDebug = false
}) => {
  const [content, setContent] = useState('');
  const [documentType, setDocumentType] = useState('general');
  const [source, setSource] = useState('');
  const [stage, setStage] = useState('');
  const [team, setTeam] = useState('');
  const [tags, setTags] = useState('');
  const [isIngesting, setIsIngesting] = useState(false);

  const plugin = useRAGBitsPlugin();

  const handleIngest = async () => {
    if (!content.trim()) {
      return;
    }

    setIsIngesting(true);

    const correlationId = `ingest-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

    try {
      ragbitsLogger.info('Starting RAGBits ingest', {
        correlation_id: correlationId,
        source_service: 'ragbits-plugin',
        target_service: 'ragbits-server',
        document_type: documentType,
        content_length: content.length
      });

      // Prepare metadata
      const metadata: Record<string, any> = {
        documentType
      };

      if (source) metadata.source = source;
      if (stage) metadata.stage = stage;
      if (team) metadata.team = team;
      if (tags) metadata.tags = tags.split(',').map(t => t.trim()).filter(t => t);

      // Call the plugin's ingest method
      const response = await plugin.ingest({
        content,
        metadata
      });

      ragbitsLogger.info('RAGBits ingest completed successfully', {
        correlation_id: correlationId,
        source_service: 'ragbits-plugin',
        document_id: response.documentId,
        execution_time: response.executionTime
      });

      onSuccess(response);
      setContent('');
      setSource('');
      setStage('');
      setTeam('');
      setTags('');
    } catch (error) {
      ragbitsLogger.error('RAGBits ingest failed', error as Error, {
        correlation_id: correlationId,
        source_service: 'ragbits-plugin',
        document_type: documentType
      });
    } finally {
      setIsIngesting(false);
    }
  };

  return (
    <div className="ragbits-ingest-panel">
      <div className="ingest-header">
        <Upload className="icon" />
        <h2>Ingest Document</h2>
      </div>

      <div className="ingest-content">
        <div className="form-section">
          <h3>Document Content</h3>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Enter document content..."
            rows={10}
            className="ingest-textarea"
          />
        </div>

        <div className="form-section">
          <h3>Metadata</h3>
          <div className="form-group">
            <label>Document Type</label>
            <select
              value={documentType}
              onChange={(e) => setDocumentType(e.target.value)}
            >
              <option value="general">General</option>
              <option value="solution">Solution</option>
              <option value="problem">Problem</option>
              <option value="test_case">Test Case</option>
              <option value="documentation">Documentation</option>
              <option value="code">Code</option>
              <option value="analysis">Analysis</option>
              <option value="report">Report</option>
              <option value="artifact">Artifact</option>
            </select>
          </div>

          <div className="form-group">
            <label>Source (Optional)</label>
            <input
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="Document source..."
            />
          </div>

          <div className="form-group">
            <label>Stage (Optional)</label>
            <input
              type="text"
              value={stage}
              onChange={(e) => setStage(e.target.value)}
              placeholder="Workflow stage..."
            />
          </div>

          <div className="form-group">
            <label>Team (Optional)</label>
            <input
              type="text"
              value={team}
              onChange={(e) => setTeam(e.target.value)}
              placeholder="Team..."
            />
          </div>

          <div className="form-group">
            <label>Tags (comma-separated)</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="tag1, tag2, tag3..."
            />
          </div>
        </div>

        {showDebug && (
          <div className="debug-info">
            <h4>Debug Information</h4>
            <pre>{JSON.stringify({
              contentLength: content.length,
              documentType,
              source,
              stage,
              team,
              tags: tags.split(',').map(t => t.trim()).filter(t => t)
            }, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="ingest-actions">
        <button className="btn btn-secondary" onClick={onClose}>
          Cancel
        </button>
        <button
          className="btn btn-primary"
          onClick={handleIngest}
          disabled={isIngesting || !content.trim()}
        >
          <FileText className="icon" />
          {isIngesting ? 'Ingesting...' : 'Ingest Document'}
        </button>
      </div>
    </div>
  );
};
