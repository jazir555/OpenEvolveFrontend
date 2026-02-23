"use strict";
// RAGBits Ingest Panel Component
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
exports.RAGBitsIngestPanel = void 0;
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
const createRAGBitsPlugin_1 = require("../utils/createRAGBitsPlugin");
const structuredLogger_1 = require("../lib/structuredLogger");
const RAGBitsIngestPanel = ({ onSuccess, onClose, showDebug = false }) => {
    const [content, setContent] = (0, react_1.useState)('');
    const [documentType, setDocumentType] = (0, react_1.useState)('general');
    const [source, setSource] = (0, react_1.useState)('');
    const [stage, setStage] = (0, react_1.useState)('');
    const [team, setTeam] = (0, react_1.useState)('');
    const [tags, setTags] = (0, react_1.useState)('');
    const [isIngesting, setIsIngesting] = (0, react_1.useState)(false);
    const plugin = (0, createRAGBitsPlugin_1.useRAGBitsPlugin)();
    const handleIngest = async () => {
        if (!content.trim()) {
            return;
        }
        setIsIngesting(true);
        const correlationId = `ingest-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        try {
            structuredLogger_1.ragbitsLogger.info('Starting RAGBits ingest', {
                correlation_id: correlationId,
                source_service: 'ragbits-plugin',
                target_service: 'ragbits-server',
                document_type: documentType,
                content_length: content.length
            });
            // Prepare metadata
            const metadata = {
                documentType
            };
            if (source)
                metadata.source = source;
            if (stage)
                metadata.stage = stage;
            if (team)
                metadata.team = team;
            if (tags)
                metadata.tags = tags.split(',').map(t => t.trim()).filter(t => t);
            // Call the plugin's ingest method
            const response = await plugin.ingest({
                content,
                metadata
            });
            structuredLogger_1.ragbitsLogger.info('RAGBits ingest completed successfully', {
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
        }
        catch (error) {
            structuredLogger_1.ragbitsLogger.error('RAGBits ingest failed', error, {
                correlation_id: correlationId,
                source_service: 'ragbits-plugin',
                document_type: documentType
            });
        }
        finally {
            setIsIngesting(false);
        }
    };
    return (<div className="ragbits-ingest-panel">
      <div className="ingest-header">
        <lucide_react_1.Upload className="icon"/>
        <h2>Ingest Document</h2>
      </div>

      <div className="ingest-content">
        <div className="form-section">
          <h3>Document Content</h3>
          <textarea value={content} onChange={(e) => setContent(e.target.value)} placeholder="Enter document content..." rows={10} className="ingest-textarea"/>
        </div>

        <div className="form-section">
          <h3>Metadata</h3>
          <div className="form-group">
            <label>Document Type</label>
            <select value={documentType} onChange={(e) => setDocumentType(e.target.value)}>
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
            <input type="text" value={source} onChange={(e) => setSource(e.target.value)} placeholder="Document source..."/>
          </div>

          <div className="form-group">
            <label>Stage (Optional)</label>
            <input type="text" value={stage} onChange={(e) => setStage(e.target.value)} placeholder="Workflow stage..."/>
          </div>

          <div className="form-group">
            <label>Team (Optional)</label>
            <input type="text" value={team} onChange={(e) => setTeam(e.target.value)} placeholder="Team..."/>
          </div>

          <div className="form-group">
            <label>Tags (comma-separated)</label>
            <input type="text" value={tags} onChange={(e) => setTags(e.target.value)} placeholder="tag1, tag2, tag3..."/>
          </div>
        </div>

        {showDebug && (<div className="debug-info">
            <h4>Debug Information</h4>
            <pre>{JSON.stringify({
                contentLength: content.length,
                documentType,
                source,
                stage,
                team,
                tags: tags.split(',').map(t => t.trim()).filter(t => t)
            }, null, 2)}</pre>
          </div>)}
      </div>

      <div className="ingest-actions">
        <button className="btn btn-secondary" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" onClick={handleIngest} disabled={isIngesting || !content.trim()}>
          <lucide_react_1.FileText className="icon"/>
          {isIngesting ? 'Ingesting...' : 'Ingest Document'}
        </button>
      </div>
    </div>);
};
exports.RAGBitsIngestPanel = RAGBitsIngestPanel;
//# sourceMappingURL=RAGBitsIngestPanel.js.map