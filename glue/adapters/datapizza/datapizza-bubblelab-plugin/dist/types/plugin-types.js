// Datapizza Plugin Types and Interfaces
// Defines all types and interfaces for the BubbleLabs Datapizza plugin
function getDefaultDatapizzaServerUrl() {
    return import.meta.env.VITE_DATAPIZZA_SERVER_URL || 'http://localhost:3000/datapizza';
}
export const DATAPIZZA_PIPELINE_TYPES = [
    {
        value: 'standard',
        label: 'Standard Pipeline',
        description: 'Standard data processing pipeline with basic transformations',
        recommendedFor: ['Simple data', 'Basic processing', 'Quick results']
    },
    {
        value: 'advanced',
        label: 'Advanced Pipeline',
        description: 'Advanced pipeline with complex transformations and optimizations',
        recommendedFor: ['Complex data', 'Large datasets', 'High performance needs']
    },
    {
        value: 'custom',
        label: 'Custom Pipeline',
        description: 'Customizable pipeline for specific use cases',
        recommendedFor: ['Specialized processing', 'Unique requirements', 'Custom workflows']
    }
];
export const DATAPIZZA_DATA_DOMAINS = [
    { value: 'structured', label: 'Structured Data', description: 'Relational databases, CSV files, spreadsheets' },
    { value: 'unstructured', label: 'Unstructured Data', description: 'Text documents, emails, social media posts' },
    { value: 'semi_structured', label: 'Semi-Structured Data', description: 'JSON, XML, HTML documents' },
    { value: 'relational', label: 'Relational Data', description: 'SQL databases, normalized data structures' },
    { value: 'document', label: 'Document Data', description: 'PDFs, Word documents, scanned text' },
    { value: 'time_series', label: 'Time Series Data', description: 'Temporal data, sensor readings, financial data' },
    { value: 'graph', label: 'Graph Data', description: 'Network data, social graphs, knowledge graphs' },
    { value: 'geospatial', label: 'Geospatial Data', description: 'GIS data, location-based information' },
    { value: 'multimedia', label: 'Multimedia Data', description: 'Images, audio, video files' },
    { value: 'general', label: 'General Data', description: 'Mixed or unspecified data types' }
];
export const DEFAULT_DATAPIZZA_CONFIG = {
    enabled: true,
    serverUrl: getDefaultDatapizzaServerUrl(),
    apiKey: '',
    timeout: 30000,
    pipelineEnabled: true,
    autoDetectDataSources: true,
    defaultPipelineType: 'standard',
    dataProcessingConfig: {
        chunkSize: 1000,
        overlapSize: 200,
        embeddingModel: 'text-embedding-ada-002',
        vectorStoreType: 'qdrant',
        maxParallelProcesses: 4
    },
    agentConfigurations: {
        agent1: {
            enabled: true,
            maxTasks: 10,
            timeout: 60
        },
        agent2: {
            enabled: true,
            parallelExecution: true,
            maxWorkers: 4
        },
        agent3: {
            enabled: true,
            critiqueLevel: 'standard'
        }
    },
    integrateWithWorkflow: true,
    integrateWithKnowledgeGraph: true,
    integrateWithExternalSources: true,
    enableCaching: true,
    cacheTTLSeconds: 3600,
    maxProcessingTime: 300,
    showAdvancedOptions: false,
    showDebugInfo: false,
    theme: 'system'
};
//# sourceMappingURL=plugin-types.js.map