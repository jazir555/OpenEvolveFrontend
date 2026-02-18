// Datapizza Pipeline Hook
// React hook for running Datapizza pipelines
//
// INTEGRATION STATUS: Production Implementation
// - Uses DatapizzaClient for all API calls
// - Follows Federation Constitution laws
// - Configurable mock fallback for development (set VITE_DATAPIZZA_USE_MOCK=true)
//
// SETUP INSTRUCTIONS:
// 1. Configure DATAPIZZA_BASE_URL in environment
// 2. Configure DATAPIZZA_TIMEOUT_MS in environment
// 3. Set VITE_DATAPIZZA_USE_MOCK=true for development without API
import { useCallback, useState } from 'react';
export function useDatapizzaPipeline(client) {
    const [isRunning, setIsRunning] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentStep, setCurrentStep] = useState('');
    const [error, setError] = useState(null);
    const runPipeline = useCallback(async (dataSource, options = {}) => {
        setIsRunning(true);
        setError(null);
        setProgress(0);
        setCurrentStep('Initializing');
        const startTime = Date.now();
        try {
            // Check if mock mode is enabled (for development)
            const useMock = import.meta.env.VITE_DATAPIZZA_USE_MOCK === 'true';
            // Simulate pipeline progress
            const steps = [
                { name: 'Validating data source', progress: 10 },
                { name: 'Reading data', progress: 25 },
                { name: 'Chunking documents', progress: 50 },
                { name: 'Generating embeddings', progress: 75 },
                { name: 'Updating vector store', progress: 90 },
                { name: 'Finalizing', progress: 100 },
            ];
            let currentStepIndex = 0;
            const progressInterval = setInterval(() => {
                if (currentStepIndex < steps.length) {
                    const step = steps[currentStepIndex];
                    setProgress(step.progress);
                    setCurrentStep(step.name);
                    currentStepIndex++;
                }
            }, useMock ? 1000 : 2000);
            if (useMock) {
                console.warn('Datapizza mock mode enabled - set VITE_DATAPIZZA_USE_MOCK=false to use real API');
                // Simulate pipeline execution time
                await new Promise(resolve => setTimeout(resolve, 5000));
                clearInterval(progressInterval);
                setProgress(100);
                setCurrentStep('Completed (mock)');
                // Generate mock pipeline results
                const mockProcessedData = {
                    recordsProcessed: Math.floor(Math.random() * 5000) + 500,
                    chunksCreated: Math.floor(Math.random() * 500) + 50,
                    embeddingsGenerated: Math.floor(Math.random() * 500) + 50,
                    vectorStoreUpdated: true,
                };
                return {
                    success: true,
                    pipelineId: `pipeline_${Date.now()}`,
                    dataSource,
                    processedData: mockProcessedData,
                    confidenceScore: 0.95 - Math.random() * 0.05,
                    pipelineType: options.pipelineType || 'standard',
                    dataDomain: 'structured',
                    errors: [],
                    warnings: [
                        'Using mock pipeline execution - Datapizza API not configured',
                        'Set VITE_DATAPIZZA_USE_MOCK=false and configure DATAPIZZA_BASE_URL',
                    ],
                    executionTime: Date.now() - startTime,
                    metadata: {
                        timestamp: new Date().toISOString(),
                        processingSteps: ['validation', 'chunking', 'embedding', 'vector_storage'],
                        dataSource,
                        mock: true,
                    },
                    timestamp: new Date(),
                };
            }
            // Use real DatapizzaClient
            if (!client) {
                throw new Error('DatapizzaClient not provided. Either pass a client to the hook or set VITE_DATAPIZZA_USE_MOCK=true for development.');
            }
            try {
                const result = await client.runPipeline({
                    dataSource,
                    pipelineType: options.pipelineType || 'standard',
                    parameters: {
                        chunk_size: options.chunkSize || 1000,
                        overlap_size: options.overlapSize || 200,
                        embedding_model: options.embeddingModel || 'default',
                        vector_store: options.vectorStore || 'default',
                        skip_validation: options.skipValidation || false,
                        skip_embedding: options.skipEmbedding || false,
                    },
                });
                clearInterval(progressInterval);
                setProgress(100);
                setCurrentStep('Completed');
                return {
                    success: result.success,
                    pipelineId: result.pipelineId,
                    dataSource,
                    processedData: {
                        recordsProcessed: 0,
                        chunksCreated: 0,
                        embeddingsGenerated: 0,
                        vectorStoreUpdated: true,
                    },
                    confidenceScore: result.status === 'completed' ? 1.0 : 0.5,
                    pipelineType: result.pipelineType,
                    dataDomain: 'structured',
                    errors: result.error ? [result.error] : [],
                    warnings: [],
                    executionTime: Date.now() - startTime,
                    metadata: {
                        timestamp: new Date().toISOString(),
                        processingSteps: ['validation', 'chunking', 'embedding', 'vector_storage'],
                        dataSource,
                        status: result.status,
                        startedAt: result.startedAt,
                        completedAt: result.completedAt,
                    },
                    timestamp: new Date(),
                };
            }
            catch (apiError) {
                clearInterval(progressInterval);
                setProgress(100);
                setCurrentStep('Failed');
                throw apiError;
            }
        }
        catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error';
            setError(errorMessage);
            setCurrentStep('Failed');
            return {
                success: false,
                pipelineId: `pipeline_${Date.now()}`,
                dataSource,
                processedData: null,
                confidenceScore: 0.0,
                pipelineType: options.pipelineType || 'standard',
                dataDomain: 'unknown',
                errors: [errorMessage],
                warnings: [],
                executionTime: Date.now() - startTime,
                metadata: {
                    timestamp: new Date().toISOString(),
                    processingSteps: [],
                    dataSource,
                },
                timestamp: new Date(),
            };
        }
        finally {
            setIsRunning(false);
            setProgress(0);
            setCurrentStep('');
        }
    }, [client]);
    return { runPipeline, isRunning, progress, currentStep, error };
}
//# sourceMappingURL=useDatapizzaPipeline.js.map