// Datapizza Processing Hook
// React hook for data processing with Datapizza
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
export function useDatapizzaProcessing(client) {
    const [isLoading, setIsLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState(null);
    const processData = useCallback(async (data, options = {}) => {
        setIsLoading(true);
        setError(null);
        setProgress(0);
        const startTime = Date.now();
        try {
            // Check if mock mode is enabled (for development)
            const useMock = import.meta.env.VITE_DATAPIZZA_USE_MOCK === 'true';
            if (useMock) {
                console.warn('Datapizza mock mode enabled - set VITE_DATAPIZZA_USE_MOCK=false to use real API');
                // Simulate progress updates
                const progressInterval = setInterval(() => {
                    setProgress((prev) => Math.min(prev + 10, 90));
                }, 500);
                // Simulate processing time
                await new Promise(resolve => setTimeout(resolve, 2000));
                clearInterval(progressInterval);
                setProgress(100);
                // Generate mock processing results
                const mockProcessedData = {
                    ...data,
                    processed: true,
                    chunks: Array.isArray(data) ? data.length : 1,
                    embeddings: Math.floor(Math.random() * 100) + 10,
                    vectorStoreUpdated: true,
                    timestamp: new Date().toISOString(),
                };
                return {
                    success: true,
                    dataId: `data_${Date.now()}`,
                    processedData: mockProcessedData,
                    confidenceScore: 0.92 - Math.random() * 0.1,
                    processingType: options.processingType || 'standard',
                    errors: [],
                    warnings: [
                        'Using mock processing - Datapizza API not configured',
                        'Set VITE_DATAPIZZA_USE_MOCK=false and configure DATAPIZZA_BASE_URL',
                    ],
                    executionTime: Date.now() - startTime,
                    metadata: {
                        timestamp: new Date().toISOString(),
                        dataType: typeof data,
                        processingSteps: ['validation', 'normalization', 'transformation'],
                        chunkCount: mockProcessedData.chunks,
                        embeddingCount: mockProcessedData.embeddings,
                        mock: true,
                    },
                    timestamp: new Date(),
                };
            }
            // Use real DatapizzaClient
            if (!client) {
                throw new Error('DatapizzaClient not provided. Either pass a client to the hook or set VITE_DATAPIZZA_USE_MOCK=true for development.');
            }
            // Simulate progress updates (real API doesn't provide progress)
            const progressInterval = setInterval(() => {
                setProgress((prev) => Math.min(prev + 5, 50));
            }, 1000);
            try {
                const result = await client.processData({
                    data,
                    processingType: options.processingType || 'standard',
                    options: {
                        chunk_size: options.chunkSize || 1000,
                        overlap_size: options.overlapSize || 200,
                        embedding_model: options.embeddingModel || 'default',
                        vector_store: options.vectorStore || 'default',
                    },
                });
                clearInterval(progressInterval);
                setProgress(100);
                return {
                    success: result.success,
                    dataId: result.dataId,
                    processedData: result.processedData,
                    confidenceScore: 1.0,
                    processingType: result.processingType,
                    errors: [],
                    warnings: [],
                    executionTime: Date.now() - startTime,
                    metadata: {
                        timestamp: new Date().toISOString(),
                        dataType: typeof data,
                        processingSteps: ['validation', 'normalization', 'transformation'],
                        ...result.metadata,
                    },
                    timestamp: new Date(),
                };
            }
            catch (apiError) {
                clearInterval(progressInterval);
                setProgress(100);
                throw apiError;
            }
        }
        catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error';
            setError(errorMessage);
            return {
                success: false,
                dataId: `data_${Date.now()}`,
                processedData: null,
                confidenceScore: 0.0,
                processingType: options.processingType || 'standard',
                errors: [errorMessage],
                warnings: [],
                executionTime: Date.now() - startTime,
                metadata: {
                    timestamp: new Date().toISOString(),
                    dataType: typeof data,
                    processingSteps: [],
                },
                timestamp: new Date(),
            };
        }
        finally {
            setIsLoading(false);
            setProgress(0);
        }
    }, [client]);
    return { processData, isLoading, progress, error };
}
//# sourceMappingURL=useDatapizzaProcessing.js.map