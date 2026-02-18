
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ConfigManager - Configuration export/import with StateSerializer support
 * Supports both legacy JSON format and new MessagePack format with compression.
 */

import { globalState } from './State';
import { ExportedConfig } from './Types';
import { getActiveDeepthinkPipeline, setActiveDeepthinkPipelineForImport, renderActiveDeepthinkPipeline } from '../Deepthink/Deepthink';
import { getSolutionPoolVersionsForExport, restoreSolutionPoolVersions } from '../Deepthink/SolutionPool';
import { getActiveAgenticState, setActiveAgenticStateForImport, renderAgenticMode } from '../Agentic/Agentic';
import { getActiveGenerativeUIState, setActiveGenerativeUIStateForImport, renderGenerativeUIMode } from '../GenerativeUI/GenerativeUI';
import { getContextualState, setContextualStateForImport, renderContextualMode } from '../Contextual/Contextual';
import { getAdaptiveDeepthinkState, setAdaptiveDeepthinkStateForImport, renderAdaptiveDeepthinkMode } from '../AdaptiveDeepthink/AdaptiveDeepthinkMode';
import { getActiveMathSolverState, setActiveMathSolverState } from '../MathSolver';
import { renderReactModePipeline } from '../React/ReactUI';
import { renderPipelines, activateTab, updateEvolutionModeDescription, updateUIAfterModeChange } from '../Refine/WebsiteUI';
import { createDefaultCustomPromptsDeepthink } from '../Deepthink/DeepthinkPrompts';
import { defaultCustomPromptsWebsite } from '../Refine/RefinePrompts';
import { createDefaultCustomPromptsReact } from '../React/ReactPrompts';
import { createDefaultCustomPromptsContextual } from '../Contextual/ContextualPrompts';
import { createDefaultCustomPromptsAdaptiveDeepthink } from '../AdaptiveDeepthink/AdaptiveDeepthinkPrompt';
import { AGENTIC_SYSTEM_PROMPT } from '../Agentic/AgenticModePrompt';
import { MATH_SOLVER_SYSTEM_PROMPT } from '../MathSolver/MathSolverPrompts';
import {
    routingManager,
    updateCustomPromptTextareasFromState,
    getSelectedModel,
    getSelectedTemperature,
    getSelectedTopP,
    getSelectedRefinementStages,
    getSelectedStrategiesCount,
    getSelectedSubStrategiesCount,
    getSelectedHypothesisCount,
    getSelectedRedTeamAggressiveness,
    getRefinementEnabled,
    getSkipSubStrategies,
    getDissectedObservationsEnabled,
    getIterativeCorrectionsEnabled,
    getProvideAllSolutionsToCorrectors,
    getPostQualityFilterEnabled,
    getAutoRefineEnabled
} from '../Routing';
import { updateControlsState } from '../UI/Controls';

// StateSerializer imports for new export/import format
import {
    serialize,
    deserialize,
    sanitizeState,
    downloadBlob,
    formatBytes,
} from './StateSerializer';

// DOM Elements Helpers
const getInitialIdeaInput = () => document.getElementById('initial-idea') as HTMLTextAreaElement;

export async function exportConfiguration() {
    const initialIdeaInput = getInitialIdeaInput();
    // Get fresh Deepthink pipeline from DeepthinkCore if in deepthink mode
    const deepthinkPipelineToExport = globalState.currentMode === 'deepthink' ? getActiveDeepthinkPipeline() : globalState.activeDeepthinkPipeline;

    const config: ExportedConfig = {
        currentMode: globalState.currentMode,
        currentEvolutionMode: globalState.currentEvolutionMode,
        initialIdea: initialIdeaInput ? initialIdeaInput.value : '',
        selectedModel: getSelectedModel(),
        selectedOriginalTemperatureIndices: [],
        pipelinesState: globalState.pipelinesState,
        activeDeepthinkPipeline: deepthinkPipelineToExport ?? null,
        activeReactPipeline: globalState.activeReactPipeline ?? null,
        embeddedAgenticState: globalState.activeReactPipeline ? (await import('../React/ReactAgenticIntegration')).getActiveReactAgenticState() : null,
        activeAgenticState: globalState.currentMode === 'agentic' ? getActiveAgenticState() : null,
        activeGenerativeUIState: globalState.currentMode === 'generativeui' ? getActiveGenerativeUIState() : null,
        activeContextualState: globalState.currentMode === 'contextual' ? getContextualState() : null,
        activeAdaptiveDeepthinkState: globalState.currentMode === 'adaptive-deepthink' ? getAdaptiveDeepthinkState() : null,
        activeMathSolverState: globalState.currentMode === 'mathsolver' ? getActiveMathSolverState() : null,
        activePipelineId: globalState.activePipelineId,
        activeDeepthinkProblemTabId: deepthinkPipelineToExport?.activeTabId ?? '',
        globalStatusText: '',
        globalStatusClass: '',
        customPromptsWebsite: globalState.customPromptsWebsiteState,
        customPromptsDeepthinkState: globalState.customPromptsDeepthinkState,
        customPromptsReact: globalState.customPromptsReactState,
        customPromptsAgentic: globalState.customPromptsAgenticState,
        customPromptsAdaptiveDeepthink: globalState.customPromptsAdaptiveDeepthinkState,
        customPromptsContextual: globalState.customPromptsContextualState,
        customPromptsMathSolver: globalState.customPromptsMathSolverState,
        isCustomPromptsOpen: false,
        // Export all model parameters
        modelParameters: {
            temperature: getSelectedTemperature(),
            topP: getSelectedTopP(),
            refinementStages: getSelectedRefinementStages(),
            strategiesCount: getSelectedStrategiesCount(),
            subStrategiesCount: getSelectedSubStrategiesCount(),
            hypothesisCount: getSelectedHypothesisCount(),
            redTeamAggressiveness: getSelectedRedTeamAggressiveness(),
            refinementEnabled: getRefinementEnabled(),
            skipSubStrategies: getSkipSubStrategies(),
            dissectedObservationsEnabled: getDissectedObservationsEnabled(),
            iterativeCorrectionsEnabled: getIterativeCorrectionsEnabled(),
            provideAllSolutionsToCorrectors: getProvideAllSolutionsToCorrectors(),
            postQualityFilterEnabled: getPostQualityFilterEnabled(),
            autoRefineEnabled: getAutoRefineEnabled()
        },
        // Export solution pool versions for evolution view
        solutionPoolVersions: deepthinkPipelineToExport ? getSolutionPoolVersionsForExport(deepthinkPipelineToExport.id) : null
    };

    // Use StateSerializer for compressed MessagePack format (NEW)
    // Falls back to JSON for backward compatibility
    const timestamp = new Date().toISOString().replace(/[:]/g, '-').split('.')[0];
    
    try {
        // Serialize with MessagePack + compression for smaller file size
        const blob = await serialize(config, {
            format: 'msgpack',
            compress: true,
            onProgress: (percent) => console.log(`Export progress: ${percent}%`)
        });
        
        const filename = `iterative-studio-config-${timestamp}.msgpack.gz`;
        downloadBlob(blob, filename);
        
        console.log(`[ConfigManager] Exported configuration: ${formatBytes(blob.size)} (compressed)`);
    } catch (error) {
        console.warn('[ConfigManager] MessagePack export failed, falling back to JSON:', error);
        
        // Fallback to JSON for backward compatibility
        const configString = JSON.stringify(config, null, 2);
        const blob = new Blob([configString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `iterative-studio-config-${timestamp}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`[ConfigManager] Exported configuration (JSON fallback)`);
    }
}

export async function handleImportConfiguration(event: Event) {
    if (globalState.isGenerating) {
        alert("Cannot import configuration while generation is in progress.");
        return;
    }
    const fileInputTarget = event.target as HTMLInputElement;
    if (!fileInputTarget.files || fileInputTarget.files.length === 0) return;
    const file = fileInputTarget.files[0];

    try {
        // Try to deserialize using StateSerializer (supports MessagePack, JSON, compression)
        const imported = await deserialize<any>(file, (percent) => {
            console.log(`[ConfigManager] Import progress: ${percent}%`);
        });
        
        // Handle both versioned and legacy formats
        let configToRestore: any;
        if (imported._version) {
            // New versioned format - use data field
            configToRestore = imported.data;
            console.log(`[ConfigManager] Imported versioned config (v${imported._version})`);
        } else if (imported.data) {
            // StateSerializer wrapped format
            configToRestore = imported.data;
            console.log(`[ConfigManager] Imported serialized config`);
        } else {
            // Legacy JSON format (direct config object)
            configToRestore = imported;
            console.log(`[ConfigManager] Imported legacy JSON config`);
        }
        
        // Sanitize the state (reset processing states, flags, etc.)
        const sanitized = sanitizeState(configToRestore) as ExportedConfig;
        
        // Restore the configuration
        await restoreConfiguration(sanitized);
        
        console.log(`[ConfigManager] Successfully imported configuration from ${file.name}`);
        
    } catch (error) {
        console.error('[ConfigManager] Import failed:', error);
        alert(`Import failed: ${error.message}. The file may be corrupted or in an unsupported format.`);
    }
}

/**
 * Restore configuration after import.
 * This function handles the actual state restoration after deserialization.
 */
async function restoreConfiguration(importedConfig: ExportedConfig) {
    const criticalFields: { key: keyof ExportedConfig; type: string }[] = [
        { key: 'currentMode', type: 'string' },
        { key: 'initialIdea', type: 'string' },
        { key: 'selectedModel', type: 'string' },
        { key: 'customPromptsWebsite', type: 'object' },
        { key: 'customPromptsReact', type: 'object' },
    ];

    if (!importedConfig) {
        throw new Error("Invalid configuration: Root object is missing.");
    }

    for (const field of criticalFields) {
        if (!(field.key in importedConfig) || typeof importedConfig[field.key] !== field.type) {
            if (field.type === 'object' && importedConfig[field.key] === undefined) {
                // Allow customPrompts to be potentially undefined, will use defaults
            } else {
                throw new Error(`Invalid configuration: Missing or malformed critical field '${field.key}'.`);
            }
        }
    }

    globalState.currentMode = importedConfig.currentMode;
    const modeRadio = document.querySelector(`input[name="app-mode"][value="${globalState.currentMode}"]`) as HTMLInputElement;
    if (modeRadio) {
        modeRadio.checked = true;
    }

    if (importedConfig.customPromptsDeepthinkState) {
        globalState.customPromptsDeepthinkState = importedConfig.customPromptsDeepthinkState;
    }
    
    // Restore evolution convergence mode
    if (importedConfig.currentEvolutionMode !== undefined) {
        globalState.currentEvolutionMode = importedConfig.currentEvolutionMode;
        const evolutionButtons = document.querySelectorAll('.evolution-convergence-button');
        evolutionButtons.forEach(button => {
            const buttonValue = (button as HTMLElement).dataset.value;
            if (buttonValue === globalState.currentEvolutionMode) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
        updateEvolutionModeDescription(globalState.currentEvolutionMode);
    }

    const initialIdeaInput = getInitialIdeaInput();
    if (initialIdeaInput) {
        initialIdeaInput.value = importedConfig.initialIdea;
    }

    if (globalState.currentMode !== 'deepthink') {
        globalState.currentProblemImageBase64 = null;
        globalState.currentProblemImageMimeType = null;
    }
    updateUIAfterModeChange();

    // Reinitialize sidebar controls after import
    if ((window as any).reinitializeSidebarControls) {
        (window as any).reinitializeSidebarControls();
    }

    // Restore model parameters AFTER sidebar controls are initialized
    if (importedConfig.modelParameters) {
        const params = importedConfig.modelParameters;
        setTimeout(() => {
            const modelConfig = routingManager.getModelConfigManager();

            if (params.temperature !== undefined) modelConfig.updateParameter('temperature', params.temperature);
            if (params.topP !== undefined) modelConfig.updateParameter('topP', params.topP);
            if (params.refinementStages !== undefined) modelConfig.updateParameter('refinementStages', params.refinementStages);
            if (params.strategiesCount !== undefined) modelConfig.updateParameter('strategiesCount', params.strategiesCount);
            if (params.subStrategiesCount !== undefined) modelConfig.updateParameter('subStrategiesCount', params.subStrategiesCount);
            if (params.hypothesisCount !== undefined) modelConfig.updateParameter('hypothesisCount', params.hypothesisCount);
            if (params.redTeamAggressiveness !== undefined) modelConfig.updateParameter('redTeamAggressiveness', params.redTeamAggressiveness);
            if (params.refinementEnabled !== undefined) modelConfig.updateParameter('refinementEnabled', params.refinementEnabled);
            if (params.skipSubStrategies !== undefined) modelConfig.updateParameter('skipSubStrategies', params.skipSubStrategies);
            if (params.dissectedObservationsEnabled !== undefined) modelConfig.updateParameter('dissectedObservationsEnabled', params.dissectedObservationsEnabled);
                    if (params.iterativeCorrectionsEnabled !== undefined) modelConfig.updateParameter('iterativeCorrectionsEnabled', params.iterativeCorrectionsEnabled);
                    if (params.provideAllSolutionsToCorrectors !== undefined) modelConfig.updateParameter('provideAllSolutionsToCorrectors', params.provideAllSolutionsToCorrectors);
                    if (params.postQualityFilterEnabled !== undefined) modelConfig.updateParameter('postQualityFilterEnabled', params.postQualityFilterEnabled);
                    if (params.autoRefineEnabled !== undefined) modelConfig.updateParameter('autoRefineEnabled', params.autoRefineEnabled);

                    // Sync UI with the restored parameters
                    const modelSelectionUI = routingManager.getModelSelectionUI();
                    if (modelSelectionUI) {
                        modelSelectionUI.syncUIWithParameters();
                    }

                    // Re-render mode-specific UI so imported parameters influence the layout
                    if (globalState.currentMode === 'deepthink' && globalState.activeDeepthinkPipeline) {
                        renderActiveDeepthinkPipeline();
                        if (globalState.activeDeepthinkPipeline.activeTabId) {
                            activateTab(globalState.activeDeepthinkPipeline.activeTabId);
                        }
                    } else if (globalState.currentMode === 'react') {
                        renderReactModePipeline();
                        if (globalState.activeReactPipeline && globalState.activeReactPipeline.activeTabId) {
                            activateTab(globalState.activeReactPipeline.activeTabId);
                        }
                    }
                }, 150); // Slightly longer delay to ensure sidebar controls are ready
            }

            if (globalState.currentMode === 'deepthink') {
                const importedPipeline = importedConfig.activeDeepthinkPipeline;
                if (!importedPipeline) {
                    alert('No Deepthink session found in this export file. The export was created without an active Deepthink analysis.');
                }
                globalState.activeDeepthinkPipeline = importedPipeline ? {
                    ...importedPipeline,
                    isStopRequested: false,
                    status: (importedPipeline.status === 'processing' || importedPipeline.status === 'stopping') ? 'idle' : importedPipeline.status,
                    activeTabId: importedConfig.activeDeepthinkProblemTabId || 'strategic-solver',
                    // Preserve judge results but reset processing states
                    initialStrategies: importedPipeline.initialStrategies?.map(strategy => ({
                        ...strategy,
                        // Reset processing states but preserve completed judge results
                        judgingStatus: strategy.judgingStatus === 'processing' || strategy.judgingStatus === 'retrying' ? 'pending' : strategy.judgingStatus,
                        // Explicitly preserve judge data
                        judgedBestSolution: strategy.judgedBestSolution,
                        judgedBestSubStrategyId: strategy.judgedBestSubStrategyId,
                        judgingRequestPrompt: strategy.judgingRequestPrompt,
                        judgingResponseText: strategy.judgingResponseText,
                        judgingError: strategy.judgingError,
                        // Preserve sub-strategy data including critique fields
                        subStrategies: strategy.subStrategies?.map(subStrategy => ({
                            ...subStrategy,
                            // Reset processing states but preserve completed data
                            selfImprovementStatus: subStrategy.selfImprovementStatus === 'processing' || subStrategy.selfImprovementStatus === 'retrying' ? 'pending' : subStrategy.selfImprovementStatus,
                            solutionCritiqueStatus: subStrategy.solutionCritiqueStatus === 'processing' || subStrategy.solutionCritiqueStatus === 'retrying' ? 'pending' : subStrategy.solutionCritiqueStatus,
                        })) || []
                    })) || [],
                    // Preserve solution critiques and dissected observations synthesis
                    solutionCritiques: Array.isArray(importedPipeline.solutionCritiques) ? importedPipeline.solutionCritiques : [],
                    solutionCritiquesStatus: importedPipeline.solutionCritiquesStatus === 'processing' ? 'pending' : importedPipeline.solutionCritiquesStatus,
                    solutionCritiquesError: importedPipeline.solutionCritiquesError,
                    dissectedObservationsSynthesis: importedPipeline.dissectedObservationsSynthesis,
                    dissectedSynthesisRequestPrompt: importedPipeline.dissectedSynthesisRequestPrompt,
                    dissectedSynthesisStatus: importedPipeline.dissectedSynthesisStatus === 'processing' || importedPipeline.dissectedSynthesisStatus === 'retrying' ? 'pending' : importedPipeline.dissectedSynthesisStatus,
                    dissectedSynthesisError: importedPipeline.dissectedSynthesisError,
                    dissectedSynthesisRetryAttempt: importedPipeline.dissectedSynthesisRetryAttempt,
                    finalJudgedBestSolution: importedPipeline.finalJudgedBestSolution,
                    finalJudgedBestStrategyId: importedPipeline.finalJudgedBestStrategyId,
                    finalJudgingRequestPrompt: importedPipeline.finalJudgingRequestPrompt,
                    finalJudgingResponseText: importedPipeline.finalJudgingResponseText,
                    finalJudgingError: importedPipeline.finalJudgingError,
                    // Preserve solution pool data (with defaults for old exports)
                    structuredSolutionPoolEnabled: importedPipeline.structuredSolutionPoolEnabled ?? false,
                    structuredSolutionPool: importedPipeline.structuredSolutionPool ?? '',
                    structuredSolutionPoolAgents: importedPipeline.structuredSolutionPoolAgents ?? [],
                    structuredSolutionPoolStatus: importedPipeline.structuredSolutionPoolStatus === 'processing' ? 'pending' : (importedPipeline.structuredSolutionPoolStatus ?? 'pending'),
                    structuredSolutionPoolError: importedPipeline.structuredSolutionPoolError ?? undefined,
                } : null;
                globalState.activePipelineId = null;

                // Sync the imported pipeline with the Deepthink module
                setActiveDeepthinkPipelineForImport(globalState.activeDeepthinkPipeline);

                // Restore solution pool versions for evolution view
                if (globalState.activeDeepthinkPipeline && importedConfig.solutionPoolVersions && importedConfig.solutionPoolVersions.length > 0) {
                    restoreSolutionPoolVersions(globalState.activeDeepthinkPipeline.id, importedConfig.solutionPoolVersions);
                }

                // Render pipeline - but if modelParameters exist, let setTimeout handle it for proper timing
                if (!importedConfig.modelParameters) {
                    renderActiveDeepthinkPipeline();
                    if (globalState.activeDeepthinkPipeline && globalState.activeDeepthinkPipeline.activeTabId) {
                        activateTab(globalState.activeDeepthinkPipeline.activeTabId);
                    }
                }
            } else if (globalState.currentMode === 'react') {
                globalState.activeReactPipeline = importedConfig.activeReactPipeline ? {
                    ...importedConfig.activeReactPipeline,
                    isStopRequested: false,
                    status: (importedConfig.activeReactPipeline.status === 'orchestrating' || importedConfig.activeReactPipeline.status === 'agentic_orchestrating' || importedConfig.activeReactPipeline.status === 'processing_workers' || importedConfig.activeReactPipeline.status === 'stopping') ? 'idle' : importedConfig.activeReactPipeline.status,
                } : null;
                globalState.activePipelineId = null;

                // Restore embedded agentic state if available
                if (importedConfig.embeddedAgenticState) {
                    import('../React/ReactAgenticIntegration').then(({ setActiveReactAgenticStateForImport }) => {
                        setActiveReactAgenticStateForImport(importedConfig.embeddedAgenticState);
                        // Re-render AFTER state is restored
                        renderReactModePipeline();

                        // Restore the active tab if available
                        if (globalState.activeReactPipeline && globalState.activeReactPipeline.activeTabId) {
                            activateTab(globalState.activeReactPipeline.activeTabId);
                        }
                    });
                } else {
                    // Re-render immediately if no agentic state
                    renderReactModePipeline();
                    if (globalState.activeReactPipeline && globalState.activeReactPipeline.activeTabId) {
                        activateTab(globalState.activeReactPipeline.activeTabId);
                    }
                }

                // Restore the active tab if available
                if (globalState.activeReactPipeline && globalState.activeReactPipeline.activeTabId) {
                    activateTab(globalState.activeReactPipeline.activeTabId);
                }
            } else if (globalState.currentMode === 'agentic') {
                // Clear other mode states for Agentic mode
                globalState.pipelinesState = [];
                globalState.activeDeepthinkPipeline = null;
                globalState.activeReactPipeline = null;
                globalState.activePipelineId = null;

                // Restore Agentic state first if available
                if (importedConfig.activeAgenticState) {
                    setActiveAgenticStateForImport(importedConfig.activeAgenticState);
                }

                // Render Agentic mode UI (after state is restored)
                renderAgenticMode();
            } else if (globalState.currentMode === 'generativeui') {
                // Import GenerativeUI state
                if (importedConfig.activeGenerativeUIState) {
                    setActiveGenerativeUIStateForImport(importedConfig.activeGenerativeUIState);
                }
                renderGenerativeUIMode();
            } else if (globalState.currentMode === 'contextual') {
                // Import Contextual state
                globalState.pipelinesState = [];
                globalState.activeReactPipeline = null;
                globalState.activeDeepthinkPipeline = null;
                globalState.activePipelineId = null;

                // Restore Contextual state if available
                if (importedConfig.activeContextualState) {
                    setContextualStateForImport(importedConfig.activeContextualState);
                }

                renderContextualMode();
            } else if (globalState.currentMode === 'adaptive-deepthink') {
                // Import Adaptive Deepthink state
                globalState.pipelinesState = [];
                globalState.activeReactPipeline = null;
                globalState.activeDeepthinkPipeline = null;
                globalState.activePipelineId = null;

                // Restore Adaptive Deepthink state if available
                if (importedConfig.activeAdaptiveDeepthinkState) {
                    setAdaptiveDeepthinkStateForImport(importedConfig.activeAdaptiveDeepthinkState);
                }

                renderAdaptiveDeepthinkMode();
            } else if (globalState.currentMode === 'mathsolver') {
                // Import MathSolver state
                globalState.pipelinesState = [];
                globalState.activeReactPipeline = null;
                globalState.activeDeepthinkPipeline = null;
                globalState.activePipelineId = null;

                // Restore MathSolver state if available (with validation)
                if (importedConfig.activeMathSolverState && 
                    typeof importedConfig.activeMathSolverState === 'object') {
                    try {
                        setActiveMathSolverState(importedConfig.activeMathSolverState);
                    } catch (error) {
                        console.warn('[ConfigManager] Failed to restore MathSolver state:', error);
                        // Continue without state - UI will show empty state
                    }
                }

                // Re-render MathSolver UI
                const { rehydrateMathSolverUI } = await import('../MathSolver');
                rehydrateMathSolverUI();
            } else { // Website mode               
                // Restore website mode pipelines state
                globalState.pipelinesState = importedConfig.pipelinesState ? importedConfig.pipelinesState.map(pipeline => ({
                    ...pipeline,
                    isStopRequested: false,
                    status: (pipeline.status === 'running' || pipeline.status === 'stopping') ? 'stopped' : pipeline.status,
                    iterations: pipeline.iterations.map(iteration => ({
                        ...iteration,
                        status: (iteration.status === 'processing' || iteration.status === 'retrying') ? 'completed' : iteration.status,
                    }))
                })) : [];
                globalState.activePipelineId = importedConfig.activePipelineId;

                // Re-render the pipelines UI
                renderPipelines();
            }


            globalState.customPromptsWebsiteState = importedConfig.customPromptsWebsite ? JSON.parse(JSON.stringify(importedConfig.customPromptsWebsite)) : JSON.parse(JSON.stringify(defaultCustomPromptsWebsite));

            const importedDeepthinkPrompts = importedConfig.customPromptsDeepthinkState || createDefaultCustomPromptsDeepthink();
            globalState.customPromptsDeepthinkState = JSON.parse(JSON.stringify(importedDeepthinkPrompts));

            const importedReactPrompts = importedConfig.customPromptsReact || createDefaultCustomPromptsReact();
            globalState.customPromptsReactState = JSON.parse(JSON.stringify(importedReactPrompts));

            const importedAgenticPrompts = importedConfig.customPromptsAgentic || { systemPrompt: AGENTIC_SYSTEM_PROMPT };
            globalState.customPromptsAgenticState = JSON.parse(JSON.stringify(importedAgenticPrompts));

            const importedAdaptiveDeepthinkPrompts = importedConfig.customPromptsAdaptiveDeepthink || createDefaultCustomPromptsAdaptiveDeepthink();
            globalState.customPromptsAdaptiveDeepthinkState = JSON.parse(JSON.stringify(importedAdaptiveDeepthinkPrompts));

            const importedContextualPrompts = importedConfig.customPromptsContextual || createDefaultCustomPromptsContextual();
            globalState.customPromptsContextualState = JSON.parse(JSON.stringify(importedContextualPrompts));

            const importedMathSolverPrompts = importedConfig.customPromptsMathSolver || { systemPrompt: MATH_SOLVER_SYSTEM_PROMPT };
            globalState.customPromptsMathSolverState = JSON.parse(JSON.stringify(importedMathSolverPrompts));

            updateCustomPromptTextareasFromState();

            updateControlsState();
        } catch (error: any) {
            // Removed console.error
        } finally {
            if (fileInputTarget) fileInputTarget.value = '';
        }
    };
    reader.onerror = () => {
        if (fileInputTarget) fileInputTarget.value = '';
    };
    reader.readAsText(file);
}
