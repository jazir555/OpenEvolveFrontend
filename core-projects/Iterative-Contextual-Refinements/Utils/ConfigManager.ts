
import { globalState } from '../Core/State';
import { ExportedConfig } from '../Core/Types';
import { getSelectedModel, getSelectedTemperature, getSelectedTopP, getSelectedRefinementStages, getSelectedStrategiesCount, getSelectedSubStrategiesCount, getSelectedHypothesisCount, getSelectedRedTeamAggressiveness, getRefinementEnabled, getSkipSubStrategies, getDissectedObservationsEnabled, getIterativeCorrectionsEnabled, getProvideAllSolutionsToCorrectors, getPostQualityFilterEnabled, getAutoRefineEnabled, routingManager } from '../Routing';
import { getSolutionPoolVersionsForExport, restoreSolutionPoolVersions } from '../Deepthink/SolutionPool';
import { updateUIAfterModeChange } from '../UI/UIManager';
import { updateCustomPromptTextareasFromState } from '../Routing';
import { updateControlsState } from '../UI/Controls';
import { renderReactModePipeline } from '../React/ReactUI';
import { renderDeepthinkConfigPanelInContainer } from '../Deepthink/DeepthinkConfigPanel';

export async function exportConfiguration() {
    const { currentMode, currentEvolutionMode, pipelinesState, activeDeepthinkPipeline, activeReactPipeline, customPromptsWebsiteState, customPromptsDeepthinkState, customPromptsReactState, customPromptsAgenticState, customPromptsAdaptiveDeepthinkState, customPromptsContextualState, activePipelineId, currentProblemImageBase64, currentProblemImageMimeType } = globalState;

    const initialIdeaInput = document.getElementById('initial-idea') as HTMLTextAreaElement;
    const globalStatusText = document.getElementById('global-status-text');

    // Deepthink specific export logic
    let deepthinkPipelineToExport = activeDeepthinkPipeline;
    if (deepthinkPipelineToExport) {
        // Ensure we export the image data if present
        if (currentProblemImageBase64) {
            deepthinkPipelineToExport.challengeImageBase64 = currentProblemImageBase64;
            deepthinkPipelineToExport.challengeImageMimeType = currentProblemImageMimeType || undefined;
        }
    }

    const config: ExportedConfig = {
        currentMode,
        currentEvolutionMode,
        initialIdea: initialIdeaInput.value,
        selectedModel: getSelectedModel(),
        selectedOriginalTemperatureIndices: pipelinesState.map(p => p.originalTemperatureIndex),
        pipelinesState,
        activeDeepthinkPipeline: deepthinkPipelineToExport ?? null,
        activeReactPipeline: activeReactPipeline ?? null,
        embeddedAgenticState: (window as any).__reactAgenticState || null,
        activeAgenticState: (window as any).__agenticState || null,
        activeGenerativeUIState: (window as any).__generativeUIState || null,
        activeContextualState: (window as any).__contextualState || null,
        activeAdaptiveDeepthinkState: (window as any).__adaptiveDeepthinkState || null,
        activePipelineId,
        activeDeepthinkProblemTabId: activeDeepthinkPipeline?.activeTabId,
        globalStatusText: globalStatusText?.textContent || '',
        globalStatusClass: globalStatusText?.className || '',
        customPromptsWebsite: customPromptsWebsiteState,
        customPromptsDeepthinkState: customPromptsDeepthinkState,
        customPromptsReact: customPromptsReactState,
        customPromptsAgentic: customPromptsAgenticState,
        customPromptsAdaptiveDeepthink: customPromptsAdaptiveDeepthinkState,
        customPromptsContextual: customPromptsContextualState,
        isCustomPromptsOpen: globalState.isCustomPromptsOpen,
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
        solutionPoolVersions: deepthinkPipelineToExport ? getSolutionPoolVersionsForExport(deepthinkPipelineToExport.id) : null
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const { downloadFile } = await import('../Components/ActionButton');
    downloadFile(blob as any, `iterative-studio-config-${Date.now()}.json`, 'application/json');
}

export async function handleImportConfiguration(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const file = input.files[0];
    const reader = new FileReader();

    reader.onload = async (e) => {
        const result = e.target?.result as string;
        try {
            const importedConfig = JSON.parse(result) as ExportedConfig;

            if (!importedConfig.currentMode || !importedConfig.pipelinesState) {
                throw new Error("Invalid configuration file format.");
            }

            globalState.currentMode = importedConfig.currentMode;
            if (importedConfig.currentEvolutionMode) globalState.currentEvolutionMode = importedConfig.currentEvolutionMode;

            const initialIdeaInput = document.getElementById('initial-idea') as HTMLTextAreaElement;
            if (initialIdeaInput) initialIdeaInput.value = importedConfig.initialIdea || '';

            updateUIAfterModeChange();

            // Restore model parameters if available
            if (importedConfig.modelParameters) {
                const params = importedConfig.modelParameters;
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

                const modelSelectionUI = routingManager.getModelSelectionUI();
                if (modelSelectionUI) {
                    modelSelectionUI.syncUIWithParameters();
                }
            }

            globalState.pipelinesState = importedConfig.pipelinesState;
            globalState.activePipelineId = importedConfig.activePipelineId;

            if (importedConfig.activeDeepthinkPipeline) {
                globalState.activeDeepthinkPipeline = importedConfig.activeDeepthinkPipeline;
                // Restore Deepthink specific state
                if (importedConfig.activeDeepthinkPipeline.challengeImageBase64) {
                    globalState.currentProblemImageBase64 = importedConfig.activeDeepthinkPipeline.challengeImageBase64;
                    globalState.currentProblemImageMimeType = importedConfig.activeDeepthinkPipeline.challengeImageMimeType || null;
                }

                // Restore solution pool versions
                if (importedConfig.solutionPoolVersions) {
                    restoreSolutionPoolVersions(importedConfig.activeDeepthinkPipeline.id, importedConfig.solutionPoolVersions);
                }
            }

            if (importedConfig.activeReactPipeline) {
                globalState.activeReactPipeline = importedConfig.activeReactPipeline;
            }

            // Restore Agentic state
            if (importedConfig.activeAgenticState) {
                (window as any).__importedAgenticState = importedConfig.activeAgenticState;
            }
            if (importedConfig.embeddedAgenticState) {
                (window as any).__importedReactAgenticState = importedConfig.embeddedAgenticState;
            }

            // Restore other states...

            // Restore custom prompts
            if (importedConfig.customPromptsWebsite) globalState.customPromptsWebsiteState = importedConfig.customPromptsWebsite;
            if (importedConfig.customPromptsDeepthinkState) globalState.customPromptsDeepthinkState = importedConfig.customPromptsDeepthinkState;
            if (importedConfig.customPromptsReact) globalState.customPromptsReactState = importedConfig.customPromptsReact;
            if (importedConfig.customPromptsAgentic) globalState.customPromptsAgenticState = importedConfig.customPromptsAgentic;
            if (importedConfig.customPromptsAdaptiveDeepthink) globalState.customPromptsAdaptiveDeepthinkState = importedConfig.customPromptsAdaptiveDeepthink;
            if (importedConfig.customPromptsContextual) globalState.customPromptsContextualState = importedConfig.customPromptsContextual;

            updateCustomPromptTextareasFromState();
            updateControlsState();

            // Re-render
            if (globalState.currentMode === 'react') {
                renderReactModePipeline();
            } else if (globalState.currentMode === 'deepthink') {
                const pipelinesContentContainer = document.getElementById('pipelines-content-container');
                renderDeepthinkConfigPanelInContainer(pipelinesContentContainer);
                // We might need to call setActiveDeepthinkPipelineForImport if it exists
                const { setActiveDeepthinkPipelineForImport } = await import('../Deepthink/Deepthink');
                if (globalState.activeDeepthinkPipeline) {
                    setActiveDeepthinkPipelineForImport(globalState.activeDeepthinkPipeline);
                }
            } else {
                // renderPipelines is called by updateUIAfterModeChange but we updated state after that.
                // So call it again?
                // updateUIAfterModeChange calls renderPipelines at the end.
                // But we called updateUIAfterModeChange BEFORE setting pipelinesState.
                // So we should call it again or call renderPipelines directly.
                const { renderPipelines } = await import('../UI/UIManager');
                renderPipelines();
            }

        } catch (error: any) {
            alert(`Failed to import configuration: ${error.message}`);
        } finally {
            input.value = '';
        }
    };
    reader.readAsText(file);
}
