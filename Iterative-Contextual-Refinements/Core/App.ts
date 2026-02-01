/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { initializeDeepthinkModule, startDeepthinkAnalysisProcess } from '../Deepthink/Deepthink';
import {
    initializeGenerativeUIMode,
    startGenerativeUIProcess
} from '../GenerativeUI/GenerativeUI';
import {
    startContextualProcess
} from '../Contextual/Contextual';
import {
    startAdaptiveDeepthinkProcess
} from '../AdaptiveDeepthink/AdaptiveDeepthinkMode';
import {
    initializeMathSolverMode,
    startMathSolverProcess
} from '../MathSolver';
import { exportConfiguration, handleImportConfiguration } from './ConfigManager';
import {
    updateUIAfterModeChange,
    initializeEvolutionConvergenceButtons
} from '../Refine/WebsiteUI';
import { openDiffModal } from '../Components/DiffModal/DiffModalController';
import {
    initializeAgenticMode,
    startAgenticProcess,
    setAgenticPromptsManager,
} from '../Agentic/Agentic';

import {
    routingManager,
    initializeRouting,
    getSelectedModel,
    getSelectedTemperature,
    getSelectedTopP,
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
    getAutoRefineEnabled,
    hasValidApiKey,
    callAI
} from '../Routing';
import {
    parseJsonSafe,
    cleanTextOutput,
    cleanOutputByType,
    parseJsonSuggestions
} from '../Parsing';
import { globalState } from './State';
import { ApplicationMode } from './Types';
import { updateControlsState } from '../UI/Controls';
import { startReactModeProcess, createAndDownloadReactProjectZip } from '../React/ReactLogic';
import { renderReactModePipeline } from '../React/ReactUI';
import { runPipeline, initPipelines } from '../Refine/WebsiteLogic';
import { renderPipelines } from '../Refine/WebsiteUI';
import { LayoutController } from '../UI/LayoutController';
import { GlobalModals } from '../UI/GlobalModals';
import { startIcrEventBridge } from '../Utils/IcrEventBridge';

export class App {
    public static init() {
        this.initializeGlobalFunctions();
        this.initializeUI();
        this.initializeEventListeners();
        startIcrEventBridge();
        LayoutController.initialize();
        GlobalModals.initialize();
    }

    private static initializeGlobalFunctions() {
        // Make function globally accessible for ReactAgenticIntegration
        (window as any).createAndDownloadReactProjectZip = createAndDownloadReactProjectZip;
        (window as any).renderReactModePipeline = renderReactModePipeline;
    }

    private static initializeUI() {
        // Initialize routing system
        initializeRouting();

        // Refresh providers to update available models
        routingManager.refreshProviders();

        this.initializeCustomPromptTextareas();
        updateUIAfterModeChange(); // Called early to set up initial UI based on default mode

        // Initialize Agentic mode
        initializeAgenticMode();
        // Initialize GenerativeUI mode
        initializeGenerativeUIMode();
        // Initialize MathSolver mode
        const pipelinesContentContainer = document.getElementById('pipelines-content-container');
        if (pipelinesContentContainer) {
            initializeMathSolverMode(pipelinesContentContainer);
        }

        initializeEvolutionConvergenceButtons();

        // Initialize deepthink module with all required dependencies
        initializeDeepthinkModule({
            getAIProvider: () => routingManager.getAIProvider(),
            callGemini: callAI,
            cleanOutputByType,
            parseJsonSuggestions: parseJsonSuggestions as any, // Only for Deepthink strategies
            parseJsonSafe,
            updateControlsState,
            escapeHtml: (str: string) => str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;'),
            getSelectedTemperature,
            getSelectedModel,
            getSelectedTopP,
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
            cleanTextOutput,
            customPromptsDeepthinkState: globalState.customPromptsDeepthinkState,
            tabsNavContainer: document.getElementById('tabs-nav-container'),
            pipelinesContentContainer: document.getElementById('pipelines-content-container'),
            setActiveDeepthinkPipeline: (pipeline: any) => {
                globalState.activeDeepthinkPipeline = pipeline as any;
            }
        });

        // Default to first mode if none specifically checked (e.g. after import or on fresh load)
        const appModeRadios = document.querySelectorAll('input[name="app-mode"]');
        let modeIsAlreadySet = false;
        appModeRadios.forEach(radio => {
            if ((radio as HTMLInputElement).checked) {
                globalState.currentMode = (radio as HTMLInputElement).value as ApplicationMode; // Ensure currentMode reflects HTML state
                modeIsAlreadySet = true;
            }
        });

        if (!modeIsAlreadySet && appModeRadios.length > 0) {
            const firstModeRadio = appModeRadios[0] as HTMLInputElement;
            if (firstModeRadio) {
                firstModeRadio.checked = true;
                globalState.currentMode = firstModeRadio.value as ApplicationMode;
            }
        }
        updateUIAfterModeChange();

        const preloader = document.getElementById('preloader');
        if (preloader) {
            preloader.classList.add('hidden');
        }

        updateControlsState();
    }

    private static initializeEventListeners() {
        const generateButton = document.getElementById('generate-button') as HTMLButtonElement;
        const initialIdeaInput = document.getElementById('initial-idea') as HTMLTextAreaElement;
        const appModeSelector = document.getElementById('app-mode-selector') as HTMLElement;
        const exportConfigButton = document.getElementById('export-config-button') as HTMLButtonElement;
        const importConfigInput = document.getElementById('import-config-input') as HTMLInputElement;

        // Keyboard shortcut: Ctrl+Enter to trigger generation
        document.addEventListener('keydown', (e: KeyboardEvent) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                // Only trigger if not inside a text input/textarea (unless it's the initial-idea)
                const activeElement = document.activeElement;
                const isInputElement = activeElement instanceof HTMLInputElement || 
                                       activeElement instanceof HTMLTextAreaElement;
                const isInitialIdea = activeElement === initialIdeaInput;
                
                if (!isInputElement || isInitialIdea) {
                    e.preventDefault();
                    if (generateButton && !generateButton.disabled) {
                        generateButton.click();
                    }
                }
            }
        });

        if (generateButton) {
            generateButton.addEventListener('click', async () => {
                console.log('Generate button clicked');
                console.log('Current mode:', globalState.currentMode);
                if (!hasValidApiKey()) { // Double check if any provider is configured
                    alert("No providers are configured. Please configure at least one AI provider using the 'Add Providers' button.");
                    return;
                }
                const initialIdea = initialIdeaInput.value.trim();
                if (!initialIdea) {
                    alert("Please enter an idea, premise, or request.");
                    return;
                }

                if (globalState.currentMode === 'deepthink') {
                    console.log('Starting Deepthink process');
                    await startDeepthinkAnalysisProcess(initialIdea, globalState.currentProblemImageBase64, globalState.currentProblemImageMimeType);
                } else if (globalState.currentMode === 'react') {
                    console.log('Starting React process');
                    try {
                        await startReactModeProcess(initialIdea);
                    } catch (e) {
                        console.error('Error starting React process:', e);
                    }
                } else if (globalState.currentMode === 'agentic') {
                    console.log('Starting Agentic process');
                    try {
                        await startAgenticProcess(initialIdea);
                    } catch (e) {
                        console.error('Error starting Agentic process:', e);
                    }
                } else if (globalState.currentMode === 'generativeui') {
                    await startGenerativeUIProcess(initialIdea);
                } else if (globalState.currentMode === 'contextual') {
                    await startContextualProcess(initialIdea, globalState.customPromptsContextualState);
                } else if (globalState.currentMode === 'adaptive-deepthink') {
                    await startAdaptiveDeepthinkProcess(initialIdea, globalState.customPromptsAdaptiveDeepthinkState, globalState.currentProblemImageBase64, globalState.currentProblemImageMimeType);
                } else if (globalState.currentMode === 'mathsolver') {
                    if (globalState.isMathSolverRunning) {
                        alert('MathSolver is already processing a problem. Please wait or cancel the current operation.');
                        return;
                    }
                    console.log('Starting MathSolver process');
                    try {
                        await startMathSolverProcess(initialIdea, {
                            preferredSolver: 'auto',
                            useKnowledgeBase: true,
                            timeout: 300
                        });
                    } catch (e) {
                        console.error('Error starting MathSolver process:', e);
                    }
                } else { // Website mode
                    console.log('Starting Website mode');
                    initPipelines();
                    renderPipelines(); // Fix: Render the pipelines UI before running them
                    console.log('Pipelines initialized:', globalState.pipelinesState.length);
                    const runningPromises = globalState.pipelinesState.map(p => runPipeline(p.id, initialIdea));

                    try {
                        await Promise.allSettled(runningPromises);
                    } finally {
                        globalState.isGenerating = false;
                        updateControlsState();
                    }
                }
            });
        }

        if (appModeSelector) {
            appModeSelector.querySelectorAll('input[name="app-mode"]').forEach(radio => {
                radio.addEventListener('change', (e) => {
                    const newMode = (e.target as HTMLInputElement).value as ApplicationMode;
                    // Track previous mode for cleanup
                    globalState.previousMode = globalState.currentMode;
                    globalState.currentMode = newMode;
                    
                    // Cleanup previous mode if it was MathSolver and we're switching away
                    if (globalState.previousMode === 'mathsolver' && newMode !== 'mathsolver') {
                        import('../MathSolver').then(({ stopMathSolverProcess }) => {
                            stopMathSolverProcess();
                        });
                    }
                    
                    updateUIAfterModeChange();
                });
            });
        }

        if (exportConfigButton) {
            exportConfigButton.addEventListener('click', exportConfiguration);
        }
        if (importConfigInput) {
            importConfigInput.addEventListener('change', handleImportConfiguration);
        }

        // Event delegation for dynamically created "Compare" buttons and "View The Argument" buttons
        const pipelinesContentContainer = document.getElementById('pipelines-content-container');
        if (pipelinesContentContainer) {
            pipelinesContentContainer.addEventListener('click', (event: Event) => {
                const target = event.target as HTMLElement;
                const button = target.closest('.compare-output-button') as HTMLElement | null;
                if (button) {
                    const pipelineId = parseInt(button.dataset.pipelineId || "-1", 10);
                    const iterationNumber = parseInt(button.dataset.iterationNumber || "-1", 10);
                    const contentType = button.dataset.contentType as ('html' | 'text');
                    if (pipelineId !== -1 && iterationNumber !== -1 && (contentType === 'html' || contentType === 'text')) {
                        openDiffModal(pipelineId, iterationNumber, contentType);
                    }
                }
            });
        }

        // Auto-refine event wiring
        const updateAutoRefineStatus = (status: string, progress?: string) => {
            const statusEl = document.getElementById('auto-refine-status-text');
            const progressEl = document.getElementById('auto-refine-progress-text');
            if (statusEl) statusEl.textContent = status;
            if (progressEl) progressEl.textContent = progress || '';
        };

        let autoRefineInProgress = false;

        const runAutoRefine = async (payload?: any) => {
            if (autoRefineInProgress) return;
            if (!getAutoRefineEnabled()) {
                updateAutoRefineStatus('Disabled', 'Enable Auto-refine to run.');
                return;
            }

            const initialIdea = initialIdeaInput.value.trim();
            if (!initialIdea) {
                updateAutoRefineStatus('Idle', 'Enter a request to refine.');
                return;
            }

            autoRefineInProgress = true;
            updateAutoRefineStatus('Running', payload?.reason ? `Reason: ${payload.reason}` : 'Refining...');

            try {
                if (globalState.currentMode === 'deepthink') {
                    await startDeepthinkAnalysisProcess(initialIdea, globalState.currentProblemImageBase64, globalState.currentProblemImageMimeType);
                } else if (globalState.currentMode === 'react') {
                    await startReactModeProcess(initialIdea);
                } else if (globalState.currentMode === 'agentic') {
                    await startAgenticProcess(initialIdea);
                } else if (globalState.currentMode === 'generativeui') {
                    await startGenerativeUIProcess(initialIdea);
                } else if (globalState.currentMode === 'contextual') {
                    await startContextualProcess(initialIdea, globalState.customPromptsContextualState);
                } else if (globalState.currentMode === 'adaptive-deepthink') {
                    await startAdaptiveDeepthinkProcess(initialIdea, globalState.customPromptsAdaptiveDeepthinkState, globalState.currentProblemImageBase64, globalState.currentProblemImageMimeType);
                } else if (globalState.currentMode === 'mathsolver') {
                    if (!globalState.isMathSolverRunning) {
                        await startMathSolverProcess(initialIdea, {
                            preferredSolver: 'auto',
                            useKnowledgeBase: true,
                            timeout: 300
                        });
                    }
                } else {
                    initPipelines();
                    renderPipelines();
                    const runningPromises = globalState.pipelinesState.map(p => runPipeline(p.id, initialIdea));
                    await Promise.allSettled(runningPromises);
                }
                updateAutoRefineStatus('Completed', 'Refinement cycle finished.');
            } catch (error: any) {
                updateAutoRefineStatus('Failed', error?.message || 'Auto-refine failed.');
            } finally {
                autoRefineInProgress = false;
                updateControlsState();
            }
        };

        const scheduleAutoRefine = (payload?: any) => {
            if (globalState.isGenerating) {
                updateAutoRefineStatus('Queued', 'Waiting for current run to finish...');
                setTimeout(() => scheduleAutoRefine(payload), 2000);
                return;
            }
            runAutoRefine(payload);
        };

        window.addEventListener('icr:refinement-needed', (event: Event) => {
            const custom = event as CustomEvent<any>;
            const payload = custom?.detail || {};
            if (payload.auto_refine === false) {
                updateAutoRefineStatus('Idle', 'Auto-refine disabled by analytics.');
                return;
            }
            scheduleAutoRefine(payload);
        });

        window.addEventListener('icr:refinement-progress', (event: Event) => {
            const custom = event as CustomEvent<{ status?: string; message?: string }>;
            updateAutoRefineStatus(custom.detail?.status || 'Running', custom.detail?.message);
        });

        window.addEventListener('icr:refinement-complete', () => {
            updateAutoRefineStatus('Completed', 'Refinement cycle finished.');
        });

        window.addEventListener('icr:refinement-error', (event: Event) => {
            const custom = event as CustomEvent<{ message?: string }>;
            updateAutoRefineStatus('Failed', custom.detail?.message || 'Auto-refine failed.');
        });
    }

    private static initializeCustomPromptTextareas() {
        // Initialize prompts manager in routing system with references to global variables
        routingManager.initializePromptsManager(
            { current: globalState.customPromptsWebsiteState },
            { current: globalState.customPromptsDeepthinkState },
            { current: globalState.customPromptsReactState },
            { current: globalState.customPromptsAgenticState },
            { current: globalState.customPromptsAdaptiveDeepthinkState },
            { current: globalState.customPromptsContextualState },
            { current: globalState.customPromptsMathSolverState }
        );

        // Set up Agentic mode with prompts manager
        const agenticPromptsManager = routingManager.getAgenticPromptsManager();
        if (agenticPromptsManager) {
            setAgenticPromptsManager(agenticPromptsManager);
        }
    }
}
