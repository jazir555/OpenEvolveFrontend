import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

/**
 * LeanAide proof status
 */
export type ProofStatus = 'idle' | 'generating' | 'verifying' | 'completed' | 'failed';

/**
 * LeanAide model configuration
 */
export interface LeanAideModelConfig {
  provider: string;
  model: string;
  api_key: string;
  temperature: number;
}

/**
 * Lean 4 code output
 */
export interface LeanCodeOutput {
  code: string;
  language: string;
  structured_output?: any;
}

/**
 * Verification result
 */
export interface VerificationResult {
  success: boolean;
  output: string;
  errors: string[];
  warnings: string[];
  elapsed_time: number;
}

/**
 * LeanAide state interface
 */
interface LeanAideState {
  // Input
  theorem: string;
  proofAttempt: string;

  // Output
  generatedProof: LeanCodeOutput | null;
  verificationResult: VerificationResult | null;

  // Configuration
  modelConfig: LeanAideModelConfig;
  selectedModels: {
    leanaide: string;
    text: string;
    image: string;
  };

  // API configuration
  apiHost: string;
  apiPort: number;

  // UI state
  status: ProofStatus;
  isLoading: boolean;
  error: string | null;
  activeTab: string;

  // Benchmark state
  benchmarkRunning: boolean;
  benchmarkProgress: number;
  benchmarkResults: any[] | null;

  // Actions
  setTheorem: (theorem: string) => void;
  setProofAttempt: (proof: string) => void;

  setGeneratedProof: (proof: LeanCodeOutput) => void;
  setVerificationResult: (result: VerificationResult) => void;

  setModelConfig: (config: Partial<LeanAideModelConfig>) => void;
  setSelectedModel: (type: 'leanaide' | 'text' | 'image', model: string) => void;

  setApiConfig: (host: string, port: number) => void;

  setStatus: (status: ProofStatus) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveTab: (tab: string) => void;

  setBenchmarkRunning: (running: boolean) => void;
  setBenchmarkProgress: (progress: number) => void;
  setBenchmarkResults: (results: any[]) => void;

  clearOutputs: () => void;
  reset: () => void;
}

/**
 * Default model configuration
 */
const defaultModelConfig: LeanAideModelConfig = {
  provider: 'openai',
  model: 'gpt-4',
  api_key: '',
  temperature: 0.7,
};

/**
 * Default API configuration
 */
const defaultApiHost = 'localhost';
const defaultApiPort = 8000;

/**
 * LeanAide store
 */
export const useLeanAideStore = create<LeanAideState>()(
  devtools(
    persist(
      (set, get) => ({
        theorem: '',
        proofAttempt: '',
        generatedProof: null,
        verificationResult: null,
        modelConfig: defaultModelConfig,
        selectedModels: {
          leanaide: 'gpt-4',
          text: 'gpt-3.5-turbo',
          image: 'gpt-4-vision-preview',
        },
        apiHost: defaultApiHost,
        apiPort: defaultApiPort,
        status: 'idle',
        isLoading: false,
        error: null,
        activeTab: 'home',
        benchmarkRunning: false,
        benchmarkProgress: 0,
        benchmarkResults: null,

        setTheorem: (theorem) => set({ theorem }),

        setProofAttempt: (proof) => set({ proofAttempt: proof }),

        setGeneratedProof: (proof) => set({ generatedProof: proof }),

        setVerificationResult: (result) => set({ verificationResult: result }),

        setModelConfig: (config) => set((state) => ({
          modelConfig: { ...state.modelConfig, ...config },
        })),

        setSelectedModel: (type, model) => set((state) => ({
          selectedModels: { ...state.selectedModels, [type]: model },
        })),

        setApiConfig: (host, port) => set({
          apiHost: host,
          apiPort: port,
        }),

        setStatus: (status) => set({ status }),

        setLoading: (loading) => set({ isLoading: loading }),

        setError: (error) => set({ error }),

        setActiveTab: (tab) => set({ activeTab: tab }),

        setBenchmarkRunning: (running) => set({ benchmarkRunning: running }),

        setBenchmarkProgress: (progress) => set({ benchmarkProgress: progress }),

        setBenchmarkResults: (results) => set({ benchmarkResults: results }),

        clearOutputs: () => set({
          generatedProof: null,
          verificationResult: null,
          error: null,
        }),

        reset: () => set({
          theorem: '',
          proofAttempt: '',
          generatedProof: null,
          verificationResult: null,
          status: 'idle',
          error: null,
        }),
      }),
      {
        name: 'leanaide-storage',
        partialize: (state) => ({
          modelConfig: state.modelConfig,
          selectedModels: state.selectedModels,
          apiHost: state.apiHost,
          apiPort: state.apiPort,
        }),
      }
    ),
    { name: 'LeanAideStore' }
  )
);
