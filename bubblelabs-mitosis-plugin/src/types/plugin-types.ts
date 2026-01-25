/**
 * Mitosis Bubble Splitting Plugin Types
 */

export interface MitosisConfig {
  enabled: boolean;
  animationDuration?: number; // in milliseconds
  bounceIntensity?: number; // 0-1 scale
  splitDelay?: number; // delay before bounce in ms
  colorVariation?: number; // 0-1 scale for color variation on split
  rotationIntensity?: number; // 0-1 scale for rotation effect during split
  opacityEffect?: boolean; // whether to include opacity changes during animation
  trailEffect?: boolean; // whether to show motion trails during animation
  easingFunction?: string; // CSS easing function for the animation
  particleEffects?: boolean; // whether to show particle effects during split
}

export interface BubbleNode {
  id: string;
  x: number;
  y: number;
  radius: number;
  color: string;
  label?: string;
  [key: string]: any;
}

export interface SplitAnimationParams {
  parentNode: BubbleNode;
  childNodes: BubbleNode[];
  containerRef: React.RefObject<HTMLDivElement>;
}

export interface EvolutionAnimationParams {
  parentNode: BubbleNode;
  childNodes: BubbleNode[];
  containerRef: React.RefObject<HTMLDivElement>;
  evolutionType?: 'survival-of-fittest' | 'standard' | 'speciation';
  survivorIndices?: number[]; // Indices of child nodes that survive (for survival-of-fittest)
  nextEvolution?: EvolutionAnimationParams; // For chaining evolutions
}

export interface MitosisPluginState {
  config: MitosisConfig;
  isAnimating: boolean;
  enabled: boolean;
  lastAnimationTime: number | null;
}

export interface PerformanceMetrics {
  avgDuration: number;
  activeAnimations: number;
  queuedAnimations: number;
}

export type AnimationPreset = 'default' | 'smooth' | 'dramatic' | 'subtle' | 'fast' | 'custom';

export interface BatchAnimationParams {
  parentNodes: BubbleNode[];
  childNodeGroups: BubbleNode[][];
  containerRef: React.RefObject<HTMLDivElement>;
}

export interface MitosisPlugin {
  initialize(config: MitosisConfig): void;
  triggerMitosisSplit(params: SplitAnimationParams): Promise<void>;
  triggerEvolutionSplit(params: EvolutionAnimationParams): Promise<void>;
  triggerBatchMitosis(params: BatchAnimationParams): Promise<void>;
  updateConfig(config: Partial<MitosisConfig>): void;
  getState(): MitosisPluginState;
  toggleEnabled(): void;
  isEnabled(): boolean;
  cleanup(): void;
  getPerformanceMetrics(): PerformanceMetrics;
  applyPreset(preset: AnimationPreset): void;
}