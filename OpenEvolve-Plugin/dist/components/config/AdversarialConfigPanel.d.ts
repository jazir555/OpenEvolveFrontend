import { default as React } from 'react';
export interface AdversarialConfig {
    attackStrategy: 'pgd' | 'fgsm' | 'cw' | 'deepfool' | 'boundary' | 'genetic';
    attackStrength: number;
    stepSize: number;
    numSteps: number;
    targetedAttack: boolean;
    targetConfidence: number;
    randomTargets: boolean;
    redTeamSize: number;
    redTeamStrategy: 'coordinated' | 'independent' | 'competitive';
    redTeamCommunication: boolean;
    blueTeamSize: number;
    blueTeamStrategy: 'static' | 'adaptive' | 'proactive';
    blueTeamLearning: boolean;
    adversarialTraining: boolean;
    inputSanitization: boolean;
    outputValidation: boolean;
    anomalyDetection: boolean;
    defenseDiversity: number;
    maxRounds: number;
    roundTimeout: number;
    victoryCondition: 'score' | 'survival' | 'objective';
    victoryThreshold: number;
    successRateThreshold: number;
    robustnessScore: number;
    coverageTarget: number;
    transferAttack: boolean;
    ensembleAttack: boolean;
    queryEfficiency: boolean;
    maxQueries: number;
    contentType: 'code' | 'text' | 'design' | 'strategy' | 'all';
}
interface AdversarialConfigPanelProps {
    config: AdversarialConfig;
    onConfigChange: (config: AdversarialConfig) => void;
}
export declare const AdversarialConfigPanel: React.FC<AdversarialConfigPanelProps>;
export default AdversarialConfigPanel;
