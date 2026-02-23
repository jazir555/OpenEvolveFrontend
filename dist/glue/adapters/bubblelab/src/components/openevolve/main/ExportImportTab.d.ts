import React from "react";
interface OpenEvolveAppState {
    protocolText: string;
    evolutionRunning: boolean;
    adversarialRunning: boolean;
    evolutionHistory: any[];
    adversarialResults: any;
    evolutionCurrentBest: string;
    evolutionStatusMessage: string;
    adversarialStatusMessage: string;
    evolutionBestScore: number;
}
interface ExportImportTabProps {
    state: OpenEvolveAppState;
    updateState: (updates: Partial<OpenEvolveAppState>) => void;
}
export declare const ExportImportTab: React.FC<ExportImportTabProps>;
export {};
//# sourceMappingURL=ExportImportTab.d.ts.map