// Datapizza State Hook
// React hook for accessing Datapizza plugin state
import { useState } from 'react';
import { DEFAULT_DATAPIZZA_CONFIG } from '../types/plugin-types';
export function useDatapizzaState() {
    const [state] = useState({
        ...DEFAULT_DATAPIZZA_CONFIG,
        status: 'idle',
        operationHistory: [],
        statistics: {
            totalOperations: 0,
            successfulOperations: 0,
            failedOperations: 0,
            averageProcessingTime: 0
        }
    });
    return state;
}
//# sourceMappingURL=useDatapizzaState.js.map