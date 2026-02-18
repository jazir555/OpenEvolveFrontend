// Datapizza Configuration Hook
// React hook for managing Datapizza plugin configuration
import { useState } from 'react';
import { DEFAULT_DATAPIZZA_CONFIG } from '../types/plugin-types';
export function useDatapizzaConfig() {
    const [config, setConfig] = useState(DEFAULT_DATAPIZZA_CONFIG);
    const updateConfig = (newConfig) => {
        setConfig(prev => ({ ...prev, ...newConfig }));
    };
    return [config, updateConfig];
}
//# sourceMappingURL=useDatapizzaConfig.js.map