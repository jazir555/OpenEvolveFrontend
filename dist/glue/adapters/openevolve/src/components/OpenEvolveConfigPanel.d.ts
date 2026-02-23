/**
 * OpenEvolve Configuration Panel
 *
 * Comprehensive React component for configuring OpenEvolve functionality
 */
import React from 'react';
import { OpenEvolvePlugin, OpenEvolvePluginState } from '../types/plugin-types';
interface OpenEvolveConfigPanelProps {
    plugin?: OpenEvolvePlugin;
    onConfigChange?: (config: OpenEvolvePluginState) => void;
}
export declare const OpenEvolveConfigPanel: React.FC<OpenEvolveConfigPanelProps>;
export {};
//# sourceMappingURL=OpenEvolveConfigPanel.d.ts.map