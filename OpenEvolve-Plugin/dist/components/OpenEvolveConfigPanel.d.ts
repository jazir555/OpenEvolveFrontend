import { default as React } from 'react';
import { OpenEvolvePlugin, OpenEvolvePluginState } from '../types/plugin-types';
interface OpenEvolveConfigPanelProps {
    plugin?: OpenEvolvePlugin;
    onConfigChange?: (config: OpenEvolvePluginState) => void;
}
export declare const OpenEvolveConfigPanel: React.FC<OpenEvolveConfigPanelProps>;
export default OpenEvolveConfigPanel;
