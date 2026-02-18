import { DatapizzaPipelineResult } from '../types/plugin-types';
import { DatapizzaClient } from '../services/DatapizzaClient';
export interface DatapizzaPipelinePanelProps {
    /** Data source to process */
    dataSource: string;
    /** Optional initial pipeline type */
    initialPipelineType?: string;
    /** Callback with pipeline result */
    onResult: (result: DatapizzaPipelineResult) => void;
    /** Callback when panel is closed */
    onClose: () => void;
    /** Show debug information */
    showDebug?: boolean;
    /** Optional DatapizzaClient instance */
    client?: DatapizzaClient;
}
export declare function DatapizzaPipelinePanel({ dataSource, initialPipelineType, onResult, onClose, showDebug, client }: DatapizzaPipelinePanelProps): any;
