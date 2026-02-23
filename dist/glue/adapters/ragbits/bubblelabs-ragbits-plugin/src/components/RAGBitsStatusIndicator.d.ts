import React from 'react';
import type { RAGBitsStatusIndicatorProps } from '../types/plugin-types';
interface StatusIndicatorProps extends RAGBitsStatusIndicatorProps {
    status: 'idle' | 'initializing' | 'ready' | 'error' | 'busy';
}
export declare const RAGBitsStatusIndicator: React.FC<StatusIndicatorProps>;
export {};
//# sourceMappingURL=RAGBitsStatusIndicator.d.ts.map