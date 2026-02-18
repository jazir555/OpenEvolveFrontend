import React from 'react';
import { type AutoformalizationConfig } from './integration/autoformalizationAnalytics';
import { type LeanAidePluginInterface } from './PluginInterface';
export interface BubbleLabIntegrationProps extends AutoformalizationConfig {
    className?: string;
}
export type LeanAideBubbleLabIntegrationProps = BubbleLabIntegrationProps;
export declare const BubbleLabLeanAideIntegration: React.FC<BubbleLabIntegrationProps>;
export declare const BubbleLabLeanAideIntegrationLazy: React.FC<BubbleLabIntegrationProps>;
export declare const registerBubbleLabIntegration: () => LeanAidePluginInterface;
export default BubbleLabLeanAideIntegration;
