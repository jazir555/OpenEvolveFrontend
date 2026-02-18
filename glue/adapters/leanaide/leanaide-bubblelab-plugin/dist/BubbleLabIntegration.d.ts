/**
 * BubbleLab UI Integration for LeanAide Autoformalization System
 *
 * This module provides the complete integration of the LeanAide autoformalization system
 * with predictive analytics into the BubbleLab UI as a comprehensive plugin system.
 */
import React from 'react';
export interface BubbleLabIntegrationProps {
    serverUrl?: string;
    apiKey?: string;
    enableAnalytics?: boolean;
    enablePredictiveFlagging?: boolean;
    enableKnowledgeGraph?: boolean;
    className?: string;
}
export declare const BubbleLabLeanAideIntegration: React.FC<BubbleLabIntegrationProps>;
export declare const BubbleLabLeanAideIntegrationLazy: React.FC<BubbleLabIntegrationProps>;
export declare const registerBubbleLabIntegration: () => LeanAidePluginInterface;
export { BubbleLabLeanAideIntegration, BubbleLabLeanAideIntegrationLazy };
export default BubbleLabLeanAideIntegration;
