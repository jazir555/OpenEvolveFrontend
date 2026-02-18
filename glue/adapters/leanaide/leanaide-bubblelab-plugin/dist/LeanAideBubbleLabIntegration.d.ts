/**
 * Main Integration Component for LeanAide Autoformalization with BubbleLab Analytics
 *
 * This component provides the complete integration between LeanAide's autoformalization system
 * and BubbleLab's analytics platform, offering a comprehensive dashboard for monitoring
 * and managing the autoformalization process.
 */
import { AnalyticsDashboard, EnhancedLeanAideVerification, KnowledgeGraphIntegration, useAutoformalizationAnalytics } from './integration/autoformalizationAnalytics';
export interface LeanAideBubbleLabIntegrationProps {
    className?: string;
}
export declare function LeanAideBubbleLabIntegration({ className }: LeanAideBubbleLabIntegrationProps): any;
export { AnalyticsDashboard, EnhancedLeanAideVerification, KnowledgeGraphIntegration, useAutoformalizationAnalytics };
export default LeanAideBubbleLabIntegration;
