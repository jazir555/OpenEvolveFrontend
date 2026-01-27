/**
 * MONITORING ALERT WORKFLOW
 *
 * System monitoring with configurable alerts and notification routing.
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema for monitoring alert workflow
 */
declare const MonitoringAlertParamsSchema: z.ZodObject<{
    /**
     * Metrics to monitor
     */
    metrics: z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        type: z.ZodEnum<["counter", "gauge", "histogram"]>;
        value: z.ZodNumber;
        threshold: z.ZodObject<{
            warning: z.ZodOptional<z.ZodNumber>;
            error: z.ZodOptional<z.ZodNumber>;
            critical: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            error?: number | undefined;
            warning?: number | undefined;
            critical?: number | undefined;
        }, {
            error?: number | undefined;
            warning?: number | undefined;
            critical?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        value: number;
        type: "gauge" | "counter" | "histogram";
        name: string;
        threshold: {
            error?: number | undefined;
            warning?: number | undefined;
            critical?: number | undefined;
        };
    }, {
        value: number;
        type: "gauge" | "counter" | "histogram";
        name: string;
        threshold: {
            error?: number | undefined;
            warning?: number | undefined;
            critical?: number | undefined;
        };
    }>, "many">;
    /**
     * Alert configuration
     */
    alertConfig: z.ZodObject<{
        severity: z.ZodEnum<["info", "warning", "error", "critical"]>;
        message: z.ZodString;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        source: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        message: string;
        source: string;
        severity: "info" | "error" | "warning" | "critical";
        tags?: string[] | undefined;
    }, {
        message: string;
        source: string;
        severity: "info" | "error" | "warning" | "critical";
        tags?: string[] | undefined;
    }>;
    /**
     * Notification channels
     */
    notifications: z.ZodObject<{
        slack: z.ZodOptional<z.ZodObject<{
            enabled: z.ZodDefault<z.ZodBoolean>;
            channel: z.ZodOptional<z.ZodString>;
            webhookUrl: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            enabled: boolean;
            channel?: string | undefined;
            webhookUrl?: string | undefined;
        }, {
            channel?: string | undefined;
            enabled?: boolean | undefined;
            webhookUrl?: string | undefined;
        }>>;
        email: z.ZodOptional<z.ZodObject<{
            enabled: z.ZodDefault<z.ZodBoolean>;
            recipients: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            enabled: boolean;
            recipients?: string[] | undefined;
        }, {
            enabled?: boolean | undefined;
            recipients?: string[] | undefined;
        }>>;
        webhook: z.ZodOptional<z.ZodObject<{
            enabled: z.ZodDefault<z.ZodBoolean>;
            url: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            enabled: boolean;
            url?: string | undefined;
        }, {
            url?: string | undefined;
            enabled?: boolean | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        slack?: {
            enabled: boolean;
            channel?: string | undefined;
            webhookUrl?: string | undefined;
        } | undefined;
        webhook?: {
            enabled: boolean;
            url?: string | undefined;
        } | undefined;
        email?: {
            enabled: boolean;
            recipients?: string[] | undefined;
        } | undefined;
    }, {
        slack?: {
            channel?: string | undefined;
            enabled?: boolean | undefined;
            webhookUrl?: string | undefined;
        } | undefined;
        webhook?: {
            url?: string | undefined;
            enabled?: boolean | undefined;
        } | undefined;
        email?: {
            enabled?: boolean | undefined;
            recipients?: string[] | undefined;
        } | undefined;
    }>;
    /**
     * Alert escalation
     */
    escalation: z.ZodOptional<z.ZodObject<{
        enabled: z.ZodDefault<z.ZodBoolean>;
        timeout: z.ZodOptional<z.ZodNumber>;
        escalateTo: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        enabled: boolean;
        timeout?: number | undefined;
        escalateTo?: string[] | undefined;
    }, {
        timeout?: number | undefined;
        enabled?: boolean | undefined;
        escalateTo?: string[] | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    metrics: {
        value: number;
        type: "gauge" | "counter" | "histogram";
        name: string;
        threshold: {
            error?: number | undefined;
            warning?: number | undefined;
            critical?: number | undefined;
        };
    }[];
    alertConfig: {
        message: string;
        source: string;
        severity: "info" | "error" | "warning" | "critical";
        tags?: string[] | undefined;
    };
    notifications: {
        slack?: {
            enabled: boolean;
            channel?: string | undefined;
            webhookUrl?: string | undefined;
        } | undefined;
        webhook?: {
            enabled: boolean;
            url?: string | undefined;
        } | undefined;
        email?: {
            enabled: boolean;
            recipients?: string[] | undefined;
        } | undefined;
    };
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    escalation?: {
        enabled: boolean;
        timeout?: number | undefined;
        escalateTo?: string[] | undefined;
    } | undefined;
}, {
    metrics: {
        value: number;
        type: "gauge" | "counter" | "histogram";
        name: string;
        threshold: {
            error?: number | undefined;
            warning?: number | undefined;
            critical?: number | undefined;
        };
    }[];
    alertConfig: {
        message: string;
        source: string;
        severity: "info" | "error" | "warning" | "critical";
        tags?: string[] | undefined;
    };
    notifications: {
        slack?: {
            channel?: string | undefined;
            enabled?: boolean | undefined;
            webhookUrl?: string | undefined;
        } | undefined;
        webhook?: {
            url?: string | undefined;
            enabled?: boolean | undefined;
        } | undefined;
        email?: {
            enabled?: boolean | undefined;
            recipients?: string[] | undefined;
        } | undefined;
    };
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    escalation?: {
        timeout?: number | undefined;
        enabled?: boolean | undefined;
        escalateTo?: string[] | undefined;
    } | undefined;
}>;
type MonitoringAlertParams = z.input<typeof MonitoringAlertParamsSchema>;
/**
 * Result schema for monitoring alert workflow
 */
declare const MonitoringAlertResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    /**
     * Alert details
     */
    alert: z.ZodOptional<z.ZodObject<{
        alertId: z.ZodString;
        severity: z.ZodEnum<["info", "warning", "error", "critical"]>;
        status: z.ZodEnum<["active", "acknowledged", "resolved", "suppressed"]>;
        message: z.ZodString;
        timestamp: z.ZodDate;
        triggeredBy: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        message: string;
        status: "active" | "acknowledged" | "resolved" | "suppressed";
        timestamp: Date;
        severity: "info" | "error" | "warning" | "critical";
        alertId: string;
        triggeredBy: string[];
    }, {
        message: string;
        status: "active" | "acknowledged" | "resolved" | "suppressed";
        timestamp: Date;
        severity: "info" | "error" | "warning" | "critical";
        alertId: string;
        triggeredBy: string[];
    }>>;
    /**
     * Notification results
     */
    notifications: z.ZodOptional<z.ZodObject<{
        slack: z.ZodOptional<z.ZodObject<{
            sent: z.ZodBoolean;
            channelId: z.ZodOptional<z.ZodString>;
            timestamp: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            sent: boolean;
            timestamp?: string | undefined;
            channelId?: string | undefined;
        }, {
            sent: boolean;
            timestamp?: string | undefined;
            channelId?: string | undefined;
        }>>;
        email: z.ZodOptional<z.ZodObject<{
            sent: z.ZodBoolean;
            recipients: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            sent: boolean;
            recipients?: string[] | undefined;
        }, {
            sent: boolean;
            recipients?: string[] | undefined;
        }>>;
        webhook: z.ZodOptional<z.ZodObject<{
            sent: z.ZodBoolean;
            response: z.ZodOptional<z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            sent: boolean;
            response?: unknown;
        }, {
            sent: boolean;
            response?: unknown;
        }>>;
    }, "strip", z.ZodTypeAny, {
        slack?: {
            sent: boolean;
            timestamp?: string | undefined;
            channelId?: string | undefined;
        } | undefined;
        webhook?: {
            sent: boolean;
            response?: unknown;
        } | undefined;
        email?: {
            sent: boolean;
            recipients?: string[] | undefined;
        } | undefined;
    }, {
        slack?: {
            sent: boolean;
            timestamp?: string | undefined;
            channelId?: string | undefined;
        } | undefined;
        webhook?: {
            sent: boolean;
            response?: unknown;
        } | undefined;
        email?: {
            sent: boolean;
            recipients?: string[] | undefined;
        } | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    alert?: {
        message: string;
        status: "active" | "acknowledged" | "resolved" | "suppressed";
        timestamp: Date;
        severity: "info" | "error" | "warning" | "critical";
        alertId: string;
        triggeredBy: string[];
    } | undefined;
    notifications?: {
        slack?: {
            sent: boolean;
            timestamp?: string | undefined;
            channelId?: string | undefined;
        } | undefined;
        webhook?: {
            sent: boolean;
            response?: unknown;
        } | undefined;
        email?: {
            sent: boolean;
            recipients?: string[] | undefined;
        } | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    alert?: {
        message: string;
        status: "active" | "acknowledged" | "resolved" | "suppressed";
        timestamp: Date;
        severity: "info" | "error" | "warning" | "critical";
        alertId: string;
        triggeredBy: string[];
    } | undefined;
    notifications?: {
        slack?: {
            sent: boolean;
            timestamp?: string | undefined;
            channelId?: string | undefined;
        } | undefined;
        webhook?: {
            sent: boolean;
            response?: unknown;
        } | undefined;
        email?: {
            sent: boolean;
            recipients?: string[] | undefined;
        } | undefined;
    } | undefined;
}>;
type MonitoringAlertResult = z.infer<typeof MonitoringAlertResultSchema>;
export declare class MonitoringAlertWorkflow extends WorkflowBubble<MonitoringAlertParams, MonitoringAlertResult> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        /**
         * Metrics to monitor
         */
        metrics: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            type: z.ZodEnum<["counter", "gauge", "histogram"]>;
            value: z.ZodNumber;
            threshold: z.ZodObject<{
                warning: z.ZodOptional<z.ZodNumber>;
                error: z.ZodOptional<z.ZodNumber>;
                critical: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                error?: number | undefined;
                warning?: number | undefined;
                critical?: number | undefined;
            }, {
                error?: number | undefined;
                warning?: number | undefined;
                critical?: number | undefined;
            }>;
        }, "strip", z.ZodTypeAny, {
            value: number;
            type: "gauge" | "counter" | "histogram";
            name: string;
            threshold: {
                error?: number | undefined;
                warning?: number | undefined;
                critical?: number | undefined;
            };
        }, {
            value: number;
            type: "gauge" | "counter" | "histogram";
            name: string;
            threshold: {
                error?: number | undefined;
                warning?: number | undefined;
                critical?: number | undefined;
            };
        }>, "many">;
        /**
         * Alert configuration
         */
        alertConfig: z.ZodObject<{
            severity: z.ZodEnum<["info", "warning", "error", "critical"]>;
            message: z.ZodString;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            source: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            message: string;
            source: string;
            severity: "info" | "error" | "warning" | "critical";
            tags?: string[] | undefined;
        }, {
            message: string;
            source: string;
            severity: "info" | "error" | "warning" | "critical";
            tags?: string[] | undefined;
        }>;
        /**
         * Notification channels
         */
        notifications: z.ZodObject<{
            slack: z.ZodOptional<z.ZodObject<{
                enabled: z.ZodDefault<z.ZodBoolean>;
                channel: z.ZodOptional<z.ZodString>;
                webhookUrl: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                enabled: boolean;
                channel?: string | undefined;
                webhookUrl?: string | undefined;
            }, {
                channel?: string | undefined;
                enabled?: boolean | undefined;
                webhookUrl?: string | undefined;
            }>>;
            email: z.ZodOptional<z.ZodObject<{
                enabled: z.ZodDefault<z.ZodBoolean>;
                recipients: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                enabled: boolean;
                recipients?: string[] | undefined;
            }, {
                enabled?: boolean | undefined;
                recipients?: string[] | undefined;
            }>>;
            webhook: z.ZodOptional<z.ZodObject<{
                enabled: z.ZodDefault<z.ZodBoolean>;
                url: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                enabled: boolean;
                url?: string | undefined;
            }, {
                url?: string | undefined;
                enabled?: boolean | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            slack?: {
                enabled: boolean;
                channel?: string | undefined;
                webhookUrl?: string | undefined;
            } | undefined;
            webhook?: {
                enabled: boolean;
                url?: string | undefined;
            } | undefined;
            email?: {
                enabled: boolean;
                recipients?: string[] | undefined;
            } | undefined;
        }, {
            slack?: {
                channel?: string | undefined;
                enabled?: boolean | undefined;
                webhookUrl?: string | undefined;
            } | undefined;
            webhook?: {
                url?: string | undefined;
                enabled?: boolean | undefined;
            } | undefined;
            email?: {
                enabled?: boolean | undefined;
                recipients?: string[] | undefined;
            } | undefined;
        }>;
        /**
         * Alert escalation
         */
        escalation: z.ZodOptional<z.ZodObject<{
            enabled: z.ZodDefault<z.ZodBoolean>;
            timeout: z.ZodOptional<z.ZodNumber>;
            escalateTo: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            enabled: boolean;
            timeout?: number | undefined;
            escalateTo?: string[] | undefined;
        }, {
            timeout?: number | undefined;
            enabled?: boolean | undefined;
            escalateTo?: string[] | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        metrics: {
            value: number;
            type: "gauge" | "counter" | "histogram";
            name: string;
            threshold: {
                error?: number | undefined;
                warning?: number | undefined;
                critical?: number | undefined;
            };
        }[];
        alertConfig: {
            message: string;
            source: string;
            severity: "info" | "error" | "warning" | "critical";
            tags?: string[] | undefined;
        };
        notifications: {
            slack?: {
                enabled: boolean;
                channel?: string | undefined;
                webhookUrl?: string | undefined;
            } | undefined;
            webhook?: {
                enabled: boolean;
                url?: string | undefined;
            } | undefined;
            email?: {
                enabled: boolean;
                recipients?: string[] | undefined;
            } | undefined;
        };
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        escalation?: {
            enabled: boolean;
            timeout?: number | undefined;
            escalateTo?: string[] | undefined;
        } | undefined;
    }, {
        metrics: {
            value: number;
            type: "gauge" | "counter" | "histogram";
            name: string;
            threshold: {
                error?: number | undefined;
                warning?: number | undefined;
                critical?: number | undefined;
            };
        }[];
        alertConfig: {
            message: string;
            source: string;
            severity: "info" | "error" | "warning" | "critical";
            tags?: string[] | undefined;
        };
        notifications: {
            slack?: {
                channel?: string | undefined;
                enabled?: boolean | undefined;
                webhookUrl?: string | undefined;
            } | undefined;
            webhook?: {
                url?: string | undefined;
                enabled?: boolean | undefined;
            } | undefined;
            email?: {
                enabled?: boolean | undefined;
                recipients?: string[] | undefined;
            } | undefined;
        };
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        escalation?: {
            timeout?: number | undefined;
            enabled?: boolean | undefined;
            escalateTo?: string[] | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        /**
         * Alert details
         */
        alert: z.ZodOptional<z.ZodObject<{
            alertId: z.ZodString;
            severity: z.ZodEnum<["info", "warning", "error", "critical"]>;
            status: z.ZodEnum<["active", "acknowledged", "resolved", "suppressed"]>;
            message: z.ZodString;
            timestamp: z.ZodDate;
            triggeredBy: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            message: string;
            status: "active" | "acknowledged" | "resolved" | "suppressed";
            timestamp: Date;
            severity: "info" | "error" | "warning" | "critical";
            alertId: string;
            triggeredBy: string[];
        }, {
            message: string;
            status: "active" | "acknowledged" | "resolved" | "suppressed";
            timestamp: Date;
            severity: "info" | "error" | "warning" | "critical";
            alertId: string;
            triggeredBy: string[];
        }>>;
        /**
         * Notification results
         */
        notifications: z.ZodOptional<z.ZodObject<{
            slack: z.ZodOptional<z.ZodObject<{
                sent: z.ZodBoolean;
                channelId: z.ZodOptional<z.ZodString>;
                timestamp: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                sent: boolean;
                timestamp?: string | undefined;
                channelId?: string | undefined;
            }, {
                sent: boolean;
                timestamp?: string | undefined;
                channelId?: string | undefined;
            }>>;
            email: z.ZodOptional<z.ZodObject<{
                sent: z.ZodBoolean;
                recipients: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                sent: boolean;
                recipients?: string[] | undefined;
            }, {
                sent: boolean;
                recipients?: string[] | undefined;
            }>>;
            webhook: z.ZodOptional<z.ZodObject<{
                sent: z.ZodBoolean;
                response: z.ZodOptional<z.ZodUnknown>;
            }, "strip", z.ZodTypeAny, {
                sent: boolean;
                response?: unknown;
            }, {
                sent: boolean;
                response?: unknown;
            }>>;
        }, "strip", z.ZodTypeAny, {
            slack?: {
                sent: boolean;
                timestamp?: string | undefined;
                channelId?: string | undefined;
            } | undefined;
            webhook?: {
                sent: boolean;
                response?: unknown;
            } | undefined;
            email?: {
                sent: boolean;
                recipients?: string[] | undefined;
            } | undefined;
        }, {
            slack?: {
                sent: boolean;
                timestamp?: string | undefined;
                channelId?: string | undefined;
            } | undefined;
            webhook?: {
                sent: boolean;
                response?: unknown;
            } | undefined;
            email?: {
                sent: boolean;
                recipients?: string[] | undefined;
            } | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        alert?: {
            message: string;
            status: "active" | "acknowledged" | "resolved" | "suppressed";
            timestamp: Date;
            severity: "info" | "error" | "warning" | "critical";
            alertId: string;
            triggeredBy: string[];
        } | undefined;
        notifications?: {
            slack?: {
                sent: boolean;
                timestamp?: string | undefined;
                channelId?: string | undefined;
            } | undefined;
            webhook?: {
                sent: boolean;
                response?: unknown;
            } | undefined;
            email?: {
                sent: boolean;
                recipients?: string[] | undefined;
            } | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        alert?: {
            message: string;
            status: "active" | "acknowledged" | "resolved" | "suppressed";
            timestamp: Date;
            severity: "info" | "error" | "warning" | "critical";
            alertId: string;
            triggeredBy: string[];
        } | undefined;
        notifications?: {
            slack?: {
                sent: boolean;
                timestamp?: string | undefined;
                channelId?: string | undefined;
            } | undefined;
            webhook?: {
                sent: boolean;
                response?: unknown;
            } | undefined;
            email?: {
                sent: boolean;
                recipients?: string[] | undefined;
            } | undefined;
        } | undefined;
    }>;
    static readonly shortDescription = "System monitoring with intelligent alerting";
    static readonly longDescription = "\n    Comprehensive system monitoring with configurable alert thresholds and multi-channel notifications.\n\n    Features:\n    - Multi-metric monitoring with customizable thresholds\n    - Severity-based alert classification (info, warning, error, critical)\n    - Multi-channel notifications (Slack, email, webhooks)\n    - Alert escalation with timeout\n    - Alert lifecycle management (active, acknowledged, resolved, suppressed)\n    - Tag-based alert organization\n\n    Use cases:\n    - Infrastructure monitoring\n    - Application performance monitoring\n    - Business metrics alerting\n    - DevOps incident response\n    - SLO/SLA monitoring\n  ";
    static readonly alias = "monitor-alert";
    constructor(params: MonitoringAlertParams, context?: BubbleContext);
    protected performAction(): Promise<MonitoringAlertResult>;
    /**
     * Evaluate metrics against thresholds
     */
    private evaluateMetrics;
    /**
     * Determine alert severity from triggered metrics
     */
    private determineSeverity;
    /**
     * Send notifications to configured channels
     */
    private sendNotifications;
    /**
     * Format Slack message
     */
    private formatSlackMessage;
    /**
     * Set up alert escalation
     */
    private setupEscalation;
    /**
     * Generate alert ID
     */
    private generateAlertId;
}
export {};
//# sourceMappingURL=monitoring-alert.workflow.d.ts.map