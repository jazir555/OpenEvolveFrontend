import { ExtendedOpenEvolvePluginState } from './extended-plugin-types';
/**
 * Enhanced Validation Types
 * Comprehensive validation for all configuration parameters
 */
export type ValidationResult = {
    valid: boolean;
    errors: ValidationError[];
    warnings: ValidationWarning[];
    recommendations: string[];
};
export type ValidationError = {
    code: string;
    message: string;
    parameter: string;
    category: 'critical' | 'major' | 'minor';
    suggestedFix?: string;
};
export type ValidationWarning = {
    code: string;
    message: string;
    parameter: string;
    category: 'performance' | 'security' | 'usability' | 'compatibility';
    suggestedImprovement?: string;
};
export type ParameterValidationRule = {
    parameter: string;
    type: 'required' | 'range' | 'enum' | 'pattern' | 'custom';
    condition: any;
    errorMessage: string;
    severity: 'critical' | 'major' | 'minor';
};
export type ConfigurationProfile = {
    name: string;
    description: string;
    category: 'performance' | 'quality' | 'security' | 'experimental';
    parameters: Record<string, any>;
    validationRules: ParameterValidationRule[];
};
/**
 * Enhanced Performance Configuration
 * Advanced performance optimization settings
 */
export interface PerformanceConfiguration {
    enabled?: boolean;
    caching?: {
        enabled: boolean;
        strategy: 'lru' | 'lfu' | 'fifo' | 'random';
        max_size: number;
        ttl: number;
        compression: 'gzip' | 'brotli' | 'none';
        cache_warmup: boolean;
        cache_eviction_policy: 'time-based' | 'size-based' | 'hybrid';
    };
    parallel_processing?: {
        enabled: boolean;
        max_workers: number;
        worker_type: 'thread' | 'process' | 'cluster';
        load_balancing: 'round-robin' | 'least-connections' | 'random';
        batch_size: number;
        timeout: number;
        retry_policy: 'exponential' | 'linear' | 'none';
    };
    memory_optimization?: {
        garbage_collection: 'auto' | 'manual' | 'aggressive';
        object_pooling: boolean;
        buffer_reuse: boolean;
        weak_references: boolean;
        memory_profiling: boolean;
        leak_detection: boolean;
    };
    memory_management?: {
        enabled?: boolean;
        garbage_collection: 'auto' | 'manual' | 'aggressive';
        object_pooling: boolean;
        buffer_reuse: boolean;
        weak_references: boolean;
        memory_profiling: boolean;
        leak_detection: boolean;
        max_memory_mb: number;
        warning_threshold: number;
        cleanup_interval: number;
    };
    network_optimization?: {
        connection_pooling: boolean;
        keep_alive: boolean;
        http2: boolean;
        compression: 'gzip' | 'brotli' | 'deflate' | 'none';
        dns_caching: boolean;
        dns_prefetch: boolean;
        request_batching: boolean;
    };
    resource_management?: {
        cpu_throttling: boolean;
        memory_limits: {
            soft: number;
            hard: number;
        };
        disk_limits: {
            soft: number;
            hard: number;
        };
        cleanup_interval: number;
        resource_monitoring: boolean;
    };
    adaptive_optimization?: {
        enabled: boolean;
        learning_rate: number;
        adaptation_interval: number;
        performance_targets: {
            latency: number;
            throughput: number;
            memory_usage: number;
        };
        strategy: 'reinforcement' | 'bayesian' | 'gradient-based';
    };
}
/**
 * Enhanced Security Configuration
 * Comprehensive security and compliance settings
 */
export interface SecurityConfiguration {
    enabled?: boolean;
    authentication?: {
        enabled: boolean;
        method: 'api-key' | 'oauth2' | 'jwt' | 'basic';
        token_expiry: number;
        refresh_tokens: boolean;
        rate_limiting: {
            requests_per_minute: number;
            burst_limit: number;
        };
        ip_whitelisting: boolean;
        ip_blacklisting: boolean;
    };
    data_protection?: {
        enabled?: boolean;
        encryption: {
            enabled?: boolean;
            at_rest: boolean;
            in_transit: boolean;
            algorithm: 'aes-256' | 'rsa-2048' | 'chacha20';
            key_rotation: number;
        };
        masking: {
            sensitive_fields: string[];
            masking_strategy: 'partial' | 'full' | 'hash';
        };
        anonymization: {
            enabled: boolean;
            fields: string[];
            strategy: 'pseudonymization' | 'generalization' | 'suppression';
        };
        redaction: {
            enabled: boolean;
            patterns: string[];
        };
    };
    compliance?: {
        enabled?: boolean;
        gdpr: boolean;
        hipaa: boolean;
        ccpa: boolean;
        soc2: boolean;
        iso_27001: boolean;
        audit_logging: {
            enabled: boolean;
            retention_days: number;
            log_level: 'basic' | 'detailed' | 'verbose';
        };
        data_retention: number;
        consent_management: boolean;
    };
    access_control?: {
        role_based: boolean;
        attribute_based: boolean;
        permission_model: 'rbac' | 'abac' | 'custom';
        default_deny: boolean;
        privilege_escalation_prevention: boolean;
    };
    network_security?: {
        tls: {
            enabled: boolean;
            min_version: '1.2' | '1.3';
            cipher_suites: string[];
        };
        firewall: {
            enabled: boolean;
            rules: string[];
        };
        ddos_protection: boolean;
        intrusion_detection: boolean;
    };
    audit?: {
        logging: {
            enabled: boolean;
            level: 'debug' | 'info' | 'warn' | 'error';
            retention: number;
        };
        monitoring: {
            enabled: boolean;
            metrics: string[];
            alerts: string[];
        };
        anomaly_detection: boolean;
    };
}
/**
 * Enhanced Monitoring Configuration
 * Comprehensive observability and monitoring
 */
export interface MonitoringConfiguration {
    enabled?: boolean;
    metrics?: {
        enabled: boolean;
        collection_interval: number;
        metrics_to_collect: string | string[];
        custom_metrics: Record<string, string>;
        aggregation: 'sum' | 'avg' | 'max' | 'min' | 'count';
        retention_days?: number;
    };
    logging?: {
        enabled: boolean;
        level: 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'fatal';
        format: 'json' | 'text' | 'structured';
        destinations: ('console' | 'file' | 'remote')[];
        rotation: {
            enabled: boolean;
            max_size: number;
            max_files: number;
        };
        sampling: {
            enabled: boolean;
            rate: number;
        };
        max_size_mb?: number;
    };
    tracing?: {
        enabled: boolean;
        sampler: 'always' | 'never' | 'probabilistic' | 'rate_limiting';
        sample_rate: number;
        max_traces_per_second: number;
        trace_context: 'w3c' | 'jaeger' | 'zipkin';
    };
    alerting?: {
        enabled: boolean;
        rules: AlertRule[];
        notifications: NotificationChannel[];
        escalation_policy: EscalationPolicy;
        thresholds?: Record<string, number>;
        destinations?: string[];
        cooldown_minutes?: number;
    };
    dashboards?: {
        enabled: boolean;
        default_dashboard: string;
        custom_dashboards: DashboardConfig[];
        refresh_interval: number;
    };
    health_checks?: {
        enabled: boolean;
        interval: number;
        endpoints: string[];
        thresholds: Record<string, number>;
    };
    profiling?: {
        enabled: boolean;
        cpu_profiling: boolean;
        memory_profiling: boolean;
        heap_snapshot: boolean;
        sampling_interval: number;
    };
}
export interface AlertRule {
    name: string;
    condition: string;
    threshold: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
    duration: number;
    cooldown: number;
}
export interface NotificationChannel {
    type: 'email' | 'slack' | 'pagerduty' | 'webhook' | 'sms';
    destination: string;
    format: 'text' | 'json' | 'html';
    enabled: boolean;
}
export interface EscalationPolicy {
    levels: EscalationLevel[];
    timeout: number;
    repeat: number;
}
export interface EscalationLevel {
    severity: 'low' | 'medium' | 'high' | 'critical';
    channels: string[];
    delay: number;
}
export interface DashboardConfig {
    name: string;
    title: string;
    layout: 'grid' | 'list' | 'custom';
    widgets: DashboardWidget[];
    refresh_interval: number;
}
export interface DashboardWidget {
    type: 'metric' | 'chart' | 'table' | 'log' | 'trace';
    title: string;
    data_source: string;
    configuration: Record<string, any>;
    position: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
}
/**
 * Enhanced Integration Configuration
 * Advanced integration patterns and APIs
 */
export interface IntegrationConfiguration {
    enabled?: boolean;
    api_integrations?: {
        rest: RestApiConfig[];
        graphql: GraphQLConfig[];
        websocket: WebSocketConfig[];
        grpc: GRPCConfig[];
    };
    rest_api?: {
        enabled?: boolean;
        timeout: number;
        max_retries: number;
        base_url: string;
        endpoints?: string[];
    };
    graphql?: {
        enabled: boolean;
        endpoint: string;
        schema?: any;
        max_batch_size?: number;
        timeout?: number;
    };
    websocket?: {
        enabled: boolean;
        url: string;
        ping_interval?: number;
        reconnect_interval?: number;
    };
    webhooks?: {
        enabled?: boolean;
        incoming: WebhookConfig[];
        outgoing: WebhookConfig[];
        retries: number;
        timeout: number;
    };
    event_streaming?: {
        enabled?: boolean;
        kafka: KafkaConfig[];
        rabbitmq: RabbitMQConfig[];
        aws_sns: AWSSNSConfig[];
        google_pubsub: GooglePubSubConfig[];
    };
    message_queues?: {
        enabled?: boolean;
        kafka: KafkaConfig[];
        rabbitmq: RabbitMQConfig[];
        aws_sns: AWSSNSConfig[];
        google_pubsub: GooglePubSubConfig[];
    };
    databases?: {
        sql: SQLDatabaseConfig[];
        nosql: NoSQLDatabaseConfig[];
        cache: CacheConfig[];
    };
    third_party_services?: {
        aws: AWSServiceConfig[];
        azure: AzureServiceConfig[];
        google_cloud: GoogleCloudServiceConfig[];
        auth: AuthServiceConfig[];
        payment: PaymentServiceConfig[];
    };
    custom_integrations?: {
        scripts: CustomScriptConfig[];
        plugins: CustomPluginConfig[];
        adapters: CustomAdapterConfig[];
    };
}
export interface RestApiConfig {
    name: string;
    base_url: string;
    endpoints: EndpointConfig[];
    authentication: ApiAuthentication;
    rate_limiting: RateLimitingConfig;
    retry_policy: RetryPolicy;
}
export interface GraphQLConfig {
    name: string;
    endpoint: string;
    schema?: any;
    authentication: ApiAuthentication;
    rate_limiting: RateLimitingConfig;
    retry_policy: RetryPolicy;
}
export interface WebSocketConfig {
    name: string;
    url: string;
    protocols?: string[];
    authentication?: ApiAuthentication;
    reconnect_policy: RetryPolicy;
}
export interface GRPCConfig {
    name: string;
    server_address: string;
    proto_files: string[];
    authentication: ApiAuthentication;
    retry_policy: RetryPolicy;
}
export interface WebhookConfig {
    name: string;
    url: string;
    events: string[];
    authentication: ApiAuthentication;
    retry_policy: RetryPolicy;
    headers: Record<string, string>;
}
export interface KafkaConfig {
    name: string;
    bootstrap_servers: string[];
    topics: string[];
    consumer_group?: string;
    authentication?: ApiAuthentication;
}
export interface RabbitMQConfig {
    name: string;
    host: string;
    port: number;
    virtual_host: string;
    queues: string[];
    authentication: ApiAuthentication;
}
export interface AWSSNSConfig {
    name: string;
    region: string;
    topic_arn: string;
    authentication: ApiAuthentication;
}
export interface GooglePubSubConfig {
    name: string;
    project: string;
    topic: string;
    authentication: ApiAuthentication;
}
export interface SQLDatabaseConfig {
    name: string;
    type: 'mysql' | 'postgresql' | 'mssql' | 'oracle' | 'sqlite';
    connection_string: string;
    pool_size: number;
}
export interface NoSQLDatabaseConfig {
    name: string;
    type: 'mongodb' | 'cassandra' | 'dynamodb' | 'cosmosdb';
    connection_string: string;
    options: Record<string, any>;
}
export interface CacheConfig {
    name: string;
    type: 'redis' | 'memcached' | 'in-memory';
    connection_string: string;
    ttl: number;
}
export interface AWSServiceConfig {
    service: string;
    region: string;
    authentication: ApiAuthentication;
    configuration: Record<string, any>;
}
export interface AzureServiceConfig {
    service: string;
    region: string;
    authentication: ApiAuthentication;
    configuration: Record<string, any>;
}
export interface GoogleCloudServiceConfig {
    service: string;
    region: string;
    authentication: ApiAuthentication;
    configuration: Record<string, any>;
}
export interface AuthServiceConfig {
    provider: 'auth0' | 'okta' | 'firebase' | 'custom';
    domain: string;
    client_id: string;
    configuration: Record<string, any>;
}
export interface PaymentServiceConfig {
    provider: 'stripe' | 'paypal' | 'braintree' | 'custom';
    api_key: string;
    configuration: Record<string, any>;
}
export interface CustomScriptConfig {
    name: string;
    script_path: string;
    runtime: 'node' | 'python' | 'bash';
    environment: Record<string, string>;
}
export interface CustomPluginConfig {
    name: string;
    version: string;
    entry_point: string;
    configuration: Record<string, any>;
}
export interface CustomAdapterConfig {
    name: string;
    adapter_type: string;
    configuration: Record<string, any>;
}
export interface EndpointConfig {
    path: string;
    method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
    description: string;
    parameters: ParameterConfig[];
    response_schema: any;
}
export interface ParameterConfig {
    name: string;
    type: 'query' | 'path' | 'header' | 'body';
    required: boolean;
    schema: any;
}
export interface ApiAuthentication {
    type: 'none' | 'api-key' | 'oauth2' | 'basic' | 'bearer';
    configuration: Record<string, any>;
}
export interface RateLimitingConfig {
    requests_per_second: number;
    burst_limit: number;
    window_size: number;
}
export interface RetryPolicy {
    max_retries: number;
    backoff: 'exponential' | 'linear' | 'constant';
    initial_delay: number;
    max_delay: number;
}
/**
 * Enhanced Error Handling Configuration
 * Comprehensive error management and recovery
 */
export interface ErrorHandlingConfiguration {
    enabled?: boolean;
    error_classification?: {
        enabled?: boolean;
        categories: string[];
        severity_levels: 'low' | 'medium' | 'high' | 'critical';
        default_severity: 'medium';
        max_history?: number;
    };
    error_recovery?: {
        enabled?: boolean;
        automatic_retry: boolean;
        fallback_strategies: FallbackStrategy[];
        circuit_breakers: CircuitBreakerConfig[];
        compensation_actions: CompensationAction[];
        max_attempts?: number;
        retry_delay?: number;
    };
    error_reporting?: {
        enabled: boolean;
        destinations: ('console' | 'file' | 'api' | 'monitoring')[];
        sampling_rate: number;
        sensitive_data_filtering: boolean;
        rate_limiting: number;
    };
    error_analysis?: {
        root_cause_analysis: boolean;
        pattern_detection: boolean;
        anomaly_detection: boolean;
        trend_analysis: boolean;
    };
    error_prevention?: {
        input_validation: boolean;
        preconditions: boolean;
        postconditions: boolean;
        invariants: boolean;
        timeout_detection: boolean;
    };
    error_context?: {
        include_stack_trace: boolean;
        include_environment: boolean;
        include_state: boolean;
        include_user_info: boolean;
        redact_sensitive_data: boolean;
    };
}
export interface FallbackStrategy {
    error_type: string;
    fallback_action: 'retry' | 'default_value' | 'alternative_method' | 'fail_silently';
    configuration: Record<string, any>;
    max_attempts: number;
}
export interface CircuitBreakerConfig {
    name: string;
    failure_threshold: number;
    reset_timeout: number;
    half_open_attempts: number;
    fallback_strategy: string;
}
export interface CompensationAction {
    error_type: string;
    compensation_type: 'rollback' | 'corrective' | 'notifications' | 'logging';
    configuration: Record<string, any>;
}
/**
 * Enhanced Extended Plugin State
 * Integrates all enhanced configurations
 */
export interface EnhancedOpenEvolvePluginState extends ExtendedOpenEvolvePluginState {
    performanceConfig?: PerformanceConfiguration;
    securityConfig?: SecurityConfiguration;
    monitoringConfig?: MonitoringConfiguration;
    integrationConfig?: IntegrationConfiguration;
    errorHandlingConfig?: ErrorHandlingConfiguration;
    advancedState?: {
        performance_metrics: PerformanceMetrics;
        security_status: SecurityStatus;
        integration_status: IntegrationStatus;
        error_statistics: ErrorStatistics;
    };
    configuration_profiles?: ConfigurationProfile[];
    active_profile?: string;
    performanceProfiles?: Record<string, PerformanceProfile | PerformanceConfiguration>;
    securityProfiles?: Record<string, SecurityProfile | SecurityConfiguration>;
    validation_state?: {
        last_validation: Date | null;
        validation_results: ValidationResult[];
        validation_history: ValidationHistoryEntry[];
    };
    validationHistory?: ValidationRecord[];
    executionStatistics?: ExecutionStatistics;
    errorStatistics?: {
        totalErrors: number;
        errorsByType: Record<string, number>;
        lastError: Error | null;
    };
    performance_state?: {
        metrics: PerformanceMetrics;
        optimization_status: OptimizationStatus;
        resource_usage: ResourceUsage;
    };
    security_state?: {
        status: SecurityStatus;
        vulnerabilities: SecurityVulnerability[];
        compliance_status: ComplianceStatus;
    };
}
/**
 * Enhanced Performance Metrics
 */
export interface PerformanceMetrics {
    execution_time: {
        average: number;
        min: number;
        max: number;
        standard_deviation: number;
    };
    memory_usage: {
        current: number;
        peak: number;
        average: number;
    };
    cpu_usage: {
        current: number;
        average: number;
        peak: number;
    };
    throughput: {
        requests_per_second: number;
        operations_per_second: number;
    };
    cache_hit_rate: number;
    error_rate: number;
    latency: {
        average: number;
        p95: number;
        p99: number;
    };
}
export interface OptimizationStatus {
    adaptive_optimization_enabled: boolean;
    current_optimization_level: number;
    performance_targets_met: boolean;
    recommendations: string[];
}
export interface ResourceUsage {
    memory: {
        used: number;
        total: number;
        percentage: number;
    };
    cpu: {
        used: number;
        total: number;
        percentage: number;
    };
    disk: {
        used: number;
        total: number;
        percentage: number;
    };
    network: {
        sent: number;
        received: number;
        bandwidth: number;
    };
}
export interface SecurityStatus {
    overall_score: number;
    vulnerabilities: {
        critical: number;
        high: number;
        medium: number;
        low: number;
    };
    compliance: {
        gdpr: boolean;
        hipaa: boolean;
        soc2: boolean;
        iso_27001: boolean;
    };
    encryption_status: 'enabled' | 'partial' | 'disabled';
    authentication_status: 'enabled' | 'partial' | 'disabled';
}
export interface SecurityVulnerability {
    id: string;
    type: 'injection' | 'xss' | 'csrf' | 'auth' | 'config' | 'crypto' | 'other';
    severity: 'critical' | 'high' | 'medium' | 'low';
    description: string;
    location: string;
    detected_at: Date;
    status: 'open' | 'fixed' | 'mitigated' | 'false_positive';
    mitigation: string;
}
export interface ComplianceStatus {
    gdpr: {
        compliant: boolean;
        issues: string[];
    };
    hipaa: {
        compliant: boolean;
        issues: string[];
    };
    soc2: {
        compliant: boolean;
        issues: string[];
    };
    iso_27001: {
        compliant: boolean;
        issues: string[];
    };
}
export interface IntegrationStatus {
    api_integrations: {
        total: number;
        healthy: number;
        unhealthy: number;
    };
    webhook_integrations: {
        total: number;
        healthy: number;
        unhealthy: number;
    };
    event_streaming: {
        total: number;
        healthy: number;
        unhealthy: number;
    };
    database_integrations: {
        total: number;
        healthy: number;
        unhealthy: number;
    };
}
export interface ErrorStatistics {
    total_errors: number;
    by_severity: {
        critical: number;
        high: number;
        medium: number;
        low: number;
    };
    by_category: Record<string, number>;
    error_rate: number;
    mean_time_to_recovery: number;
}
export interface ValidationHistoryEntry {
    timestamp: Date;
    configuration_version: string;
    validation_result: ValidationResult;
    actions_taken: string[];
}
/**
 * Enhanced Default Configuration
 * Complete default values for all enhanced features
 */
export declare const ENHANCED_OPENEVOLVE_PLUGIN_CONSTANTS: {
    PERFORMANCE_DEFAULTS: {
        caching: {
            enabled: boolean;
            strategy: string;
            max_size: number;
            ttl: number;
            compression: string;
            cache_warmup: boolean;
            cache_eviction_policy: string;
        };
        parallel_processing: {
            enabled: boolean;
            max_workers: number;
            worker_type: string;
            load_balancing: string;
            batch_size: number;
            timeout: number;
            retry_policy: string;
        };
        memory_optimization: {
            garbage_collection: string;
            object_pooling: boolean;
            buffer_reuse: boolean;
            weak_references: boolean;
            memory_profiling: boolean;
            leak_detection: boolean;
        };
        network_optimization: {
            connection_pooling: boolean;
            keep_alive: boolean;
            http2: boolean;
            compression: string;
            dns_caching: boolean;
            dns_prefetch: boolean;
            request_batching: boolean;
        };
        resource_management: {
            cpu_throttling: boolean;
            memory_limits: {
                soft: number;
                hard: number;
            };
            disk_limits: {
                soft: number;
                hard: number;
            };
            cleanup_interval: number;
            resource_monitoring: boolean;
        };
        adaptive_optimization: {
            enabled: boolean;
            learning_rate: number;
            adaptation_interval: number;
            performance_targets: {
                latency: number;
                throughput: number;
                memory_usage: number;
            };
            strategy: string;
        };
    };
    SECURITY_DEFAULTS: {
        authentication: {
            enabled: boolean;
            method: string;
            token_expiry: number;
            refresh_tokens: boolean;
            rate_limiting: {
                requests_per_minute: number;
                burst_limit: number;
            };
            ip_whitelisting: boolean;
            ip_blacklisting: boolean;
        };
        data_protection: {
            encryption: {
                enabled: boolean;
                at_rest: boolean;
                in_transit: boolean;
                algorithm: string;
                key_rotation: number;
            };
            masking: {
                sensitive_fields: string[];
                masking_strategy: string;
            };
            anonymization: {
                enabled: boolean;
                fields: any[];
                strategy: string;
            };
            redaction: {
                enabled: boolean;
                patterns: string[];
            };
        };
        compliance: {
            gdpr: boolean;
            hipaa: boolean;
            ccpa: boolean;
            soc2: boolean;
            iso_27001: boolean;
            audit_logging: {
                enabled: boolean;
                retention_days: number;
                log_level: string;
            };
            data_retention: number;
            consent_management: boolean;
        };
        access_control: {
            role_based: boolean;
            attribute_based: boolean;
            permission_model: string;
            default_deny: boolean;
            privilege_escalation_prevention: boolean;
        };
        network_security: {
            tls: {
                enabled: boolean;
                min_version: string;
                cipher_suites: string[];
            };
            firewall: {
                enabled: boolean;
                rules: string[];
            };
            ddos_protection: boolean;
            intrusion_detection: boolean;
        };
        audit: {
            logging: {
                enabled: boolean;
                level: string;
                retention: number;
            };
            monitoring: {
                enabled: boolean;
                metrics: string[];
                alerts: string[];
            };
            anomaly_detection: boolean;
        };
    };
    MONITORING_DEFAULTS: {
        metrics: {
            enabled: boolean;
            collection_interval: number;
            metrics_to_collect: string[];
            custom_metrics: {};
            aggregation: string;
        };
        logging: {
            enabled: boolean;
            level: string;
            format: string;
            destinations: string[];
            rotation: {
                enabled: boolean;
                max_size: number;
                max_files: number;
            };
            sampling: {
                enabled: boolean;
                rate: number;
            };
        };
        tracing: {
            enabled: boolean;
            sampler: string;
            sample_rate: number;
            max_traces_per_second: number;
            trace_context: string;
        };
        alerting: {
            enabled: boolean;
            rules: {
                name: string;
                condition: string;
                threshold: number;
                severity: string;
                duration: number;
                cooldown: number;
            }[];
            notifications: {
                type: string;
                destination: string;
                format: string;
                enabled: boolean;
            }[];
            escalation_policy: {
                levels: {
                    severity: string;
                    channels: string[];
                    delay: number;
                }[];
                timeout: number;
                repeat: number;
            };
        };
        dashboards: {
            enabled: boolean;
            default_dashboard: string;
            custom_dashboards: any[];
            refresh_interval: number;
        };
        health_checks: {
            enabled: boolean;
            interval: number;
            endpoints: string[];
            thresholds: {
                response_time: number;
                availability: number;
            };
        };
        profiling: {
            enabled: boolean;
            cpu_profiling: boolean;
            memory_profiling: boolean;
            heap_snapshot: boolean;
            sampling_interval: number;
        };
    };
    INTEGRATION_DEFAULTS: {
        api_integrations: {
            rest: any[];
            graphql: any[];
            websocket: any[];
            grpc: any[];
        };
        webhooks: {
            incoming: any[];
            outgoing: any[];
            retries: number;
            timeout: number;
        };
        event_streaming: {
            kafka: any[];
            rabbitmq: any[];
            aws_sns: any[];
            google_pubsub: any[];
        };
        databases: {
            sql: any[];
            nosql: any[];
            cache: any[];
        };
        third_party_services: {
            aws: any[];
            azure: any[];
            google_cloud: any[];
            auth: any[];
            payment: any[];
        };
        custom_integrations: {
            scripts: any[];
            plugins: any[];
            adapters: any[];
        };
    };
    ERROR_HANDLING_DEFAULTS: {
        error_classification: {
            categories: string[];
            severity_levels: string[];
            default_severity: string;
        };
        error_recovery: {
            automatic_retry: boolean;
            fallback_strategies: ({
                error_type: string;
                fallback_action: string;
                configuration: {
                    max_attempts: number;
                    default?: undefined;
                };
                max_attempts: number;
            } | {
                error_type: string;
                fallback_action: string;
                configuration: {
                    default: any;
                    max_attempts?: undefined;
                };
                max_attempts: number;
            })[];
            circuit_breakers: {
                name: string;
                failure_threshold: number;
                reset_timeout: number;
                half_open_attempts: number;
                fallback_strategy: string;
            }[];
            compensation_actions: {
                error_type: string;
                compensation_type: string;
                configuration: {
                    max_retries: number;
                };
            }[];
        };
        error_reporting: {
            enabled: boolean;
            destinations: string[];
            sampling_rate: number;
            sensitive_data_filtering: boolean;
            rate_limiting: number;
        };
        error_analysis: {
            root_cause_analysis: boolean;
            pattern_detection: boolean;
            anomaly_detection: boolean;
            trend_analysis: boolean;
        };
        error_prevention: {
            input_validation: boolean;
            preconditions: boolean;
            postconditions: boolean;
            invariants: boolean;
            timeout_detection: boolean;
        };
        error_context: {
            include_stack_trace: boolean;
            include_environment: boolean;
            include_state: boolean;
            include_user_info: boolean;
            redact_sensitive_data: boolean;
        };
    };
    QUALITY_DIVERSITY_DEFAULTS: {
        feature_dimensions: any;
        feature_bins: number;
        archive_size: number;
        novelty_threshold: number;
        behavior_dimensions: any[];
        diversity_metric: string;
        diversity_reference_size: number;
        adaptive_feature_dimensions: boolean;
        double_selection: boolean;
        qd_algorithm: string;
        behavior_descriptor_type: string;
        archive_learning_rate: number;
        quality_threshold: number;
        diversity_weight: number;
        behavior_space: string;
        distance_metric: string;
        archive_update_freq: number;
        exploration_bonus: number;
        pareto_layers: number;
    };
    MULTI_OBJECTIVE_DEFAULTS: {
        objectives: any;
        objective_weights: any[];
        pareto_front_size: number;
        dominance_metric: string;
        constraint_handling: string;
        reference_point: any[];
        crowding_distance: boolean;
        epsilon_dominance: number;
        decomposition_method: string;
        scalarization_function: string;
        dominance_type: string;
        epsilon_values: any[];
        scalarization: string;
        constraint_tolerance: number;
        hypervolume_ref: any[];
    };
    ISLAND_MODEL_DEFAULTS: {
        num_islands: number;
        migration_interval: number;
        migration_rate: number;
        migration_topology: string;
        ring_topology: boolean;
        controlled_gene_flow: boolean;
        island_diversity_metric: string;
        migration_selection: string;
        island_initialization: string;
        island_specialization: boolean;
        migration_size: number;
        migration_policy: string;
        replacement_policy: string;
        island_sizes: any[];
        heterogeneous_islands: boolean;
        synchronous_migration: boolean;
        adaptive_migration: boolean;
    };
    SELECTION_DEFAULTS: {
        elite_ratio: number;
        exploration_ratio: number;
        exploitation_ratio: number;
        multi_strategy_sampling: boolean;
        selection_pressure: number;
        tournament_size: number;
        crossover_rate: number;
        mutation_rate: number;
        elitism_count: number;
        selection_method: string;
        reproduction_method: string;
        parent_selection: string;
        random_ratio: number;
        survivor_selection: string;
        replacement_rate: number;
        selection_pressure_decay: number;
        diversity_selection: boolean;
        age_based_selection: boolean;
    };
    EVALUATION_DEFAULTS: {
        cascade_evaluation: boolean;
        cascade_thresholds: number[];
        parallel_evaluations: number;
        evaluator_timeout: number;
        max_retries_eval: number;
        use_llm_feedback: boolean;
        llm_feedback_weight: number;
        evaluator_models: any[];
        evaluator_system_message: string;
        ensemble_size: number;
        consensus_threshold: number;
        evaluation_criteria: any[];
        custom_evaluator: any;
        evaluation_batch_size: number;
        cache_evaluations: boolean;
        cache_size: number;
        evaluation_noise: number;
        fitness_scaling: string;
        normalization: boolean;
        multi_criteria_eval: boolean;
        evaluation_budget: number;
        incremental_eval: boolean;
        surrogate_model: boolean;
        active_learning: boolean;
        uncertainty_sampling: boolean;
    };
    PROMPT_ENGINEERING_DEFAULTS: {
        prompt_template: string;
        system_prompt: string;
        context_length: number;
        prompt_optimization: boolean;
        template_stochasticity: boolean;
        meta_prompting: boolean;
        few_shot_examples: number;
        chain_of_thought: boolean;
        self_consistency: boolean;
        prompt_ensembling: boolean;
        dynamic_prompting: boolean;
        prompt_compression: boolean;
    };
    ARTIFACT_MANAGEMENT_DEFAULTS: {
        enable_artifacts: boolean;
        artifact_types: string[];
        max_artifact_size: number;
        artifact_validation: boolean;
        artifact_compression: boolean;
        artifact_versioning: boolean;
        artifact_metadata: boolean;
        artifact_cleanup: boolean;
        artifact_storage: string;
        artifact_encryption: boolean;
    };
    RESOURCE_MANAGEMENT_DEFAULTS: {
        memory_limit_mb: number;
        cpu_limit: number;
        max_time: number;
        disk_limit_mb: number;
        network_limit_mbps: number;
        api_call_limit: number;
        token_limit: number;
        cost_limit_usd: number;
        resource_monitoring: boolean;
        auto_scaling: boolean;
        checkpoint_interval: number;
    };
    DATABASE_STORAGE_DEFAULTS: {
        db_path: string;
        db_type: string;
        connection_string: string;
        max_connections: number;
        connection_timeout: number;
        query_timeout: number;
        batch_size: number;
        compression: boolean;
        encryption: boolean;
        backup_enabled: boolean;
    };
    EVOLUTION_TRACING_DEFAULTS: {
        trace_enabled: boolean;
        trace_level: string;
        trace_format: string;
        trace_file: string;
        trace_compression: boolean;
        trace_rotation: boolean;
        max_trace_size_mb: number;
        trace_buffer_size: number;
        real_time_tracing: boolean;
        trace_sampling: number;
        include_population: boolean;
        include_fitness: boolean;
    };
    EARLY_STOPPING_DEFAULTS: {
        early_stopping: boolean;
        early_stopping_patience: number;
        min_improvement: number;
        improvement_window: number;
        plateau_threshold: number;
        convergence_check: boolean;
        diversity_threshold: number;
        stagnation_limit: number;
        adaptive_stopping: boolean;
    };
    DISTRIBUTED_PROCESSING_DEFAULTS: {
        distributed: boolean;
        num_workers: number;
        worker_timeout: number;
        load_balancing: string;
        fault_tolerance: boolean;
        worker_restart: boolean;
        communication_backend: string;
        message_compression: boolean;
        heartbeat_interval: number;
        cluster_scaling: boolean;
    };
    ADVANCED_RESEARCH_DEFAULTS: {
        novelty_search: boolean;
        curiosity_driven: boolean;
        meta_learning: boolean;
        transfer_learning: boolean;
        continual_learning: boolean;
        few_shot_adaptation: boolean;
        zero_shot_transfer: boolean;
        domain_adaptation: boolean;
        multi_task_learning: boolean;
        lifelong_learning: boolean;
        neural_architecture_search: boolean;
        hyperparameter_optimization: boolean;
        automated_ml: boolean;
        explainable_ai: boolean;
        federated_learning: boolean;
        differential_privacy: boolean;
        quantum_computing: boolean;
        neuromorphic_computing: boolean;
        edge_computing: boolean;
        green_ai: boolean;
    };
    CUSTOM_REQUIREMENTS_DEFAULTS: {
        custom_fitness: string;
        custom_operators: any[];
        custom_constraints: any[];
        domain_knowledge: string;
        expert_rules: any[];
        business_logic: string;
        regulatory_compliance: any[];
        ethical_guidelines: any[];
    };
    UI_VISUALIZATION_DEFAULTS: {
        enable_visualization: boolean;
        plot_frequency: number;
        plot_types: string[];
        interactive_plots: boolean;
        real_time_updates: boolean;
        export_plots: boolean;
        plot_format: string;
        dashboard_enabled: boolean;
    };
    EXPERIMENTAL_DEFAULTS: {
        experimental_features: boolean;
        beta_algorithms: boolean;
        research_mode: boolean;
        debug_mode: boolean;
        profiling_enabled: boolean;
        memory_profiling: boolean;
        experimental_logging: boolean;
    };
    PLUGIN_NAME: string;
    PLUGIN_VERSION: string;
    PLUGIN_DESCRIPTION: string;
    PLUGIN_AUTHOR: string;
    PLUGIN_LICENSE: string;
    DEFAULT_EVOLUTION_MODE: import('./plugin-types').EvolutionStrategy;
    DEFAULT_ADVERSARIAL_MODE: import('./plugin-types').AdversarialStrategy;
    DEFAULT_DECOMPOSITION_STRATEGY: import('./plugin-types').DecompositionStrategy;
    DEFAULT_MAX_ITERATIONS: number;
    DEFAULT_POPULATION_SIZE: number;
    DEFAULT_TEMPERATURE: number;
    DEFAULT_MAX_TOKENS: number;
    DEFAULT_MODEL_ID: string;
    DEFAULT_API_BASE: string;
    DEFAULT_TIMEOUT: number;
    DEFAULT_MAX_RETRIES: number;
    DEFAULT_RETRY_DELAY: number;
    DEFAULT_MDAP_MAKER_ENABLED: boolean;
    DEFAULT_MDAP_MAKER_AUTO_SELECT: boolean;
    DEFAULT_MDAP_MAKER_MAX_DEPTH: number;
    DEFAULT_MDAP_MAKER_K_AHEAD: number;
    DEFAULT_MDAP_MAKER_RED_FLAGGING: boolean;
    DEFAULT_MDAP_MAKER_ADAPTIVE_K: boolean;
    DEFAULT_MDAP_MAKER_PROVIDER: string;
    DEFAULT_MDAP_MAKER_MODEL: string;
    DEFAULT_MDAP_MAKER_KEYWORDS: string[];
    EXECUTION_METHODS: string[];
    DEFAULT_EXECUTION_METHOD: string;
    EVOLUTION_STRATEGIES: string[];
    ADVERSARIAL_STRATEGIES: string[];
    DECOMPOSITION_STRATEGIES: string[];
};
/**
 * Enhanced Default Configuration
 * Complete configuration with all enhanced features
 */
export declare const DEFAULT_ENHANCED_OPENEVOLVE_CONFIG: EnhancedOpenEvolvePluginState;
/**
 * Performance Profile
 */
export interface PerformanceProfile {
    name: string;
    settings: PerformanceConfiguration;
}
/**
 * Security Profile
 */
export interface SecurityProfile {
    name: string;
    settings: SecurityConfiguration;
}
/**
 * Validation Record
 */
export interface ValidationRecord {
    validationId: string;
    validationType: 'performance' | 'security' | 'monitoring' | 'integration' | 'error_handling';
    success: boolean;
    errorMessage?: string;
    timestamp: number;
}
/**
 * Execution Statistics
 */
export interface ExecutionStatistics {
    totalExecutions: number;
    successfulExecutions: number;
    failedExecutions: number;
    averageExecutionTime: number;
}
/**
 * Enhanced Execution Options
 */
export interface EnhancedExecutionOptions {
    performanceProfile?: string;
    securityProfile?: string;
    monitoringEnabled?: boolean;
    integrationMode?: 'auto' | 'manual' | 'disabled';
}
/**
 * Enhanced Execution Result
 */
export interface EnhancedExecutionResult {
    success: boolean;
    result?: any;
    error?: Error;
    metrics?: PerformanceMetrics;
    securityStatus?: SecurityStatus;
}
/**
 * Enhanced OpenEvolve Plugin Interface
 * Defines the contract for enhanced plugin instances
 */
export interface EnhancedOpenEvolvePlugin {
    getState: () => EnhancedOpenEvolvePluginState;
    setState: (state: EnhancedOpenEvolvePluginState) => void;
    subscribe: (listener: (state: EnhancedOpenEvolvePluginState) => void) => () => void;
    updateConfig: (updates: Partial<EnhancedOpenEvolvePluginState>) => boolean;
    resetConfig: () => boolean;
    getEnhancedState: () => EnhancedOpenEvolvePluginState;
    subscribeToEnhancedState: (listener: (state: EnhancedOpenEvolvePluginState) => void) => () => void;
    updateEnhancedConfig: (updates: Partial<EnhancedOpenEvolvePluginState>) => boolean;
    resetEnhancedConfig: () => boolean;
    validatePerformanceConfig: (config?: PerformanceConfiguration) => boolean;
    optimizePerformance: () => Promise<boolean>;
    getPerformanceMetrics: () => Record<string, number>;
    validateSecurityConfig: (config?: SecurityConfiguration) => boolean;
    auditSecurity: () => Promise<Record<string, boolean>>;
    performSecurityAudit: () => Promise<SecurityStatus>;
    getSecurityStatus: () => any;
    validateMonitoringConfig: (config?: MonitoringConfiguration) => boolean;
    setupMonitoring: () => Promise<boolean>;
    getMonitoringData: () => Record<string, any>;
    startMonitoring?: () => void;
    stopMonitoring?: () => void;
    validateIntegrationConfig: (config?: IntegrationConfiguration) => boolean;
    testIntegration: (name: string) => Promise<boolean>;
    enableIntegration?: (name: string) => boolean;
    disableIntegration?: (name: string) => boolean;
    getIntegrationStatus: () => any;
    setupIntegrations?: () => Promise<boolean>;
    cleanupIntegrations?: () => Promise<boolean>;
    validateErrorHandlingConfig: (config?: ErrorHandlingConfiguration) => boolean;
    getErrorReport: () => Record<string, any>;
    getErrorStatistics: () => ErrorStatistics;
    clearErrors: () => void;
    handleError?: (error: Error) => void;
    classifyError?: (error: Error) => string;
    logError?: (error: Error, context?: any) => void;
    reportError?: (error: Error) => void;
    attemptErrorRecovery?: (error: Error) => Promise<boolean>;
    addValidationResult: (result: ValidationRecord) => void;
    clearValidationHistory: () => boolean;
    addPerformanceProfile: (name: string, settings: PerformanceConfiguration) => boolean;
    removePerformanceProfile: (name: string) => boolean;
    addSecurityProfile: (name: string, settings: SecurityConfiguration) => boolean;
    removeSecurityProfile: (name: string) => boolean;
    executeEvolutionWithEnhancedFeatures: (goal: string, options?: EnhancedExecutionOptions) => Promise<EnhancedExecutionResult>;
    getMemoryUsage: () => any;
    getCacheStats: () => any;
}
