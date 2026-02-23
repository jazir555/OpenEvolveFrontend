/**
 * Centralized Environment Variable Schema
 *
 * Following the Federation Constitution:
 * - Law of Configuration Explicitness: No magic defaults
 * - All configurable values must be via environment variables
 * - Crashes immediately if required vars are missing
 *
 * This is the SINGLE SOURCE OF TRUTH for all environment variables.
 * Corresponds to: glue/ENVIRONMENT_VARIABLES.md and .env.schema
 */
import { EnvVar } from './env-validator';
/**
 * Core OpenEvolve Environment Variables
 */
export declare const CORE_ENV_VARS: EnvVar[];
/**
 * Infrastructure Environment Variables
 */
export declare const INFRA_ENV_VARS: EnvVar[];
/**
 * API Gateway Environment Variables
 */
export declare const API_GATEWAY_ENV_VARS: EnvVar[];
/**
 * Adapter Environment Variables
 */
export declare const BUBBLELAB_ADAPTER_ENV_VARS: EnvVar[];
export declare const GRAPHITI_ADAPTER_ENV_VARS: EnvVar[];
export declare const VECTORDB_ADAPTER_ENV_VARS: EnvVar[];
export declare const OPENEVOLVE_ADAPTER_ENV_VARS: EnvVar[];
export declare const ICR_ADAPTER_ENV_VARS: EnvVar[];
export declare const LEANAIDE_ADAPTER_ENV_VARS: EnvVar[];
export declare const Z3_ADAPTER_ENV_VARS: EnvVar[];
/**
 * RESE Adapter Environment Variables
 */
export declare const RESE_DEE_ENV_VARS: EnvVar[];
export declare const RESE_LLTDL_ENV_VARS: EnvVar[];
export declare const RESE_SCE_ENV_VARS: EnvVar[];
export declare const RESE_PHASE_ENV_VARS: EnvVar[];
/**
 * Knowledge Engine Environment Variables
 */
export declare const KNOWLEDGE_ENGINE_ENV_VARS: EnvVar[];
/**
 * Plugin Environment Variables
 */
export declare const PLUGIN_ENV_VARS: EnvVar[];
export declare const DATAPIZZA_PLUGIN_ENV_VARS: EnvVar[];
/**
 * LLM Environment Variables (used by multiple components)
 */
export declare const LLM_ENV_VARS: EnvVar[];
/**
 * Event Bus Environment Variables
 */
export declare const EVENT_BUS_ENV_VARS: EnvVar[];
/**
 * Observability Environment Variables
 */
export declare const OBSERVABILITY_ENV_VARS: EnvVar[];
/**
 * PES (Prompt Evolution Strategy) Environment Variables
 */
export declare const PES_ENV_VARS: EnvVar[];
/**
 * Orchestration Environment Variables
 */
export declare const ORCHESTRATION_ENV_VARS: EnvVar[];
/**
 * All Environment Variables Combined
 * Useful for validating the entire configuration at once
 */
export declare const ALL_ENV_VARS: EnvVar[];
/**
 * Helper function to get schema by component name
 */
export declare function getSchemaForComponent(componentName: string): EnvVar[];
//# sourceMappingURL=env-schema.d.ts.map