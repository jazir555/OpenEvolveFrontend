/**
 * BubbleLab Adapter Contract Tests
 *
 * Purpose: Validate BubbleLab API contracts to prevent breaking changes
 * Compliance: Phase 2 - The Contract (Defense)
 *
 * These tests run on adapter startup to verify the API returns expected fields
 * If contracts are violated, the adapter refuses to start (Law of Runtime Truth)
 */
import { z } from 'zod';
/**
 * Health Check Response Contract
 */
declare const HealthCheckContract: z.ZodObject<{
    status: z.ZodEnum<["ok", "healthy", "error"]>;
    version: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: "error" | "healthy" | "ok";
    version?: string | undefined;
}, {
    status: "error" | "healthy" | "ok";
    version?: string | undefined;
}>;
/**
 * BubbleFlow List Response Contract
 */
declare const BubbleFlowContract: z.ZodObject<{
    id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    eventType: z.ZodString;
    webhookActive: z.ZodBoolean;
    createdAt: z.ZodOptional<z.ZodString>;
    updatedAt: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    name: string;
    id: string | number;
    eventType: string;
    webhookActive: boolean;
    description?: string | undefined;
    createdAt?: string | undefined;
    updatedAt?: string | undefined;
}, {
    name: string;
    id: string | number;
    eventType: string;
    webhookActive: boolean;
    description?: string | undefined;
    createdAt?: string | undefined;
    updatedAt?: string | undefined;
}>;
/**
 * BubbleFlow Create Response Contract
 */
declare const BubbleFlowCreateResponseContract: z.ZodObject<{
    id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    name: z.ZodString;
    requiredCredentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodString, "many">>>;
    webhookUrl: z.ZodOptional<z.ZodString>;
    createdAt: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    name: string;
    id: string | number;
    createdAt?: string | undefined;
    requiredCredentials?: Record<string, string[]> | undefined;
    webhookUrl?: string | undefined;
}, {
    name: string;
    id: string | number;
    createdAt?: string | undefined;
    requiredCredentials?: Record<string, string[]> | undefined;
    webhookUrl?: string | undefined;
}>;
/**
 * Execution Response Contract
 */
declare const ExecutionResponseContract: z.ZodObject<{
    execution_id: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodNumber]>>;
    output: z.ZodOptional<z.ZodAny>;
    error: z.ZodOptional<z.ZodString>;
    status: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    error?: string | undefined;
    execution_id?: string | number | undefined;
    status?: string | undefined;
    output?: any;
}, {
    error?: string | undefined;
    execution_id?: string | number | undefined;
    status?: string | undefined;
    output?: any;
}>;
/**
 * Execution History Response Contract
 */
declare const ExecutionHistoryContract: z.ZodObject<{
    executions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodNumber]>>;
        status: z.ZodString;
        startedAt: z.ZodString;
        completedAt: z.ZodOptional<z.ZodString>;
        output: z.ZodOptional<z.ZodAny>;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        status: string;
        startedAt: string;
        error?: string | undefined;
        id?: string | number | undefined;
        output?: any;
        completedAt?: string | undefined;
    }, {
        status: string;
        startedAt: string;
        error?: string | undefined;
        id?: string | number | undefined;
        output?: any;
        completedAt?: string | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    executions?: {
        status: string;
        startedAt: string;
        error?: string | undefined;
        id?: string | number | undefined;
        output?: any;
        completedAt?: string | undefined;
    }[] | undefined;
}, {
    executions?: {
        status: string;
        startedAt: string;
        error?: string | undefined;
        id?: string | number | undefined;
        output?: any;
        completedAt?: string | undefined;
    }[] | undefined;
}>;
/**
 * Validate all API contracts before starting adapter
 * This function should be called during adapter initialization
 *
 * @returns true if all contracts are valid
 * @throws Error if any contract is violated
 */
export declare function validateAllContracts(): boolean;
export { HealthCheckContract, BubbleFlowContract, BubbleFlowCreateResponseContract, ExecutionResponseContract, ExecutionHistoryContract, };
//# sourceMappingURL=contract.test.d.ts.map