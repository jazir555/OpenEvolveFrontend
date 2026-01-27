import { z } from '@hono/zod-openapi';
export declare const hackathonOfferSchema: z.ZodObject<{
    isActive: z.ZodBoolean;
    expiresAt: z.ZodString;
    redeemedAt: z.ZodString;
}, "strip", z.ZodTypeAny, {
    isActive: boolean;
    expiresAt: string;
    redeemedAt: string;
}, {
    isActive: boolean;
    expiresAt: string;
    redeemedAt: string;
}>;
export type HackathonOffer = z.infer<typeof hackathonOfferSchema>;
export declare const specialOfferSchema: z.ZodObject<{
    isActive: z.ZodBoolean;
    plan: z.ZodString;
    expiresAt: z.ZodNullable<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    isActive: boolean;
    plan: string;
    expiresAt: string | null;
}, {
    isActive: boolean;
    plan: string;
    expiresAt: string | null;
}>;
export type SpecialOffer = z.infer<typeof specialOfferSchema>;
export declare const redeemCouponRequestSchema: z.ZodObject<{
    code: z.ZodString;
}, "strip", z.ZodTypeAny, {
    code: string;
}, {
    code: string;
}>;
export type RedeemCouponRequest = z.infer<typeof redeemCouponRequestSchema>;
export declare const redeemCouponResponseSchema: z.ZodObject<{
    success: z.ZodBoolean;
    message: z.ZodString;
    expiresAt: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    success: boolean;
    expiresAt?: string | undefined;
}, {
    message: string;
    success: boolean;
    expiresAt?: string | undefined;
}>;
export type RedeemCouponResponse = z.infer<typeof redeemCouponResponseSchema>;
export declare const subscriptionStatusResponseSchema: z.ZodObject<{
    userId: z.ZodString;
    plan: z.ZodString;
    planDisplayName: z.ZodString;
    features: z.ZodArray<z.ZodString, "many">;
    usage: z.ZodObject<{
        executionCount: z.ZodNumber;
        executionLimit: z.ZodNumber;
        creditLimit: z.ZodNumber;
        activeFlowLimit: z.ZodNumber;
        estimatedMonthlyCost: z.ZodNumber;
        resetDate: z.ZodString;
        tokenUsage: z.ZodArray<z.ZodObject<{
            modelName: z.ZodString;
            inputTokens: z.ZodNumber;
            outputTokens: z.ZodNumber;
            totalTokens: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            modelName: string;
            inputTokens: number;
            outputTokens: number;
            totalTokens: number;
        }, {
            modelName: string;
            inputTokens: number;
            outputTokens: number;
            totalTokens: number;
        }>, "many">;
        serviceUsage: z.ZodArray<z.ZodObject<{
            service: z.ZodNativeEnum<typeof import("./types").CredentialType>;
            subService: z.ZodOptional<z.ZodString>;
            unit: z.ZodString;
            usage: z.ZodNumber;
            unitCost: z.ZodNumber;
            totalCost: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            service: import("./types").CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }, {
            service: import("./types").CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        serviceUsage: {
            service: import("./types").CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[];
        executionCount: number;
        executionLimit: number;
        creditLimit: number;
        activeFlowLimit: number;
        estimatedMonthlyCost: number;
        resetDate: string;
        tokenUsage: {
            modelName: string;
            inputTokens: number;
            outputTokens: number;
            totalTokens: number;
        }[];
    }, {
        serviceUsage: {
            service: import("./types").CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[];
        executionCount: number;
        executionLimit: number;
        creditLimit: number;
        activeFlowLimit: number;
        estimatedMonthlyCost: number;
        resetDate: string;
        tokenUsage: {
            modelName: string;
            inputTokens: number;
            outputTokens: number;
            totalTokens: number;
        }[];
    }>;
    isActive: z.ZodBoolean;
    hackathonOffer: z.ZodOptional<z.ZodObject<{
        isActive: z.ZodBoolean;
        expiresAt: z.ZodString;
        redeemedAt: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        isActive: boolean;
        expiresAt: string;
        redeemedAt: string;
    }, {
        isActive: boolean;
        expiresAt: string;
        redeemedAt: string;
    }>>;
    specialOffer: z.ZodOptional<z.ZodObject<{
        isActive: z.ZodBoolean;
        plan: z.ZodString;
        expiresAt: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        isActive: boolean;
        plan: string;
        expiresAt: string | null;
    }, {
        isActive: boolean;
        plan: string;
        expiresAt: string | null;
    }>>;
}, "strip", z.ZodTypeAny, {
    usage: {
        serviceUsage: {
            service: import("./types").CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[];
        executionCount: number;
        executionLimit: number;
        creditLimit: number;
        activeFlowLimit: number;
        estimatedMonthlyCost: number;
        resetDate: string;
        tokenUsage: {
            modelName: string;
            inputTokens: number;
            outputTokens: number;
            totalTokens: number;
        }[];
    };
    isActive: boolean;
    plan: string;
    userId: string;
    planDisplayName: string;
    features: string[];
    hackathonOffer?: {
        isActive: boolean;
        expiresAt: string;
        redeemedAt: string;
    } | undefined;
    specialOffer?: {
        isActive: boolean;
        plan: string;
        expiresAt: string | null;
    } | undefined;
}, {
    usage: {
        serviceUsage: {
            service: import("./types").CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[];
        executionCount: number;
        executionLimit: number;
        creditLimit: number;
        activeFlowLimit: number;
        estimatedMonthlyCost: number;
        resetDate: string;
        tokenUsage: {
            modelName: string;
            inputTokens: number;
            outputTokens: number;
            totalTokens: number;
        }[];
    };
    isActive: boolean;
    plan: string;
    userId: string;
    planDisplayName: string;
    features: string[];
    hackathonOffer?: {
        isActive: boolean;
        expiresAt: string;
        redeemedAt: string;
    } | undefined;
    specialOffer?: {
        isActive: boolean;
        plan: string;
        expiresAt: string | null;
    } | undefined;
}>;
export type SubscriptionStatusResponse = z.infer<typeof subscriptionStatusResponseSchema>;
//# sourceMappingURL=subscription-status-schema.d.ts.map