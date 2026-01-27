import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const StripeBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createPaymentIntent">;
    amount: z.ZodNumber;
    currency: z.ZodDefault<z.ZodString>;
    customer: z.ZodOptional<z.ZodString>;
    paymentMethod: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    confirm: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    captureMethod: z.ZodDefault<z.ZodOptional<z.ZodEnum<["automatic", "manual"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createPaymentIntent";
    currency: string;
    amount: number;
    confirm: boolean;
    captureMethod: "manual" | "automatic";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    customer?: string | undefined;
    paymentMethod?: string | undefined;
}, {
    operation: "createPaymentIntent";
    amount: number;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    currency?: string | undefined;
    customer?: string | undefined;
    paymentMethod?: string | undefined;
    confirm?: boolean | undefined;
    captureMethod?: "manual" | "automatic" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"confirmPayment">;
    paymentIntentId: z.ZodString;
    paymentMethod: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "confirmPayment";
    paymentIntentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    paymentMethod?: string | undefined;
}, {
    operation: "confirmPayment";
    paymentIntentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    paymentMethod?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"refundPayment">;
    paymentIntentId: z.ZodString;
    amount: z.ZodOptional<z.ZodNumber>;
    reason: z.ZodOptional<z.ZodEnum<["duplicate", "fraudulent", "requested_by_customer", "other"]>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "refundPayment";
    paymentIntentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    amount?: number | undefined;
    reason?: "other" | "duplicate" | "fraudulent" | "requested_by_customer" | undefined;
}, {
    operation: "refundPayment";
    paymentIntentId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    amount?: number | undefined;
    reason?: "other" | "duplicate" | "fraudulent" | "requested_by_customer" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createCustomer">;
    email: z.ZodOptional<z.ZodString>;
    name: z.ZodOptional<z.ZodString>;
    phone: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createCustomer";
    description?: string | undefined;
    name?: string | undefined;
    email?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    phone?: string | undefined;
    metadata?: Record<string, string> | undefined;
}, {
    operation: "createCustomer";
    description?: string | undefined;
    name?: string | undefined;
    email?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    phone?: string | undefined;
    metadata?: Record<string, string> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getCustomer">;
    customerId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getCustomer";
    customerId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getCustomer";
    customerId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateCustomer">;
    customerId: z.ZodString;
    email: z.ZodOptional<z.ZodString>;
    name: z.ZodOptional<z.ZodString>;
    phone: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateCustomer";
    customerId: string;
    description?: string | undefined;
    name?: string | undefined;
    email?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    phone?: string | undefined;
    metadata?: Record<string, string> | undefined;
}, {
    operation: "updateCustomer";
    customerId: string;
    description?: string | undefined;
    name?: string | undefined;
    email?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    phone?: string | undefined;
    metadata?: Record<string, string> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createSubscription">;
    customer: z.ZodString;
    priceId: z.ZodString;
    quantity: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    trialPeriodDays: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    paymentBehavior: z.ZodOptional<z.ZodEnum<["default_incomplete", "allow_incomplete", "error_if_incomplete"]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createSubscription";
    customer: string;
    priceId: string;
    quantity: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    trialPeriodDays?: number | undefined;
    paymentBehavior?: "default_incomplete" | "allow_incomplete" | "error_if_incomplete" | undefined;
}, {
    operation: "createSubscription";
    customer: string;
    priceId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    quantity?: number | undefined;
    trialPeriodDays?: number | undefined;
    paymentBehavior?: "default_incomplete" | "allow_incomplete" | "error_if_incomplete" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"cancelSubscription">;
    subscriptionId: z.ZodString;
    cancelAtPeriodEnd: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "cancelSubscription";
    subscriptionId: string;
    cancelAtPeriodEnd: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "cancelSubscription";
    subscriptionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cancelAtPeriodEnd?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateSubscription">;
    subscriptionId: z.ZodString;
    priceId: z.ZodOptional<z.ZodString>;
    quantity: z.ZodOptional<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    prorationBehavior: z.ZodOptional<z.ZodEnum<["create_prorations", "always_invoice", "none"]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateSubscription";
    subscriptionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    priceId?: string | undefined;
    quantity?: number | undefined;
    prorationBehavior?: "none" | "create_prorations" | "always_invoice" | undefined;
}, {
    operation: "updateSubscription";
    subscriptionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    priceId?: string | undefined;
    quantity?: number | undefined;
    prorationBehavior?: "none" | "create_prorations" | "always_invoice" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createInvoice">;
    customer: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    autoAdvance: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    collectionMethod: z.ZodDefault<z.ZodOptional<z.ZodEnum<["charge_automatically", "send_invoice"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createInvoice";
    customer: string;
    autoAdvance: boolean;
    collectionMethod: "charge_automatically" | "send_invoice";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
}, {
    operation: "createInvoice";
    customer: string;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    autoAdvance?: boolean | undefined;
    collectionMethod?: "charge_automatically" | "send_invoice" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getInvoice">;
    invoiceId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getInvoice";
    invoiceId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getInvoice";
    invoiceId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listInvoices">;
    customer: z.ZodOptional<z.ZodString>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    startingAfter: z.ZodOptional<z.ZodString>;
    status: z.ZodOptional<z.ZodEnum<["draft", "open", "paid", "uncollectible", "void"]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listInvoices";
    limit: number;
    status?: "open" | "void" | "draft" | "paid" | "uncollectible" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    customer?: string | undefined;
    startingAfter?: string | undefined;
}, {
    operation: "listInvoices";
    status?: "open" | "void" | "draft" | "paid" | "uncollectible" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    customer?: string | undefined;
    startingAfter?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createProduct">;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    images: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    statementDescriptor: z.ZodOptional<z.ZodString>;
    unitLabel: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "createProduct";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    images?: string[] | undefined;
    metadata?: Record<string, string> | undefined;
    statementDescriptor?: string | undefined;
    unitLabel?: string | undefined;
}, {
    name: string;
    operation: "createProduct";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    images?: string[] | undefined;
    metadata?: Record<string, string> | undefined;
    statementDescriptor?: string | undefined;
    unitLabel?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createPrice">;
    product: z.ZodString;
    unitAmount: z.ZodNumber;
    currency: z.ZodDefault<z.ZodString>;
    recurring: z.ZodOptional<z.ZodObject<{
        interval: z.ZodEnum<["day", "week", "month", "year"]>;
        intervalCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        usageType: z.ZodDefault<z.ZodOptional<z.ZodEnum<["licensed", "metered"]>>>;
    }, "strip", z.ZodTypeAny, {
        interval: "week" | "month" | "year" | "day";
        intervalCount: number;
        usageType: "licensed" | "metered";
    }, {
        interval: "week" | "month" | "year" | "day";
        intervalCount?: number | undefined;
        usageType?: "licensed" | "metered" | undefined;
    }>>;
    nickname: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createPrice";
    currency: string;
    product: string;
    unitAmount: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    nickname?: string | undefined;
    recurring?: {
        interval: "week" | "month" | "year" | "day";
        intervalCount: number;
        usageType: "licensed" | "metered";
    } | undefined;
}, {
    operation: "createPrice";
    product: string;
    unitAmount: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata?: Record<string, string> | undefined;
    currency?: string | undefined;
    nickname?: string | undefined;
    recurring?: {
        interval: "week" | "month" | "year" | "day";
        intervalCount?: number | undefined;
        usageType?: "licensed" | "metered" | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"handleWebhook">;
    payload: z.ZodString;
    signature: z.ZodString;
    secret: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "handleWebhook";
    payload: string;
    signature: string;
    secret: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "handleWebhook";
    payload: string;
    signature: string;
    secret: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
export type StripeBubbleParams = z.input<typeof StripeBubbleParamsSchema>;
declare const StripeBubbleResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createPaymentIntent">;
    result: z.ZodObject<{
        id: z.ZodString;
        amount: z.ZodNumber;
        currency: z.ZodString;
        status: z.ZodString;
        clientSecret: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createPaymentIntent";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    };
}, {
    operation: "createPaymentIntent";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"confirmPayment">;
    result: z.ZodObject<{
        id: z.ZodString;
        amount: z.ZodNumber;
        currency: z.ZodString;
        status: z.ZodString;
        clientSecret: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "confirmPayment";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    };
}, {
    operation: "confirmPayment";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        amount: number;
        description?: string | undefined;
        clientSecret?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"refundPayment">;
    result: z.ZodObject<{
        id: z.ZodString;
        amount: z.ZodNumber;
        currency: z.ZodString;
        status: z.ZodString;
        paymentIntentId: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        currency: string;
        amount: number;
        paymentIntentId: string;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        currency: string;
        amount: number;
        paymentIntentId: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "refundPayment";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        currency: string;
        amount: number;
        paymentIntentId: string;
    };
}, {
    operation: "refundPayment";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        currency: string;
        amount: number;
        paymentIntentId: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createCustomer">;
    result: z.ZodObject<{
        id: z.ZodString;
        email: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        phone: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    }, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createCustomer";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    };
}, {
    operation: "createCustomer";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getCustomer">;
    result: z.ZodObject<{
        id: z.ZodString;
        email: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        phone: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    }, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getCustomer";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    };
}, {
    operation: "getCustomer";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateCustomer">;
    result: z.ZodObject<{
        id: z.ZodString;
        email: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        phone: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    }, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "updateCustomer";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    };
}, {
    operation: "updateCustomer";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        phone?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createSubscription">;
    result: z.ZodObject<{
        id: z.ZodString;
        customerId: z.ZodString;
        status: z.ZodString;
        currentPeriodStart: z.ZodString;
        currentPeriodEnd: z.ZodString;
        cancelAtPeriodEnd: z.ZodBoolean;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createSubscription";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    };
}, {
    operation: "createSubscription";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"cancelSubscription">;
    result: z.ZodObject<{
        id: z.ZodString;
        customerId: z.ZodString;
        status: z.ZodString;
        currentPeriodStart: z.ZodString;
        currentPeriodEnd: z.ZodString;
        cancelAtPeriodEnd: z.ZodBoolean;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "cancelSubscription";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    };
}, {
    operation: "cancelSubscription";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateSubscription">;
    result: z.ZodObject<{
        id: z.ZodString;
        customerId: z.ZodString;
        status: z.ZodString;
        currentPeriodStart: z.ZodString;
        currentPeriodEnd: z.ZodString;
        cancelAtPeriodEnd: z.ZodBoolean;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "updateSubscription";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    };
}, {
    operation: "updateSubscription";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        customerId: string;
        cancelAtPeriodEnd: boolean;
        currentPeriodStart: string;
        currentPeriodEnd: string;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createInvoice">;
    result: z.ZodObject<{
        id: z.ZodString;
        number: z.ZodOptional<z.ZodString>;
        status: z.ZodString;
        amountDue: z.ZodNumber;
        currency: z.ZodString;
        customer: z.ZodString;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createInvoice";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    };
}, {
    operation: "createInvoice";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getInvoice">;
    result: z.ZodObject<{
        id: z.ZodString;
        number: z.ZodOptional<z.ZodString>;
        status: z.ZodString;
        amountDue: z.ZodNumber;
        currency: z.ZodString;
        customer: z.ZodString;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getInvoice";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    };
}, {
    operation: "getInvoice";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        customer: string;
        amountDue: number;
        number?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listInvoices">;
    result: z.ZodObject<{
        invoices: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            number: z.ZodOptional<z.ZodString>;
            status: z.ZodString;
            amountDue: z.ZodNumber;
            currency: z.ZodString;
            customer: z.ZodString;
            createdAt: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            status: string;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }, {
            status: string;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }>, "many">;
        hasMore: z.ZodBoolean;
        count: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        count: number;
        hasMore: boolean;
        invoices: {
            status: string;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }[];
    }, {
        error: string;
        success: boolean;
        count: number;
        hasMore: boolean;
        invoices: {
            status: string;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "listInvoices";
    result: {
        error: string;
        success: boolean;
        count: number;
        hasMore: boolean;
        invoices: {
            status: string;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }[];
    };
}, {
    operation: "listInvoices";
    result: {
        error: string;
        success: boolean;
        count: number;
        hasMore: boolean;
        invoices: {
            status: string;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }[];
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createProduct">;
    result: z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        active: z.ZodOptional<z.ZodBoolean>;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        name: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        active?: boolean | undefined;
    }, {
        error: string;
        name: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        active?: boolean | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createProduct";
    result: {
        error: string;
        name: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        active?: boolean | undefined;
    };
}, {
    operation: "createProduct";
    result: {
        error: string;
        name: string;
        success: boolean;
        id: string;
        createdAt: string;
        description?: string | undefined;
        active?: boolean | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createPrice">;
    result: z.ZodObject<{
        id: z.ZodString;
        productId: z.ZodString;
        unitAmount: z.ZodNumber;
        currency: z.ZodString;
        recurring: z.ZodOptional<z.ZodAny>;
        active: z.ZodBoolean;
        createdAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        unitAmount: number;
        active: boolean;
        productId: string;
        recurring?: any;
    }, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        unitAmount: number;
        active: boolean;
        productId: string;
        recurring?: any;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "createPrice";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        unitAmount: number;
        active: boolean;
        productId: string;
        recurring?: any;
    };
}, {
    operation: "createPrice";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        currency: string;
        unitAmount: number;
        active: boolean;
        productId: string;
        recurring?: any;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"handleWebhook">;
    result: z.ZodObject<{
        id: z.ZodString;
        type: z.ZodString;
        data: z.ZodAny;
        processed: z.ZodBoolean;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        type: string;
        success: boolean;
        id: string;
        processed: boolean;
        data?: any;
    }, {
        error: string;
        type: string;
        success: boolean;
        id: string;
        processed: boolean;
        data?: any;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "handleWebhook";
    result: {
        error: string;
        type: string;
        success: boolean;
        id: string;
        processed: boolean;
        data?: any;
    };
}, {
    operation: "handleWebhook";
    result: {
        error: string;
        type: string;
        success: boolean;
        id: string;
        processed: boolean;
        data?: any;
    };
}>]>;
type StripeBubbleResult = z.output<typeof StripeBubbleResultSchema>;
/**
 * Stripe Bubble - Complete Service Bubble Implementation
 *
 * Provides comprehensive integration with the Stripe API for payment processing,
 * customer management, subscriptions, invoicing, and webhook handling.
 *
 * @template T - Stripe bubble parameters type
 */
export declare class StripeBubble<T extends StripeBubbleParams = StripeBubbleParams> extends ServiceBubble<T, any> {
    static readonly type: "service";
    static readonly service = "stripe";
    static readonly authType: "apikey";
    static readonly bubbleName = "stripe";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createPaymentIntent">;
        amount: z.ZodNumber;
        currency: z.ZodDefault<z.ZodString>;
        customer: z.ZodOptional<z.ZodString>;
        paymentMethod: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        confirm: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        captureMethod: z.ZodDefault<z.ZodOptional<z.ZodEnum<["automatic", "manual"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createPaymentIntent";
        currency: string;
        amount: number;
        confirm: boolean;
        captureMethod: "manual" | "automatic";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        customer?: string | undefined;
        paymentMethod?: string | undefined;
    }, {
        operation: "createPaymentIntent";
        amount: number;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        currency?: string | undefined;
        customer?: string | undefined;
        paymentMethod?: string | undefined;
        confirm?: boolean | undefined;
        captureMethod?: "manual" | "automatic" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"confirmPayment">;
        paymentIntentId: z.ZodString;
        paymentMethod: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "confirmPayment";
        paymentIntentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        paymentMethod?: string | undefined;
    }, {
        operation: "confirmPayment";
        paymentIntentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        paymentMethod?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"refundPayment">;
        paymentIntentId: z.ZodString;
        amount: z.ZodOptional<z.ZodNumber>;
        reason: z.ZodOptional<z.ZodEnum<["duplicate", "fraudulent", "requested_by_customer", "other"]>>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "refundPayment";
        paymentIntentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        amount?: number | undefined;
        reason?: "other" | "duplicate" | "fraudulent" | "requested_by_customer" | undefined;
    }, {
        operation: "refundPayment";
        paymentIntentId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        amount?: number | undefined;
        reason?: "other" | "duplicate" | "fraudulent" | "requested_by_customer" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createCustomer">;
        email: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        phone: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createCustomer";
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        phone?: string | undefined;
        metadata?: Record<string, string> | undefined;
    }, {
        operation: "createCustomer";
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        phone?: string | undefined;
        metadata?: Record<string, string> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getCustomer">;
        customerId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getCustomer";
        customerId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getCustomer";
        customerId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateCustomer">;
        customerId: z.ZodString;
        email: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        phone: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateCustomer";
        customerId: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        phone?: string | undefined;
        metadata?: Record<string, string> | undefined;
    }, {
        operation: "updateCustomer";
        customerId: string;
        description?: string | undefined;
        name?: string | undefined;
        email?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        phone?: string | undefined;
        metadata?: Record<string, string> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createSubscription">;
        customer: z.ZodString;
        priceId: z.ZodString;
        quantity: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        trialPeriodDays: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        paymentBehavior: z.ZodOptional<z.ZodEnum<["default_incomplete", "allow_incomplete", "error_if_incomplete"]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createSubscription";
        customer: string;
        priceId: string;
        quantity: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        trialPeriodDays?: number | undefined;
        paymentBehavior?: "default_incomplete" | "allow_incomplete" | "error_if_incomplete" | undefined;
    }, {
        operation: "createSubscription";
        customer: string;
        priceId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        quantity?: number | undefined;
        trialPeriodDays?: number | undefined;
        paymentBehavior?: "default_incomplete" | "allow_incomplete" | "error_if_incomplete" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"cancelSubscription">;
        subscriptionId: z.ZodString;
        cancelAtPeriodEnd: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "cancelSubscription";
        subscriptionId: string;
        cancelAtPeriodEnd: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "cancelSubscription";
        subscriptionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cancelAtPeriodEnd?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateSubscription">;
        subscriptionId: z.ZodString;
        priceId: z.ZodOptional<z.ZodString>;
        quantity: z.ZodOptional<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        prorationBehavior: z.ZodOptional<z.ZodEnum<["create_prorations", "always_invoice", "none"]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateSubscription";
        subscriptionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        priceId?: string | undefined;
        quantity?: number | undefined;
        prorationBehavior?: "none" | "create_prorations" | "always_invoice" | undefined;
    }, {
        operation: "updateSubscription";
        subscriptionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        priceId?: string | undefined;
        quantity?: number | undefined;
        prorationBehavior?: "none" | "create_prorations" | "always_invoice" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createInvoice">;
        customer: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        autoAdvance: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        collectionMethod: z.ZodDefault<z.ZodOptional<z.ZodEnum<["charge_automatically", "send_invoice"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createInvoice";
        customer: string;
        autoAdvance: boolean;
        collectionMethod: "charge_automatically" | "send_invoice";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
    }, {
        operation: "createInvoice";
        customer: string;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        autoAdvance?: boolean | undefined;
        collectionMethod?: "charge_automatically" | "send_invoice" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getInvoice">;
        invoiceId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getInvoice";
        invoiceId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getInvoice";
        invoiceId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listInvoices">;
        customer: z.ZodOptional<z.ZodString>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        startingAfter: z.ZodOptional<z.ZodString>;
        status: z.ZodOptional<z.ZodEnum<["draft", "open", "paid", "uncollectible", "void"]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listInvoices";
        limit: number;
        status?: "open" | "void" | "draft" | "paid" | "uncollectible" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        customer?: string | undefined;
        startingAfter?: string | undefined;
    }, {
        operation: "listInvoices";
        status?: "open" | "void" | "draft" | "paid" | "uncollectible" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        customer?: string | undefined;
        startingAfter?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createProduct">;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        images: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        statementDescriptor: z.ZodOptional<z.ZodString>;
        unitLabel: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "createProduct";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        images?: string[] | undefined;
        metadata?: Record<string, string> | undefined;
        statementDescriptor?: string | undefined;
        unitLabel?: string | undefined;
    }, {
        name: string;
        operation: "createProduct";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        images?: string[] | undefined;
        metadata?: Record<string, string> | undefined;
        statementDescriptor?: string | undefined;
        unitLabel?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createPrice">;
        product: z.ZodString;
        unitAmount: z.ZodNumber;
        currency: z.ZodDefault<z.ZodString>;
        recurring: z.ZodOptional<z.ZodObject<{
            interval: z.ZodEnum<["day", "week", "month", "year"]>;
            intervalCount: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
            usageType: z.ZodDefault<z.ZodOptional<z.ZodEnum<["licensed", "metered"]>>>;
        }, "strip", z.ZodTypeAny, {
            interval: "week" | "month" | "year" | "day";
            intervalCount: number;
            usageType: "licensed" | "metered";
        }, {
            interval: "week" | "month" | "year" | "day";
            intervalCount?: number | undefined;
            usageType?: "licensed" | "metered" | undefined;
        }>>;
        nickname: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createPrice";
        currency: string;
        product: string;
        unitAmount: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        nickname?: string | undefined;
        recurring?: {
            interval: "week" | "month" | "year" | "day";
            intervalCount: number;
            usageType: "licensed" | "metered";
        } | undefined;
    }, {
        operation: "createPrice";
        product: string;
        unitAmount: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata?: Record<string, string> | undefined;
        currency?: string | undefined;
        nickname?: string | undefined;
        recurring?: {
            interval: "week" | "month" | "year" | "day";
            intervalCount?: number | undefined;
            usageType?: "licensed" | "metered" | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"handleWebhook">;
        payload: z.ZodString;
        signature: z.ZodString;
        secret: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "handleWebhook";
        payload: string;
        signature: string;
        secret: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "handleWebhook";
        payload: string;
        signature: string;
        secret: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createPaymentIntent">;
        result: z.ZodObject<{
            id: z.ZodString;
            amount: z.ZodNumber;
            currency: z.ZodString;
            status: z.ZodString;
            clientSecret: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createPaymentIntent";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        };
    }, {
        operation: "createPaymentIntent";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"confirmPayment">;
        result: z.ZodObject<{
            id: z.ZodString;
            amount: z.ZodNumber;
            currency: z.ZodString;
            status: z.ZodString;
            clientSecret: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "confirmPayment";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        };
    }, {
        operation: "confirmPayment";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            amount: number;
            description?: string | undefined;
            clientSecret?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"refundPayment">;
        result: z.ZodObject<{
            id: z.ZodString;
            amount: z.ZodNumber;
            currency: z.ZodString;
            status: z.ZodString;
            paymentIntentId: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            currency: string;
            amount: number;
            paymentIntentId: string;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            currency: string;
            amount: number;
            paymentIntentId: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "refundPayment";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            currency: string;
            amount: number;
            paymentIntentId: string;
        };
    }, {
        operation: "refundPayment";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            currency: string;
            amount: number;
            paymentIntentId: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createCustomer">;
        result: z.ZodObject<{
            id: z.ZodString;
            email: z.ZodOptional<z.ZodString>;
            name: z.ZodOptional<z.ZodString>;
            phone: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        }, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createCustomer";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        };
    }, {
        operation: "createCustomer";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getCustomer">;
        result: z.ZodObject<{
            id: z.ZodString;
            email: z.ZodOptional<z.ZodString>;
            name: z.ZodOptional<z.ZodString>;
            phone: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        }, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getCustomer";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        };
    }, {
        operation: "getCustomer";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateCustomer">;
        result: z.ZodObject<{
            id: z.ZodString;
            email: z.ZodOptional<z.ZodString>;
            name: z.ZodOptional<z.ZodString>;
            phone: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        }, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateCustomer";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        };
    }, {
        operation: "updateCustomer";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            name?: string | undefined;
            email?: string | undefined;
            phone?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createSubscription">;
        result: z.ZodObject<{
            id: z.ZodString;
            customerId: z.ZodString;
            status: z.ZodString;
            currentPeriodStart: z.ZodString;
            currentPeriodEnd: z.ZodString;
            cancelAtPeriodEnd: z.ZodBoolean;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createSubscription";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        };
    }, {
        operation: "createSubscription";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"cancelSubscription">;
        result: z.ZodObject<{
            id: z.ZodString;
            customerId: z.ZodString;
            status: z.ZodString;
            currentPeriodStart: z.ZodString;
            currentPeriodEnd: z.ZodString;
            cancelAtPeriodEnd: z.ZodBoolean;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "cancelSubscription";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        };
    }, {
        operation: "cancelSubscription";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateSubscription">;
        result: z.ZodObject<{
            id: z.ZodString;
            customerId: z.ZodString;
            status: z.ZodString;
            currentPeriodStart: z.ZodString;
            currentPeriodEnd: z.ZodString;
            cancelAtPeriodEnd: z.ZodBoolean;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateSubscription";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        };
    }, {
        operation: "updateSubscription";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            customerId: string;
            cancelAtPeriodEnd: boolean;
            currentPeriodStart: string;
            currentPeriodEnd: string;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createInvoice">;
        result: z.ZodObject<{
            id: z.ZodString;
            number: z.ZodOptional<z.ZodString>;
            status: z.ZodString;
            amountDue: z.ZodNumber;
            currency: z.ZodString;
            customer: z.ZodString;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createInvoice";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        };
    }, {
        operation: "createInvoice";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getInvoice">;
        result: z.ZodObject<{
            id: z.ZodString;
            number: z.ZodOptional<z.ZodString>;
            status: z.ZodString;
            amountDue: z.ZodNumber;
            currency: z.ZodString;
            customer: z.ZodString;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getInvoice";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        };
    }, {
        operation: "getInvoice";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            customer: string;
            amountDue: number;
            number?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listInvoices">;
        result: z.ZodObject<{
            invoices: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                number: z.ZodOptional<z.ZodString>;
                status: z.ZodString;
                amountDue: z.ZodNumber;
                currency: z.ZodString;
                customer: z.ZodString;
                createdAt: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                status: string;
                id: string;
                createdAt: string;
                currency: string;
                customer: string;
                amountDue: number;
                number?: string | undefined;
            }, {
                status: string;
                id: string;
                createdAt: string;
                currency: string;
                customer: string;
                amountDue: number;
                number?: string | undefined;
            }>, "many">;
            hasMore: z.ZodBoolean;
            count: z.ZodNumber;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            count: number;
            hasMore: boolean;
            invoices: {
                status: string;
                id: string;
                createdAt: string;
                currency: string;
                customer: string;
                amountDue: number;
                number?: string | undefined;
            }[];
        }, {
            error: string;
            success: boolean;
            count: number;
            hasMore: boolean;
            invoices: {
                status: string;
                id: string;
                createdAt: string;
                currency: string;
                customer: string;
                amountDue: number;
                number?: string | undefined;
            }[];
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "listInvoices";
        result: {
            error: string;
            success: boolean;
            count: number;
            hasMore: boolean;
            invoices: {
                status: string;
                id: string;
                createdAt: string;
                currency: string;
                customer: string;
                amountDue: number;
                number?: string | undefined;
            }[];
        };
    }, {
        operation: "listInvoices";
        result: {
            error: string;
            success: boolean;
            count: number;
            hasMore: boolean;
            invoices: {
                status: string;
                id: string;
                createdAt: string;
                currency: string;
                customer: string;
                amountDue: number;
                number?: string | undefined;
            }[];
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createProduct">;
        result: z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            active: z.ZodOptional<z.ZodBoolean>;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            name: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            active?: boolean | undefined;
        }, {
            error: string;
            name: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            active?: boolean | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createProduct";
        result: {
            error: string;
            name: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            active?: boolean | undefined;
        };
    }, {
        operation: "createProduct";
        result: {
            error: string;
            name: string;
            success: boolean;
            id: string;
            createdAt: string;
            description?: string | undefined;
            active?: boolean | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createPrice">;
        result: z.ZodObject<{
            id: z.ZodString;
            productId: z.ZodString;
            unitAmount: z.ZodNumber;
            currency: z.ZodString;
            recurring: z.ZodOptional<z.ZodAny>;
            active: z.ZodBoolean;
            createdAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            unitAmount: number;
            active: boolean;
            productId: string;
            recurring?: any;
        }, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            unitAmount: number;
            active: boolean;
            productId: string;
            recurring?: any;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "createPrice";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            unitAmount: number;
            active: boolean;
            productId: string;
            recurring?: any;
        };
    }, {
        operation: "createPrice";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            currency: string;
            unitAmount: number;
            active: boolean;
            productId: string;
            recurring?: any;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"handleWebhook">;
        result: z.ZodObject<{
            id: z.ZodString;
            type: z.ZodString;
            data: z.ZodAny;
            processed: z.ZodBoolean;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            type: string;
            success: boolean;
            id: string;
            processed: boolean;
            data?: any;
        }, {
            error: string;
            type: string;
            success: boolean;
            id: string;
            processed: boolean;
            data?: any;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "handleWebhook";
        result: {
            error: string;
            type: string;
            success: boolean;
            id: string;
            processed: boolean;
            data?: any;
        };
    }, {
        operation: "handleWebhook";
        result: {
            error: string;
            type: string;
            success: boolean;
            id: string;
            processed: boolean;
            data?: any;
        };
    }>]>;
    static readonly shortDescription = "Complete Stripe integration for payments and billing";
    static readonly longDescription = "\n    Comprehensive Stripe service bubble for all payment operations.\n\n    Operations:\n    1. createPaymentIntent - Create payment intents for one-time payments\n    2. confirmPayment - Confirm and process payment intents\n    3. refundPayment - Create refunds for payments\n    4. createCustomer - Create new customer records\n    5. getCustomer - Retrieve customer information\n    6. updateCustomer - Update customer details\n    7. createSubscription - Create recurring subscriptions\n    8. cancelSubscription - Cancel subscriptions\n    9. updateSubscription - Modify subscription details\n    10. createInvoice - Create and send invoices\n    11. getInvoice - Retrieve invoice details\n    12. listInvoices - List customer invoices\n    13. createProduct - Create products\n    14. createPrice - Create product prices\n    15. handleWebhook - Verify and process webhooks\n\n    Features:\n    - Full payment lifecycle management\n    - Subscription and recurring billing\n    - Invoice generation and management\n    - Product and price management\n    - Webhook signature verification\n    - Customer management\n    - Refund processing\n    - Resilience patterns with automatic retries\n  ";
    static readonly alias = "stripe";
    private client;
    private resilience;
    /**
     * Create a new Stripe Bubble instance
     * @param params - Operation parameters
     * @param context - Bubble execution context
     */
    constructor(params: T, context?: BubbleContext);
    /**
     * Test the validity of the Stripe API credentials
     * @returns Promise that resolves to true if credentials are valid, false otherwise
     */
    testCredential(): Promise<boolean>;
    /**
     * Extract the Stripe API key from credentials
     * @returns Stripe API key or undefined if not found
     * @throws AuthenticationError if credentials are invalid or missing
     */
    protected chooseCredential(): string | undefined;
    /**
     * Execute the Stripe operation specified in params
     * @param context - Bubble execution context (unused)
     * @returns Promise that resolves with the operation result
     * @throws AuthenticationError if API key is missing
     */
    protected performAction(context?: BubbleContext): Promise<Extract<StripeBubbleResult, {
        operation: T['operation'];
    }>>;
    /**
     * Create a payment intent for one-time payment
     * @param params - Payment intent parameters
     * @returns Promise that resolves with the payment intent result
     */
    private createPaymentIntent;
    /**
     * Confirm a payment intent
     * @param params - Confirm payment parameters
     * @returns Promise that resolves with the payment intent result
     */
    private confirmPayment;
    /**
     * Create a refund for a payment
     * @param params - Refund parameters
     * @returns Promise that resolves with the refund result
     */
    private refundPayment;
    /**
     * Create a new customer in Stripe
     * @param params - Customer creation parameters
     * @returns Promise that resolves with the customer result
     */
    private createCustomer;
    private getCustomer;
    private updateCustomer;
    private createSubscription;
    private cancelSubscription;
    private updateSubscription;
    private createInvoice;
    private getInvoice;
    private listInvoices;
    private createProduct;
    private createPrice;
    private handleWebhook;
    /**
     * Create an error result object
     * @param error - Error message
     * @returns Error result object
     */
    private errorResult;
}
export {};
//# sourceMappingURL=stripe-bubble.d.ts.map