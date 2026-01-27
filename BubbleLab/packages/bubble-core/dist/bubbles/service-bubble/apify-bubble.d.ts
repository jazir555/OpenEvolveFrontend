import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const ApifyBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"runActor">;
    actorId: z.ZodString;
    input: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    buildId: z.ZodOptional<z.ZodString>;
    memory: z.ZodOptional<z.ZodNumber>;
    timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    waitForFinish: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    build: z.ZodDefault<z.ZodOptional<z.ZodEnum<["latest", "specific"]>>>;
    buildNumber: z.ZodOptional<z.ZodString>;
    maxItems: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    maxItems: number;
    input: Record<string, unknown>;
    operation: "runActor";
    actorId: string;
    waitForFinish: boolean;
    build: "latest" | "specific";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    memory?: number | undefined;
    buildId?: string | undefined;
    buildNumber?: string | undefined;
}, {
    input: Record<string, unknown>;
    operation: "runActor";
    actorId: string;
    timeout?: number | undefined;
    maxItems?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    waitForFinish?: boolean | undefined;
    memory?: number | undefined;
    buildId?: string | undefined;
    build?: "latest" | "specific" | undefined;
    buildNumber?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getActor">;
    actorId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getActor";
    actorId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getActor";
    actorId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listActors">;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    search: z.ZodOptional<z.ZodString>;
    sortBy: z.ZodDefault<z.ZodOptional<z.ZodEnum<["createdAt", "modifiedAt", "usageStats"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listActors";
    limit: number;
    offset: number;
    sortBy: "createdAt" | "modifiedAt" | "usageStats";
    search?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "listActors";
    search?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    sortBy?: "createdAt" | "modifiedAt" | "usageStats" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"buildActor">;
    actorId: z.ZodString;
    buildTag: z.ZodOptional<z.ZodString>;
    version: z.ZodOptional<z.ZodString>;
    waitForFinish: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "buildActor";
    actorId: string;
    waitForFinish: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    version?: string | undefined;
    buildTag?: string | undefined;
}, {
    operation: "buildActor";
    actorId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    waitForFinish?: boolean | undefined;
    version?: string | undefined;
    buildTag?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRun">;
    runId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getRun";
    runId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getRun";
    runId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"waitForRun">;
    runId: z.ZodString;
    waitFor: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    waitInterval: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "waitForRun";
    runId: string;
    waitFor: number;
    waitInterval: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "waitForRun";
    runId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    waitFor?: number | undefined;
    waitInterval?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"stopRun">;
    runId: z.ZodString;
    gracefully: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "stopRun";
    runId: string;
    gracefully: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "stopRun";
    runId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    gracefully?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listRuns">;
    actorId: z.ZodOptional<z.ZodString>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    status: z.ZodOptional<z.ZodEnum<["READY", "RUNNING", "SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listRuns";
    limit: number;
    offset: number;
    status?: "READY" | "RUNNING" | "SUCCEEDED" | "FAILED" | "TIMED-OUT" | "ABORTED" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    actorId?: string | undefined;
}, {
    operation: "listRuns";
    status?: "READY" | "RUNNING" | "SUCCEEDED" | "FAILED" | "TIMED-OUT" | "ABORTED" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    actorId?: string | undefined;
    offset?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDataset">;
    datasetId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getDataset";
    datasetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getDataset";
    datasetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDatasetItems">;
    datasetId: z.ZodString;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    clean: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["json", "csv", "xml", "xlsx", "html"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    format: "xml" | "html" | "json" | "csv" | "xlsx";
    operation: "getDatasetItems";
    limit: number;
    datasetId: string;
    offset: number;
    clean: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getDatasetItems";
    datasetId: string;
    format?: "xml" | "html" | "json" | "csv" | "xlsx" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    clean?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"downloadDataset">;
    datasetId: z.ZodString;
    format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["json", "csv", "xlsx", "html"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    format: "html" | "json" | "csv" | "xlsx";
    operation: "downloadDataset";
    datasetId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "downloadDataset";
    datasetId: string;
    format?: "html" | "json" | "csv" | "xlsx" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"webScrape">;
    url: z.ZodString;
    selectors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    waitForSelector: z.ZodOptional<z.ZodString>;
    timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    proxyConfiguration: z.ZodOptional<z.ZodObject<{
        useApifyProxy: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        proxyGroups: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        countryCode: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        useApifyProxy: boolean;
        countryCode?: string | undefined;
        proxyGroups?: string[] | undefined;
    }, {
        countryCode?: string | undefined;
        useApifyProxy?: boolean | undefined;
        proxyGroups?: string[] | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    url: string;
    operation: "webScrape";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    selectors?: string[] | undefined;
    waitForSelector?: string | undefined;
    proxyConfiguration?: {
        useApifyProxy: boolean;
        countryCode?: string | undefined;
        proxyGroups?: string[] | undefined;
    } | undefined;
}, {
    url: string;
    operation: "webScrape";
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    selectors?: string[] | undefined;
    waitForSelector?: string | undefined;
    proxyConfiguration?: {
        countryCode?: string | undefined;
        useApifyProxy?: boolean | undefined;
        proxyGroups?: string[] | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"crawlWebsite">;
    startUrls: z.ZodArray<z.ZodString, "many">;
    maxPages: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    proxyConfiguration: z.ZodOptional<z.ZodObject<{
        useApifyProxy: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        proxyGroups: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        countryCode: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        useApifyProxy: boolean;
        countryCode?: string | undefined;
        proxyGroups?: string[] | undefined;
    }, {
        countryCode?: string | undefined;
        useApifyProxy?: boolean | undefined;
        proxyGroups?: string[] | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "crawlWebsite";
    startUrls: string[];
    maxPages: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    proxyConfiguration?: {
        useApifyProxy: boolean;
        countryCode?: string | undefined;
        proxyGroups?: string[] | undefined;
    } | undefined;
}, {
    operation: "crawlWebsite";
    startUrls: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxPages?: number | undefined;
    proxyConfiguration?: {
        countryCode?: string | undefined;
        useApifyProxy?: boolean | undefined;
        proxyGroups?: string[] | undefined;
    } | undefined;
}>]>;
type ApifyBubbleParams = z.input<typeof ApifyBubbleParamsSchema>;
declare const ApifyBubbleResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"runActor">;
    result: z.ZodObject<{
        runId: z.ZodString;
        status: z.ZodString;
        actorId: z.ZodString;
        startedAt: z.ZodOptional<z.ZodString>;
        finishedAt: z.ZodOptional<z.ZodString>;
        datasetId: z.ZodOptional<z.ZodString>;
        itemsCount: z.ZodOptional<z.ZodNumber>;
        consoleUrl: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        actorId: string;
        runId: string;
        consoleUrl: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        startedAt?: string | undefined;
        finishedAt?: string | undefined;
    }, {
        error: string;
        status: string;
        success: boolean;
        actorId: string;
        runId: string;
        consoleUrl: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        startedAt?: string | undefined;
        finishedAt?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "runActor";
    result: {
        error: string;
        status: string;
        success: boolean;
        actorId: string;
        runId: string;
        consoleUrl: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        startedAt?: string | undefined;
        finishedAt?: string | undefined;
    };
}, {
    operation: "runActor";
    result: {
        error: string;
        status: string;
        success: boolean;
        actorId: string;
        runId: string;
        consoleUrl: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        startedAt?: string | undefined;
        finishedAt?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getActor">;
    result: z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        versions: z.ZodArray<z.ZodObject<{
            versionNumber: z.ZodString;
            buildStatus: z.ZodString;
            createdAt: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            createdAt: string;
            versionNumber: string;
            buildStatus: string;
        }, {
            createdAt: string;
            versionNumber: string;
            buildStatus: string;
        }>, "many">;
        defaultRunOptions: z.ZodOptional<z.ZodObject<{
            build: z.ZodString;
            timeoutSecs: z.ZodNumber;
            memoryMbytes: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            build: string;
            timeoutSecs: number;
            memoryMbytes: number;
        }, {
            build: string;
            timeoutSecs: number;
            memoryMbytes: number;
        }>>;
        stats: z.ZodOptional<z.ZodObject<{
            totalRuns: z.ZodNumber;
            usersCount: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            totalRuns: number;
            usersCount: number;
        }, {
            totalRuns: number;
            usersCount: number;
        }>>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        name: string;
        success: boolean;
        id: string;
        versions: {
            createdAt: string;
            versionNumber: string;
            buildStatus: string;
        }[];
        description?: string | undefined;
        stats?: {
            totalRuns: number;
            usersCount: number;
        } | undefined;
        defaultRunOptions?: {
            build: string;
            timeoutSecs: number;
            memoryMbytes: number;
        } | undefined;
    }, {
        error: string;
        name: string;
        success: boolean;
        id: string;
        versions: {
            createdAt: string;
            versionNumber: string;
            buildStatus: string;
        }[];
        description?: string | undefined;
        stats?: {
            totalRuns: number;
            usersCount: number;
        } | undefined;
        defaultRunOptions?: {
            build: string;
            timeoutSecs: number;
            memoryMbytes: number;
        } | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getActor";
    result: {
        error: string;
        name: string;
        success: boolean;
        id: string;
        versions: {
            createdAt: string;
            versionNumber: string;
            buildStatus: string;
        }[];
        description?: string | undefined;
        stats?: {
            totalRuns: number;
            usersCount: number;
        } | undefined;
        defaultRunOptions?: {
            build: string;
            timeoutSecs: number;
            memoryMbytes: number;
        } | undefined;
    };
}, {
    operation: "getActor";
    result: {
        error: string;
        name: string;
        success: boolean;
        id: string;
        versions: {
            createdAt: string;
            versionNumber: string;
            buildStatus: string;
        }[];
        description?: string | undefined;
        stats?: {
            totalRuns: number;
            usersCount: number;
        } | undefined;
        defaultRunOptions?: {
            build: string;
            timeoutSecs: number;
            memoryMbytes: number;
        } | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getRun">;
    result: z.ZodObject<{
        id: z.ZodString;
        status: z.ZodString;
        actorId: z.ZodString;
        startedAt: z.ZodString;
        finishedAt: z.ZodOptional<z.ZodString>;
        datasetId: z.ZodOptional<z.ZodString>;
        itemsCount: z.ZodOptional<z.ZodNumber>;
        usage: z.ZodOptional<z.ZodObject<{
            computeUnits: z.ZodNumber;
            duration: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            duration: number;
            computeUnits: number;
        }, {
            duration: number;
            computeUnits: number;
        }>>;
        consoleUrl: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        actorId: string;
        consoleUrl: string;
        startedAt: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        usage?: {
            duration: number;
            computeUnits: number;
        } | undefined;
        finishedAt?: string | undefined;
    }, {
        error: string;
        status: string;
        success: boolean;
        id: string;
        actorId: string;
        consoleUrl: string;
        startedAt: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        usage?: {
            duration: number;
            computeUnits: number;
        } | undefined;
        finishedAt?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getRun";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        actorId: string;
        consoleUrl: string;
        startedAt: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        usage?: {
            duration: number;
            computeUnits: number;
        } | undefined;
        finishedAt?: string | undefined;
    };
}, {
    operation: "getRun";
    result: {
        error: string;
        status: string;
        success: boolean;
        id: string;
        actorId: string;
        consoleUrl: string;
        startedAt: string;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        usage?: {
            duration: number;
            computeUnits: number;
        } | undefined;
        finishedAt?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDataset">;
    result: z.ZodObject<{
        id: z.ZodString;
        name: z.ZodOptional<z.ZodString>;
        itemCount: z.ZodNumber;
        createdAt: z.ZodString;
        modifiedAt: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        modifiedAt: string;
        itemCount: number;
        name?: string | undefined;
    }, {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        modifiedAt: string;
        itemCount: number;
        name?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getDataset";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        modifiedAt: string;
        itemCount: number;
        name?: string | undefined;
    };
}, {
    operation: "getDataset";
    result: {
        error: string;
        success: boolean;
        id: string;
        createdAt: string;
        modifiedAt: string;
        itemCount: number;
        name?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getDatasetItems">;
    result: z.ZodObject<{
        items: z.ZodArray<z.ZodUnknown, "many">;
        count: z.ZodNumber;
        limit: z.ZodNumber;
        offset: z.ZodNumber;
        total: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        items: unknown[];
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        total?: number | undefined;
    }, {
        error: string;
        items: unknown[];
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        total?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getDatasetItems";
    result: {
        error: string;
        items: unknown[];
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        total?: number | undefined;
    };
}, {
    operation: "getDatasetItems";
    result: {
        error: string;
        items: unknown[];
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        total?: number | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"webScrape">;
    result: z.ZodObject<{
        url: z.ZodString;
        content: z.ZodOptional<z.ZodString>;
        data: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        screenshot: z.ZodOptional<z.ZodString>;
        pageHtml: z.ZodOptional<z.ZodString>;
        itemsCount: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    }, {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "webScrape";
    result: {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    };
}, {
    operation: "webScrape";
    result: {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"puppeteerScrape">;
    result: z.ZodObject<{
        url: z.ZodString;
        content: z.ZodOptional<z.ZodString>;
        data: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        screenshot: z.ZodOptional<z.ZodString>;
        pageHtml: z.ZodOptional<z.ZodString>;
        itemsCount: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    }, {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "puppeteerScrape";
    result: {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    };
}, {
    operation: "puppeteerScrape";
    result: {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"cheerioScrape">;
    result: z.ZodObject<{
        url: z.ZodString;
        content: z.ZodOptional<z.ZodString>;
        data: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        screenshot: z.ZodOptional<z.ZodString>;
        pageHtml: z.ZodOptional<z.ZodString>;
        itemsCount: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    }, {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "cheerioScrape";
    result: {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    };
}, {
    operation: "cheerioScrape";
    result: {
        error: string;
        url: string;
        success: boolean;
        content?: string | undefined;
        data?: unknown[] | undefined;
        itemsCount?: number | undefined;
        screenshot?: string | undefined;
        pageHtml?: string | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listActors">;
    result: z.ZodObject<{
        actors: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            stats: z.ZodOptional<z.ZodObject<{
                totalRuns: z.ZodNumber;
                usersCount: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                totalRuns: number;
                usersCount: number;
            }, {
                totalRuns: number;
                usersCount: number;
            }>>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            description?: string | undefined;
            username?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
        }, {
            name: string;
            id: string;
            description?: string | undefined;
            username?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
        }>, "many">;
        count: z.ZodNumber;
        limit: z.ZodNumber;
        offset: z.ZodNumber;
        total: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        actors: {
            name: string;
            id: string;
            description?: string | undefined;
            username?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
        }[];
        total?: number | undefined;
    }, {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        actors: {
            name: string;
            id: string;
            description?: string | undefined;
            username?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
        }[];
        total?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "listActors";
    result: {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        actors: {
            name: string;
            id: string;
            description?: string | undefined;
            username?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
        }[];
        total?: number | undefined;
    };
}, {
    operation: "listActors";
    result: {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        actors: {
            name: string;
            id: string;
            description?: string | undefined;
            username?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
        }[];
        total?: number | undefined;
    };
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getActorRuns">;
    result: z.ZodObject<{
        runs: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            status: z.ZodString;
            startedAt: z.ZodString;
            finishedAt: z.ZodOptional<z.ZodString>;
            itemsCount: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            status: string;
            id: string;
            startedAt: string;
            itemsCount?: number | undefined;
            finishedAt?: string | undefined;
        }, {
            status: string;
            id: string;
            startedAt: string;
            itemsCount?: number | undefined;
            finishedAt?: string | undefined;
        }>, "many">;
        count: z.ZodNumber;
        limit: z.ZodNumber;
        offset: z.ZodNumber;
        total: z.ZodOptional<z.ZodNumber>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        runs: {
            status: string;
            id: string;
            startedAt: string;
            itemsCount?: number | undefined;
            finishedAt?: string | undefined;
        }[];
        total?: number | undefined;
    }, {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        runs: {
            status: string;
            id: string;
            startedAt: string;
            itemsCount?: number | undefined;
            finishedAt?: string | undefined;
        }[];
        total?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    operation: "getActorRuns";
    result: {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        runs: {
            status: string;
            id: string;
            startedAt: string;
            itemsCount?: number | undefined;
            finishedAt?: string | undefined;
        }[];
        total?: number | undefined;
    };
}, {
    operation: "getActorRuns";
    result: {
        error: string;
        success: boolean;
        limit: number;
        count: number;
        offset: number;
        runs: {
            status: string;
            id: string;
            startedAt: string;
            itemsCount?: number | undefined;
            finishedAt?: string | undefined;
        }[];
        total?: number | undefined;
    };
}>]>;
type ApifyBubbleResult = z.output<typeof ApifyBubbleResultSchema>;
export declare class ApifyBubble<T extends ApifyBubbleParams = ApifyBubbleParams> extends ServiceBubble<T, any> {
    static readonly type: "service";
    static readonly service = "apify";
    static readonly authType: "apikey";
    static readonly bubbleName = "apify";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"runActor">;
        actorId: z.ZodString;
        input: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        buildId: z.ZodOptional<z.ZodString>;
        memory: z.ZodOptional<z.ZodNumber>;
        timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        waitForFinish: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        build: z.ZodDefault<z.ZodOptional<z.ZodEnum<["latest", "specific"]>>>;
        buildNumber: z.ZodOptional<z.ZodString>;
        maxItems: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        maxItems: number;
        input: Record<string, unknown>;
        operation: "runActor";
        actorId: string;
        waitForFinish: boolean;
        build: "latest" | "specific";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        memory?: number | undefined;
        buildId?: string | undefined;
        buildNumber?: string | undefined;
    }, {
        input: Record<string, unknown>;
        operation: "runActor";
        actorId: string;
        timeout?: number | undefined;
        maxItems?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        waitForFinish?: boolean | undefined;
        memory?: number | undefined;
        buildId?: string | undefined;
        build?: "latest" | "specific" | undefined;
        buildNumber?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getActor">;
        actorId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getActor";
        actorId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getActor";
        actorId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listActors">;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        search: z.ZodOptional<z.ZodString>;
        sortBy: z.ZodDefault<z.ZodOptional<z.ZodEnum<["createdAt", "modifiedAt", "usageStats"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listActors";
        limit: number;
        offset: number;
        sortBy: "createdAt" | "modifiedAt" | "usageStats";
        search?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "listActors";
        search?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        sortBy?: "createdAt" | "modifiedAt" | "usageStats" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"buildActor">;
        actorId: z.ZodString;
        buildTag: z.ZodOptional<z.ZodString>;
        version: z.ZodOptional<z.ZodString>;
        waitForFinish: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "buildActor";
        actorId: string;
        waitForFinish: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        version?: string | undefined;
        buildTag?: string | undefined;
    }, {
        operation: "buildActor";
        actorId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        waitForFinish?: boolean | undefined;
        version?: string | undefined;
        buildTag?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRun">;
        runId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRun";
        runId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getRun";
        runId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"waitForRun">;
        runId: z.ZodString;
        waitFor: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        waitInterval: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "waitForRun";
        runId: string;
        waitFor: number;
        waitInterval: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "waitForRun";
        runId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        waitFor?: number | undefined;
        waitInterval?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"stopRun">;
        runId: z.ZodString;
        gracefully: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "stopRun";
        runId: string;
        gracefully: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "stopRun";
        runId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        gracefully?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listRuns">;
        actorId: z.ZodOptional<z.ZodString>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        status: z.ZodOptional<z.ZodEnum<["READY", "RUNNING", "SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listRuns";
        limit: number;
        offset: number;
        status?: "READY" | "RUNNING" | "SUCCEEDED" | "FAILED" | "TIMED-OUT" | "ABORTED" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        actorId?: string | undefined;
    }, {
        operation: "listRuns";
        status?: "READY" | "RUNNING" | "SUCCEEDED" | "FAILED" | "TIMED-OUT" | "ABORTED" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        actorId?: string | undefined;
        offset?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDataset">;
        datasetId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getDataset";
        datasetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getDataset";
        datasetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDatasetItems">;
        datasetId: z.ZodString;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        clean: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["json", "csv", "xml", "xlsx", "html"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        format: "xml" | "html" | "json" | "csv" | "xlsx";
        operation: "getDatasetItems";
        limit: number;
        datasetId: string;
        offset: number;
        clean: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getDatasetItems";
        datasetId: string;
        format?: "xml" | "html" | "json" | "csv" | "xlsx" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        clean?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"downloadDataset">;
        datasetId: z.ZodString;
        format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["json", "csv", "xlsx", "html"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        format: "html" | "json" | "csv" | "xlsx";
        operation: "downloadDataset";
        datasetId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "downloadDataset";
        datasetId: string;
        format?: "html" | "json" | "csv" | "xlsx" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"webScrape">;
        url: z.ZodString;
        selectors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        waitForSelector: z.ZodOptional<z.ZodString>;
        timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        proxyConfiguration: z.ZodOptional<z.ZodObject<{
            useApifyProxy: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
            proxyGroups: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            countryCode: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            useApifyProxy: boolean;
            countryCode?: string | undefined;
            proxyGroups?: string[] | undefined;
        }, {
            countryCode?: string | undefined;
            useApifyProxy?: boolean | undefined;
            proxyGroups?: string[] | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        url: string;
        operation: "webScrape";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        selectors?: string[] | undefined;
        waitForSelector?: string | undefined;
        proxyConfiguration?: {
            useApifyProxy: boolean;
            countryCode?: string | undefined;
            proxyGroups?: string[] | undefined;
        } | undefined;
    }, {
        url: string;
        operation: "webScrape";
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        selectors?: string[] | undefined;
        waitForSelector?: string | undefined;
        proxyConfiguration?: {
            countryCode?: string | undefined;
            useApifyProxy?: boolean | undefined;
            proxyGroups?: string[] | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"crawlWebsite">;
        startUrls: z.ZodArray<z.ZodString, "many">;
        maxPages: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        proxyConfiguration: z.ZodOptional<z.ZodObject<{
            useApifyProxy: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
            proxyGroups: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            countryCode: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            useApifyProxy: boolean;
            countryCode?: string | undefined;
            proxyGroups?: string[] | undefined;
        }, {
            countryCode?: string | undefined;
            useApifyProxy?: boolean | undefined;
            proxyGroups?: string[] | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "crawlWebsite";
        startUrls: string[];
        maxPages: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        proxyConfiguration?: {
            useApifyProxy: boolean;
            countryCode?: string | undefined;
            proxyGroups?: string[] | undefined;
        } | undefined;
    }, {
        operation: "crawlWebsite";
        startUrls: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxPages?: number | undefined;
        proxyConfiguration?: {
            countryCode?: string | undefined;
            useApifyProxy?: boolean | undefined;
            proxyGroups?: string[] | undefined;
        } | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"runActor">;
        result: z.ZodObject<{
            runId: z.ZodString;
            status: z.ZodString;
            actorId: z.ZodString;
            startedAt: z.ZodOptional<z.ZodString>;
            finishedAt: z.ZodOptional<z.ZodString>;
            datasetId: z.ZodOptional<z.ZodString>;
            itemsCount: z.ZodOptional<z.ZodNumber>;
            consoleUrl: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            actorId: string;
            runId: string;
            consoleUrl: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            startedAt?: string | undefined;
            finishedAt?: string | undefined;
        }, {
            error: string;
            status: string;
            success: boolean;
            actorId: string;
            runId: string;
            consoleUrl: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            startedAt?: string | undefined;
            finishedAt?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "runActor";
        result: {
            error: string;
            status: string;
            success: boolean;
            actorId: string;
            runId: string;
            consoleUrl: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            startedAt?: string | undefined;
            finishedAt?: string | undefined;
        };
    }, {
        operation: "runActor";
        result: {
            error: string;
            status: string;
            success: boolean;
            actorId: string;
            runId: string;
            consoleUrl: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            startedAt?: string | undefined;
            finishedAt?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getActor">;
        result: z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            versions: z.ZodArray<z.ZodObject<{
                versionNumber: z.ZodString;
                buildStatus: z.ZodString;
                createdAt: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                createdAt: string;
                versionNumber: string;
                buildStatus: string;
            }, {
                createdAt: string;
                versionNumber: string;
                buildStatus: string;
            }>, "many">;
            defaultRunOptions: z.ZodOptional<z.ZodObject<{
                build: z.ZodString;
                timeoutSecs: z.ZodNumber;
                memoryMbytes: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                build: string;
                timeoutSecs: number;
                memoryMbytes: number;
            }, {
                build: string;
                timeoutSecs: number;
                memoryMbytes: number;
            }>>;
            stats: z.ZodOptional<z.ZodObject<{
                totalRuns: z.ZodNumber;
                usersCount: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                totalRuns: number;
                usersCount: number;
            }, {
                totalRuns: number;
                usersCount: number;
            }>>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            name: string;
            success: boolean;
            id: string;
            versions: {
                createdAt: string;
                versionNumber: string;
                buildStatus: string;
            }[];
            description?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
            defaultRunOptions?: {
                build: string;
                timeoutSecs: number;
                memoryMbytes: number;
            } | undefined;
        }, {
            error: string;
            name: string;
            success: boolean;
            id: string;
            versions: {
                createdAt: string;
                versionNumber: string;
                buildStatus: string;
            }[];
            description?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
            defaultRunOptions?: {
                build: string;
                timeoutSecs: number;
                memoryMbytes: number;
            } | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getActor";
        result: {
            error: string;
            name: string;
            success: boolean;
            id: string;
            versions: {
                createdAt: string;
                versionNumber: string;
                buildStatus: string;
            }[];
            description?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
            defaultRunOptions?: {
                build: string;
                timeoutSecs: number;
                memoryMbytes: number;
            } | undefined;
        };
    }, {
        operation: "getActor";
        result: {
            error: string;
            name: string;
            success: boolean;
            id: string;
            versions: {
                createdAt: string;
                versionNumber: string;
                buildStatus: string;
            }[];
            description?: string | undefined;
            stats?: {
                totalRuns: number;
                usersCount: number;
            } | undefined;
            defaultRunOptions?: {
                build: string;
                timeoutSecs: number;
                memoryMbytes: number;
            } | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getRun">;
        result: z.ZodObject<{
            id: z.ZodString;
            status: z.ZodString;
            actorId: z.ZodString;
            startedAt: z.ZodString;
            finishedAt: z.ZodOptional<z.ZodString>;
            datasetId: z.ZodOptional<z.ZodString>;
            itemsCount: z.ZodOptional<z.ZodNumber>;
            usage: z.ZodOptional<z.ZodObject<{
                computeUnits: z.ZodNumber;
                duration: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                duration: number;
                computeUnits: number;
            }, {
                duration: number;
                computeUnits: number;
            }>>;
            consoleUrl: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            actorId: string;
            consoleUrl: string;
            startedAt: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            usage?: {
                duration: number;
                computeUnits: number;
            } | undefined;
            finishedAt?: string | undefined;
        }, {
            error: string;
            status: string;
            success: boolean;
            id: string;
            actorId: string;
            consoleUrl: string;
            startedAt: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            usage?: {
                duration: number;
                computeUnits: number;
            } | undefined;
            finishedAt?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getRun";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            actorId: string;
            consoleUrl: string;
            startedAt: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            usage?: {
                duration: number;
                computeUnits: number;
            } | undefined;
            finishedAt?: string | undefined;
        };
    }, {
        operation: "getRun";
        result: {
            error: string;
            status: string;
            success: boolean;
            id: string;
            actorId: string;
            consoleUrl: string;
            startedAt: string;
            datasetId?: string | undefined;
            itemsCount?: number | undefined;
            usage?: {
                duration: number;
                computeUnits: number;
            } | undefined;
            finishedAt?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDataset">;
        result: z.ZodObject<{
            id: z.ZodString;
            name: z.ZodOptional<z.ZodString>;
            itemCount: z.ZodNumber;
            createdAt: z.ZodString;
            modifiedAt: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            modifiedAt: string;
            itemCount: number;
            name?: string | undefined;
        }, {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            modifiedAt: string;
            itemCount: number;
            name?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getDataset";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            modifiedAt: string;
            itemCount: number;
            name?: string | undefined;
        };
    }, {
        operation: "getDataset";
        result: {
            error: string;
            success: boolean;
            id: string;
            createdAt: string;
            modifiedAt: string;
            itemCount: number;
            name?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getDatasetItems">;
        result: z.ZodObject<{
            items: z.ZodArray<z.ZodUnknown, "many">;
            count: z.ZodNumber;
            limit: z.ZodNumber;
            offset: z.ZodNumber;
            total: z.ZodOptional<z.ZodNumber>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            items: unknown[];
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            total?: number | undefined;
        }, {
            error: string;
            items: unknown[];
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            total?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getDatasetItems";
        result: {
            error: string;
            items: unknown[];
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            total?: number | undefined;
        };
    }, {
        operation: "getDatasetItems";
        result: {
            error: string;
            items: unknown[];
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            total?: number | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"webScrape">;
        result: z.ZodObject<{
            url: z.ZodString;
            content: z.ZodOptional<z.ZodString>;
            data: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            screenshot: z.ZodOptional<z.ZodString>;
            pageHtml: z.ZodOptional<z.ZodString>;
            itemsCount: z.ZodOptional<z.ZodNumber>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        }, {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "webScrape";
        result: {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        };
    }, {
        operation: "webScrape";
        result: {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"puppeteerScrape">;
        result: z.ZodObject<{
            url: z.ZodString;
            content: z.ZodOptional<z.ZodString>;
            data: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            screenshot: z.ZodOptional<z.ZodString>;
            pageHtml: z.ZodOptional<z.ZodString>;
            itemsCount: z.ZodOptional<z.ZodNumber>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        }, {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "puppeteerScrape";
        result: {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        };
    }, {
        operation: "puppeteerScrape";
        result: {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"cheerioScrape">;
        result: z.ZodObject<{
            url: z.ZodString;
            content: z.ZodOptional<z.ZodString>;
            data: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            screenshot: z.ZodOptional<z.ZodString>;
            pageHtml: z.ZodOptional<z.ZodString>;
            itemsCount: z.ZodOptional<z.ZodNumber>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        }, {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "cheerioScrape";
        result: {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        };
    }, {
        operation: "cheerioScrape";
        result: {
            error: string;
            url: string;
            success: boolean;
            content?: string | undefined;
            data?: unknown[] | undefined;
            itemsCount?: number | undefined;
            screenshot?: string | undefined;
            pageHtml?: string | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listActors">;
        result: z.ZodObject<{
            actors: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                name: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                stats: z.ZodOptional<z.ZodObject<{
                    totalRuns: z.ZodNumber;
                    usersCount: z.ZodNumber;
                }, "strip", z.ZodTypeAny, {
                    totalRuns: number;
                    usersCount: number;
                }, {
                    totalRuns: number;
                    usersCount: number;
                }>>;
            }, "strip", z.ZodTypeAny, {
                name: string;
                id: string;
                description?: string | undefined;
                username?: string | undefined;
                stats?: {
                    totalRuns: number;
                    usersCount: number;
                } | undefined;
            }, {
                name: string;
                id: string;
                description?: string | undefined;
                username?: string | undefined;
                stats?: {
                    totalRuns: number;
                    usersCount: number;
                } | undefined;
            }>, "many">;
            count: z.ZodNumber;
            limit: z.ZodNumber;
            offset: z.ZodNumber;
            total: z.ZodOptional<z.ZodNumber>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            actors: {
                name: string;
                id: string;
                description?: string | undefined;
                username?: string | undefined;
                stats?: {
                    totalRuns: number;
                    usersCount: number;
                } | undefined;
            }[];
            total?: number | undefined;
        }, {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            actors: {
                name: string;
                id: string;
                description?: string | undefined;
                username?: string | undefined;
                stats?: {
                    totalRuns: number;
                    usersCount: number;
                } | undefined;
            }[];
            total?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "listActors";
        result: {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            actors: {
                name: string;
                id: string;
                description?: string | undefined;
                username?: string | undefined;
                stats?: {
                    totalRuns: number;
                    usersCount: number;
                } | undefined;
            }[];
            total?: number | undefined;
        };
    }, {
        operation: "listActors";
        result: {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            actors: {
                name: string;
                id: string;
                description?: string | undefined;
                username?: string | undefined;
                stats?: {
                    totalRuns: number;
                    usersCount: number;
                } | undefined;
            }[];
            total?: number | undefined;
        };
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getActorRuns">;
        result: z.ZodObject<{
            runs: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                status: z.ZodString;
                startedAt: z.ZodString;
                finishedAt: z.ZodOptional<z.ZodString>;
                itemsCount: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                status: string;
                id: string;
                startedAt: string;
                itemsCount?: number | undefined;
                finishedAt?: string | undefined;
            }, {
                status: string;
                id: string;
                startedAt: string;
                itemsCount?: number | undefined;
                finishedAt?: string | undefined;
            }>, "many">;
            count: z.ZodNumber;
            limit: z.ZodNumber;
            offset: z.ZodNumber;
            total: z.ZodOptional<z.ZodNumber>;
            success: z.ZodBoolean;
            error: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            runs: {
                status: string;
                id: string;
                startedAt: string;
                itemsCount?: number | undefined;
                finishedAt?: string | undefined;
            }[];
            total?: number | undefined;
        }, {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            runs: {
                status: string;
                id: string;
                startedAt: string;
                itemsCount?: number | undefined;
                finishedAt?: string | undefined;
            }[];
            total?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        operation: "getActorRuns";
        result: {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            runs: {
                status: string;
                id: string;
                startedAt: string;
                itemsCount?: number | undefined;
                finishedAt?: string | undefined;
            }[];
            total?: number | undefined;
        };
    }, {
        operation: "getActorRuns";
        result: {
            error: string;
            success: boolean;
            limit: number;
            count: number;
            offset: number;
            runs: {
                status: string;
                id: string;
                startedAt: string;
                itemsCount?: number | undefined;
                finishedAt?: string | undefined;
            }[];
            total?: number | undefined;
        };
    }>]>;
    static readonly shortDescription = "Web scraping and automation platform";
    static readonly longDescription = "\n    Apify Service Bubble for web scraping, crawling, and automation.\n\n    Operations (12):\n    1. runActor - Execute any Apify actor with custom input parameters\n    2. getActor - Retrieve actor details, versions, and statistics\n    3. listActors - Browse and discover available actors\n    4. buildActor - Build actor from source code\n    5. getRun - Check status and details of an actor run\n    6. waitForRun - Wait for run completion with polling\n    7. stopRun - Stop a running actor gracefully or immediately\n    8. listRuns - List historical runs for an actor\n    9. getDataset - Get dataset metadata and information\n    10. getDatasetItems - Fetch scraped data from datasets\n    11. downloadDataset - Download dataset in various formats\n    12. webScrape - Quick web scraping with selectors\n    13. crawlWebsite - Crawl entire websites with proxy support\n\n    Features:\n    - Full resilience patterns with circuit breaker and retry logic\n    - SSRF protection with URL validation\n    - Actor and run ID validation\n    - Memory management (128-8192 MB)\n    - Rate limiting with exponential backoff\n    - Proxy configuration support\n    - Dataset download in multiple formats\n    - Real-time run monitoring and control\n    - Error sanitization for security\n  ";
    static readonly alias = "apify";
    private client;
    private resilience;
    constructor(params: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<ApifyBubbleResult, {
        operation: T['operation'];
    }>>;
    private runActor;
    private getActor;
    private getRun;
    private getDataset;
    private getDatasetItems;
    private webScrape;
    private puppeteerScrape;
    private cheerioScrape;
    private listActors;
    private getActorRuns;
    private buildActor;
    private waitForRun;
    private stopRun;
    private listRuns;
    private downloadDataset;
    private crawlWebsite;
    private waitForCompletion;
    private errorResult;
}
export {};
//# sourceMappingURL=apify-bubble.d.ts.map