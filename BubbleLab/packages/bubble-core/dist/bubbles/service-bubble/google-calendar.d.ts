import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const GoogleCalendarParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"list_calendars">;
    max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    page_token: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_calendars";
    max_results: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    page_token?: string | undefined;
}, {
    operation: "list_calendars";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    max_results?: number | undefined;
    page_token?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_events">;
    calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    time_min: z.ZodOptional<z.ZodString>;
    time_max: z.ZodOptional<z.ZodString>;
    q: z.ZodOptional<z.ZodString>;
    single_events: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    order_by: z.ZodDefault<z.ZodOptional<z.ZodEnum<["startTime", "updated"]>>>;
    page_token: z.ZodOptional<z.ZodString>;
    max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_events";
    max_results: number;
    order_by: "updated" | "startTime";
    calendar_id: string;
    single_events: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    page_token?: string | undefined;
    time_min?: string | undefined;
    time_max?: string | undefined;
    q?: string | undefined;
}, {
    operation: "list_events";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    max_results?: number | undefined;
    order_by?: "updated" | "startTime" | undefined;
    page_token?: string | undefined;
    calendar_id?: string | undefined;
    time_min?: string | undefined;
    time_max?: string | undefined;
    q?: string | undefined;
    single_events?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_event">;
    calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    event_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_event";
    calendar_id: string;
    event_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_event";
    event_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    calendar_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_event">;
    calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    summary: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    location: z.ZodOptional<z.ZodString>;
    start: z.ZodObject<{
        dateTime: z.ZodOptional<z.ZodString>;
        date: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }>;
    end: z.ZodObject<{
        dateTime: z.ZodOptional<z.ZodString>;
        date: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }>;
    attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
        email: z.ZodString;
        optional: z.ZodOptional<z.ZodBoolean>;
        responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
        displayName: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }, {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }>, "many">>;
    conference: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_event";
    calendar_id: string;
    summary: string;
    start: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    };
    end: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    };
    conference: boolean;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    location?: string | undefined;
    attendees?: {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }[] | undefined;
}, {
    operation: "create_event";
    summary: string;
    start: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    };
    end: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    };
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    calendar_id?: string | undefined;
    location?: string | undefined;
    attendees?: {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }[] | undefined;
    conference?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_event">;
    calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    event_id: z.ZodString;
    summary: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    location: z.ZodOptional<z.ZodString>;
    start: z.ZodOptional<z.ZodObject<{
        dateTime: z.ZodOptional<z.ZodString>;
        date: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }>>;
    end: z.ZodOptional<z.ZodObject<{
        dateTime: z.ZodOptional<z.ZodString>;
        date: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }, {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    }>>;
    attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
        email: z.ZodString;
        optional: z.ZodOptional<z.ZodBoolean>;
        responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
        displayName: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }, {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_event";
    calendar_id: string;
    event_id: string;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    summary?: string | undefined;
    location?: string | undefined;
    start?: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    } | undefined;
    end?: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    } | undefined;
    attendees?: {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }[] | undefined;
}, {
    operation: "update_event";
    event_id: string;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    calendar_id?: string | undefined;
    summary?: string | undefined;
    location?: string | undefined;
    start?: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    } | undefined;
    end?: {
        date?: string | undefined;
        timeZone?: string | undefined;
        dateTime?: string | undefined;
    } | undefined;
    attendees?: {
        email: string;
        displayName?: string | undefined;
        optional?: boolean | undefined;
        responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_event">;
    calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    event_id: z.ZodString;
    send_updates: z.ZodDefault<z.ZodOptional<z.ZodEnum<["all", "externalOnly", "none"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_event";
    calendar_id: string;
    event_id: string;
    send_updates: "all" | "externalOnly" | "none";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_event";
    event_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    calendar_id?: string | undefined;
    send_updates?: "all" | "externalOnly" | "none" | undefined;
}>]>;
declare const GoogleCalendarResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"list_calendars">;
    success: z.ZodBoolean;
    calendars: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
        selected: z.ZodOptional<z.ZodBoolean>;
        accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodString;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
        selected: z.ZodOptional<z.ZodBoolean>;
        accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodString;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
        selected: z.ZodOptional<z.ZodBoolean>;
        accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    next_page_token: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_calendars";
    next_page_token?: string | undefined;
    calendars?: z.objectOutputType<{
        id: z.ZodString;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
        selected: z.ZodOptional<z.ZodBoolean>;
        accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_calendars";
    next_page_token?: string | undefined;
    calendars?: z.objectInputType<{
        id: z.ZodString;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        timeZone: z.ZodOptional<z.ZodString>;
        selected: z.ZodOptional<z.ZodBoolean>;
        accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_events">;
    success: z.ZodBoolean;
    events: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    next_page_token: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_events";
    next_page_token?: string | undefined;
    events?: z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_events";
    next_page_token?: string | undefined;
    events?: z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_event">;
    success: z.ZodBoolean;
    event: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_event";
    event?: z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_event";
    event?: z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_event">;
    success: z.ZodBoolean;
    event: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_event";
    event?: z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_event";
    event?: z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_event">;
    success: z.ZodBoolean;
    event: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_event";
    event?: z.objectOutputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "update_event";
    event?: z.objectInputType<{
        id: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
        htmlLink: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        organizer: z.ZodOptional<z.ZodObject<{
            email: z.ZodOptional<z.ZodString>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email?: string | undefined;
            displayName?: string | undefined;
        }, {
            email?: string | undefined;
            displayName?: string | undefined;
        }>>;
        hangoutLink: z.ZodOptional<z.ZodString>;
        conferenceData: z.ZodOptional<z.ZodAny>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_event">;
    success: z.ZodBoolean;
    deleted: z.ZodOptional<z.ZodBoolean>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_event";
    deleted?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_event";
    deleted?: boolean | undefined;
}>]>;
type GoogleCalendarResult = z.output<typeof GoogleCalendarResultSchema>;
type GoogleCalendarParams = z.input<typeof GoogleCalendarParamsSchema>;
export type GoogleCalendarOperationResult<T extends GoogleCalendarParams['operation']> = Extract<GoogleCalendarResult, {
    operation: T;
}>;
export type GoogleCalendarParamsInput = z.input<typeof GoogleCalendarParamsSchema>;
export declare class GoogleCalendarBubble<T extends GoogleCalendarParams = GoogleCalendarParams> extends ServiceBubble<T, Extract<GoogleCalendarResult, {
    operation: T['operation'];
}>> {
    static readonly type: "service";
    static readonly service = "google-calendar";
    static readonly authType: "oauth";
    static readonly bubbleName = "google-calendar";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"list_calendars">;
        max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        page_token: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_calendars";
        max_results: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        page_token?: string | undefined;
    }, {
        operation: "list_calendars";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        max_results?: number | undefined;
        page_token?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_events">;
        calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        time_min: z.ZodOptional<z.ZodString>;
        time_max: z.ZodOptional<z.ZodString>;
        q: z.ZodOptional<z.ZodString>;
        single_events: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        order_by: z.ZodDefault<z.ZodOptional<z.ZodEnum<["startTime", "updated"]>>>;
        page_token: z.ZodOptional<z.ZodString>;
        max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_events";
        max_results: number;
        order_by: "updated" | "startTime";
        calendar_id: string;
        single_events: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        page_token?: string | undefined;
        time_min?: string | undefined;
        time_max?: string | undefined;
        q?: string | undefined;
    }, {
        operation: "list_events";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        max_results?: number | undefined;
        order_by?: "updated" | "startTime" | undefined;
        page_token?: string | undefined;
        calendar_id?: string | undefined;
        time_min?: string | undefined;
        time_max?: string | undefined;
        q?: string | undefined;
        single_events?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_event">;
        calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        event_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_event";
        calendar_id: string;
        event_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_event";
        event_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        calendar_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_event">;
        calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        summary: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>;
        end: z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        conference: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_event";
        calendar_id: string;
        summary: string;
        start: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        };
        end: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        };
        conference: boolean;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        location?: string | undefined;
        attendees?: {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }[] | undefined;
    }, {
        operation: "create_event";
        summary: string;
        start: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        };
        end: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        };
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        calendar_id?: string | undefined;
        location?: string | undefined;
        attendees?: {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }[] | undefined;
        conference?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_event">;
        calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        event_id: z.ZodString;
        summary: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        start: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        end: z.ZodOptional<z.ZodObject<{
            dateTime: z.ZodOptional<z.ZodString>;
            date: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }, {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        }>>;
        attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
            email: z.ZodString;
            optional: z.ZodOptional<z.ZodBoolean>;
            responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
            displayName: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }, {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_event";
        calendar_id: string;
        event_id: string;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        summary?: string | undefined;
        location?: string | undefined;
        start?: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        } | undefined;
        end?: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        } | undefined;
        attendees?: {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }[] | undefined;
    }, {
        operation: "update_event";
        event_id: string;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        calendar_id?: string | undefined;
        summary?: string | undefined;
        location?: string | undefined;
        start?: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        } | undefined;
        end?: {
            date?: string | undefined;
            timeZone?: string | undefined;
            dateTime?: string | undefined;
        } | undefined;
        attendees?: {
            email: string;
            displayName?: string | undefined;
            optional?: boolean | undefined;
            responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_event">;
        calendar_id: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        event_id: z.ZodString;
        send_updates: z.ZodDefault<z.ZodOptional<z.ZodEnum<["all", "externalOnly", "none"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_event";
        calendar_id: string;
        event_id: string;
        send_updates: "all" | "externalOnly" | "none";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_event";
        event_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        calendar_id?: string | undefined;
        send_updates?: "all" | "externalOnly" | "none" | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"list_calendars">;
        success: z.ZodBoolean;
        calendars: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
            selected: z.ZodOptional<z.ZodBoolean>;
            accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodString;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
            selected: z.ZodOptional<z.ZodBoolean>;
            accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodString;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
            selected: z.ZodOptional<z.ZodBoolean>;
            accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        next_page_token: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_calendars";
        next_page_token?: string | undefined;
        calendars?: z.objectOutputType<{
            id: z.ZodString;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
            selected: z.ZodOptional<z.ZodBoolean>;
            accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_calendars";
        next_page_token?: string | undefined;
        calendars?: z.objectInputType<{
            id: z.ZodString;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            timeZone: z.ZodOptional<z.ZodString>;
            selected: z.ZodOptional<z.ZodBoolean>;
            accessRole: z.ZodOptional<z.ZodEnum<["freeBusyReader", "reader", "writer", "owner"]>>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_events">;
        success: z.ZodBoolean;
        events: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        next_page_token: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_events";
        next_page_token?: string | undefined;
        events?: z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_events";
        next_page_token?: string | undefined;
        events?: z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_event">;
        success: z.ZodBoolean;
        event: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_event";
        event?: z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_event";
        event?: z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_event">;
        success: z.ZodBoolean;
        event: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_event";
        event?: z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_event";
        event?: z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_event">;
        success: z.ZodBoolean;
        event: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_event";
        event?: z.objectOutputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_event";
        event?: z.objectInputType<{
            id: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
            htmlLink: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
            summary: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            start: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            end: z.ZodOptional<z.ZodObject<{
                dateTime: z.ZodOptional<z.ZodString>;
                date: z.ZodOptional<z.ZodString>;
                timeZone: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }, {
                date?: string | undefined;
                timeZone?: string | undefined;
                dateTime?: string | undefined;
            }>>;
            attendees: z.ZodOptional<z.ZodArray<z.ZodObject<{
                email: z.ZodString;
                optional: z.ZodOptional<z.ZodBoolean>;
                responseStatus: z.ZodOptional<z.ZodEnum<["needsAction", "declined", "tentative", "accepted"]>>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }, {
                email: string;
                displayName?: string | undefined;
                optional?: boolean | undefined;
                responseStatus?: "needsAction" | "declined" | "tentative" | "accepted" | undefined;
            }>, "many">>;
            organizer: z.ZodOptional<z.ZodObject<{
                email: z.ZodOptional<z.ZodString>;
                displayName: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                email?: string | undefined;
                displayName?: string | undefined;
            }, {
                email?: string | undefined;
                displayName?: string | undefined;
            }>>;
            hangoutLink: z.ZodOptional<z.ZodString>;
            conferenceData: z.ZodOptional<z.ZodAny>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_event">;
        success: z.ZodBoolean;
        deleted: z.ZodOptional<z.ZodBoolean>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_event";
        deleted?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_event";
        deleted?: boolean | undefined;
    }>]>;
    static readonly shortDescription = "Google Calendar integration for managing events";
    static readonly longDescription = "\n    Google Calendar service integration for listing, creating, updating and deleting events.\n    Use cases:\n    - List calendars and events with filters and pagination\n    - Create meetings with attendees and optional Google Meet link\n    - Update or delete existing events and notify attendees\n    Security Features:\n    - OAuth 2.0 with scoped access to Calendar\n  ";
    static readonly alias = "gcal";
    constructor(params?: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    private makeCalendarApiRequest;
    protected performAction(context?: BubbleContext): Promise<Extract<GoogleCalendarResult, {
        operation: T['operation'];
    }>>;
    private listCalendars;
    private listEvents;
    private getEvent;
    private buildEventBody;
    private createEvent;
    private updateEvent;
    private deleteEvent;
    protected chooseCredential(): string | undefined;
}
export {};
//# sourceMappingURL=google-calendar.d.ts.map