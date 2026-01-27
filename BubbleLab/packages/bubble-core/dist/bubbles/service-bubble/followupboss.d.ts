import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const FUBParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"list_people">;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    sort: z.ZodOptional<z.ZodString>;
    fields: z.ZodOptional<z.ZodString>;
    includeTrash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_people";
    limit: number;
    offset: number;
    includeTrash: boolean;
    sort?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string | undefined;
}, {
    operation: "list_people";
    sort?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    includeTrash?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_person">;
    person_id: z.ZodNumber;
    fields: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_person";
    person_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string | undefined;
}, {
    operation: "get_person";
    person_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    fields?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_person">;
    firstName: z.ZodOptional<z.ZodString>;
    lastName: z.ZodOptional<z.ZodString>;
    emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
        value: z.ZodString;
        type: z.ZodOptional<z.ZodString>;
        isPrimary: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }>, "many">>;
    phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
        value: z.ZodString;
        type: z.ZodOptional<z.ZodString>;
        isPrimary: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }>, "many">>;
    stage: z.ZodOptional<z.ZodString>;
    source: z.ZodOptional<z.ZodString>;
    assignedTo: z.ZodOptional<z.ZodNumber>;
    tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_person";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    tags?: string[] | undefined;
    source?: string | undefined;
    firstName?: string | undefined;
    lastName?: string | undefined;
    emails?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    phones?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    stage?: string | undefined;
    assignedTo?: number | undefined;
}, {
    operation: "create_person";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    tags?: string[] | undefined;
    source?: string | undefined;
    firstName?: string | undefined;
    lastName?: string | undefined;
    emails?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    phones?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    stage?: string | undefined;
    assignedTo?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_person">;
    person_id: z.ZodNumber;
    firstName: z.ZodOptional<z.ZodString>;
    lastName: z.ZodOptional<z.ZodString>;
    emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
        value: z.ZodString;
        type: z.ZodOptional<z.ZodString>;
        isPrimary: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }>, "many">>;
    phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
        value: z.ZodString;
        type: z.ZodOptional<z.ZodString>;
        isPrimary: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }, {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }>, "many">>;
    stage: z.ZodOptional<z.ZodString>;
    source: z.ZodOptional<z.ZodString>;
    assignedTo: z.ZodOptional<z.ZodNumber>;
    tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_person";
    person_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    tags?: string[] | undefined;
    source?: string | undefined;
    firstName?: string | undefined;
    lastName?: string | undefined;
    emails?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    phones?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    stage?: string | undefined;
    assignedTo?: number | undefined;
}, {
    operation: "update_person";
    person_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    tags?: string[] | undefined;
    source?: string | undefined;
    firstName?: string | undefined;
    lastName?: string | undefined;
    emails?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    phones?: {
        value: string;
        type?: string | undefined;
        isPrimary?: boolean | undefined;
    }[] | undefined;
    stage?: string | undefined;
    assignedTo?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_person">;
    person_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_person";
    person_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_person";
    person_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_tasks">;
    personId: z.ZodOptional<z.ZodNumber>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_tasks";
    limit: number;
    offset: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    personId?: number | undefined;
}, {
    operation: "list_tasks";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    personId?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_task">;
    task_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_task";
    task_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_task";
    task_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_task">;
    personId: z.ZodOptional<z.ZodNumber>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    dueDate: z.ZodOptional<z.ZodString>;
    assignedTo: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "create_task";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    assignedTo?: number | undefined;
    personId?: number | undefined;
    dueDate?: string | undefined;
}, {
    name: string;
    operation: "create_task";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    assignedTo?: number | undefined;
    personId?: number | undefined;
    dueDate?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_task">;
    task_id: z.ZodNumber;
    name: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    dueDate: z.ZodOptional<z.ZodString>;
    completed: z.ZodOptional<z.ZodBoolean>;
    assignedTo: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_task";
    task_id: number;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    assignedTo?: number | undefined;
    dueDate?: string | undefined;
    completed?: boolean | undefined;
}, {
    operation: "update_task";
    task_id: number;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    assignedTo?: number | undefined;
    dueDate?: string | undefined;
    completed?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_task">;
    task_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_task";
    task_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_task";
    task_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_notes">;
    personId: z.ZodOptional<z.ZodNumber>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_notes";
    limit: number;
    offset: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    personId?: number | undefined;
}, {
    operation: "list_notes";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    personId?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_note">;
    personId: z.ZodNumber;
    subject: z.ZodOptional<z.ZodString>;
    body: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_note";
    body: string;
    personId: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    subject?: string | undefined;
}, {
    operation: "create_note";
    body: string;
    personId: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    subject?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_note">;
    note_id: z.ZodNumber;
    subject: z.ZodOptional<z.ZodString>;
    body: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_note";
    note_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    subject?: string | undefined;
    body?: string | undefined;
}, {
    operation: "update_note";
    note_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    subject?: string | undefined;
    body?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_note">;
    note_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_note";
    note_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_note";
    note_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_deals">;
    personId: z.ZodOptional<z.ZodNumber>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_deals";
    limit: number;
    offset: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    personId?: number | undefined;
}, {
    operation: "list_deals";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    personId?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_deal">;
    deal_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_deal";
    deal_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_deal";
    deal_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_deal">;
    personId: z.ZodOptional<z.ZodNumber>;
    name: z.ZodOptional<z.ZodString>;
    price: z.ZodOptional<z.ZodNumber>;
    stage: z.ZodOptional<z.ZodString>;
    closeDate: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_deal";
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    price?: number | undefined;
    stage?: string | undefined;
    personId?: number | undefined;
    closeDate?: string | undefined;
}, {
    operation: "create_deal";
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    price?: number | undefined;
    stage?: string | undefined;
    personId?: number | undefined;
    closeDate?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_deal">;
    deal_id: z.ZodNumber;
    name: z.ZodOptional<z.ZodString>;
    price: z.ZodOptional<z.ZodNumber>;
    stage: z.ZodOptional<z.ZodString>;
    closeDate: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_deal";
    deal_id: number;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    price?: number | undefined;
    stage?: string | undefined;
    closeDate?: string | undefined;
}, {
    operation: "update_deal";
    deal_id: number;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    price?: number | undefined;
    stage?: string | undefined;
    closeDate?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_events">;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    personId: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_events";
    limit: number;
    offset: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    personId?: number | undefined;
}, {
    operation: "list_events";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    personId?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_event">;
    event_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_event";
    event_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_event";
    event_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_event">;
    type: z.ZodString;
    source: z.ZodOptional<z.ZodString>;
    message: z.ZodOptional<z.ZodString>;
    person: z.ZodObject<{
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value: string;
        }, {
            value: string;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value: string;
        }, {
            value: string;
        }>, "many">>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        tags?: string[] | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
        }[] | undefined;
        phones?: {
            value: string;
        }[] | undefined;
    }, {
        tags?: string[] | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
        }[] | undefined;
        phones?: {
            value: string;
        }[] | undefined;
    }>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    type: string;
    operation: "create_event";
    person: {
        tags?: string[] | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
        }[] | undefined;
        phones?: {
            value: string;
        }[] | undefined;
    };
    message?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    source?: string | undefined;
}, {
    type: string;
    operation: "create_event";
    person: {
        tags?: string[] | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
        }[] | undefined;
        phones?: {
            value: string;
        }[] | undefined;
    };
    message?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    source?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_calls">;
    personId: z.ZodOptional<z.ZodNumber>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_calls";
    limit: number;
    offset: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    personId?: number | undefined;
}, {
    operation: "list_calls";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    personId?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_call">;
    personId: z.ZodNumber;
    outcome: z.ZodOptional<z.ZodString>;
    note: z.ZodOptional<z.ZodString>;
    duration: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_call";
    personId: number;
    duration?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    outcome?: string | undefined;
    note?: string | undefined;
}, {
    operation: "create_call";
    personId: number;
    duration?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    outcome?: string | undefined;
    note?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_appointments">;
    personId: z.ZodOptional<z.ZodNumber>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_appointments";
    limit: number;
    offset: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    personId?: number | undefined;
}, {
    operation: "list_appointments";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    personId?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_appointment">;
    personId: z.ZodOptional<z.ZodNumber>;
    title: z.ZodString;
    startTime: z.ZodString;
    endTime: z.ZodOptional<z.ZodString>;
    location: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    title: string;
    operation: "create_appointment";
    startTime: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    location?: string | undefined;
    personId?: number | undefined;
    endTime?: string | undefined;
}, {
    title: string;
    operation: "create_appointment";
    startTime: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    location?: string | undefined;
    personId?: number | undefined;
    endTime?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_webhooks">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_webhooks";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "list_webhooks";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_webhook">;
    webhook_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_webhook";
    webhook_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_webhook";
    webhook_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_webhook">;
    event: z.ZodEnum<["peopleCreated", "peopleUpdated", "peopleDeleted", "peopleTagsCreated", "peopleStageUpdated", "peopleRelationshipCreated", "peopleRelationshipUpdated", "peopleRelationshipDeleted", "notesCreated", "notesUpdated", "notesDeleted", "emailsCreated", "emailsUpdated", "emailsDeleted", "tasksCreated", "tasksUpdated", "tasksDeleted", "appointmentsCreated", "appointmentsUpdated", "appointmentsDeleted", "textMessagesCreated", "textMessagesUpdated", "textMessagesDeleted", "callsCreated", "callsUpdated", "callsDeleted", "dealsCreated", "dealsUpdated", "dealsDeleted", "eventsCreated", "stageCreated", "stageUpdated", "stageDeleted", "pipelineCreated", "pipelineUpdated", "pipelineDeleted", "pipelineStageCreated", "pipelineStageUpdated", "pipelineStageDeleted", "customFieldsCreated", "customFieldsUpdated", "customFieldsDeleted", "dealCustomFieldsCreated", "dealCustomFieldsUpdated", "dealCustomFieldsDeleted", "emEventsOpened", "emEventsClicked", "emEventsUnsubscribed", "reactionCreated", "reactionDeleted", "threadedReplyCreated", "threadedReplyUpdated", "threadedReplyDeleted"]>;
    url: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    url: string;
    operation: "create_webhook";
    event: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    url: string;
    operation: "create_webhook";
    event: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_webhook">;
    webhook_id: z.ZodNumber;
    event: z.ZodOptional<z.ZodEnum<["peopleCreated", "peopleUpdated", "peopleDeleted", "peopleTagsCreated", "peopleStageUpdated", "peopleRelationshipCreated", "peopleRelationshipUpdated", "peopleRelationshipDeleted", "notesCreated", "notesUpdated", "notesDeleted", "emailsCreated", "emailsUpdated", "emailsDeleted", "tasksCreated", "tasksUpdated", "tasksDeleted", "appointmentsCreated", "appointmentsUpdated", "appointmentsDeleted", "textMessagesCreated", "textMessagesUpdated", "textMessagesDeleted", "callsCreated", "callsUpdated", "callsDeleted", "dealsCreated", "dealsUpdated", "dealsDeleted", "eventsCreated", "stageCreated", "stageUpdated", "stageDeleted", "pipelineCreated", "pipelineUpdated", "pipelineDeleted", "pipelineStageCreated", "pipelineStageUpdated", "pipelineStageDeleted", "customFieldsCreated", "customFieldsUpdated", "customFieldsDeleted", "dealCustomFieldsCreated", "dealCustomFieldsUpdated", "dealCustomFieldsDeleted", "emEventsOpened", "emEventsClicked", "emEventsUnsubscribed", "reactionCreated", "reactionDeleted", "threadedReplyCreated", "threadedReplyUpdated", "threadedReplyDeleted"]>>;
    url: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_webhook";
    webhook_id: number;
    url?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    event?: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted" | undefined;
}, {
    operation: "update_webhook";
    webhook_id: number;
    url?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    event?: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted" | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_webhook">;
    webhook_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_webhook";
    webhook_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_webhook";
    webhook_id: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
declare const FUBResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"list_people">;
    success: z.ZodBoolean;
    people: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    _metadata: z.ZodOptional<z.ZodObject<{
        total: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_people";
    people?: z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_people";
    people?: z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_person">;
    success: z.ZodBoolean;
    person: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_person";
    person?: z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_person";
    person?: z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_person">;
    success: z.ZodBoolean;
    person: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_person";
    person?: z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_person";
    person?: z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_person">;
    success: z.ZodBoolean;
    person: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_person";
    person?: z.objectOutputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "update_person";
    person?: z.objectInputType<{
        id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        created: z.ZodOptional<z.ZodString>;
        updated: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_person">;
    success: z.ZodBoolean;
    deleted_id: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_person";
    deleted_id?: number | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_person";
    deleted_id?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_tasks">;
    success: z.ZodBoolean;
    tasks: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    _metadata: z.ZodOptional<z.ZodObject<{
        total: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_tasks";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    tasks?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_tasks";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    tasks?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_task">;
    success: z.ZodBoolean;
    task: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_task";
    task?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_task";
    task?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_task">;
    success: z.ZodBoolean;
    task: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_task";
    task?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_task";
    task?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_task">;
    success: z.ZodBoolean;
    task: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_task";
    task?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "update_task";
    task?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_task">;
    success: z.ZodBoolean;
    deleted_id: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_task";
    deleted_id?: number | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_task";
    deleted_id?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_notes">;
    success: z.ZodBoolean;
    notes: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    _metadata: z.ZodOptional<z.ZodObject<{
        total: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_notes";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    notes?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_notes";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    notes?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_note">;
    success: z.ZodBoolean;
    note: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_note";
    note?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_note";
    note?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_note">;
    success: z.ZodBoolean;
    note: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_note";
    note?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "update_note";
    note?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_note">;
    success: z.ZodBoolean;
    deleted_id: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_note";
    deleted_id?: number | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_note";
    deleted_id?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_deals">;
    success: z.ZodBoolean;
    deals: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    _metadata: z.ZodOptional<z.ZodObject<{
        total: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_deals";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    deals?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_deals";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    deals?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_deal">;
    success: z.ZodBoolean;
    deal: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_deal";
    deal?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_deal";
    deal?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_deal">;
    success: z.ZodBoolean;
    deal: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_deal";
    deal?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_deal";
    deal?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_deal">;
    success: z.ZodBoolean;
    deal: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_deal";
    deal?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "update_deal";
    deal?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_events">;
    success: z.ZodBoolean;
    events: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    _metadata: z.ZodOptional<z.ZodObject<{
        total: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_events";
    events?: z.objectOutputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_events";
    events?: z.objectInputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_event">;
    success: z.ZodBoolean;
    event: z.ZodOptional<z.ZodObject<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_event";
    event?: z.objectOutputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_event";
    event?: z.objectInputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_event">;
    success: z.ZodBoolean;
    event: z.ZodOptional<z.ZodObject<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_event";
    event?: z.objectOutputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_event";
    event?: z.objectInputType<{
        id: z.ZodOptional<z.ZodNumber>;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodOptional<z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_calls">;
    success: z.ZodBoolean;
    calls: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    _metadata: z.ZodOptional<z.ZodObject<{
        total: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_calls";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    calls?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_calls";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    calls?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_call">;
    success: z.ZodBoolean;
    call: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_call";
    call?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_call";
    call?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_appointments">;
    success: z.ZodBoolean;
    appointments: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    _metadata: z.ZodOptional<z.ZodObject<{
        total: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
        offset: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }, {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_appointments";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    appointments?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_appointments";
    _metadata?: {
        limit?: number | undefined;
        offset?: number | undefined;
        total?: number | undefined;
    } | undefined;
    appointments?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_appointment">;
    success: z.ZodBoolean;
    appointment: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_appointment";
    appointment?: z.objectOutputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_appointment";
    appointment?: z.objectInputType<{
        id: z.ZodNumber;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodOptional<z.ZodString>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        created: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_webhooks">;
    success: z.ZodBoolean;
    webhooks: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_webhooks";
    webhooks?: z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_webhooks";
    webhooks?: z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_webhook">;
    success: z.ZodBoolean;
    webhook: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_webhook";
    webhook?: z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_webhook";
    webhook?: z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_webhook">;
    success: z.ZodBoolean;
    webhook: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_webhook";
    webhook?: z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_webhook";
    webhook?: z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_webhook">;
    success: z.ZodBoolean;
    webhook: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_webhook";
    webhook?: z.objectOutputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    error: string;
    success: boolean;
    operation: "update_webhook";
    webhook?: z.objectInputType<{
        id: z.ZodNumber;
        event: z.ZodString;
        url: z.ZodString;
        status: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_webhook">;
    success: z.ZodBoolean;
    deleted_id: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_webhook";
    deleted_id?: number | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_webhook";
    deleted_id?: number | undefined;
}>]>;
type FUBResult = z.output<typeof FUBResultSchema>;
type FUBParams = z.input<typeof FUBParamsSchema>;
export type FUBOperationResult<T extends FUBParams['operation']> = Extract<FUBResult, {
    operation: T;
}>;
export type FUBParamsInput = z.input<typeof FUBParamsSchema>;
export declare class FollowUpBossBubble<T extends FUBParams = FUBParams> extends ServiceBubble<T, Extract<FUBResult, {
    operation: T['operation'];
}>> {
    static readonly type: "service";
    static readonly service = "followupboss";
    static readonly authType: "oauth";
    static readonly bubbleName = "followupboss";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"list_people">;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        sort: z.ZodOptional<z.ZodString>;
        fields: z.ZodOptional<z.ZodString>;
        includeTrash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_people";
        limit: number;
        offset: number;
        includeTrash: boolean;
        sort?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string | undefined;
    }, {
        operation: "list_people";
        sort?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        includeTrash?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_person">;
        person_id: z.ZodNumber;
        fields: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_person";
        person_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string | undefined;
    }, {
        operation: "get_person";
        person_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        fields?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_person">;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_person";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        tags?: string[] | undefined;
        source?: string | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        phones?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        stage?: string | undefined;
        assignedTo?: number | undefined;
    }, {
        operation: "create_person";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        tags?: string[] | undefined;
        source?: string | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        phones?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        stage?: string | undefined;
        assignedTo?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_person">;
        person_id: z.ZodNumber;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
            value: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            isPrimary: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }, {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }>, "many">>;
        stage: z.ZodOptional<z.ZodString>;
        source: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_person";
        person_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        tags?: string[] | undefined;
        source?: string | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        phones?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        stage?: string | undefined;
        assignedTo?: number | undefined;
    }, {
        operation: "update_person";
        person_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        tags?: string[] | undefined;
        source?: string | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        emails?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        phones?: {
            value: string;
            type?: string | undefined;
            isPrimary?: boolean | undefined;
        }[] | undefined;
        stage?: string | undefined;
        assignedTo?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_person">;
        person_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_person";
        person_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_person";
        person_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_tasks">;
        personId: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_tasks";
        limit: number;
        offset: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        personId?: number | undefined;
    }, {
        operation: "list_tasks";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        personId?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_task">;
        task_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_task";
        task_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_task";
        task_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_task">;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "create_task";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        assignedTo?: number | undefined;
        personId?: number | undefined;
        dueDate?: string | undefined;
    }, {
        name: string;
        operation: "create_task";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        assignedTo?: number | undefined;
        personId?: number | undefined;
        dueDate?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_task">;
        task_id: z.ZodNumber;
        name: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        dueDate: z.ZodOptional<z.ZodString>;
        completed: z.ZodOptional<z.ZodBoolean>;
        assignedTo: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_task";
        task_id: number;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        assignedTo?: number | undefined;
        dueDate?: string | undefined;
        completed?: boolean | undefined;
    }, {
        operation: "update_task";
        task_id: number;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        assignedTo?: number | undefined;
        dueDate?: string | undefined;
        completed?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_task">;
        task_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_task";
        task_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_task";
        task_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_notes">;
        personId: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_notes";
        limit: number;
        offset: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        personId?: number | undefined;
    }, {
        operation: "list_notes";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        personId?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_note">;
        personId: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_note";
        body: string;
        personId: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        subject?: string | undefined;
    }, {
        operation: "create_note";
        body: string;
        personId: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        subject?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_note">;
        note_id: z.ZodNumber;
        subject: z.ZodOptional<z.ZodString>;
        body: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_note";
        note_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        subject?: string | undefined;
        body?: string | undefined;
    }, {
        operation: "update_note";
        note_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        subject?: string | undefined;
        body?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_note">;
        note_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_note";
        note_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_note";
        note_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_deals">;
        personId: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_deals";
        limit: number;
        offset: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        personId?: number | undefined;
    }, {
        operation: "list_deals";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        personId?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_deal">;
        deal_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_deal";
        deal_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_deal";
        deal_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_deal">;
        personId: z.ZodOptional<z.ZodNumber>;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_deal";
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        price?: number | undefined;
        stage?: string | undefined;
        personId?: number | undefined;
        closeDate?: string | undefined;
    }, {
        operation: "create_deal";
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        price?: number | undefined;
        stage?: string | undefined;
        personId?: number | undefined;
        closeDate?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_deal">;
        deal_id: z.ZodNumber;
        name: z.ZodOptional<z.ZodString>;
        price: z.ZodOptional<z.ZodNumber>;
        stage: z.ZodOptional<z.ZodString>;
        closeDate: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_deal";
        deal_id: number;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        price?: number | undefined;
        stage?: string | undefined;
        closeDate?: string | undefined;
    }, {
        operation: "update_deal";
        deal_id: number;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        price?: number | undefined;
        stage?: string | undefined;
        closeDate?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_events">;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        personId: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_events";
        limit: number;
        offset: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        personId?: number | undefined;
    }, {
        operation: "list_events";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        personId?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_event">;
        event_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_event";
        event_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_event";
        event_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_event">;
        type: z.ZodString;
        source: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodString>;
        person: z.ZodObject<{
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
            }, {
                value: string;
            }>, "many">>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }, {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        }>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        operation: "create_event";
        person: {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        };
        message?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        source?: string | undefined;
    }, {
        type: string;
        operation: "create_event";
        person: {
            tags?: string[] | undefined;
            firstName?: string | undefined;
            lastName?: string | undefined;
            emails?: {
                value: string;
            }[] | undefined;
            phones?: {
                value: string;
            }[] | undefined;
        };
        message?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        source?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_calls">;
        personId: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_calls";
        limit: number;
        offset: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        personId?: number | undefined;
    }, {
        operation: "list_calls";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        personId?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_call">;
        personId: z.ZodNumber;
        outcome: z.ZodOptional<z.ZodString>;
        note: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_call";
        personId: number;
        duration?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        outcome?: string | undefined;
        note?: string | undefined;
    }, {
        operation: "create_call";
        personId: number;
        duration?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        outcome?: string | undefined;
        note?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_appointments">;
        personId: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_appointments";
        limit: number;
        offset: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        personId?: number | undefined;
    }, {
        operation: "list_appointments";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        personId?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_appointment">;
        personId: z.ZodOptional<z.ZodNumber>;
        title: z.ZodString;
        startTime: z.ZodString;
        endTime: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        operation: "create_appointment";
        startTime: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        location?: string | undefined;
        personId?: number | undefined;
        endTime?: string | undefined;
    }, {
        title: string;
        operation: "create_appointment";
        startTime: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        location?: string | undefined;
        personId?: number | undefined;
        endTime?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_webhooks">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_webhooks";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "list_webhooks";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_webhook">;
        webhook_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_webhook";
        webhook_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_webhook";
        webhook_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_webhook">;
        event: z.ZodEnum<["peopleCreated", "peopleUpdated", "peopleDeleted", "peopleTagsCreated", "peopleStageUpdated", "peopleRelationshipCreated", "peopleRelationshipUpdated", "peopleRelationshipDeleted", "notesCreated", "notesUpdated", "notesDeleted", "emailsCreated", "emailsUpdated", "emailsDeleted", "tasksCreated", "tasksUpdated", "tasksDeleted", "appointmentsCreated", "appointmentsUpdated", "appointmentsDeleted", "textMessagesCreated", "textMessagesUpdated", "textMessagesDeleted", "callsCreated", "callsUpdated", "callsDeleted", "dealsCreated", "dealsUpdated", "dealsDeleted", "eventsCreated", "stageCreated", "stageUpdated", "stageDeleted", "pipelineCreated", "pipelineUpdated", "pipelineDeleted", "pipelineStageCreated", "pipelineStageUpdated", "pipelineStageDeleted", "customFieldsCreated", "customFieldsUpdated", "customFieldsDeleted", "dealCustomFieldsCreated", "dealCustomFieldsUpdated", "dealCustomFieldsDeleted", "emEventsOpened", "emEventsClicked", "emEventsUnsubscribed", "reactionCreated", "reactionDeleted", "threadedReplyCreated", "threadedReplyUpdated", "threadedReplyDeleted"]>;
        url: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        url: string;
        operation: "create_webhook";
        event: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        url: string;
        operation: "create_webhook";
        event: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_webhook">;
        webhook_id: z.ZodNumber;
        event: z.ZodOptional<z.ZodEnum<["peopleCreated", "peopleUpdated", "peopleDeleted", "peopleTagsCreated", "peopleStageUpdated", "peopleRelationshipCreated", "peopleRelationshipUpdated", "peopleRelationshipDeleted", "notesCreated", "notesUpdated", "notesDeleted", "emailsCreated", "emailsUpdated", "emailsDeleted", "tasksCreated", "tasksUpdated", "tasksDeleted", "appointmentsCreated", "appointmentsUpdated", "appointmentsDeleted", "textMessagesCreated", "textMessagesUpdated", "textMessagesDeleted", "callsCreated", "callsUpdated", "callsDeleted", "dealsCreated", "dealsUpdated", "dealsDeleted", "eventsCreated", "stageCreated", "stageUpdated", "stageDeleted", "pipelineCreated", "pipelineUpdated", "pipelineDeleted", "pipelineStageCreated", "pipelineStageUpdated", "pipelineStageDeleted", "customFieldsCreated", "customFieldsUpdated", "customFieldsDeleted", "dealCustomFieldsCreated", "dealCustomFieldsUpdated", "dealCustomFieldsDeleted", "emEventsOpened", "emEventsClicked", "emEventsUnsubscribed", "reactionCreated", "reactionDeleted", "threadedReplyCreated", "threadedReplyUpdated", "threadedReplyDeleted"]>>;
        url: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_webhook";
        webhook_id: number;
        url?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        event?: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted" | undefined;
    }, {
        operation: "update_webhook";
        webhook_id: number;
        url?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        event?: "peopleCreated" | "peopleUpdated" | "peopleDeleted" | "peopleTagsCreated" | "peopleStageUpdated" | "peopleRelationshipCreated" | "peopleRelationshipUpdated" | "peopleRelationshipDeleted" | "notesCreated" | "notesUpdated" | "notesDeleted" | "emailsCreated" | "emailsUpdated" | "emailsDeleted" | "tasksCreated" | "tasksUpdated" | "tasksDeleted" | "appointmentsCreated" | "appointmentsUpdated" | "appointmentsDeleted" | "textMessagesCreated" | "textMessagesUpdated" | "textMessagesDeleted" | "callsCreated" | "callsUpdated" | "callsDeleted" | "dealsCreated" | "dealsUpdated" | "dealsDeleted" | "eventsCreated" | "stageCreated" | "stageUpdated" | "stageDeleted" | "pipelineCreated" | "pipelineUpdated" | "pipelineDeleted" | "pipelineStageCreated" | "pipelineStageUpdated" | "pipelineStageDeleted" | "customFieldsCreated" | "customFieldsUpdated" | "customFieldsDeleted" | "dealCustomFieldsCreated" | "dealCustomFieldsUpdated" | "dealCustomFieldsDeleted" | "emEventsOpened" | "emEventsClicked" | "emEventsUnsubscribed" | "reactionCreated" | "reactionDeleted" | "threadedReplyCreated" | "threadedReplyUpdated" | "threadedReplyDeleted" | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_webhook">;
        webhook_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_webhook";
        webhook_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_webhook";
        webhook_id: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"list_people">;
        success: z.ZodBoolean;
        people: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        _metadata: z.ZodOptional<z.ZodObject<{
            total: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
            offset: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_people";
        people?: z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_people";
        people?: z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_person">;
        success: z.ZodBoolean;
        person: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_person";
        person?: z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_person";
        person?: z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_person">;
        success: z.ZodBoolean;
        person: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_person";
        person?: z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_person";
        person?: z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_person">;
        success: z.ZodBoolean;
        person: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_person";
        person?: z.objectOutputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_person";
        person?: z.objectInputType<{
            id: z.ZodNumber;
            firstName: z.ZodOptional<z.ZodString>;
            lastName: z.ZodOptional<z.ZodString>;
            emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                value: z.ZodString;
                type: z.ZodOptional<z.ZodString>;
                isPrimary: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }, {
                value: string;
                type?: string | undefined;
                isPrimary?: boolean | undefined;
            }>, "many">>;
            stage: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            created: z.ZodOptional<z.ZodString>;
            updated: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_person">;
        success: z.ZodBoolean;
        deleted_id: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_person";
        deleted_id?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_person";
        deleted_id?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_tasks">;
        success: z.ZodBoolean;
        tasks: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        _metadata: z.ZodOptional<z.ZodObject<{
            total: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
            offset: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_tasks";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        tasks?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_tasks";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        tasks?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_task">;
        success: z.ZodBoolean;
        task: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_task";
        task?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_task";
        task?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_task">;
        success: z.ZodBoolean;
        task: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_task";
        task?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_task";
        task?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_task">;
        success: z.ZodBoolean;
        task: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_task";
        task?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_task";
        task?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            dueDate: z.ZodOptional<z.ZodString>;
            completed: z.ZodOptional<z.ZodBoolean>;
            assignedTo: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_task">;
        success: z.ZodBoolean;
        deleted_id: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_task";
        deleted_id?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_task";
        deleted_id?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_notes">;
        success: z.ZodBoolean;
        notes: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        _metadata: z.ZodOptional<z.ZodObject<{
            total: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
            offset: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_notes";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        notes?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_notes";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        notes?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_note">;
        success: z.ZodBoolean;
        note: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_note";
        note?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_note";
        note?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_note">;
        success: z.ZodBoolean;
        note: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_note";
        note?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_note";
        note?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            subject: z.ZodOptional<z.ZodString>;
            body: z.ZodString;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_note">;
        success: z.ZodBoolean;
        deleted_id: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_note";
        deleted_id?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_note";
        deleted_id?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_deals">;
        success: z.ZodBoolean;
        deals: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        _metadata: z.ZodOptional<z.ZodObject<{
            total: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
            offset: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_deals";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        deals?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_deals";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        deals?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_deal">;
        success: z.ZodBoolean;
        deal: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_deal";
        deal?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_deal";
        deal?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_deal">;
        success: z.ZodBoolean;
        deal: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_deal";
        deal?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_deal";
        deal?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_deal">;
        success: z.ZodBoolean;
        deal: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_deal";
        deal?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_deal";
        deal?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodOptional<z.ZodString>;
            price: z.ZodOptional<z.ZodNumber>;
            stage: z.ZodOptional<z.ZodString>;
            closeDate: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_events">;
        success: z.ZodBoolean;
        events: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        _metadata: z.ZodOptional<z.ZodObject<{
            total: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
            offset: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_events";
        events?: z.objectOutputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_events";
        events?: z.objectInputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_event">;
        success: z.ZodBoolean;
        event: z.ZodOptional<z.ZodObject<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_event";
        event?: z.objectOutputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_event";
        event?: z.objectInputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_event">;
        success: z.ZodBoolean;
        event: z.ZodOptional<z.ZodObject<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_event";
        event?: z.objectOutputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_event";
        event?: z.objectInputType<{
            id: z.ZodOptional<z.ZodNumber>;
            type: z.ZodString;
            source: z.ZodOptional<z.ZodString>;
            message: z.ZodOptional<z.ZodString>;
            person: z.ZodOptional<z.ZodObject<{
                firstName: z.ZodOptional<z.ZodString>;
                lastName: z.ZodOptional<z.ZodString>;
                emails: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                phones: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                }, {
                    value: string;
                }>, "many">>;
                tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }, {
                tags?: string[] | undefined;
                firstName?: string | undefined;
                lastName?: string | undefined;
                emails?: {
                    value: string;
                }[] | undefined;
                phones?: {
                    value: string;
                }[] | undefined;
            }>>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_calls">;
        success: z.ZodBoolean;
        calls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        _metadata: z.ZodOptional<z.ZodObject<{
            total: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
            offset: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_calls";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        calls?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_calls";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        calls?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_call">;
        success: z.ZodBoolean;
        call: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_call";
        call?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_call";
        call?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodNumber;
            outcome: z.ZodOptional<z.ZodString>;
            note: z.ZodOptional<z.ZodString>;
            duration: z.ZodOptional<z.ZodNumber>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_appointments">;
        success: z.ZodBoolean;
        appointments: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        _metadata: z.ZodOptional<z.ZodObject<{
            total: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
            offset: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }, {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_appointments";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        appointments?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_appointments";
        _metadata?: {
            limit?: number | undefined;
            offset?: number | undefined;
            total?: number | undefined;
        } | undefined;
        appointments?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_appointment">;
        success: z.ZodBoolean;
        appointment: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_appointment";
        appointment?: z.objectOutputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_appointment";
        appointment?: z.objectInputType<{
            id: z.ZodNumber;
            personId: z.ZodOptional<z.ZodNumber>;
            title: z.ZodOptional<z.ZodString>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            location: z.ZodOptional<z.ZodString>;
            created: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_webhooks">;
        success: z.ZodBoolean;
        webhooks: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_webhooks";
        webhooks?: z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_webhooks";
        webhooks?: z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_webhook">;
        success: z.ZodBoolean;
        webhook: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_webhook";
        webhook?: z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_webhook";
        webhook?: z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_webhook">;
        success: z.ZodBoolean;
        webhook: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_webhook";
        webhook?: z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_webhook";
        webhook?: z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_webhook">;
        success: z.ZodBoolean;
        webhook: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_webhook";
        webhook?: z.objectOutputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_webhook";
        webhook?: z.objectInputType<{
            id: z.ZodNumber;
            event: z.ZodString;
            url: z.ZodString;
            status: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_webhook">;
        success: z.ZodBoolean;
        deleted_id: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_webhook";
        deleted_id?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_webhook";
        deleted_id?: number | undefined;
    }>]>;
    static readonly shortDescription = "Follow Up Boss CRM integration";
    static readonly longDescription = "\n    Follow Up Boss CRM integration for real estate professionals.\n    Use cases:\n    - Manage contacts/people with full CRUD operations\n    - Create and track tasks\n    - Add notes to contacts\n    - Manage deals in the pipeline\n    - Log calls and appointments\n    - Create events (preferred method for new leads)\n    - Automate lead management workflows\n  ";
    static readonly alias = "fub";
    constructor(params?: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    private makeFUBApiRequest;
    protected performAction(context?: BubbleContext): Promise<Extract<FUBResult, {
        operation: T['operation'];
    }>>;
    private listPeople;
    private getPerson;
    private createPerson;
    private updatePerson;
    private deletePerson;
    private listTasks;
    private getTask;
    private createTask;
    private updateTask;
    private deleteTask;
    private listNotes;
    private createNote;
    private updateNote;
    private deleteNote;
    private listDeals;
    private getDeal;
    private createDeal;
    private updateDeal;
    private listEvents;
    private getEvent;
    private createEvent;
    private listCalls;
    private createCall;
    private listAppointments;
    private createAppointment;
    private listWebhooks;
    private getWebhook;
    private createWebhook;
    private updateWebhook;
    private deleteWebhook;
    protected chooseCredential(): string | undefined;
}
export {};
//# sourceMappingURL=followupboss.d.ts.map