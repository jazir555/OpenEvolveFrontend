import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// Define common FUB schemas
const FUBPersonSchema = z
    .object({
    id: z.number().describe('Unique person identifier'),
    firstName: z.string().optional().describe('First name'),
    lastName: z.string().optional().describe('Last name'),
    emails: z
        .array(z.object({
        value: z.string(),
        type: z.string().optional(),
        isPrimary: z.boolean().optional(),
    }))
        .optional()
        .describe('Email addresses'),
    phones: z
        .array(z.object({
        value: z.string(),
        type: z.string().optional(),
        isPrimary: z.boolean().optional(),
    }))
        .optional()
        .describe('Phone numbers'),
    stage: z.string().optional().describe('Current stage in pipeline'),
    source: z.string().optional().describe('Lead source'),
    assignedTo: z.number().optional().describe('Assigned user ID'),
    tags: z.array(z.string()).optional().describe('Tags applied to person'),
    created: z.string().optional().describe('Creation timestamp'),
    updated: z.string().optional().describe('Last update timestamp'),
})
    .passthrough()
    .describe('FUB person/contact object');
const FUBTaskSchema = z
    .object({
    id: z.number().describe('Unique task identifier'),
    personId: z.number().optional().describe('Associated person ID'),
    name: z.string().describe('Task name/title'),
    description: z.string().optional().describe('Task description'),
    dueDate: z.string().optional().describe('Due date (YYYY-MM-DD)'),
    completed: z.boolean().optional().describe('Whether task is completed'),
    assignedTo: z.number().optional().describe('Assigned user ID'),
    created: z.string().optional().describe('Creation timestamp'),
})
    .passthrough()
    .describe('FUB task object');
const FUBNoteSchema = z
    .object({
    id: z.number().describe('Unique note identifier'),
    personId: z.number().describe('Associated person ID'),
    subject: z.string().optional().describe('Note subject'),
    body: z.string().describe('Note content'),
    created: z.string().optional().describe('Creation timestamp'),
})
    .passthrough()
    .describe('FUB note object');
const FUBDealSchema = z
    .object({
    id: z.number().describe('Unique deal identifier'),
    personId: z.number().optional().describe('Associated person ID'),
    name: z.string().optional().describe('Deal name'),
    price: z.number().optional().describe('Deal price/value'),
    stage: z.string().optional().describe('Deal stage'),
    closeDate: z.string().optional().describe('Expected close date'),
    created: z.string().optional().describe('Creation timestamp'),
})
    .passthrough()
    .describe('FUB deal object');
const FUBEventSchema = z
    .object({
    id: z.number().optional().describe('Event identifier'),
    type: z
        .string()
        .describe('Event type (e.g., "Showing Request", "Registration")'),
    source: z.string().optional().describe('Event source'),
    message: z.string().optional().describe('Event message'),
    person: z
        .object({
        firstName: z.string().optional(),
        lastName: z.string().optional(),
        emails: z.array(z.object({ value: z.string() })).optional(),
        phones: z.array(z.object({ value: z.string() })).optional(),
        tags: z.array(z.string()).optional(),
    })
        .optional()
        .describe('Person data for the event'),
    created: z.string().optional().describe('Creation timestamp'),
})
    .passthrough()
    .describe('FUB event object');
const FUBCallSchema = z
    .object({
    id: z.number().describe('Unique call identifier'),
    personId: z.number().describe('Associated person ID'),
    outcome: z.string().optional().describe('Call outcome'),
    note: z.string().optional().describe('Call notes'),
    duration: z.number().optional().describe('Call duration in seconds'),
    created: z.string().optional().describe('Creation timestamp'),
})
    .passthrough()
    .describe('FUB call object');
const FUBAppointmentSchema = z
    .object({
    id: z.number().describe('Unique appointment identifier'),
    personId: z.number().optional().describe('Associated person ID'),
    title: z.string().optional().describe('Appointment title'),
    startTime: z.string().optional().describe('Start time'),
    endTime: z.string().optional().describe('End time'),
    location: z.string().optional().describe('Location'),
    created: z.string().optional().describe('Creation timestamp'),
})
    .passthrough()
    .describe('FUB appointment object');
const FUBWebhookSchema = z
    .object({
    id: z.number().describe('Unique webhook identifier'),
    event: z.string().describe('Webhook event type'),
    url: z.string().describe('Callback URL'),
    status: z.string().optional().describe('Webhook status (Active/Inactive)'),
})
    .passthrough()
    .describe('FUB webhook object');
// Supported webhook event types
const FUB_WEBHOOK_EVENTS = [
    'peopleCreated',
    'peopleUpdated',
    'peopleDeleted',
    'peopleTagsCreated',
    'peopleStageUpdated',
    'peopleRelationshipCreated',
    'peopleRelationshipUpdated',
    'peopleRelationshipDeleted',
    'notesCreated',
    'notesUpdated',
    'notesDeleted',
    'emailsCreated',
    'emailsUpdated',
    'emailsDeleted',
    'tasksCreated',
    'tasksUpdated',
    'tasksDeleted',
    'appointmentsCreated',
    'appointmentsUpdated',
    'appointmentsDeleted',
    'textMessagesCreated',
    'textMessagesUpdated',
    'textMessagesDeleted',
    'callsCreated',
    'callsUpdated',
    'callsDeleted',
    'dealsCreated',
    'dealsUpdated',
    'dealsDeleted',
    'eventsCreated',
    'stageCreated',
    'stageUpdated',
    'stageDeleted',
    'pipelineCreated',
    'pipelineUpdated',
    'pipelineDeleted',
    'pipelineStageCreated',
    'pipelineStageUpdated',
    'pipelineStageDeleted',
    'customFieldsCreated',
    'customFieldsUpdated',
    'customFieldsDeleted',
    'dealCustomFieldsCreated',
    'dealCustomFieldsUpdated',
    'dealCustomFieldsDeleted',
    'emEventsOpened',
    'emEventsClicked',
    'emEventsUnsubscribed',
    'reactionCreated',
    'reactionDeleted',
    'threadedReplyCreated',
    'threadedReplyUpdated',
    'threadedReplyDeleted',
];
// Define the parameters schema for FUB operations
const FUBParamsSchema = z.discriminatedUnion('operation', [
    // People operations
    z.object({
        operation: z.literal('list_people').describe('List people/contacts'),
        limit: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(25)
            .describe('Number of results to return'),
        offset: z
            .number()
            .optional()
            .default(0)
            .describe('Number of results to skip'),
        sort: z.string().optional().describe('Sort field'),
        fields: z
            .string()
            .optional()
            .describe('Comma-separated fields to return (use "allFields" for all)'),
        includeTrash: z
            .boolean()
            .optional()
            .default(false)
            .describe('Include people in Trash stage'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('get_person').describe('Get a specific person by ID'),
        person_id: z.number().describe('Person ID to retrieve'),
        fields: z.string().optional().describe('Comma-separated fields to return'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z
            .literal('create_person')
            .describe('Create a new person/contact'),
        firstName: z.string().optional().describe('First name'),
        lastName: z.string().optional().describe('Last name'),
        emails: z
            .array(z.object({
            value: z.string().email(),
            type: z.string().optional(),
            isPrimary: z.boolean().optional(),
        }))
            .optional()
            .describe('Email addresses'),
        phones: z
            .array(z.object({
            value: z.string(),
            type: z.string().optional(),
            isPrimary: z.boolean().optional(),
        }))
            .optional()
            .describe('Phone numbers'),
        stage: z.string().optional().describe('Initial stage'),
        source: z.string().optional().describe('Lead source'),
        assignedTo: z.number().optional().describe('Assigned user ID'),
        tags: z.array(z.string()).optional().describe('Tags to apply'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('update_person').describe('Update an existing person'),
        person_id: z.number().describe('Person ID to update'),
        firstName: z.string().optional().describe('First name'),
        lastName: z.string().optional().describe('Last name'),
        emails: z
            .array(z.object({
            value: z.string().email(),
            type: z.string().optional(),
            isPrimary: z.boolean().optional(),
        }))
            .optional()
            .describe('Email addresses'),
        phones: z
            .array(z.object({
            value: z.string(),
            type: z.string().optional(),
            isPrimary: z.boolean().optional(),
        }))
            .optional()
            .describe('Phone numbers'),
        stage: z.string().optional().describe('Stage'),
        source: z.string().optional().describe('Lead source'),
        assignedTo: z.number().optional().describe('Assigned user ID'),
        tags: z.array(z.string()).optional().describe('Tags'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('delete_person').describe('Delete a person'),
        person_id: z.number().describe('Person ID to delete'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Task operations
    z.object({
        operation: z.literal('list_tasks').describe('List tasks'),
        personId: z.number().optional().describe('Filter by person ID'),
        limit: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(25)
            .describe('Number of results to return'),
        offset: z
            .number()
            .optional()
            .default(0)
            .describe('Number of results to skip'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('get_task').describe('Get a specific task by ID'),
        task_id: z.number().describe('Task ID to retrieve'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('create_task').describe('Create a new task'),
        personId: z.number().optional().describe('Associated person ID'),
        name: z.string().min(1).describe('Task name/title'),
        description: z.string().optional().describe('Task description'),
        dueDate: z.string().optional().describe('Due date (YYYY-MM-DD)'),
        assignedTo: z.number().optional().describe('Assigned user ID'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('update_task').describe('Update an existing task'),
        task_id: z.number().describe('Task ID to update'),
        name: z.string().optional().describe('Task name/title'),
        description: z.string().optional().describe('Task description'),
        dueDate: z.string().optional().describe('Due date (YYYY-MM-DD)'),
        completed: z.boolean().optional().describe('Whether task is completed'),
        assignedTo: z.number().optional().describe('Assigned user ID'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('delete_task').describe('Delete a task'),
        task_id: z.number().describe('Task ID to delete'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Note operations
    z.object({
        operation: z.literal('list_notes').describe('List notes'),
        personId: z.number().optional().describe('Filter by person ID'),
        limit: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(25)
            .describe('Number of results to return'),
        offset: z
            .number()
            .optional()
            .default(0)
            .describe('Number of results to skip'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('create_note').describe('Create a new note'),
        personId: z.number().describe('Associated person ID'),
        subject: z.string().optional().describe('Note subject'),
        body: z.string().min(1).describe('Note content'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('update_note').describe('Update an existing note'),
        note_id: z.number().describe('Note ID to update'),
        subject: z.string().optional().describe('Note subject'),
        body: z.string().optional().describe('Note content'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('delete_note').describe('Delete a note'),
        note_id: z.number().describe('Note ID to delete'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Deal operations
    z.object({
        operation: z.literal('list_deals').describe('List deals'),
        personId: z.number().optional().describe('Filter by person ID'),
        limit: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(25)
            .describe('Number of results to return'),
        offset: z
            .number()
            .optional()
            .default(0)
            .describe('Number of results to skip'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('get_deal').describe('Get a specific deal by ID'),
        deal_id: z.number().describe('Deal ID to retrieve'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('create_deal').describe('Create a new deal'),
        personId: z.number().optional().describe('Associated person ID'),
        name: z.string().optional().describe('Deal name'),
        price: z.number().optional().describe('Deal price/value'),
        stage: z.string().optional().describe('Deal stage'),
        closeDate: z
            .string()
            .optional()
            .describe('Expected close date (YYYY-MM-DD)'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('update_deal').describe('Update an existing deal'),
        deal_id: z.number().describe('Deal ID to update'),
        name: z.string().optional().describe('Deal name'),
        price: z.number().optional().describe('Deal price/value'),
        stage: z.string().optional().describe('Deal stage'),
        closeDate: z
            .string()
            .optional()
            .describe('Expected close date (YYYY-MM-DD)'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Event operations (preferred for new leads)
    z.object({
        operation: z.literal('list_events').describe('List/search events'),
        limit: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(25)
            .describe('Number of results to return'),
        offset: z
            .number()
            .optional()
            .default(0)
            .describe('Number of results to skip'),
        personId: z.number().optional().describe('Filter by person ID'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('get_event').describe('Get a specific event by ID'),
        event_id: z.number().describe('Event ID to retrieve'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z
            .literal('create_event')
            .describe('Create an event (preferred for new leads)'),
        type: z
            .string()
            .min(1)
            .describe('Event type (e.g., "Showing Request", "Registration")'),
        source: z.string().optional().describe('Event source'),
        message: z.string().optional().describe('Event message'),
        person: z
            .object({
            firstName: z.string().optional(),
            lastName: z.string().optional(),
            emails: z.array(z.object({ value: z.string().email() })).optional(),
            phones: z.array(z.object({ value: z.string() })).optional(),
            tags: z.array(z.string()).optional(),
        })
            .describe('Person data for the event'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Call operations
    z.object({
        operation: z.literal('list_calls').describe('List calls'),
        personId: z.number().optional().describe('Filter by person ID'),
        limit: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(25)
            .describe('Number of results to return'),
        offset: z
            .number()
            .optional()
            .default(0)
            .describe('Number of results to skip'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('create_call').describe('Log a call'),
        personId: z.number().describe('Associated person ID'),
        outcome: z.string().optional().describe('Call outcome'),
        note: z.string().optional().describe('Call notes'),
        duration: z.number().optional().describe('Call duration in seconds'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Appointment operations
    z.object({
        operation: z.literal('list_appointments').describe('List appointments'),
        personId: z.number().optional().describe('Filter by person ID'),
        limit: z
            .number()
            .min(1)
            .max(100)
            .optional()
            .default(25)
            .describe('Number of results to return'),
        offset: z
            .number()
            .optional()
            .default(0)
            .describe('Number of results to skip'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z
            .literal('create_appointment')
            .describe('Create an appointment'),
        personId: z.number().optional().describe('Associated person ID'),
        title: z.string().min(1).describe('Appointment title'),
        startTime: z.string().describe('Start time (ISO 8601)'),
        endTime: z.string().optional().describe('End time (ISO 8601)'),
        location: z.string().optional().describe('Location'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Webhook operations
    z.object({
        operation: z.literal('list_webhooks').describe('List registered webhooks'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z
            .literal('get_webhook')
            .describe('Get a specific webhook by ID'),
        webhook_id: z.number().describe('Webhook ID to retrieve'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('create_webhook').describe('Register a new webhook'),
        event: z
            .enum(FUB_WEBHOOK_EVENTS)
            .describe('Webhook event type to subscribe to'),
        url: z
            .string()
            .url()
            .describe('HTTPS callback URL for webhook notifications'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z
            .literal('update_webhook')
            .describe('Update an existing webhook'),
        webhook_id: z.number().describe('Webhook ID to update'),
        event: z
            .enum(FUB_WEBHOOK_EVENTS)
            .optional()
            .describe('New webhook event type'),
        url: z.string().url().optional().describe('New HTTPS callback URL'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    z.object({
        operation: z.literal('delete_webhook').describe('Delete a webhook'),
        webhook_id: z.number().describe('Webhook ID to delete'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
]);
// Define result schemas for different operations
const FUBResultSchema = z.discriminatedUnion('operation', [
    // People results
    z.object({
        operation: z.literal('list_people'),
        success: z.boolean(),
        people: z.array(FUBPersonSchema).optional(),
        _metadata: z
            .object({
            total: z.number().optional(),
            limit: z.number().optional(),
            offset: z.number().optional(),
        })
            .optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('get_person'),
        success: z.boolean(),
        person: FUBPersonSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_person'),
        success: z.boolean(),
        person: FUBPersonSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('update_person'),
        success: z.boolean(),
        person: FUBPersonSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('delete_person'),
        success: z.boolean(),
        deleted_id: z.number().optional(),
        error: z.string(),
    }),
    // Task results
    z.object({
        operation: z.literal('list_tasks'),
        success: z.boolean(),
        tasks: z.array(FUBTaskSchema).optional(),
        _metadata: z
            .object({
            total: z.number().optional(),
            limit: z.number().optional(),
            offset: z.number().optional(),
        })
            .optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('get_task'),
        success: z.boolean(),
        task: FUBTaskSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_task'),
        success: z.boolean(),
        task: FUBTaskSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('update_task'),
        success: z.boolean(),
        task: FUBTaskSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('delete_task'),
        success: z.boolean(),
        deleted_id: z.number().optional(),
        error: z.string(),
    }),
    // Note results
    z.object({
        operation: z.literal('list_notes'),
        success: z.boolean(),
        notes: z.array(FUBNoteSchema).optional(),
        _metadata: z
            .object({
            total: z.number().optional(),
            limit: z.number().optional(),
            offset: z.number().optional(),
        })
            .optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_note'),
        success: z.boolean(),
        note: FUBNoteSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('update_note'),
        success: z.boolean(),
        note: FUBNoteSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('delete_note'),
        success: z.boolean(),
        deleted_id: z.number().optional(),
        error: z.string(),
    }),
    // Deal results
    z.object({
        operation: z.literal('list_deals'),
        success: z.boolean(),
        deals: z.array(FUBDealSchema).optional(),
        _metadata: z
            .object({
            total: z.number().optional(),
            limit: z.number().optional(),
            offset: z.number().optional(),
        })
            .optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('get_deal'),
        success: z.boolean(),
        deal: FUBDealSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_deal'),
        success: z.boolean(),
        deal: FUBDealSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('update_deal'),
        success: z.boolean(),
        deal: FUBDealSchema.optional(),
        error: z.string(),
    }),
    // Event results
    z.object({
        operation: z.literal('list_events'),
        success: z.boolean(),
        events: z.array(FUBEventSchema).optional(),
        _metadata: z
            .object({
            total: z.number().optional(),
            limit: z.number().optional(),
            offset: z.number().optional(),
        })
            .optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('get_event'),
        success: z.boolean(),
        event: FUBEventSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_event'),
        success: z.boolean(),
        event: FUBEventSchema.optional(),
        error: z.string(),
    }),
    // Call results
    z.object({
        operation: z.literal('list_calls'),
        success: z.boolean(),
        calls: z.array(FUBCallSchema).optional(),
        _metadata: z
            .object({
            total: z.number().optional(),
            limit: z.number().optional(),
            offset: z.number().optional(),
        })
            .optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_call'),
        success: z.boolean(),
        call: FUBCallSchema.optional(),
        error: z.string(),
    }),
    // Appointment results
    z.object({
        operation: z.literal('list_appointments'),
        success: z.boolean(),
        appointments: z.array(FUBAppointmentSchema).optional(),
        _metadata: z
            .object({
            total: z.number().optional(),
            limit: z.number().optional(),
            offset: z.number().optional(),
        })
            .optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_appointment'),
        success: z.boolean(),
        appointment: FUBAppointmentSchema.optional(),
        error: z.string(),
    }),
    // Webhook results
    z.object({
        operation: z.literal('list_webhooks'),
        success: z.boolean(),
        webhooks: z.array(FUBWebhookSchema).optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('get_webhook'),
        success: z.boolean(),
        webhook: FUBWebhookSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('create_webhook'),
        success: z.boolean(),
        webhook: FUBWebhookSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('update_webhook'),
        success: z.boolean(),
        webhook: FUBWebhookSchema.optional(),
        error: z.string(),
    }),
    z.object({
        operation: z.literal('delete_webhook'),
        success: z.boolean(),
        deleted_id: z.number().optional(),
        error: z.string(),
    }),
]);
export class FollowUpBossBubble extends ServiceBubble {
    static type = 'service';
    static service = 'followupboss';
    static authType = 'oauth';
    static bubbleName = 'followupboss';
    static schema = FUBParamsSchema;
    static resultSchema = FUBResultSchema;
    static shortDescription = 'Follow Up Boss CRM integration';
    static longDescription = `
    Follow Up Boss CRM integration for real estate professionals.
    Use cases:
    - Manage contacts/people with full CRUD operations
    - Create and track tasks
    - Add notes to contacts
    - Manage deals in the pipeline
    - Log calls and appointments
    - Create events (preferred method for new leads)
    - Automate lead management workflows
  `;
    static alias = 'fub';
    constructor(params = { operation: 'list_people' }, context) {
        super(params, context);
    }
    async testCredential() {
        const credential = this.chooseCredential();
        try {
            const response = await fetch('https://api.followupboss.com/v1/me', {
                headers: {
                    Authorization: `Bearer ${credential}`,
                    'Content-Type': 'application/json',
                    'X-System': process.env.FUB_SYSTEM_NAME || 'Bubble-Lab',
                    'X-System-Key': process.env.FUB_SYSTEM_KEY || '',
                },
            });
            return response.ok;
        }
        catch {
            return false;
        }
    }
    async makeFUBApiRequest(endpoint, method = 'GET', body) {
        const url = `https://api.followupboss.com/v1${endpoint}`;
        const requestHeaders = {
            Authorization: `Bearer ${this.chooseCredential()}`,
            'Content-Type': 'application/json',
            'X-System': process.env.FUB_SYSTEM_NAME || 'Bubble-Lab',
            'X-System-Key': process.env.FUB_SYSTEM_KEY || '',
        };
        const requestInit = {
            method,
            headers: requestHeaders,
        };
        if (body && method !== 'GET') {
            requestInit.body = JSON.stringify(body);
        }
        const response = await fetch(url, requestInit);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`FUB API error: ${response.status} ${response.statusText} - ${errorText}`);
        }
        // Handle empty responses
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }
        else {
            return await response.text();
        }
    }
    async performAction(context) {
        void context;
        const { operation } = this.params;
        try {
            const result = await (async () => {
                switch (operation) {
                    // People operations
                    case 'list_people':
                        return await this.listPeople(this.params);
                    case 'get_person':
                        return await this.getPerson(this.params);
                    case 'create_person':
                        return await this.createPerson(this.params);
                    case 'update_person':
                        return await this.updatePerson(this.params);
                    case 'delete_person':
                        return await this.deletePerson(this.params);
                    // Task operations
                    case 'list_tasks':
                        return await this.listTasks(this.params);
                    case 'get_task':
                        return await this.getTask(this.params);
                    case 'create_task':
                        return await this.createTask(this.params);
                    case 'update_task':
                        return await this.updateTask(this.params);
                    case 'delete_task':
                        return await this.deleteTask(this.params);
                    // Note operations
                    case 'list_notes':
                        return await this.listNotes(this.params);
                    case 'create_note':
                        return await this.createNote(this.params);
                    case 'update_note':
                        return await this.updateNote(this.params);
                    case 'delete_note':
                        return await this.deleteNote(this.params);
                    // Deal operations
                    case 'list_deals':
                        return await this.listDeals(this.params);
                    case 'get_deal':
                        return await this.getDeal(this.params);
                    case 'create_deal':
                        return await this.createDeal(this.params);
                    case 'update_deal':
                        return await this.updateDeal(this.params);
                    // Event operations
                    case 'list_events':
                        return await this.listEvents(this.params);
                    case 'get_event':
                        return await this.getEvent(this.params);
                    case 'create_event':
                        return await this.createEvent(this.params);
                    // Call operations
                    case 'list_calls':
                        return await this.listCalls(this.params);
                    case 'create_call':
                        return await this.createCall(this.params);
                    // Appointment operations
                    case 'list_appointments':
                        return await this.listAppointments(this.params);
                    case 'create_appointment':
                        return await this.createAppointment(this.params);
                    // Webhook operations
                    case 'list_webhooks':
                        return await this.listWebhooks(this.params);
                    case 'get_webhook':
                        return await this.getWebhook(this.params);
                    case 'create_webhook':
                        return await this.createWebhook(this.params);
                    case 'update_webhook':
                        return await this.updateWebhook(this.params);
                    case 'delete_webhook':
                        return await this.deleteWebhook(this.params);
                    default:
                        throw new Error(`Unsupported operation: ${operation}`);
                }
            })();
            return result;
        }
        catch (error) {
            return {
                operation,
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
    // People operations
    async listPeople(params) {
        const queryParams = new URLSearchParams();
        if (params.limit)
            queryParams.set('limit', params.limit.toString());
        if (params.offset)
            queryParams.set('offset', params.offset.toString());
        if (params.sort)
            queryParams.set('sort', params.sort);
        if (params.fields)
            queryParams.set('fields', params.fields);
        if (params.includeTrash)
            queryParams.set('includeTrash', 'true');
        const response = (await this.makeFUBApiRequest(`/people?${queryParams.toString()}`));
        return {
            operation: 'list_people',
            success: true,
            people: response.people,
            _metadata: response._metadata,
            error: '',
        };
    }
    async getPerson(params) {
        const queryParams = params.fields ? `?fields=${params.fields}` : '';
        const response = (await this.makeFUBApiRequest(`/people/${params.person_id}${queryParams}`));
        return {
            operation: 'get_person',
            success: true,
            person: response,
            error: '',
        };
    }
    async createPerson(params) {
        const { operation: _, credentials: __, ...personData } = params;
        const response = (await this.makeFUBApiRequest('/people', 'POST', personData));
        return {
            operation: 'create_person',
            success: true,
            person: response,
            error: '',
        };
    }
    async updatePerson(params) {
        const { operation: _, credentials: __, person_id, ...personData } = params;
        const response = (await this.makeFUBApiRequest(`/people/${person_id}`, 'PUT', personData));
        return {
            operation: 'update_person',
            success: true,
            person: response,
            error: '',
        };
    }
    async deletePerson(params) {
        await this.makeFUBApiRequest(`/people/${params.person_id}`, 'DELETE');
        return {
            operation: 'delete_person',
            success: true,
            deleted_id: params.person_id,
            error: '',
        };
    }
    // Task operations
    async listTasks(params) {
        const queryParams = new URLSearchParams();
        if (params.personId)
            queryParams.set('personId', params.personId.toString());
        if (params.limit)
            queryParams.set('limit', params.limit.toString());
        if (params.offset)
            queryParams.set('offset', params.offset.toString());
        const response = (await this.makeFUBApiRequest(`/tasks?${queryParams.toString()}`));
        return {
            operation: 'list_tasks',
            success: true,
            tasks: response.tasks,
            _metadata: response._metadata,
            error: '',
        };
    }
    async getTask(params) {
        const response = (await this.makeFUBApiRequest(`/tasks/${params.task_id}`));
        return {
            operation: 'get_task',
            success: true,
            task: response,
            error: '',
        };
    }
    async createTask(params) {
        const { operation: _, credentials: __, ...taskData } = params;
        const response = (await this.makeFUBApiRequest('/tasks', 'POST', taskData));
        return {
            operation: 'create_task',
            success: true,
            task: response,
            error: '',
        };
    }
    async updateTask(params) {
        const { operation: _, credentials: __, task_id, ...taskData } = params;
        const response = (await this.makeFUBApiRequest(`/tasks/${task_id}`, 'PUT', taskData));
        return {
            operation: 'update_task',
            success: true,
            task: response,
            error: '',
        };
    }
    async deleteTask(params) {
        await this.makeFUBApiRequest(`/tasks/${params.task_id}`, 'DELETE');
        return {
            operation: 'delete_task',
            success: true,
            deleted_id: params.task_id,
            error: '',
        };
    }
    // Note operations
    async listNotes(params) {
        const queryParams = new URLSearchParams();
        if (params.personId)
            queryParams.set('personId', params.personId.toString());
        if (params.limit)
            queryParams.set('limit', params.limit.toString());
        if (params.offset)
            queryParams.set('offset', params.offset.toString());
        const response = (await this.makeFUBApiRequest(`/notes?${queryParams.toString()}`));
        return {
            operation: 'list_notes',
            success: true,
            notes: response.notes,
            _metadata: response._metadata,
            error: '',
        };
    }
    async createNote(params) {
        const { operation: _, credentials: __, ...noteData } = params;
        const response = (await this.makeFUBApiRequest('/notes', 'POST', noteData));
        return {
            operation: 'create_note',
            success: true,
            note: response,
            error: '',
        };
    }
    async updateNote(params) {
        const { operation: _, credentials: __, note_id, ...noteData } = params;
        const response = (await this.makeFUBApiRequest(`/notes/${note_id}`, 'PUT', noteData));
        return {
            operation: 'update_note',
            success: true,
            note: response,
            error: '',
        };
    }
    async deleteNote(params) {
        await this.makeFUBApiRequest(`/notes/${params.note_id}`, 'DELETE');
        return {
            operation: 'delete_note',
            success: true,
            deleted_id: params.note_id,
            error: '',
        };
    }
    // Deal operations
    async listDeals(params) {
        const queryParams = new URLSearchParams();
        if (params.personId)
            queryParams.set('personId', params.personId.toString());
        if (params.limit)
            queryParams.set('limit', params.limit.toString());
        if (params.offset)
            queryParams.set('offset', params.offset.toString());
        const response = (await this.makeFUBApiRequest(`/deals?${queryParams.toString()}`));
        return {
            operation: 'list_deals',
            success: true,
            deals: response.deals,
            _metadata: response._metadata,
            error: '',
        };
    }
    async getDeal(params) {
        const response = (await this.makeFUBApiRequest(`/deals/${params.deal_id}`));
        return {
            operation: 'get_deal',
            success: true,
            deal: response,
            error: '',
        };
    }
    async createDeal(params) {
        const { operation: _, credentials: __, ...dealData } = params;
        const response = (await this.makeFUBApiRequest('/deals', 'POST', dealData));
        return {
            operation: 'create_deal',
            success: true,
            deal: response,
            error: '',
        };
    }
    async updateDeal(params) {
        const { operation: _, credentials: __, deal_id, ...dealData } = params;
        const response = (await this.makeFUBApiRequest(`/deals/${deal_id}`, 'PUT', dealData));
        return {
            operation: 'update_deal',
            success: true,
            deal: response,
            error: '',
        };
    }
    // Event operations
    async listEvents(params) {
        const queryParams = new URLSearchParams();
        if (params.limit)
            queryParams.set('limit', params.limit.toString());
        if (params.offset)
            queryParams.set('offset', params.offset.toString());
        if (params.personId)
            queryParams.set('personId', params.personId.toString());
        const response = (await this.makeFUBApiRequest(`/events?${queryParams.toString()}`));
        return {
            operation: 'list_events',
            success: true,
            events: response.events,
            _metadata: response._metadata,
            error: '',
        };
    }
    async getEvent(params) {
        const response = (await this.makeFUBApiRequest(`/events/${params.event_id}`));
        return {
            operation: 'get_event',
            success: true,
            event: response,
            error: '',
        };
    }
    async createEvent(params) {
        const { operation: _, credentials: __, ...eventData } = params;
        const response = (await this.makeFUBApiRequest('/events', 'POST', eventData));
        return {
            operation: 'create_event',
            success: true,
            event: response,
            error: '',
        };
    }
    // Call operations
    async listCalls(params) {
        const queryParams = new URLSearchParams();
        if (params.personId)
            queryParams.set('personId', params.personId.toString());
        if (params.limit)
            queryParams.set('limit', params.limit.toString());
        if (params.offset)
            queryParams.set('offset', params.offset.toString());
        const response = (await this.makeFUBApiRequest(`/calls?${queryParams.toString()}`));
        return {
            operation: 'list_calls',
            success: true,
            calls: response.calls,
            _metadata: response._metadata,
            error: '',
        };
    }
    async createCall(params) {
        const { operation: _, credentials: __, ...callData } = params;
        const response = (await this.makeFUBApiRequest('/calls', 'POST', callData));
        return {
            operation: 'create_call',
            success: true,
            call: response,
            error: '',
        };
    }
    // Appointment operations
    async listAppointments(params) {
        const queryParams = new URLSearchParams();
        if (params.personId)
            queryParams.set('personId', params.personId.toString());
        if (params.limit)
            queryParams.set('limit', params.limit.toString());
        if (params.offset)
            queryParams.set('offset', params.offset.toString());
        const response = (await this.makeFUBApiRequest(`/appointments?${queryParams.toString()}`));
        return {
            operation: 'list_appointments',
            success: true,
            appointments: response.appointments,
            _metadata: response._metadata,
            error: '',
        };
    }
    async createAppointment(params) {
        const { operation: _, credentials: __, ...appointmentData } = params;
        const response = (await this.makeFUBApiRequest('/appointments', 'POST', appointmentData));
        return {
            operation: 'create_appointment',
            success: true,
            appointment: response,
            error: '',
        };
    }
    // Webhook operations
    async listWebhooks(params) {
        void params;
        const response = (await this.makeFUBApiRequest('/webhooks'));
        return {
            operation: 'list_webhooks',
            success: true,
            webhooks: response.webhooks,
            error: '',
        };
    }
    async getWebhook(params) {
        const response = (await this.makeFUBApiRequest(`/webhooks/${params.webhook_id}`));
        return {
            operation: 'get_webhook',
            success: true,
            webhook: response,
            error: '',
        };
    }
    async createWebhook(params) {
        const { operation: _, credentials: __, ...webhookData } = params;
        const response = (await this.makeFUBApiRequest('/webhooks', 'POST', webhookData));
        return {
            operation: 'create_webhook',
            success: true,
            webhook: response,
            error: '',
        };
    }
    async updateWebhook(params) {
        const { operation: _, credentials: __, webhook_id, ...webhookData } = params;
        const response = (await this.makeFUBApiRequest(`/webhooks/${webhook_id}`, 'PUT', webhookData));
        return {
            operation: 'update_webhook',
            success: true,
            webhook: response,
            error: '',
        };
    }
    async deleteWebhook(params) {
        await this.makeFUBApiRequest(`/webhooks/${params.webhook_id}`, 'DELETE');
        return {
            operation: 'delete_webhook',
            success: true,
            deleted_id: params.webhook_id,
            error: '',
        };
    }
    chooseCredential() {
        const { credentials } = this.params;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('No Follow Up Boss credentials provided');
        }
        return credentials[CredentialType.FUB_CRED];
    }
}
//# sourceMappingURL=followupboss.js.map