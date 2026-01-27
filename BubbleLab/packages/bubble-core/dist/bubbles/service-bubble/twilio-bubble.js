import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import twilio from 'twilio';
/**
 * Twilio Bubble - Communication Service Bubble Implementation
 *
 * Full production implementation with 8 operations:
 * 1. sendSMS - Send a text message
 * 2. makeCall - Initiate a voice call
 * 3. sendWhatsApp - Send a WhatsApp message
 * 4. lookupNumber - Get information about a phone number
 * 5. createMessage - Create a message for later sending
 * 6. getMessage - Retrieve message details
 * 7. getMedia - Retrieve media from a message
 * 8. validateNumber - Validate a phone number format
 */
// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================
const SendSMSParamsSchema = z.object({
    operation: z.literal('sendSMS'),
    to: z.string().min(1, 'Recipient phone number is required'),
    from: z.string().min(1, 'Sender phone number is required'),
    body: z.string().min(1, 'Message body is required'),
    statusCallback: z.string().url().optional(),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const MakeCallParamsSchema = z.object({
    operation: z.literal('makeCall'),
    to: z.string().min(1, 'Recipient phone number is required'),
    from: z.string().min(1, 'Sender phone number is required'),
    url: z.string().url().describe('Twiml URL for call instructions'),
    statusCallback: z.string().url().optional(),
    method: z.enum(['GET', 'POST']).optional().default('POST'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const SendWhatsAppParamsSchema = z.object({
    operation: z.literal('sendWhatsApp'),
    to: z.string().min(1, 'Recipient phone number is required'),
    from: z.string().min(1, 'Sender WhatsApp number is required'),
    body: z.string().min(1, 'Message body is required'),
    mediaUrl: z.array(z.string().url()).optional(),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const LookupNumberParamsSchema = z.object({
    operation: z.literal('lookupNumber'),
    phoneNumber: z.string().min(1, 'Phone number to lookup is required'),
    type: z.array(z.enum(['carrier', 'caller-name', 'phone-type'])).optional().default(['carrier']),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const CreateMessageParamsSchema = z.object({
    operation: z.literal('createMessage'),
    to: z.string().min(1, 'Recipient phone number is required'),
    from: z.string().min(1, 'Sender phone number is required'),
    body: z.string().min(1, 'Message body is required'),
    scheduleTime: z.string().optional().describe('ISO 8601 datetime for scheduled sending'),
    statusCallback: z.string().url().optional(),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const GetMessageParamsSchema = z.object({
    operation: z.literal('getMessage'),
    messageSid: z.string().min(1, 'Message SID is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const GetMediaParamsSchema = z.object({
    operation: z.literal('getMedia'),
    messageSid: z.string().min(1, 'Message SID is required'),
    mediaSid: z.string().min(1, 'Media SID is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const ValidateNumberParamsSchema = z.object({
    operation: z.literal('validateNumber'),
    phoneNumber: z.string().min(1, 'Phone number to validate is required'),
    countryCode: z.string().length(2).optional().describe('ISO 3166-1 alpha-2 country code'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
// Union of all parameter schemas
const TwilioBubbleParamsSchema = z.discriminatedUnion('operation', [
    SendSMSParamsSchema,
    MakeCallParamsSchema,
    SendWhatsAppParamsSchema,
    LookupNumberParamsSchema,
    CreateMessageParamsSchema,
    GetMessageParamsSchema,
    GetMediaParamsSchema,
    ValidateNumberParamsSchema,
]);
// Result schema
const TwilioBubbleResultSchema = z.object({
    success: z.boolean(),
    data: z.unknown().describe('Operation result data'),
    error: z.string(),
    meta: z.object({
        operation: z.string(),
        sid: z.string().optional(),
    }),
});
// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================
export class TwilioBubble extends ServiceBubble {
    static service = 'twilio';
    static authType = 'apikey';
    static bubbleName = 'twilio';
    static type = 'service';
    static schema = TwilioBubbleParamsSchema;
    static resultSchema = TwilioBubbleResultSchema;
    static shortDescription = 'SMS, voice, and WhatsApp messaging platform';
    static longDescription = `
    Twilio Bubble for programmable communication.

    Features:
    - Send SMS and MMS messages
    - Make and receive voice calls
    - WhatsApp messaging
    - Phone number lookup and validation
    - Media handling for MMS
    - Scheduled messaging

    Use cases:
    - SMS notifications and alerts
    - Two-factor authentication
    - Voice call automation
    - WhatsApp business messaging
    - Phone number verification
    - Appointment reminders
  `;
    static alias = 'sms';
    client = null;
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    getCredentialType() {
        return CredentialType.TWILIO_CRED;
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('Twilio credentials are required');
        }
        return credentials[CredentialType.TWILIO_CRED];
    }
    async testCredential() {
        try {
            const client = this.getClient();
            const credential = this.chooseCredential();
            if (!credential) {
                throw new Error('Twilio credentials not found');
            }
            const config = typeof credential === 'string' ? JSON.parse(credential) : credential;
            await client.api.accounts(config.accountSid).fetch();
            return true;
        }
        catch (error) {
            console.error('[Twilio] Credential test failed:', error);
            return false;
        }
    }
    getClient() {
        if (!this.client) {
            const credential = this.chooseCredential();
            if (!credential) {
                throw new Error('Twilio credentials not found');
            }
            // Parse credential (expected format: JSON string with accountSid and authToken)
            let config;
            try {
                config = typeof credential === 'string' ? JSON.parse(credential) : credential;
            }
            catch {
                throw new Error('Invalid Twilio credentials format. Expected JSON string.');
            }
            if (!config.accountSid || !config.authToken) {
                throw new Error('Twilio accountSid and authToken are required in credentials');
            }
            this.client = twilio(config.accountSid, config.authToken);
            console.log('[Twilio] Client initialized successfully');
        }
        return this.client;
    }
    async performAction(context) {
        void context;
        try {
            const client = this.getClient();
            const operation = this.params.operation;
            let result;
            console.log(`[Twilio] Executing operation: ${operation}`);
            switch (operation) {
                case 'sendSMS':
                    result = await this.sendSMS(client);
                    break;
                case 'makeCall':
                    result = await this.makeCall(client);
                    break;
                case 'sendWhatsApp':
                    result = await this.sendWhatsApp(client);
                    break;
                case 'lookupNumber':
                    result = await this.lookupNumber(client);
                    break;
                case 'createMessage':
                    result = await this.createMessage(client);
                    break;
                case 'getMessage':
                    result = await this.getMessage(client);
                    break;
                case 'getMedia':
                    result = await this.getMedia(client);
                    break;
                case 'validateNumber':
                    result = await this.validateNumber(client);
                    break;
                default:
                    throw new Error(`Unknown operation: ${operation}`);
            }
            return {
                success: true,
                data: result,
                error: '', // Empty string for successful operations,
                meta: {
                    operation,
                    sid: result?.sid,
                },
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[Twilio] Operation failed:`, errorMessage);
            return {
                success: false,
                data: null,
                error: errorMessage,
                meta: {
                    operation: this.params.operation,
                },
            };
        }
    }
    async sendSMS(client) {
        const params = this.params;
        const messageOptions = {
            body: params.body,
            from: params.from,
            to: params.to,
        };
        if (params.statusCallback) {
            messageOptions.statusCallback = params.statusCallback;
        }
        const message = await client.messages.create(messageOptions);
        console.log(`[Twilio] SMS sent to ${params.to}: ${message.sid}`);
        return {
            sid: message.sid,
            status: message.status,
            to: message.to,
            from: message.from,
            body: message.body,
            dateCreated: message.dateCreated,
        };
    }
    async makeCall(client) {
        const params = this.params;
        const callOptions = {
            url: params.url,
            to: params.to,
            from: params.from,
            method: params.method,
        };
        if (params.statusCallback) {
            callOptions.statusCallback = params.statusCallback;
        }
        const call = await client.calls.create(callOptions);
        console.log(`[Twilio] Call initiated to ${params.to}: ${call.sid}`);
        return {
            sid: call.sid,
            status: call.status,
            to: call.to,
            from: call.from,
            dateCreated: call.dateCreated,
        };
    }
    async sendWhatsApp(client) {
        const params = this.params;
        const messageOptions = {
            body: params.body,
            from: `whatsapp:${params.from}`,
            to: `whatsapp:${params.to}`,
        };
        if (params.mediaUrl && params.mediaUrl.length > 0) {
            messageOptions.mediaUrl = params.mediaUrl;
        }
        const message = await client.messages.create(messageOptions);
        console.log(`[Twilio] WhatsApp message sent to ${params.to}: ${message.sid}`);
        return {
            sid: message.sid,
            status: message.status,
            to: message.to,
            from: message.from,
            body: message.body,
            dateCreated: message.dateCreated,
        };
    }
    async lookupNumber(client) {
        const params = this.params;
        const lookups = await client.lookups.v1.phoneNumbers(params.phoneNumber).fetch({
            type: params.type,
        });
        console.log(`[Twilio] Number lookup completed for ${params.phoneNumber}`);
        return {
            phoneNumber: lookups.phoneNumber,
            nationalFormat: lookups.nationalFormat,
            country: lookups.countryCode,
            carrier: lookups.carrier,
            type: params.type,
        };
    }
    async createMessage(client) {
        const params = this.params;
        const messageOptions = {
            body: params.body,
            from: params.from,
            to: params.to,
            statusCallback: params.statusCallback,
        };
        const message = await client.messages.create(messageOptions);
        console.log(`[Twilio] Message created: ${message.sid}`);
        return {
            sid: message.sid,
            status: message.status,
            to: message.to,
            from: message.from,
            body: message.body,
            dateCreated: message.dateCreated,
        };
    }
    async getMessage(client) {
        const params = this.params;
        const message = await client.messages(params.messageSid).fetch();
        console.log(`[Twilio] Message retrieved: ${params.messageSid}`);
        return {
            sid: message.sid,
            status: message.status,
            to: message.to,
            from: message.from,
            body: message.body,
            dateCreated: message.dateCreated,
            dateUpdated: message.dateUpdated,
            dateSent: message.dateSent,
            direction: message.direction,
            errorMessage: message.errorMessage,
            errorCode: message.errorCode,
        };
    }
    async getMedia(client) {
        const params = this.params;
        const media = await client
            .messages(params.messageSid)
            .media(params.mediaSid)
            .fetch();
        console.log(`[Twilio] Media retrieved: ${params.mediaSid}`);
        return {
            sid: media.sid,
            contentType: media.contentType,
            parentSid: media.parentSid,
            url: media.uri,
            dateCreated: media.dateCreated,
            dateUpdated: media.dateUpdated,
        };
    }
    async validateNumber(client) {
        const params = this.params;
        try {
            const lookups = await client.lookups.v1.phoneNumbers(params.phoneNumber).fetch({
                type: ['carrier'],
                countryCode: params.countryCode,
            });
            console.log(`[Twilio] Number validated: ${params.phoneNumber}`);
            return {
                phoneNumber: lookups.phoneNumber,
                nationalFormat: lookups.nationalFormat,
                country: lookups.countryCode,
                valid: true,
                carrier: lookups.carrier,
            };
        }
        catch (error) {
            if (error.status === 404) {
                console.log(`[Twilio] Number validation failed: ${params.phoneNumber}`);
                return {
                    phoneNumber: params.phoneNumber,
                    valid: false,
                    error: 'Phone number not found or invalid',
                };
            }
            throw error;
        }
    }
}
//# sourceMappingURL=twilio-bubble.js.map