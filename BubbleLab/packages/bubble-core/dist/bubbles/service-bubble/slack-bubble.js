import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
/**
 * Slack Bubble - Team Communication Service Bubble Implementation
 *
 * Full production implementation with 10 operations:
 * 1. sendMessage - Send a message to a channel
 * 2. sendDM - Send a direct message to a user
 * 3. updateMessage - Update an existing message
 * 4. deleteMessage - Delete a message
 * 5. addReaction - Add a reaction to a message
 * 6. removeReaction - Remove a reaction from a message
 * 7. getChannelInfo - Get information about a channel
 * 8. listChannels - List all channels in the workspace
 * 9. getUserInfo - Get information about a user
 * 10. uploadFile - Upload a file to a channel
 */
// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================
const SendMessageParamsSchema = z.object({
    operation: z.literal('sendMessage'),
    channel: z.string().min(1, 'Channel ID or name is required'),
    text: z.string().min(1, 'Message text is required'),
    threadTs: z.string().optional().describe('Thread parent timestamp to reply in thread'),
    blocks: z.array(z.unknown()).optional().describe('Array of Slack block kit blocks'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const SendDMParamsSchema = z.object({
    operation: z.literal('sendDM'),
    userId: z.string().min(1, 'User ID is required'),
    text: z.string().min(1, 'Message text is required'),
    blocks: z.array(z.unknown()).optional().describe('Array of Slack block kit blocks'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const UpdateMessageParamsSchema = z.object({
    operation: z.literal('updateMessage'),
    channel: z.string().min(1, 'Channel ID is required'),
    timestamp: z.string().min(1, 'Message timestamp is required'),
    text: z.string().min(1, 'Updated message text is required'),
    blocks: z.array(z.unknown()).optional().describe('Array of Slack block kit blocks'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const DeleteMessageParamsSchema = z.object({
    operation: z.literal('deleteMessage'),
    channel: z.string().min(1, 'Channel ID is required'),
    timestamp: z.string().min(1, 'Message timestamp is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const AddReactionParamsSchema = z.object({
    operation: z.literal('addReaction'),
    channel: z.string().min(1, 'Channel ID is required'),
    timestamp: z.string().min(1, 'Message timestamp is required'),
    reaction: z.string().min(1, 'Reaction emoji name (without colons)'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const RemoveReactionParamsSchema = z.object({
    operation: z.literal('removeReaction'),
    channel: z.string().min(1, 'Channel ID is required'),
    timestamp: z.string().min(1, 'Message timestamp is required'),
    reaction: z.string().min(1, 'Reaction emoji name (without colons)'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const GetChannelInfoParamsSchema = z.object({
    operation: z.literal('getChannelInfo'),
    channelId: z.string().min(1, 'Channel ID is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const ListChannelsParamsSchema = z.object({
    operation: z.literal('listChannels'),
    limit: z.number().int().positive().optional().default(100),
    types: z.array(z.enum(['public_channel', 'private_channel', 'mpim', 'im'])).optional(),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const GetUserInfoParamsSchema = z.object({
    operation: z.literal('getUserInfo'),
    userId: z.string().min(1, 'User ID is required'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
const UploadFileParamsSchema = z.object({
    operation: z.literal('uploadFile'),
    channel: z.string().min(1, 'Channel ID is required'),
    fileContent: z.union([z.string(), z.instanceof(Buffer)]).describe('File content'),
    filename: z.string().min(1, 'Filename is required'),
    filetype: z.string().optional().describe('File type (e.g., txt, png, pdf)'),
    title: z.string().optional().describe('Title of the file'),
    initialComment: z.string().optional().describe('Initial comment to add'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});
// Union of all parameter schemas
const SlackBubbleParamsSchema = z.discriminatedUnion('operation', [
    SendMessageParamsSchema,
    SendDMParamsSchema,
    UpdateMessageParamsSchema,
    DeleteMessageParamsSchema,
    AddReactionParamsSchema,
    RemoveReactionParamsSchema,
    GetChannelInfoParamsSchema,
    ListChannelsParamsSchema,
    GetUserInfoParamsSchema,
    UploadFileParamsSchema,
]);
// Result schema
const SlackBubbleResultSchema = z.object({
    success: z.boolean(),
    data: z.unknown().describe('Operation result data'),
    error: z.string(),
    meta: z.object({
        operation: z.string(),
        channel: z.string().optional(),
        timestamp: z.string().optional(),
    }),
});
// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================
export class SlackBubble extends ServiceBubble {
    static service = 'slack';
    static authType = 'oauth';
    static bubbleName = 'slack';
    static type = 'service';
    static schema = SlackBubbleParamsSchema;
    static resultSchema = SlackBubbleResultSchema;
    static shortDescription = 'Team communication and collaboration platform';
    static longDescription = `
    Slack Bubble for team communication and messaging.

    Features:
    - Send messages to channels and direct messages
    - Rich formatting with Block Kit
    - Threaded conversations
    - File sharing and uploads
    - Message reactions
    - Channel and user information retrieval
    - Real-time webhooks support

    Use cases:
    - Team notifications and alerts
    - Automated status updates
    - Incident management
    - Approval workflows
    - Daily standups and reports
    - Integration notifications
  `;
    static alias = 'chat';
    botToken = null;
    baseUrl = 'https://slack.com/api';
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    getCredentialType() {
        return CredentialType.SLACK_CRED;
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            throw new Error('Slack credentials are required');
        }
        return credentials[CredentialType.SLACK_CRED];
    }
    async testCredential() {
        try {
            const token = this.getToken();
            const response = await fetch(`${this.baseUrl}/auth.test`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
            });
            const data = await response.json();
            return data.ok === true;
        }
        catch (error) {
            console.error('[Slack] Credential test failed:', error);
            return false;
        }
    }
    getToken() {
        if (!this.botToken) {
            const credential = this.chooseCredential();
            if (!credential) {
                throw new Error('Slack credentials not found');
            }
            // Parse credential (expected format: JSON string with botToken)
            let config;
            try {
                config = typeof credential === 'string' ? JSON.parse(credential) : credential;
            }
            catch {
                throw new Error('Invalid Slack credentials format. Expected JSON string.');
            }
            if (!config.botToken && !config.accessToken) {
                throw new Error('Slack bot token is required in credentials');
            }
            this.botToken = config.botToken || config.accessToken;
            console.log('[Slack] Token initialized successfully');
        }
        if (!this.botToken) {
            throw new Error('Slack bot token initialization failed');
        }
        return this.botToken;
    }
    async performAction(context) {
        void context;
        try {
            const operation = this.params.operation;
            let result;
            console.log(`[Slack] Executing operation: ${operation}`);
            switch (operation) {
                case 'sendMessage':
                    result = await this.sendMessage();
                    break;
                case 'sendDM':
                    result = await this.sendDM();
                    break;
                case 'updateMessage':
                    result = await this.updateMessage();
                    break;
                case 'deleteMessage':
                    result = await this.deleteMessage();
                    break;
                case 'addReaction':
                    result = await this.addReaction();
                    break;
                case 'removeReaction':
                    result = await this.removeReaction();
                    break;
                case 'getChannelInfo':
                    result = await this.getChannelInfo();
                    break;
                case 'listChannels':
                    result = await this.listChannels();
                    break;
                case 'getUserInfo':
                    result = await this.getUserInfo();
                    break;
                case 'uploadFile':
                    result = await this.uploadFile();
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
                    channel: this.extractChannel(),
                    timestamp: result.ts,
                },
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`[Slack] Operation failed:`, errorMessage);
            return {
                success: false,
                data: null,
                error: errorMessage,
                meta: {
                    operation: this.params.operation,
                    channel: this.extractChannel(),
                },
            };
        }
    }
    async makeRequest(endpoint, body) {
        const token = this.getToken();
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });
        const data = await response.json();
        if (!data.ok) {
            throw new Error(data.error || 'Slack API request failed');
        }
        return data;
    }
    async sendMessage() {
        const params = this.params;
        const body = {
            channel: params.channel,
            text: params.text,
        };
        if (params.threadTs) {
            body.thread_ts = params.threadTs;
        }
        if (params.blocks) {
            body.blocks = params.blocks;
        }
        const result = await this.makeRequest('/chat.postMessage', body);
        console.log(`[Slack] Message sent to channel ${params.channel}: ${result.ts}`);
        return {
            channel: params.channel,
            timestamp: result.ts,
            messageTs: result.ts,
            status: 'sent',
        };
    }
    async sendDM() {
        const params = this.params;
        const body = {
            channel: params.userId,
            text: params.text,
        };
        if (params.blocks) {
            body.blocks = params.blocks;
        }
        const result = await this.makeRequest('/chat.postMessage', body);
        console.log(`[Slack] DM sent to user ${params.userId}: ${result.ts}`);
        return {
            userId: params.userId,
            timestamp: result.ts,
            messageTs: result.ts,
            status: 'sent',
        };
    }
    async updateMessage() {
        const params = this.params;
        const body = {
            channel: params.channel,
            ts: params.timestamp,
            text: params.text,
        };
        if (params.blocks) {
            body.blocks = params.blocks;
        }
        const result = await this.makeRequest('/chat.update', body);
        console.log(`[Slack] Message updated: ${params.timestamp}`);
        return {
            channel: params.channel,
            timestamp: params.timestamp,
            messageTs: result.ts,
            status: 'updated',
        };
    }
    async deleteMessage() {
        const params = this.params;
        const body = {
            channel: params.channel,
            ts: params.timestamp,
        };
        await this.makeRequest('/chat.delete', body);
        console.log(`[Slack] Message deleted: ${params.timestamp}`);
        return {
            channel: params.channel,
            timestamp: params.timestamp,
            status: 'deleted',
        };
    }
    async addReaction() {
        const params = this.params;
        const body = {
            channel: params.channel,
            timestamp: params.timestamp,
            name: params.reaction,
        };
        await this.makeRequest('/reactions.add', body);
        console.log(`[Slack] Reaction added: :${params.reaction}: to ${params.timestamp}`);
        return {
            channel: params.channel,
            timestamp: params.timestamp,
            reaction: params.reaction,
            status: 'added',
        };
    }
    async removeReaction() {
        const params = this.params;
        const body = {
            channel: params.channel,
            timestamp: params.timestamp,
            name: params.reaction,
        };
        await this.makeRequest('/reactions.remove', body);
        console.log(`[Slack] Reaction removed: :${params.reaction}: from ${params.timestamp}`);
        return {
            channel: params.channel,
            timestamp: params.timestamp,
            reaction: params.reaction,
            status: 'removed',
        };
    }
    async getChannelInfo() {
        const params = this.params;
        const body = {
            channel: params.channelId,
        };
        const result = await this.makeRequest('/conversations.info', body);
        console.log(`[Slack] Retrieved channel info: ${params.channelId}`);
        return {
            channel: result.channel,
            info: {
                id: result.channel.id,
                name: result.channel.name,
                topic: result.channel.topic,
                purpose: result.channel.purpose,
                members: result.channel.num_members,
            },
        };
    }
    async listChannels() {
        const params = this.params;
        const body = {
            limit: params.limit,
        };
        if (params.types) {
            body.types = params.types.join(',');
        }
        const result = await this.makeRequest('/conversations.list', body);
        console.log(`[Slack] Listed ${result.channels.length} channels`);
        return {
            channels: result.channels.map((ch) => ({
                id: ch.id,
                name: ch.name,
                topic: ch.topic?.value || '',
                isPrivate: ch.is_private,
                memberCount: ch.num_members,
            })),
            count: result.channels.length,
        };
    }
    async getUserInfo() {
        const params = this.params;
        const body = {
            user: params.userId,
        };
        const result = await this.makeRequest('/users.info', body);
        console.log(`[Slack] Retrieved user info: ${params.userId}`);
        return {
            user: {
                id: result.user.id,
                name: result.user.name,
                displayName: result.user.real_name,
                email: result.user.profile?.email,
                title: result.user.profile?.title,
                timezone: result.user.tz,
            },
        };
    }
    async uploadFile() {
        const params = this.params;
        const token = this.getToken();
        // Prepare form data
        const formData = new FormData();
        formData.append('channels', params.channel);
        formData.append('filename', params.filename);
        if (typeof params.fileContent === 'string') {
            formData.append('file', new Blob([params.fileContent]), params.filename);
        }
        else {
            formData.append('file', new Blob([params.fileContent]), params.filename);
        }
        if (params.filetype) {
            formData.append('filetype', params.filetype);
        }
        if (params.title) {
            formData.append('title', params.title);
        }
        if (params.initialComment) {
            formData.append('initial_comment', params.initialComment);
        }
        const response = await fetch(`${this.baseUrl}/files.upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
            },
            body: formData,
        });
        const data = await response.json();
        if (!data.ok) {
            throw new Error(data.error || 'File upload failed');
        }
        console.log(`[Slack] File uploaded to ${params.channel}: ${data.file?.id}`);
        return {
            channel: params.channel,
            fileId: data.file?.id,
            fileUrl: data.file?.url_private,
            status: 'uploaded',
        };
    }
    extractChannel() {
        const params = this.params;
        return params.channel || params.channelId;
    }
}
//# sourceMappingURL=slack-bubble.js.map