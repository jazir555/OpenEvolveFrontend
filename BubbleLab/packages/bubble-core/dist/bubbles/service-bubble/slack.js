import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { validateFilePath, ValidationError as CommonValidationError } from '../common/validators.js';
import { AuthenticationError, ExternalServiceError } from '../common/error-handlers.js';
// Slack API base URL
const SLACK_API_BASE = 'https://slack.com/api';
// Slack operation types are defined inline in the discriminated union below
// Define Slack channel types
const ChannelTypes = z
    .enum(['public_channel', 'private_channel', 'mpim', 'im'])
    .describe('Types of Slack channels: public_channel, private_channel, mpim (multi-person direct message), im (direct message)');
// Define message attachment schema
const MessageAttachmentSchema = z.object({
    color: z
        .string()
        .optional()
        .describe('Color bar accent (hex color or good/warning/danger)'),
    pretext: z
        .string()
        .optional()
        .describe('Text that appears before the main attachment content'),
    author_name: z
        .string()
        .optional()
        .describe('Author name displayed at the top'),
    author_link: z
        .string()
        .url()
        .optional()
        .describe('URL to link the author name'),
    author_icon: z.string().url().optional().describe('Author icon image URL'),
    title: z.string().optional().describe('Attachment title text'),
    title_link: z.string().url().optional().describe('URL to link the title'),
    text: z.string().optional().describe('Main attachment text content'),
    fields: z
        .array(z.object({
        title: z.string().describe('Field title'),
        value: z.string().describe('Field value'),
        short: z
            .boolean()
            .optional()
            .describe('Whether field should be displayed side-by-side'),
    }))
        .optional()
        .describe('Array of field objects for structured data'),
    image_url: z.string().url().optional().describe('URL of image to display'),
    thumb_url: z.string().url().optional().describe('URL of thumbnail image'),
    footer: z.string().optional().describe('Footer text'),
    footer_icon: z.string().url().optional().describe('Footer icon URL'),
    ts: z.number().optional().describe('Timestamp for the attachment'),
});
// Define block kit elements (simplified)
const BlockElementSchema = z
    .object({
    type: z
        .string()
        .describe('Block element type (section, divider, button, etc.)'),
    text: z
        .object({
        type: z.enum(['plain_text', 'mrkdwn']).describe('Text formatting type'),
        text: z.string().describe('The actual text content'),
        emoji: z.boolean().optional(),
        verbatim: z.boolean().optional(),
    })
        .optional()
        .describe('Text object for the block element'),
    // Add elements field for context blocks
    elements: z
        .array(z.object({
        type: z
            .enum(['plain_text', 'mrkdwn', 'image'])
            .describe('Element type'),
        text: z.string().optional().describe('Text content'),
        image_url: z
            .string()
            .optional()
            .describe('Image URL for image elements'),
        alt_text: z
            .string()
            .optional()
            .describe('Alt text for image elements'),
        emoji: z.boolean().optional(),
        verbatim: z.boolean().optional(),
    }))
        .optional()
        .describe('Elements array for context blocks'),
})
    .passthrough() // Allow additional properties for different block types
    .describe('Block Kit element for rich message formatting');
// Define the parameters schema for different Slack operations
const SlackParamsSchema = z.discriminatedUnion('operation', [
    // Send message operation
    z.object({
        operation: z
            .literal('send_message')
            .describe('Send a message to a Slack channel or DM'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890), channel name (e.g., general or #general), or user ID for DM'),
        text: z
            .string()
            .min(1, 'Message text is required')
            .describe('Message text content'),
        username: z
            .string()
            .optional()
            .describe('Override bot username for this message'),
        icon_emoji: z
            .string()
            .optional()
            .describe('Override bot icon with emoji (e.g., :robot_face:)'),
        icon_url: z
            .string()
            .url()
            .optional()
            .describe('Override bot icon with custom image URL'),
        attachments: z
            .array(MessageAttachmentSchema)
            .optional()
            .describe('Legacy message attachments'),
        blocks: z
            .array(BlockElementSchema)
            .optional()
            .describe('Block Kit structured message blocks'),
        thread_ts: z
            .string()
            .optional()
            .describe('Timestamp of parent message to reply in thread'),
        reply_broadcast: z
            .boolean()
            .optional()
            .default(false)
            .describe('Broadcast thread reply to channel'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
        unfurl_links: z
            .boolean()
            .optional()
            .default(true)
            .describe('Enable automatic link unfurling'),
        unfurl_media: z
            .boolean()
            .optional()
            .default(true)
            .describe('Enable automatic media unfurling'),
    }),
    // List channels operation
    z.object({
        operation: z
            .literal('list_channels')
            .describe('List all channels in the Slack workspace'),
        types: z
            .array(ChannelTypes)
            .optional()
            .default(['public_channel', 'private_channel'])
            .describe('Types of channels to include in results'),
        exclude_archived: z
            .boolean()
            .optional()
            .default(true)
            .describe('Exclude archived channels from results'),
        limit: z
            .number()
            .min(1)
            .max(1000)
            .optional()
            .default(50)
            .describe('Maximum number of channels to return (1-1000)'),
        cursor: z
            .string()
            .optional()
            .describe('Cursor for pagination to get next set of results'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get channel info operation
    z.object({
        operation: z
            .literal('get_channel_info')
            .describe('Get detailed information about a specific channel'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890) or channel name (e.g., general or #general)'),
        include_locale: z
            .boolean()
            .optional()
            .default(false)
            .describe('Include locale information in the response'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get user info operation
    z.object({
        operation: z
            .literal('get_user_info')
            .describe('Get detailed information about a specific user'),
        user: z
            .string()
            .min(1, 'User ID is required')
            .describe('User ID to get information about'),
        include_locale: z
            .boolean()
            .optional()
            .default(false)
            .describe('Include locale information in the response'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // List users operation
    z.object({
        operation: z
            .literal('list_users')
            .describe('List all users in the Slack workspace'),
        limit: z
            .number()
            .min(1)
            .max(1000)
            .optional()
            .default(50)
            .describe('Maximum number of users to return (1-1000)'),
        cursor: z
            .string()
            .optional()
            .describe('Cursor for pagination to get next set of results'),
        include_locale: z
            .boolean()
            .optional()
            .default(false)
            .describe('Include locale information in the response'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get conversation history operation
    z.object({
        operation: z
            .literal('get_conversation_history')
            .describe('Retrieve message history from a channel or direct message'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890) or channel name (e.g., general or #general)'),
        latest: z
            .string()
            .optional()
            .describe('End of time range of messages to include (timestamp)'),
        oldest: z
            .string()
            .optional()
            .describe('Start of time range of messages to include (timestamp)'),
        inclusive: z
            .boolean()
            .optional()
            .default(false)
            .describe('Include messages with latest or oldest timestamps in results'),
        limit: z
            .number()
            .min(1)
            .max(1000)
            .optional()
            .default(20)
            .describe('Maximum number of messages to return (1-1000)'),
        cursor: z
            .string()
            .optional()
            .describe('Cursor for pagination to get next set of results'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Get thread replies operation
    z.object({
        operation: z
            .literal('get_thread_replies')
            .describe('Retrieve all replies to a thread in a channel'),
        channel: z
            .string()
            .min(1, 'Channel ID is required')
            .describe('Channel ID where the thread exists'),
        ts: z
            .string()
            .min(1, 'Thread timestamp is required')
            .describe('Timestamp of the parent message to get replies for'),
        latest: z
            .string()
            .optional()
            .describe('End of time range of messages to include (timestamp)'),
        oldest: z
            .string()
            .optional()
            .describe('Start of time range of messages to include (timestamp)'),
        inclusive: z
            .boolean()
            .optional()
            .default(false)
            .describe('Include messages with latest or oldest timestamps in results'),
        limit: z
            .number()
            .min(1)
            .max(1000)
            .optional()
            .default(100)
            .describe('Maximum number of messages to return (1-1000)'),
        cursor: z
            .string()
            .optional()
            .describe('Cursor for pagination to get next set of results'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Update message operation
    z.object({
        operation: z
            .literal('update_message')
            .describe('Update an existing message in a channel'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890) or channel name (e.g., general or #general) where the message is located'),
        ts: z
            .string()
            .min(1, 'Message timestamp is required')
            .describe('Timestamp of the message to update'),
        text: z.string().optional().describe('New text content for the message'),
        attachments: z
            .array(MessageAttachmentSchema)
            .optional()
            .describe('New legacy message attachments'),
        blocks: z
            .array(BlockElementSchema)
            .optional()
            .describe('New Block Kit structured message blocks'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Delete message operation
    z.object({
        operation: z
            .literal('delete_message')
            .describe('Delete a message from a channel'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890) or channel name (e.g., general or #general) where the message is located'),
        ts: z
            .string()
            .min(1, 'Message timestamp is required')
            .describe('Timestamp of the message to delete'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Add reaction operation
    z.object({
        operation: z
            .literal('add_reaction')
            .describe('Add an emoji reaction to a message'),
        name: z
            .string()
            .min(1, 'Emoji name is required')
            .describe('Emoji name without colons (e.g., thumbsup, heart)'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890) or channel name (e.g., general or #general) where the message is located'),
        timestamp: z
            .string()
            .min(1, 'Message timestamp is required')
            .describe('Timestamp of the message to react to'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Remove reaction operation
    z.object({
        operation: z
            .literal('remove_reaction')
            .describe('Remove an emoji reaction from a message'),
        name: z
            .string()
            .min(1, 'Emoji name is required')
            .describe('Emoji name without colons (e.g., thumbsup, heart)'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890) or channel name (e.g., general or #general) where the message is located'),
        timestamp: z
            .string()
            .min(1, 'Message timestamp is required')
            .describe('Timestamp of the message to remove reaction from'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Join channel operation
    z.object({
        operation: z
            .literal('join_channel')
            .describe('Join a public Slack channel'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890) or channel name (e.g., general or #general) to join'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
    // Upload file operation
    z.object({
        operation: z
            .literal('upload_file')
            .describe('Upload a file to a Slack channel'),
        channel: z
            .string()
            .min(1, 'Channel ID or name is required')
            .describe('Channel ID (e.g., C1234567890), channel name (e.g., general or #general), or user ID for DM'),
        file_path: z
            .string()
            .min(1, 'File path is required')
            .max(500, 'File path too long (max 500 characters)')
            .refine((path) => {
            // SECURITY: Prevent path traversal attacks
            const normalizedPath = path.replace(/\\/g, '/');
            // Block path traversal attempts
            if (normalizedPath.includes('..')) {
                return false;
            }
            // Block absolute paths (only allow relative paths from working directory)
            if (normalizedPath.startsWith('/')) {
                return false;
            }
            // Block Windows drive letters
            if (/^[a-zA-Z]:/.test(normalizedPath)) {
                return false;
            }
            // Only allow safe characters in paths
            if (!/^[\w\-./ ]+$/.test(normalizedPath)) {
                return false;
            }
            return true;
        }, 'File path contains invalid characters or path traversal sequences')
            .describe('Local file path to upload (relative paths only, no ../ or absolute paths)'),
        filename: z
            .string()
            .optional()
            .describe('Override filename for the upload'),
        title: z.string().optional().describe('Title for the file'),
        initial_comment: z
            .string()
            .optional()
            .describe('Initial comment to post with the file'),
        thread_ts: z
            .string()
            .optional()
            .describe('Timestamp of parent message to upload file in thread'),
        credentials: z
            .record(z.nativeEnum(CredentialType), z.string())
            .optional()
            .describe('Object mapping credential types to values (injected at runtime)'),
    }),
]);
// Define Slack API response schemas for type safety
const SlackChannelSchema = z
    .object({
    id: z.string().describe('Unique channel identifier'),
    name: z.string().describe('Channel name without # prefix'),
    is_channel: z
        .boolean()
        .optional()
        .describe('True if this is a public channel'),
    is_group: z
        .boolean()
        .optional()
        .describe('True if this is a private channel'),
    is_im: z.boolean().optional().describe('True if this is a direct message'),
    is_mpim: z
        .boolean()
        .optional()
        .describe('True if this is a multi-person direct message'),
    is_private: z
        .boolean()
        .optional()
        .describe('True if this is a private channel'),
    created: z.number().describe('Unix timestamp when channel was created'),
    is_archived: z.boolean().describe('True if channel is archived'),
    is_general: z
        .boolean()
        .optional()
        .describe('True if this is the #general channel'),
    unlinked: z
        .number()
        .optional()
        .describe('Unix timestamp when channel was unlinked'),
    name_normalized: z.string().optional().describe('Normalized channel name'),
    is_shared: z
        .boolean()
        .optional()
        .describe('True if channel is shared with other workspaces'),
    is_ext_shared: z
        .boolean()
        .optional()
        .describe('True if channel is shared externally'),
    is_org_shared: z
        .boolean()
        .optional()
        .describe('True if channel is shared across organization'),
    shared_team_ids: z
        .array(z.string())
        .optional()
        .describe('IDs of teams this channel is shared with'),
    pending_shared: z
        .array(z.string())
        .optional()
        .describe('Pending shared connections'),
    pending_connected_team_ids: z
        .array(z.string())
        .optional()
        .describe('Pending team connection IDs'),
    is_pending_ext_shared: z
        .boolean()
        .optional()
        .describe('True if external sharing is pending'),
    is_member: z
        .boolean()
        .optional()
        .describe('True if the bot is a member of this channel'),
    is_open: z.boolean().optional().describe('True if the channel is open'),
    topic: z
        .object({
        value: z.string().describe('Topic text'),
        creator: z.string().describe('User ID who set the topic'),
        last_set: z.number().describe('Unix timestamp when topic was last set'),
    })
        .optional()
        .describe('Channel topic information'),
    purpose: z
        .object({
        value: z.string().describe('Purpose text'),
        creator: z.string().describe('User ID who set the purpose'),
        last_set: z
            .number()
            .describe('Unix timestamp when purpose was last set'),
    })
        .optional()
        .describe('Channel purpose information'),
    num_members: z
        .number()
        .optional()
        .describe('Number of members in the channel'),
})
    .describe('Slack channel object with metadata');
const SlackUserSchema = z
    .object({
    id: z.string().describe('Unique user identifier'),
    team_id: z.string().optional().describe('Team/workspace ID'),
    name: z.string().describe('Username (handle without @)'),
    deleted: z.boolean().optional().describe('True if user account is deleted'),
    color: z.string().optional().describe('Color code for user in UI'),
    real_name: z.string().optional().describe('Users real name'),
    tz: z.string().optional().describe('Timezone identifier'),
    tz_label: z.string().optional().describe('Human-readable timezone label'),
    tz_offset: z
        .number()
        .optional()
        .describe('Timezone offset from UTC in seconds'),
    profile: z
        .object({
        title: z.string().optional().describe('Job title'),
        phone: z.string().optional().describe('Phone number'),
        skype: z.string().optional().describe('Skype username'),
        real_name: z.string().optional().describe('Real name from profile'),
        real_name_normalized: z
            .string()
            .optional()
            .describe('Normalized real name'),
        display_name: z.string().optional().describe('Display name'),
        display_name_normalized: z
            .string()
            .optional()
            .describe('Normalized display name'),
        fields: z
            .record(z.unknown())
            .optional()
            .describe('Custom profile fields'),
        status_text: z.string().optional().describe('Current status text'),
        status_emoji: z.string().optional().describe('Current status emoji'),
        status_expiration: z
            .number()
            .optional()
            .describe('Unix timestamp when status expires'),
        avatar_hash: z.string().optional().describe('Hash for avatar image'),
        image_original: z
            .string()
            .optional()
            .describe('URL of original avatar image'),
        is_custom_image: z
            .boolean()
            .optional()
            .describe('True if using custom avatar'),
        email: z.string().optional().describe('Email address'),
        first_name: z.string().optional().describe('First name'),
        last_name: z.string().optional().describe('Last name'),
        image_24: z.string().optional().describe('24x24 pixel avatar URL'),
        image_32: z.string().optional().describe('32x32 pixel avatar URL'),
        image_48: z.string().optional().describe('48x48 pixel avatar URL'),
        image_72: z.string().optional().describe('72x72 pixel avatar URL'),
        image_192: z.string().optional().describe('192x192 pixel avatar URL'),
        image_512: z.string().optional().describe('512x512 pixel avatar URL'),
        image_1024: z
            .string()
            .optional()
            .describe('1024x1024 pixel avatar URL'),
    })
        .optional()
        .describe('User profile information'),
    is_admin: z
        .boolean()
        .optional()
        .describe('True if user is workspace admin'),
    is_owner: z
        .boolean()
        .optional()
        .describe('True if user is workspace owner'),
    is_primary_owner: z
        .boolean()
        .optional()
        .describe('True if user is primary workspace owner'),
    is_restricted: z
        .boolean()
        .optional()
        .describe('True if user is restricted (single-channel guest)'),
    is_ultra_restricted: z
        .boolean()
        .optional()
        .describe('True if user is ultra restricted (multi-channel guest)'),
    is_bot: z.boolean().optional().describe('True if this is a bot user'),
    is_app_user: z.boolean().optional().describe('True if this is an app user'),
    updated: z
        .number()
        .optional()
        .describe('Unix timestamp when user was last updated'),
    has_2fa: z
        .boolean()
        .optional()
        .describe('True if user has two-factor authentication enabled'),
})
    .describe('Slack user object with profile and permissions');
const SlackMessageSchema = z
    .object({
    type: z.string().describe('Message type (usually "message")'),
    ts: z.string().describe('Message timestamp (unique identifier)'),
    user: z.string().optional().describe('User ID who sent the message'),
    bot_id: z
        .string()
        .optional()
        .describe('Bot ID if message was sent by a bot'),
    bot_profile: z
        .object({
        name: z.string().optional().describe('Bot display name'),
    })
        .optional()
        .describe('Bot profile information if message was sent by a bot'),
    username: z
        .string()
        .optional()
        .describe('Username of the bot or user who sent the message'),
    text: z.string().optional().describe('Message text content'),
    thread_ts: z
        .string()
        .optional()
        .describe('Timestamp of parent message if this is a thread reply'),
    parent_user_id: z
        .string()
        .optional()
        .describe('User ID of thread parent message author'),
    reply_count: z
        .number()
        .optional()
        .describe('Number of replies in this thread'),
    reply_users_count: z
        .number()
        .optional()
        .describe('Number of unique users who replied in thread'),
    latest_reply: z
        .string()
        .optional()
        .describe('Timestamp of most recent reply in thread'),
    reply_users: z
        .array(z.string())
        .optional()
        .describe('Array of user IDs who replied in thread'),
    is_locked: z.boolean().optional().describe('True if thread is locked'),
    subscribed: z
        .boolean()
        .optional()
        .describe('True if current user is subscribed to thread'),
    attachments: z
        .array(z.unknown())
        .optional()
        .describe('Legacy message attachments'),
    blocks: z
        .array(z.unknown())
        .optional()
        .describe('Block Kit structured content'),
    reactions: z
        .array(z.object({
        name: z.string().describe('Emoji name without colons'),
        users: z
            .array(z.string())
            .describe('User IDs who reacted with this emoji'),
        count: z.number().describe('Total count of this reaction'),
    }))
        .optional()
        .describe('Array of emoji reactions on this message'),
})
    .describe('Slack message object with content and metadata');
// Define result schemas for different operations
const SlackResultSchema = z.discriminatedUnion('operation', [
    z.object({
        operation: z
            .literal('send_message')
            .describe('Send a message to a Slack channel or DM'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        channel: z
            .string()
            .optional()
            .describe('Channel ID where the message was sent'),
        ts: z.string().optional().describe('Timestamp of the sent message'),
        message: SlackMessageSchema.optional().describe('Details of the sent message'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('list_channels')
            .describe('List all channels in the Slack workspace'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        channels: z
            .array(SlackChannelSchema)
            .optional()
            .describe('Array of channel objects'),
        response_metadata: z
            .object({
            next_cursor: z
                .string()
                .describe('Cursor for pagination to get next set of results'),
        })
            .optional()
            .describe('Metadata for pagination'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('get_channel_info')
            .describe('Get detailed information about a specific channel'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        channel: SlackChannelSchema.optional().describe('Channel information object'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('get_user_info')
            .describe('Get detailed information about a specific user'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        user: SlackUserSchema.optional().describe('User information object'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('list_users')
            .describe('List all users in the Slack workspace'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        members: z
            .array(SlackUserSchema)
            .optional()
            .describe('Array of user objects'),
        response_metadata: z
            .object({
            next_cursor: z
                .string()
                .describe('Cursor for pagination to get next set of results'),
        })
            .optional()
            .describe('Metadata for pagination'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('get_conversation_history')
            .describe('Retrieve message history from a channel or direct message'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        messages: z
            .array(SlackMessageSchema)
            .optional()
            .describe('Array of message objects'),
        has_more: z
            .boolean()
            .optional()
            .describe('Whether there are more messages to retrieve'),
        response_metadata: z
            .object({
            next_cursor: z
                .string()
                .describe('Cursor for pagination to get next set of results'),
        })
            .optional()
            .describe('Metadata for pagination'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('get_thread_replies')
            .describe('Retrieve all replies to a thread in a channel'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        messages: z
            .array(SlackMessageSchema)
            .optional()
            .describe('Array of message objects in the thread'),
        has_more: z
            .boolean()
            .optional()
            .describe('Whether there are more messages to retrieve'),
        response_metadata: z
            .object({
            next_cursor: z
                .string()
                .describe('Cursor for pagination to get next set of results'),
        })
            .optional()
            .describe('Metadata for pagination'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('update_message')
            .describe('Update an existing message in a channel'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        channel: z
            .string()
            .optional()
            .describe('Channel ID where the message was updated'),
        ts: z.string().optional().describe('Timestamp of the updated message'),
        text: z.string().optional().describe('Updated text content of the message'),
        message: SlackMessageSchema.optional().describe('Details of the updated message'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('delete_message')
            .describe('Delete a message from a channel'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        channel: z
            .string()
            .optional()
            .describe('Channel ID where the message was deleted'),
        ts: z.string().optional().describe('Timestamp of the deleted message'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('add_reaction')
            .describe('Add an emoji reaction to a message'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('remove_reaction')
            .describe('Remove an emoji reaction from a message'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('join_channel')
            .describe('Join a public Slack channel'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        channel: SlackChannelSchema.optional().describe('Channel information object after joining'),
        already_in_channel: z
            .boolean()
            .optional()
            .describe('Whether the bot was already a member of the channel'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
    z.object({
        operation: z
            .literal('upload_file')
            .describe('Upload a file to a Slack channel'),
        ok: z.boolean().describe('Whether the Slack API call was successful'),
        file: z
            .object({
            id: z.string().describe('Unique file identifier'),
            created: z.number().describe('Unix timestamp when file was created'),
            timestamp: z.number().describe('Unix timestamp when file was uploaded'),
            name: z.string().describe('Original filename'),
            title: z.string().optional().describe('File title'),
            mimetype: z.string().describe('MIME type of the file'),
            filetype: z.string().describe('File type extension'),
            pretty_type: z.string().describe('Human-readable file type'),
            user: z.string().describe('User ID who uploaded the file'),
            editable: z.boolean().describe('Whether the file is editable'),
            size: z.number().describe('File size in bytes'),
            mode: z.string().describe('File sharing mode'),
            is_external: z
                .boolean()
                .describe('Whether file is from external source'),
            external_type: z.string().describe('External file type if applicable'),
            is_public: z.boolean().describe('Whether file is publicly accessible'),
            public_url_shared: z.boolean().describe('Whether public URL is shared'),
            display_as_bot: z
                .boolean()
                .describe('Whether file is displayed as uploaded by bot'),
            username: z.string().describe('Username of uploader'),
            url_private: z.string().describe('Private URL to access file'),
            url_private_download: z.string().describe('Private download URL'),
            permalink: z.string().describe('Permanent link to file'),
            permalink_public: z
                .string()
                .optional()
                .describe('Public permanent link'),
            shares: z
                .object({
                public: z
                    .record(z.array(z.object({
                    reply_users: z
                        .array(z.string())
                        .describe('User IDs who replied'),
                    reply_users_count: z
                        .number()
                        .describe('Number of unique users who replied'),
                    reply_count: z.number().describe('Total number of replies'),
                    ts: z.string().describe('Timestamp of the share'),
                    channel_name: z.string().describe('Name of the channel'),
                    team_id: z.string().describe('Team ID'),
                })))
                    .optional()
                    .describe('Public channel shares'),
                private: z
                    .record(z.array(z.object({
                    reply_users: z
                        .array(z.string())
                        .describe('User IDs who replied'),
                    reply_users_count: z
                        .number()
                        .describe('Number of unique users who replied'),
                    reply_count: z.number().describe('Total number of replies'),
                    ts: z.string().describe('Timestamp of the share'),
                    channel_name: z.string().describe('Name of the channel'),
                    team_id: z.string().describe('Team ID'),
                })))
                    .optional()
                    .describe('Private channel shares'),
            })
                .optional()
                .describe('Information about where file is shared'),
            channels: z
                .array(z.string())
                .optional()
                .describe('Channel IDs where file is shared'),
            groups: z
                .array(z.string())
                .optional()
                .describe('Private group IDs where file is shared'),
            ims: z
                .array(z.string())
                .optional()
                .describe('Direct message IDs where file is shared'),
            has_rich_preview: z
                .boolean()
                .optional()
                .describe('Whether file has rich preview'),
        })
            .optional()
            .describe('File information object'),
        error: z.string().describe('Error message if operation failed'),
        success: z.boolean().describe('Whether the operation was successful'),
    }),
]);
export class SlackBubble extends ServiceBubble {
    /**
     * Test the validity of the Slack credential by making a test API call
     * @returns Promise that resolves to true if credential is valid, false otherwise
     * @throws ExternalServiceError if Slack API is unreachable
     */
    async testCredential() {
        const response = await this.makeSlackApiCall('auth.test', {});
        return response.ok;
    }
    static type = 'service';
    static service = 'slack';
    static authType = 'apikey';
    static bubbleName = 'slack';
    static schema = SlackParamsSchema;
    static resultSchema = SlackResultSchema;
    static shortDescription = 'Slack integration for messaging and workspace management';
    static longDescription = `
    Comprehensive Slack integration bubble for managing messages, channels, and users.
    Use cases:
    - Send messages to channels or direct messages
    - Retrieve channel information and list channels
    - Get user information and list workspace members
    - Manage conversation history and message operations
    - Add/remove reactions and manage message interactions
    
    Security Features:
    - Token-based authentication
    - Parameter validation and sanitization
    - Rate limiting awareness
    - Comprehensive error handling
  `;
    static alias = 'slack';
    constructor(params = {
        operation: 'list_channels',
    }, context, instanceId) {
        super(params, context, instanceId);
    }
    async performAction(context) {
        // Context is available but not currently used in this implementation
        void context;
        const { operation } = this.params;
        try {
            const result = await (async () => {
                switch (operation) {
                    case 'send_message':
                        return await this.sendMessage(this.params);
                    case 'list_channels':
                        return await this.listChannels(this.params);
                    case 'get_channel_info':
                        return await this.getChannelInfo(this.params);
                    case 'get_user_info':
                        return await this.getUserInfo(this.params);
                    case 'list_users':
                        return await this.listUsers(this.params);
                    case 'get_conversation_history':
                        return await this.getConversationHistory(this.params);
                    case 'get_thread_replies':
                        return await this.getThreadReplies(this.params);
                    case 'update_message':
                        return await this.updateMessage(this.params);
                    case 'delete_message':
                        return await this.deleteMessage(this.params);
                    case 'add_reaction':
                        return await this.addReaction(this.params);
                    case 'remove_reaction':
                        return await this.removeReaction(this.params);
                    case 'upload_file':
                        return await this.uploadFile(this.params);
                    case 'join_channel':
                        return await this.joinChannel(this.params);
                    default:
                        throw new Error(`Unsupported operation: ${operation}`);
                }
            })();
            // The result is guaranteed to match T['operation'] because of the discriminated union
            return result;
        }
        catch (error) {
            const failedOperation = this.params.operation;
            return {
                success: false,
                ok: false,
                operation: failedOperation,
                error: error instanceof Error
                    ? error.message
                    : 'Unknown error occurred in SlackBubble',
            };
        }
    }
    /**
     * Helper method to resolve channel names to channel IDs.
     * If the input looks like a channel ID (starts with C, G, or D), returns it as-is.
     * Otherwise, searches for a channel with the given name.
     */
    async resolveChannelId(channelInput) {
        // Check if input is already a channel ID (starts with C, G, D, etc.)
        if (/^[CGD][A-Z0-9]+$/i.test(channelInput)) {
            return channelInput;
        }
        // Remove # prefix if present
        const channelName = channelInput.replace(/^#/, '');
        // Get all channels to find the matching name
        const response = await this.makeSlackApiCall('conversations.list', {
            types: 'public_channel,private_channel',
            exclude_archived: 'true',
            limit: '1000', // Get a large batch to find the channel
        }, 'GET');
        if (!response.ok) {
            throw new Error(`Failed to list channels: ${response.error}`);
        }
        const channels = response.channels;
        const matchedChannel = channels.find((channel) => channel.name.toLowerCase() === channelName.toLowerCase());
        if (!matchedChannel) {
            throw new Error(`Channel "${channelName}" not found. Available channels: ${channels.map((c) => c.name).join(', ')}`);
        }
        return matchedChannel.id;
    }
    async sendMessage(params) {
        const { channel, text, username, icon_emoji, icon_url, attachments, blocks, thread_ts, reply_broadcast, unfurl_links, unfurl_media, } = params;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const body = {
            channel: resolvedChannel,
            text,
            unfurl_links,
            unfurl_media,
        };
        if (username)
            body.username = username;
        if (icon_emoji)
            body.icon_emoji = icon_emoji;
        if (icon_url)
            body.icon_url = icon_url;
        if (attachments)
            body.attachments = JSON.stringify(attachments);
        if (blocks)
            body.blocks = JSON.stringify(blocks);
        if (thread_ts) {
            body.thread_ts = thread_ts;
            body.reply_broadcast = reply_broadcast;
        }
        console.log('sending blocks', body.blocks);
        const response = await this.makeSlackApiCall('chat.postMessage', body);
        return {
            operation: 'send_message',
            ok: response.ok,
            channel: response.ok ? response.channel : undefined,
            ts: response.ok ? response.ts : undefined,
            message: response.ok && response.message
                ? SlackMessageSchema.parse(response.message)
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async listChannels(params) {
        // Parse the params to apply defaults
        const parsed = SlackParamsSchema.parse(params);
        const { types, exclude_archived, limit, cursor } = parsed;
        const queryParams = {
            types: types.join(','),
            exclude_archived: exclude_archived.toString(),
            limit: limit.toString(),
        };
        if (cursor)
            queryParams.cursor = cursor;
        const response = await this.makeSlackApiCall('conversations.list', queryParams, 'GET');
        return {
            operation: 'list_channels',
            ok: response.ok,
            channels: response.ok && response.channels
                ? z.array(SlackChannelSchema).parse(response.channels)
                : undefined,
            response_metadata: response.ok && response.response_metadata
                ? {
                    next_cursor: response.response_metadata.next_cursor,
                }
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async getChannelInfo(params) {
        // Parse the params to apply defaults
        const parsed = SlackParamsSchema.parse(params);
        const { channel, include_locale } = parsed;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const queryParams = {
            channel: resolvedChannel,
            include_locale: include_locale.toString(),
        };
        const response = await this.makeSlackApiCall('conversations.info', queryParams, 'GET');
        return {
            operation: 'get_channel_info',
            ok: response.ok,
            channel: response.ok && response.channel
                ? SlackChannelSchema.parse(response.channel)
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async getUserInfo(params) {
        // Parse the params to apply defaults
        const parsed = SlackParamsSchema.parse(params);
        const { user, include_locale } = parsed;
        const queryParams = {
            user,
            include_locale: include_locale.toString(),
        };
        const response = await this.makeSlackApiCall('users.info', queryParams, 'GET');
        return {
            operation: 'get_user_info',
            ok: response.ok,
            user: response.ok && response.user
                ? SlackUserSchema.parse(response.user)
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async listUsers(params) {
        // Parse the params to apply defaults
        const parsed = SlackParamsSchema.parse(params);
        const { limit, cursor, include_locale } = parsed;
        const queryParams = {
            limit: limit.toString(),
            include_locale: include_locale.toString(),
        };
        if (cursor)
            queryParams.cursor = cursor;
        const response = await this.makeSlackApiCall('users.list', queryParams, 'GET');
        return {
            operation: 'list_users',
            ok: response.ok,
            members: response.ok && response.members
                ? z.array(SlackUserSchema).parse(response.members)
                : undefined,
            response_metadata: response.ok && response.response_metadata
                ? {
                    next_cursor: response.response_metadata.next_cursor,
                }
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async getConversationHistory(params) {
        // Parse the params to apply defaults
        const parsed = SlackParamsSchema.parse(params);
        const { channel, latest, oldest, inclusive, limit, cursor } = parsed;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const queryParams = {
            channel: resolvedChannel,
            inclusive: inclusive.toString(),
            limit: limit.toString(),
        };
        if (latest)
            queryParams.latest = latest;
        if (oldest)
            queryParams.oldest = oldest;
        if (cursor)
            queryParams.cursor = cursor;
        const response = await this.makeSlackApiCall('conversations.history', queryParams, 'GET');
        return {
            operation: 'get_conversation_history',
            ok: response.ok,
            messages: response.ok && response.messages
                ? z.array(SlackMessageSchema).parse(response.messages)
                : undefined,
            has_more: response.ok ? response.has_more : undefined,
            response_metadata: response.ok && response.response_metadata
                ? {
                    next_cursor: response.response_metadata.next_cursor,
                }
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async getThreadReplies(params) {
        // Parse the params to apply defaults
        const parsed = SlackParamsSchema.parse(params);
        const { channel, ts, latest, oldest, inclusive, limit, cursor } = parsed;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const queryParams = {
            channel: resolvedChannel,
            ts: ts,
        };
        if (latest)
            queryParams.latest = latest;
        if (oldest)
            queryParams.oldest = oldest;
        if (inclusive !== undefined)
            queryParams.inclusive = inclusive.toString();
        if (limit !== undefined)
            queryParams.limit = limit.toString();
        if (cursor)
            queryParams.cursor = cursor;
        const response = await this.makeSlackApiCall('conversations.replies', queryParams, 'GET');
        return {
            operation: 'get_thread_replies',
            ok: response.ok,
            messages: response.ok && response.messages
                ? z.array(SlackMessageSchema).parse(response.messages)
                : undefined,
            has_more: response.ok ? response.has_more : undefined,
            response_metadata: response.ok && response.response_metadata
                ? {
                    next_cursor: response.response_metadata.next_cursor,
                }
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async updateMessage(params) {
        const { channel, ts, text, attachments, blocks } = params;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const body = {
            channel: resolvedChannel,
            ts,
        };
        if (text)
            body.text = text;
        if (attachments)
            body.attachments = JSON.stringify(attachments);
        if (blocks)
            body.blocks = JSON.stringify(blocks);
        const response = await this.makeSlackApiCall('chat.update', body);
        return {
            operation: 'update_message',
            ok: response.ok,
            channel: response.ok ? response.channel : undefined,
            ts: response.ok ? response.ts : undefined,
            text: response.ok ? response.text : undefined,
            message: response.ok && response.message
                ? SlackMessageSchema.parse(response.message)
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async deleteMessage(params) {
        const { channel, ts } = params;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const body = {
            channel: resolvedChannel,
            ts,
        };
        const response = await this.makeSlackApiCall('chat.delete', body);
        return {
            operation: 'delete_message',
            ok: response.ok,
            channel: response.ok ? response.channel : undefined,
            ts: response.ok ? response.ts : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async addReaction(params) {
        const { name, channel, timestamp } = params;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const body = {
            name,
            channel: resolvedChannel,
            timestamp,
        };
        const response = await this.makeSlackApiCall('reactions.add', body);
        return {
            operation: 'add_reaction',
            ok: response.ok,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    async removeReaction(params) {
        const { name, channel, timestamp } = params;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const body = {
            name,
            channel: resolvedChannel,
            timestamp,
        };
        const response = await this.makeSlackApiCall('reactions.remove', body);
        return {
            operation: 'remove_reaction',
            ok: response.ok,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    /**
     * Upload a file to a Slack channel
     * @param params - Upload file parameters including channel and file path
     * @returns Promise resolving to upload result with file metadata
     * @throws AuthenticationError if credentials are invalid
     * @throws ExternalServiceError if Slack API call fails
     */
    async uploadFile(params) {
        const { channel, file_path, filename, title, initial_comment, thread_ts } = params;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        // SECURITY: Validate file path using common validator
        try {
            validateFilePath(file_path, false); // false = no absolute paths
        }
        catch (error) {
            if (error instanceof CommonValidationError) {
                return {
                    operation: 'upload_file',
                    ok: false,
                    error: error.message,
                    success: false,
                };
            }
            throw error;
        }
        // SECURITY: Block sensitive file extensions
        const sensitiveExtensions = [
            '.key', '.pem', '.crt', '.p12', '.pfx', // Certificates/keys
            '.env', '.config', '.conf', // Config files
            '.sh', '.bash', '.ps1', '.bat', '.cmd', // Scripts
            '.exe', '.dll', '.so', '.dylib', // Binaries
        ];
        const lowerPath = file_path.toLowerCase();
        if (sensitiveExtensions.some((ext) => lowerPath.endsWith(ext))) {
            return {
                operation: 'upload_file',
                ok: false,
                error: `Files with extension ${sensitiveExtensions.join(', ')} are not allowed for security reasons`,
                success: false,
            };
        }
        // Read the file
        const fs = await import('fs/promises');
        const path = await import('path');
        try {
            // SECURITY: Additional check - verify file exists and is readable
            try {
                await fs.access(file_path, fs.constants.R_OK);
            }
            catch {
                return {
                    operation: 'upload_file',
                    ok: false,
                    error: 'File does not exist or is not readable',
                    success: false,
                };
            }
            // SECURITY: Get file stats to verify it's a regular file (not a directory or special file)
            const stats = await fs.stat(file_path);
            if (!stats.isFile()) {
                return {
                    operation: 'upload_file',
                    ok: false,
                    error: 'Path does not point to a regular file',
                    success: false,
                };
            }
            // SECURITY: Limit file size to prevent DoS (max 50MB)
            const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
            if (stats.size > MAX_FILE_SIZE) {
                return {
                    operation: 'upload_file',
                    ok: false,
                    error: `File too large: ${stats.size} bytes. Maximum size is ${MAX_FILE_SIZE} bytes (50MB)`,
                    success: false,
                };
            }
            const fileBuffer = await fs.readFile(file_path);
            const actualFilename = filename || path.basename(file_path);
            const fileSize = fileBuffer.length;
            // Step 1: Get upload URL
            const uploadUrlResponse = await this.makeSlackApiCall('files.getUploadURLExternal', {
                filename: actualFilename,
                length: fileSize.toString(),
            });
            if (!uploadUrlResponse.ok) {
                throw new Error(`Failed to get upload URL: ${uploadUrlResponse.error}`);
            }
            const { upload_url, file_id } = uploadUrlResponse;
            // Step 2: Upload file to the URL
            const uploadResponse = await fetch(upload_url, {
                method: 'POST',
                body: fileBuffer,
                headers: {
                    'Content-Type': 'application/octet-stream',
                },
            });
            if (!uploadResponse.ok) {
                throw new Error(`Failed to upload file: ${uploadResponse.statusText}`);
            }
            // Step 3: Complete the upload
            const completeParams = {
                files: JSON.stringify([
                    {
                        id: file_id,
                        title: title || actualFilename,
                    },
                ]),
            };
            // Add optional parameters
            if (resolvedChannel)
                completeParams.channel_id = resolvedChannel;
            if (initial_comment)
                completeParams.initial_comment = initial_comment;
            if (thread_ts)
                completeParams.thread_ts = thread_ts;
            const completeResponse = await this.makeSlackApiCall('files.completeUploadExternal', completeParams);
            if (!completeResponse.ok) {
                throw new Error(`Failed to complete upload: ${completeResponse.error}`);
            }
            // Extract file info from response
            const files = completeResponse.files || [];
            const uploadedFile = files[0] || {};
            return {
                operation: 'upload_file',
                ok: true,
                file: {
                    id: uploadedFile.id || file_id,
                    created: uploadedFile.created || Date.now() / 1000,
                    timestamp: uploadedFile.timestamp || Date.now() / 1000,
                    name: uploadedFile.name || actualFilename,
                    title: uploadedFile.title || title || actualFilename,
                    mimetype: uploadedFile.mimetype || 'image/png',
                    filetype: uploadedFile.filetype || 'png',
                    pretty_type: uploadedFile.pretty_type || 'PNG',
                    user: uploadedFile.user || '',
                    editable: uploadedFile.editable || false,
                    size: uploadedFile.size || fileSize,
                    mode: uploadedFile.mode || 'hosted',
                    is_external: uploadedFile.is_external || false,
                    external_type: uploadedFile.external_type || '',
                    is_public: uploadedFile.is_public || false,
                    public_url_shared: uploadedFile.public_url_shared || false,
                    display_as_bot: uploadedFile.display_as_bot || false,
                    username: uploadedFile.username || '',
                    url_private: uploadedFile.url_private || '',
                    url_private_download: uploadedFile.url_private_download || '',
                    permalink: uploadedFile.permalink || '',
                    permalink_public: uploadedFile.permalink_public || '',
                    shares: uploadedFile.shares || {},
                    channels: uploadedFile.channels || [resolvedChannel],
                    groups: uploadedFile.groups || [],
                    ims: uploadedFile.ims || [],
                    has_rich_preview: uploadedFile.has_rich_preview || false,
                },
                error: '',
                success: true,
            };
        }
        catch (error) {
            return {
                operation: 'upload_file',
                ok: false,
                error: error instanceof Error ? error.message : 'Unknown file upload error',
                success: false,
            };
        }
    }
    async joinChannel(params) {
        const { channel } = params;
        // Resolve channel name to ID if needed
        const resolvedChannel = await this.resolveChannelId(channel);
        const body = {
            channel: resolvedChannel,
        };
        const response = await this.makeSlackApiCall('conversations.join', body);
        return {
            operation: 'join_channel',
            ok: response.ok,
            channel: response.ok && response.channel
                ? SlackChannelSchema.parse(response.channel)
                : undefined,
            already_in_channel: response.ok
                ? response.already_in_channel
                : undefined,
            error: !response.ok ? JSON.stringify(response, null, 2) : '',
            success: response.ok,
        };
    }
    /**
     * Choose the appropriate credential for Slack API calls
     * @returns The Slack authentication token
     * @throws AuthenticationError if credentials are not provided
     */
    chooseCredential() {
        const { credentials } = this.params;
        // If no credentials were injected, return undefined
        if (!credentials || typeof credentials !== 'object') {
            throw new AuthenticationError('No slack credentials provided');
        }
        // Slack bubble always uses Slack credentials
        return credentials[CredentialType.SLACK_CRED];
    }
    /**
     * Make an API call to the Slack API
     * @param endpoint - The Slack API endpoint to call
     * @param params - Query parameters or request body
     * @param method - HTTP method (GET or POST)
     * @returns Promise resolving to Slack API response
     * @throws AuthenticationError if authentication token is missing
     * @throws ExternalServiceError if API call fails
     */
    async makeSlackApiCall(endpoint, params, method = 'POST') {
        const url = `${SLACK_API_BASE}/${endpoint}`;
        // Use chooseCredential to get the appropriate credential
        const authToken = this.chooseCredential();
        if (!authToken) {
            throw new AuthenticationError('Slack authentication token is required');
        }
        const headers = {
            Authorization: `Bearer ${authToken}`,
            'Content-Type': method === 'POST'
                ? 'application/json'
                : 'application/x-www-form-urlencoded',
        };
        let fetchConfig;
        if (method === 'GET') {
            const searchParams = new URLSearchParams();
            for (const [key, value] of Object.entries(params)) {
                if (value !== undefined && value !== null) {
                    searchParams.append(key, String(value));
                }
            }
            fetchConfig = {
                method: 'GET',
                headers,
            };
            const urlWithParams = `${url}?${searchParams.toString()}`;
            const response = await fetch(urlWithParams, fetchConfig);
            const data = (await response.json());
            if (!response.ok && !data.ok) {
                throw new ExternalServiceError('slack', `Slack API error: ${data.error || 'Unknown error'}`);
            }
            return data;
        }
        else {
            // Most Slack POST endpoints expect form-encoded data, not JSON
            // Only specific endpoints like chat.postMessage with blocks expect JSON
            const needsJson = ['chat.postMessage', 'chat.update'].includes(endpoint) &&
                (params.blocks || params.attachments);
            if (needsJson) {
                fetchConfig = {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(params),
                };
            }
            else {
                // Use form-encoded for most endpoints
                const formData = new URLSearchParams();
                for (const [key, value] of Object.entries(params)) {
                    if (value !== undefined && value !== null) {
                        formData.append(key, String(value));
                    }
                }
                fetchConfig = {
                    method: 'POST',
                    headers: {
                        ...headers,
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: formData.toString(),
                };
            }
            const response = await fetch(url, fetchConfig);
            const data = (await response.json());
            if (!response.ok && !data.ok) {
                throw new ExternalServiceError('slack', `Slack API error: ${data.error || 'Unknown error'}`);
            }
            return data;
        }
    }
}
//# sourceMappingURL=slack.js.map