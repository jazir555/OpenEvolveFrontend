/**
 * Slack Bubble Unit Tests
 * File: service-bubble/slack.test.ts
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SlackBubble } from './slack.js';
import { CredentialType } from '@bubblelab/shared-schemas';

describe('SlackBubble', () => {
  let mockFetch: any;

  beforeEach(() => {
    // Mock fetch API for HTTP requests
    mockFetch = vi.fn();
    global.fetch = mockFetch;

    // Clear any mock state
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Construction and Initialization', () => {
    it('should create instance with valid parameters for send_message operation', () => {
      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Test message',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      expect(bubble).toBeDefined();
      expect(bubble.params.operation).toBe('send_message');
      expect(bubble.params.channel).toBe('C123456');
    });

    it('should create instance with valid parameters for list_channels operation', () => {
      const bubble = new SlackBubble({
        operation: 'list_channels',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      expect(bubble).toBeDefined();
      expect(bubble.params.operation).toBe('list_channels');
    });

    it('should validate required parameters for send_message', () => {
      expect(() => {
        new SlackBubble({
          operation: 'send_message',
          // Missing channel and text
        } as any);
      }).toThrow();
    });

    it('should set default values for optional parameters', () => {
      const bubble = new SlackBubble({
        operation: 'list_channels',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      expect(bubble.params.limit).toBeDefined();
      expect(bubble.params.exclude_archived).toBeDefined();
    });
  });

  describe('Authentication', () => {
    it('should use provided credentials for API calls', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: true, channels: [] }),
      });

      const bubble = new SlackBubble({
        operation: 'list_channels',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      await bubble.act();

      // Verify credentials were used
      expect(mockFetch).toHaveBeenCalled();
      const authHeader = mockFetch.mock.calls[0][1].headers['Authorization'];
      expect(authHeader).toContain('xoxb-test-token');
    });

    it('should handle missing credentials gracefully', async () => {
      const bubble = new SlackBubble({
        operation: 'list_channels',
        credentials: undefined,
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('credentials');
    });

    it('should test credential validity with auth.test', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: true, team: 'TestTeam', user: 'TestBot' }),
      });

      const bubble = new SlackBubble({
        operation: 'list_channels',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'valid-token',
          }),
        },
      });

      const isValid = await bubble.testCredential();

      expect(isValid).toBe(true);
    });

    it('should invalidate bad credentials', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: false, error: 'invalid_auth' }),
      });

      const bubble = new SlackBubble({
        operation: 'list_channels',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'invalid-token',
          }),
        },
      });

      const isValid = await bubble.testCredential();

      expect(isValid).toBe(false);
    });
  });

  describe('Message Operations', () => {
    it('should send message to channel successfully', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channel: 'C123456',
          ts: '1234567890.123456',
          message: { type: 'message', text: 'Test message', ts: '1234567890.123456' },
        }),
      });

      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Test message',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.ok).toBe(true);
      expect(result.operation).toBe('send_message');
      expect(result.ts).toBeDefined();
    });

    it('should send message with blocks', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channel: 'C123456',
          ts: '1234567890.123456',
        }),
      });

      const blocks = [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: '*Test Message*',
          },
        },
      ];

      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Fallback text',
        blocks,
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
    });

    it('should send thread reply', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channel: 'C123456',
          ts: '1234567890.123457',
        }),
      });

      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Thread reply',
        thread_ts: '1234567890.123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
    });

    it('should update existing message', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channel: 'C123456',
          ts: '1234567890.123456',
          text: 'Updated message',
        }),
      });

      const bubble = new SlackBubble({
        operation: 'update_message',
        channel: 'C123456',
        ts: '1234567890.123456',
        text: 'Updated message',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('update_message');
      expect(result.text).toBe('Updated message');
    });

    it('should delete message', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channel: 'C123456',
          ts: '1234567890.123456',
        }),
      });

      const bubble = new SlackBubble({
        operation: 'delete_message',
        channel: 'C123456',
        ts: '1234567890.123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('delete_message');
    });
  });

  describe('Channel Operations', () => {
    it('should list channels', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channels: [
            {
              id: 'C123456',
              name: 'general',
              is_channel: true,
              is_private: false,
              num_members: 10,
            },
            {
              id: 'C789012',
              name: 'random',
              is_channel: true,
              is_private: false,
              num_members: 5,
            },
          ],
        }),
      });

      const bubble = new SlackBubble({
        operation: 'list_channels',
        limit: 10,
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.channels).toHaveLength(2);
      expect(result.channels[0].name).toBe('general');
    });

    it('should get channel info', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channel: {
            id: 'C123456',
            name: 'test-channel',
            topic: { value: 'Test topic' },
            purpose: { value: 'Test purpose' },
            num_members: 10,
          },
        }),
      });

      const bubble = new SlackBubble({
        operation: 'get_channel_info',
        channel: 'C123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.channel.name).toBe('test-channel');
      expect(result.channel.num_members).toBe(10);
    });

    it('should join channel', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          channel: {
            id: 'C123456',
            name: 'test-channel',
            is_member: true,
          },
        }),
      });

      const bubble = new SlackBubble({
        operation: 'join_channel',
        channel: 'C123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('join_channel');
    });
  });

  describe('User Operations', () => {
    it('should get user info', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          user: {
            id: 'U123456',
            name: 'testuser',
            real_name: 'Test User',
            profile: {
              email: 'test@example.com',
              title: 'Developer',
            },
            tz: 'America/New_York',
          },
        }),
      });

      const bubble = new SlackBubble({
        operation: 'get_user_info',
        user: 'U123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.user.name).toBe('testuser');
      expect(result.user.profile.email).toBe('test@example.com');
    });

    it('should list users', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          members: [
            {
              id: 'U123456',
              name: 'user1',
              real_name: 'User One',
            },
            {
              id: 'U789012',
              name: 'user2',
              real_name: 'User Two',
            },
          ],
        }),
      });

      const bubble = new SlackBubble({
        operation: 'list_users',
        limit: 10,
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.members).toHaveLength(2);
      expect(result.members[0].name).toBe('user1');
    });
  });

  describe('Reaction Operations', () => {
    it('should add reaction to message', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: true }),
      });

      const bubble = new SlackBubble({
        operation: 'add_reaction',
        name: 'thumbsup',
        channel: 'C123456',
        timestamp: '1234567890.123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('add_reaction');
    });

    it('should remove reaction from message', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: true }),
      });

      const bubble = new SlackBubble({
        operation: 'remove_reaction',
        name: 'thumbsup',
        channel: 'C123456',
        timestamp: '1234567890.123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('remove_reaction');
    });
  });

  describe('Security - File Upload Validation', () => {
    it('should reject path traversal attempts (..)', async () => {
      const bubble = new SlackBubble({
        operation: 'upload_file',
        channel: 'C123456',
        file_path: '../../../etc/passwd',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('..');
    });

    it('should reject absolute paths', async () => {
      const bubble = new SlackBubble({
        operation: 'upload_file',
        channel: 'C123456',
        file_path: '/etc/passwd',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Absolute paths');
    });

    it('should reject paths with ~ character', async () => {
      const bubble = new SlackBubble({
        operation: 'upload_file',
        channel: 'C123456',
        file_path: '~/.ssh/id_rsa',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('~');
    });

    it('should reject sensitive file extensions', async () => {
      const bubble = new SlackBubble({
        operation: 'upload_file',
        channel: 'C123456',
        file_path: 'config.env',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('.env');
    });

    it('should reject files with .key extension', async () => {
      const bubble = new SlackBubble({
        operation: 'upload_file',
        channel: 'C123456',
        file_path: 'private.key',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('.key');
    });

    it('should reject executable file extensions', async () => {
      const bubble = new SlackBubble({
        operation: 'upload_file',
        channel: 'C123456',
        file_path: 'malware.exe',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('.exe');
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid credentials', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: false, error: 'invalid_auth' }),
      });

      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Test',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'invalid-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('invalid_auth');
    });

    it('should handle channel not found', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: false, error: 'channel_not_found' }),
      });

      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C999999',
        text: 'Test',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
    });

    it('should handle rate limiting', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ ok: false, error: 'ratelimited' }),
      });

      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Test',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('ratelimited');
    });

    it('should handle malformed credentials JSON', async () => {
      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Test',
        credentials: {
          [CredentialType.SLACK_CRED]: 'not-json',
        },
      });

      // Should handle gracefully during credential parsing
      await expect(bubble.act()).rejects.toThrow();
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Test',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Network error');
    });
  });

  describe('Input Validation', () => {
    it('should validate URL format', () => {
      const bubble = new SlackBubble({
        operation: 'send_message',
        channel: 'C123456',
        text: 'Test',
        icon_url: 'not-a-url',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      // Should fail schema validation
      expect(() => SlackBubble.schema.parse(bubble.params)).toThrow();
    });

    it('should validate limit range', () => {
      expect(() => {
        new SlackBubble({
          operation: 'list_channels',
          limit: 2000, // Above maximum of 1000
          credentials: {
            [CredentialType.SLACK_CRED]: JSON.stringify({
              botToken: 'xoxb-test-token',
            }),
          },
        } as any);
      }).toThrow();
    });

    it('should validate file_path length', async () => {
      const longPath = 'a'.repeat(5000);
      const bubble = new SlackBubble({
        operation: 'upload_file',
        channel: 'C123456',
        file_path: longPath,
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(false);
      expect(result.error).toContain('too long');
    });
  });

  describe('Conversation History', () => {
    it('should get conversation history', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          messages: [
            {
              type: 'message',
              ts: '1234567890.123456',
              user: 'U123456',
              text: 'Test message',
            },
          ],
          has_more: false,
        }),
      });

      const bubble = new SlackBubble({
        operation: 'get_conversation_history',
        channel: 'C123456',
        limit: 10,
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.messages).toHaveLength(1);
      expect(result.messages[0].text).toBe('Test message');
    });

    it('should get thread replies', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          ok: true,
          messages: [
            {
              type: 'message',
              ts: '1234567890.123457',
              user: 'U123456',
              text: 'Thread reply',
            },
          ],
          has_more: false,
        }),
      });

      const bubble = new SlackBubble({
        operation: 'get_thread_replies',
        channel: 'C123456',
        ts: '1234567890.123456',
        credentials: {
          [CredentialType.SLACK_CRED]: JSON.stringify({
            botToken: 'xoxb-test-token',
          }),
        },
      });

      const result = await bubble.act();

      expect(result.success).toBe(true);
      expect(result.operation).toBe('get_thread_replies');
    });
  });
});
