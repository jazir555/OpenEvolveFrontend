import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
import sgMail from '@sendgrid/mail';

/**
 * SendGrid Bubble - Email Service Bubble Implementation
 *
 * Full production implementation with 8 operations:
 * 1. sendEmail - Send a single email
 * 2. sendBulkEmails - Send multiple emails in bulk
 * 3. sendTemplate - Send email using a template
 * 4. addContact - Add a contact to SendGrid
 * 5. getContact - Retrieve contact information
 * 6. deleteContact - Delete a contact
 * 7. createList - Create a contact list
 * 8. addToList - Add contacts to a list
 */

// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================

const SendEmailParamsSchema = z.object({
  operation: z.literal('sendEmail'),
  to: z.union([z.string().email(), z.array(z.string().email())]).describe('Recipient email address(es)'),
  from: z.string().email().describe('Sender email address'),
  subject: z.string().min(1, 'Subject is required'),
  text: z.string().optional().describe('Plain text content'),
  html: z.string().optional().describe('HTML content'),
  attachments: z
    .array(
      z.object({
        filename: z.string(),
        content: z.string().describe('Base64 encoded content'),
        type: z.string().optional(),
        disposition: z.string().optional(),
      })
    )
    .optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SendBulkEmailsParamsSchema = z.object({
  operation: z.literal('sendBulkEmails'),
  messages: z
    .array(
      z.object({
        to: z.union([z.string().email(), z.array(z.string().email())]),
        from: z.string().email(),
        subject: z.string().min(1),
        text: z.string().optional(),
        html: z.string().optional(),
        attachments: z.array(z.unknown()).optional(),
      })
    )
    .min(1, 'At least one message is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const SendTemplateParamsSchema = z.object({
  operation: z.literal('sendTemplate'),
  to: z.union([z.string().email(), z.array(z.string().email())]),
  from: z.string().email(),
  templateId: z.string().min(1, 'Template ID is required'),
  dynamicData: z.record(z.unknown()).optional().describe('Template dynamic data'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const AddContactParamsSchema = z.object({
  operation: z.literal('addContact'),
  email: z.string().email().describe('Contact email address'),
  firstName: z.string().optional(),
  lastName: z.string().optional(),
  customFields: z.record(z.unknown()).optional().describe('Custom field data'),
  listIds: z.array(z.string()).optional().describe('List IDs to add contact to'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const GetContactParamsSchema = z.object({
  operation: z.literal('getContact'),
  email: z.string().email().describe('Contact email address'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const DeleteContactParamsSchema = z.object({
  operation: z.literal('deleteContact'),
  email: z.string().email().describe('Contact email address'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const CreateListParamsSchema = z.object({
  operation: z.literal('createList'),
  name: z.string().min(1, 'List name is required'),
  description: z.string().optional(),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

const AddToListParamsSchema = z.object({
  operation: z.literal('addToList'),
  listId: z.string().min(1, 'List ID is required'),
  emails: z.array(z.string().email()).min(1, 'At least one email is required'),
  credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
});

// Union of all parameter schemas
const SendGridBubbleParamsSchema = z.discriminatedUnion('operation', [
  SendEmailParamsSchema,
  SendBulkEmailsParamsSchema,
  SendTemplateParamsSchema,
  AddContactParamsSchema,
  GetContactParamsSchema,
  DeleteContactParamsSchema,
  CreateListParamsSchema,
  AddToListParamsSchema,
]);

type SendGridBubbleParams = z.input<typeof SendGridBubbleParamsSchema>;

// Result schema
const SendGridBubbleResultSchema = z.object({
  success: z.boolean(),
  data: z.unknown().describe('Operation result data'),
  error: z.string(),
  meta: z.object({
    operation: z.string(),
    messageId: z.string().optional(),
  }),
});

type SendGridBubbleResult = z.output<typeof SendGridBubbleResultSchema>;

// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================

export class SendGridBubble extends ServiceBubble<
  SendGridBubbleParams,
  SendGridBubbleResult
> {
  static readonly service = 'sendgrid';
  static readonly authType = 'apikey' as const;
  static readonly bubbleName: BubbleName = 'sendgrid';
  static readonly type = 'service' as const;
  static readonly schema = SendGridBubbleParamsSchema;
  static readonly resultSchema = SendGridBubbleResultSchema;
  static readonly shortDescription =
    'Email delivery and marketing automation platform';
  static readonly longDescription = `
    SendGrid Bubble for transactional and marketing emails.

    Features:
    - Send single and bulk emails
    - Template-based emails with dynamic content
    - Contact management and segmentation
    - Contact lists for campaigns
    - Attachments and HTML content
    - High deliverability rates

    Use cases:
    - Transactional emails (passwords, notifications)
    - Marketing campaigns
    - Newsletter distribution
    - User onboarding emails
    - Automated email sequences
  `;
  static readonly alias = 'email';

  constructor(
    params: SendGridBubbleParams,
    context?: BubbleContext,
    instanceId?: string
  ) {
    super(params, context, instanceId);
  }

  protected getCredentialType(): CredentialType {
    return CredentialType.SENDGRID_CRED;
  }

  protected chooseCredential(): string | undefined {
    const credentials = this.params.credentials;
    if (!credentials || typeof credentials !== 'object') {
      throw new Error('SendGrid credentials are required');
    }
    return credentials[CredentialType.SENDGRID_CRED];
  }

  public async testCredential(): Promise<boolean> {
    try {
      const apiKey = this.chooseCredential();
      if (!apiKey) {
        return false;
      }

      sgMail.setApiKey(apiKey);
      // SendGrid doesn't have a simple test method, so we just verify the key format
      return apiKey.startsWith('SG.');
    } catch (error) {
      console.error('[SendGrid] Credential test failed:', error);
      return false;
    }
  }

  private getApiKey(): string {
    const apiKey = this.chooseCredential();
    if (!apiKey) {
      throw new Error('SendGrid API key not found');
    }
    sgMail.setApiKey(apiKey);
    return apiKey;
  }

  protected async performAction(context?: BubbleContext): Promise<SendGridBubbleResult> {
    void context;

    try {
      this.getApiKey();
      const operation = this.params.operation;
      let result: any;

      console.log(`[SendGrid] Executing operation: ${operation}`);

      switch (operation) {
        case 'sendEmail':
          result = await this.sendEmail();
          break;

        case 'sendBulkEmails':
          result = await this.sendBulkEmails();
          break;

        case 'sendTemplate':
          result = await this.sendTemplate();
          break;

        case 'addContact':
          result = await this.addContact();
          break;

        case 'getContact':
          result = await this.getContact();
          break;

        case 'deleteContact':
          result = await this.deleteContact();
          break;

        case 'createList':
          result = await this.createList();
          break;

        case 'addToList':
          result = await this.addToList();
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
          messageId: result?.messageId,
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[SendGrid] Operation failed:`, errorMessage);

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

  private async sendEmail(): Promise<any> {
    const params = this.params as z.output<typeof SendEmailParamsSchema>;

    const msg: any = {
      to: params.to,
      from: params.from,
      subject: params.subject,
    };

    if (params.text) {
      msg.text = params.text;
    }

    if (params.html) {
      msg.html = params.html;
    }

    if (params.attachments) {
      msg.attachments = params.attachments;
    }

    const response = await sgMail.send(msg);

    console.log(`[SendGrid] Email sent to ${params.to}: ${response[0].statusCode}`);

    return {
      statusCode: response[0].statusCode,
      messageId: response[0].headers['x-message-id'],
      to: params.to,
      subject: params.subject,
    };
  }

  private async sendBulkEmails(): Promise<any> {
    const params = this.params as z.output<typeof SendBulkEmailsParamsSchema>;

    const responses = await sgMail.send(params.messages as any);

    console.log(`[SendGrid] Sent ${params.messages.length} bulk emails`);

    return {
      totalSent: params.messages.length,
      results: responses.map((response: any, index: number) => ({
        to: params.messages[index].to,
        statusCode: response.statusCode,
        messageId: response.headers['x-message-id'],
      })),
    };
  }

  private async sendTemplate(): Promise<any> {
    const params = this.params as z.output<typeof SendTemplateParamsSchema>;

    const msg: any = {
      to: params.to,
      from: params.from,
      templateId: params.templateId,
    };

    if (params.dynamicData) {
      msg.dynamicTemplateData = params.dynamicData;
    }

    const response = await sgMail.send(msg);

    console.log(`[SendGrid] Template email sent to ${params.to}: ${response[0].statusCode}`);

    return {
      statusCode: response[0].statusCode,
      messageId: response[0].headers['x-message-id'],
      to: params.to,
      templateId: params.templateId,
    };
  }

  private async addContact(): Promise<any> {
    const params = this.params as z.output<typeof AddContactParamsSchema>;
    const apiKey = this.getApiKey();

    // SendGrid Marketing API requires different base URL
    const contactData: any = {
      contacts: [
        {
          email: params.email,
          first_name: params.firstName,
          last_name: params.lastName,
        },
      ],
    };

    if (params.customFields) {
      contactData.contacts[0].custom_fields = params.customFields;
    }

    if (params.listIds && params.listIds.length > 0) {
      contactData.list_ids = params.listIds;
    }

    const response = await fetch('https://api.sendgrid.com/v3/marketing/contacts', {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(contactData),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.errors?.[0]?.message || 'Failed to add contact');
    }

    console.log(`[SendGrid] Contact added: ${params.email}`);

    return {
      email: params.email,
      job_id: result.job_id,
      status: 'created',
    };
  }

  private async getContact(): Promise<any> {
    const params = this.params as z.output<typeof GetContactParamsSchema>;
    const apiKey = this.getApiKey();

    const response = await fetch(
      `https://api.sendgrid.com/v3/marketing/contacts/emails/${params.email}`,
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.errors?.[0]?.message || 'Failed to get contact');
    }

    console.log(`[SendGrid] Contact retrieved: ${params.email}`);

    return {
      contact: result.contact,
      email: params.email,
    };
  }

  private async deleteContact(): Promise<any> {
    const params = this.params as z.output<typeof DeleteContactParamsSchema>;
    const apiKey = this.getApiKey();

    const response = await fetch('https://api.sendgrid.com/v3/marketing/contacts', {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        emails: [params.email],
      }),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.errors?.[0]?.message || 'Failed to delete contact');
    }

    console.log(`[SendGrid] Contact deleted: ${params.email}`);

    return {
      email: params.email,
      status: 'deleted',
      jobId: result.job_id,
    };
  }

  private async createList(): Promise<any> {
    const params = this.params as z.output<typeof CreateListParamsSchema>;
    const apiKey = this.getApiKey();

    const response = await fetch('https://api.sendgrid.com/v3/marketing/lists', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: params.name,
        description: params.description,
      }),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.errors?.[0]?.message || 'Failed to create list');
    }

    console.log(`[SendGrid] List created: ${params.name}`);

    return {
      id: result.id,
      name: result.name,
      description: result.description,
      contactCount: result.contact_count,
    };
  }

  private async addToList(): Promise<any> {
    const params = this.params as z.output<typeof AddToListParamsSchema>;
    const apiKey = this.getApiKey();

    const contactData: any = {
      list_ids: [params.listId],
      contacts: params.emails.map((email) => ({ email })),
    };

    const response = await fetch('https://api.sendgrid.com/v3/marketing/contacts', {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(contactData),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.errors?.[0]?.message || 'Failed to add contacts to list');
    }

    console.log(`[SendGrid] Added ${params.emails.length} contacts to list ${params.listId}`);

    return {
      listId: params.listId,
      emailCount: params.emails.length,
      jobId: result.job_id,
      status: 'added',
    };
  }
}

