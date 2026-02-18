// Template for Website Lead Generation (Firecrawl, AI Agent, Google Drive)

import { RECOMMENDED_MODELS } from '@bubblelab/shared-schemas';

export const templateCode = `import {
  BubbleFlow,
  FirecrawlBubble,
  GoogleDriveBubble,
  AIAgentBubble,
  type WebhookEvent,
} from '@bubblelab/bubble-core';

export interface Lead {
  company: string;
  founder: string;
  role?: string;
  url?: string;
  reason: string;
  outreachHook: string;
}

export interface Output {
  leadCount: number;
  message: string;
  leads: Lead[];
  reportLink?: string;
}

export interface WebsiteLeadPayload extends WebhookEvent {
  /**
   * The YC directory or target listing URL to scrape.
   * @canBeFile false
   */
  targetUrl?: string;
  /**
   * The positioning document that describes your product and ICP.
   * @canBeFile true
   */
  positioningDocument: string;
  /**
   * Maximum number of leads to return.
   * @canBeFile false
   */
  maxLeads?: number;
  /**
   * Optional Google Drive folder ID for the report file.
   * @canBeFile false
   */
  outputFolderId?: string;
}

export class WebsiteLeadGeneration extends BubbleFlow<'webhook/http'> {
  private async scrapeDirectory(url: string): Promise<string> {
    const firecrawl = new FirecrawlBubble({
      operation: 'scrape',
      url,
      formats: ['markdown', 'links'],
      onlyMainContent: true,
    });

    const result = await firecrawl.action();
    if (!result.success || !result.data?.result) {
      throw new Error(result.error || 'Failed to scrape directory');
    }

    const markdown = result.data.result.markdown || '';
    const links = result.data.result.links || [];
    return \`\${markdown}\n\nLinks:\n\${links.join('\n')}\`;
  }

  private async generateLeads(
    positioningDocument: string,
    directoryContent: string,
    maxLeads: number
  ): Promise<Lead[]> {
    const systemPrompt = \`
You are a lead researcher. Given a positioning document and a YC directory page scrape,
identify founders who match the ideal customer profile. Return JSON only.
Each lead must include: company, founder, role (optional), url (optional), reason, outreachHook.
Limit to \${maxLeads} leads.
\`;

    const message = \`
Positioning Document:\n\${positioningDocument}\n\nDirectory Content:\n\${directoryContent}
\`;

    const agent = new AIAgentBubble({
      message,
      systemPrompt,
      model: { model: '${RECOMMENDED_MODELS.FAST}', jsonMode: true },
      tools: [],
    });

    const result = await agent.action();
    if (!result.success || !result.data?.response) {
      throw new Error(result.error || 'Lead analysis failed');
    }

    const parsed = JSON.parse(result.data.response);
    if (!Array.isArray(parsed)) {
      throw new Error('AI response is not a valid array');
    }

    return parsed as Lead[];
  }

  private async uploadReport(
    leads: Lead[],
    folderId?: string
  ): Promise<{ link?: string }> {
    const reportContent = JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        leads,
      },
      null,
      2
    );

    const drive = new GoogleDriveBubble({
      operation: 'uploadFile',
      fileName: \`lead-report-\${Date.now()}.json\`,
      content: reportContent,
      mimeType: 'application/json',
      parents: folderId ? [folderId] : undefined,
    });

    const result = await drive.action();
    if (!result.success || !result.data?.file) {
      throw new Error(result.data?.error || 'Failed to upload report');
    }

    return {
      link: result.data.file.webViewLink || result.data.file.webContentLink,
    };
  }

  async handle(payload: WebsiteLeadPayload): Promise<Output> {
    const {
      targetUrl = 'https://www.ycombinator.com/companies?batch=S25',
      positioningDocument,
      maxLeads = 15,
      outputFolderId,
    } = payload;

    if (!positioningDocument) {
      throw new Error('positioningDocument is required');
    }

    const directoryContent = await this.scrapeDirectory(targetUrl);
    const leads = await this.generateLeads(
      positioningDocument,
      directoryContent,
      maxLeads
    );

    const report = await this.uploadReport(leads, outputFolderId);

    return {
      leadCount: leads.length,
      message: leads.length
        ? 'Lead report generated successfully.'
        : 'No leads found from the directory.',
      leads,
      reportLink: report.link,
    };
  }
}`;

export const metadata = {
  inputsSchema: JSON.stringify({
    type: 'object',
    properties: {
      targetUrl: {
        type: 'string',
        description: 'YC directory URL or target listing page to scrape.',
      },
      positioningDocument: {
        type: 'string',
        description: 'Positioning document defining your ideal customer profile.',
      },
      maxLeads: {
        type: 'number',
        description: 'Maximum leads to return (default: 15).',
      },
      outputFolderId: {
        type: 'string',
        description: 'Optional Google Drive folder ID for the report file.',
      },
    },
    required: ['positioningDocument'],
  }),
  requiredCredentials: {
    firecrawl: ['read'],
    'google-drive': ['write'],
    'ai-agent': ['generate'],
  },
};
