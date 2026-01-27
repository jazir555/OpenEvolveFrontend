import { ServiceBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * ApifyBubble - Web scraping and automation via Apify actors
 */
export class ApifyBubble extends ServiceBubble<ApifyParams, ApifyResult> {
  bubbleName = 'apify';
  type = 'service';
  alias = 'Apify';
  credentialType = 'apify_api_key';

  params = {
    token: z.string().min(1),
    timeout: z.number().int().positive().default(300000)
  };

  private client: any = null;

  async connect() {
    const { ApifyClient } = await import('apify');
    this.client = new ApifyClient({ token: this.params.token });
  }

  async runActor(params: { actorId: string; input: any }): Promise<ApifyResult> {
    try {
      const result = await this.client.actor(params.actorId).call(input);
      return { success: true, run: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getActor(params: { actorId: string }): Promise<ApifyResult> {
    try {
      const result = await this.client.actor(params.actorId).get();
      return { success: true, actor: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getRun(params: { runId: string }): Promise<ApifyResult> {
    try {
      const result = await this.client.run(params.runId).get();
      return { success: true, run: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getDataset(params: { datasetId: string; limit?: number; offset?: number }): Promise<ApifyResult> {
    try {
      const result = await this.client.dataset(params.datasetId).listItems({
        limit: params.limit || 100,
        offset: params.offset || 0
      });
      return { success: true, items: result.items, total: result.total };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async getDatasetItems(params: { datasetId: string; limit?: number; offset?: number; clean?: boolean }): Promise<ApifyResult> {
    try {
      const result = await this.client.dataset(params.datasetId).listItems({
        limit: params.limit || 100,
        offset: params.offset || 0,
        clean: params.clean !== false
      });
      return { success: true, items: result.items };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async webScrape(params: { url: string; proxy?: boolean }): Promise<ApifyResult> {
    try {
      const result = await this.client.actor('apify/web-scraper').call({
        startUrls: [{ url: params.url }],
        useProxy: params.proxy !== false
      });
      const dataset = await this.client.dataset(result.defaultDatasetId).listItems();
      return { success: true, items: dataset.items };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async puppeteerScrape(params: { url: string; proxy?: boolean }): Promise<ApifyResult> {
    try {
      const result = await this.client.actor('apify/puppeteer-scraper').call({
        startUrls: [{ url: params.url }],
        useProxy: params.proxy !== false
      });
      const dataset = await this.client.dataset(result.defaultDatasetId).listItems();
      return { success: true, items: dataset.items };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async cheerioScrape(params: { url: string; proxy?: boolean }): Promise<ApifyResult> {
    try {
      const result = await this.client.actor('apify/cheerio-scraper').call({
        startUrls: [{ url: params.url }],
        useProxy: params.proxy !== false
      });
      const dataset = await this.client.dataset(result.defaultDatasetId).listItems();
      return { success: true, items: dataset.items };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface ApifyParams {
  token: string;
  timeout?: number;
}

export interface ApifyResult {
  success: boolean;
  run?: any;
  actor?: any;
  items?: any[];
  total?: number;
  error?: string;
}
