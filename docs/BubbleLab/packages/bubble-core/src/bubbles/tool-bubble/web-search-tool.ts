import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * WebSearchTool - Production web search operations with multiple API support
 *
 * Supports:
 * - SerpAPI (Google, Bing, Yahoo, DuckDuckGo)
 * - Google Custom Search API
 * - Bing Search API
 * - DuckDuckGo (free, no API key needed)
 */
export class WebSearchTool extends ToolBubble<WebSearchParams, WebSearchResult> {
  bubbleName = 'web-search';
  type = 'tool';
  alias = 'web-search';

  private apiKey: string;
  private apiProvider: 'serpapi' | 'google' | 'bing' | 'duckduckgo';
  private baseUrl: string;

  params = {
    apiKey: z.string().optional(),
    apiProvider: z.enum(['serpapi', 'google', 'bing', 'duckduckgo']).default('duckduckgo'),
    timeout: z.number().int().positive().default(30000),
    maxResults: z.number().int().positive().max(100).default(10)
  };

  constructor(params: WebSearchParams = {}) {
    super(params);
    this.apiKey = params.apiKey || process.env.SERPAPI_API_KEY || process.env.GOOGLE_API_KEY || '';
    this.apiProvider = (params.apiProvider as any) || 'duckduckgo';
    this.baseUrl = this.getBaseUrl();
  }

  private getBaseUrl(): string {
    switch (this.apiProvider) {
      case 'serpapi':
        return 'https://serpapi.com/search';
      case 'google':
        return 'https://www.googleapis.com/customsearch/v1';
      case 'bing':
        return 'https://api.bing.microsoft.com/v7.0/search';
      case 'duckduckgo':
        return 'https://duckduckgo.com/html/';
      default:
        return 'https://duckduckgo.com/html/';
    }
  }

  async execute(input: any): Promise<WebSearchResult> {
    try {
      const query = input.query || input.q;
      if (!query) {
        throw new Error('Query is required');
      }

      const result = await this.search({
        query,
        num: input.maxResults || input.num || 10,
        start: input.start || 0
      });

      return { success: true, results: result.results };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        details: {
          provider: this.apiProvider,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  /**
   * Perform web search using configured provider
   */
  async search(params: {
    query: string;
    num?: number;
    start?: number;
    safe?: 'active' | 'off';
    filter?: string;
  }): Promise<WebSearchResult> {
    try {
      switch (this.apiProvider) {
        case 'serpapi':
          return await this.searchSerpAPI(params);
        case 'google':
          return await this.searchGoogle(params);
        case 'bing':
          return await this.searchBing(params);
        case 'duckduckgo':
          return await this.searchDuckDuckGo(params);
        default:
          return await this.searchDuckDuckGo(params);
      }
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        details: { provider: this.apiProvider }
      };
    }
  }

  /**
   * Search using SerpAPI (supports Google, Bing, Yahoo, etc.)
   */
  private async searchSerpAPI(params: any): Promise<WebSearchResult> {
    if (!this.apiKey) {
      throw new Error('SerpAPI key is required. Set SERPAPI_API_KEY environment variable.');
    }

    const url = new URL(this.baseUrl);
    url.searchParams.append('q', params.query);
    url.searchParams.append('api_key', this.apiKey);
    url.searchParams.append('engine', 'google');
    if (params.num) url.searchParams.append('num', params.num.toString());
    if (params.start) url.searchParams.append('start', params.start.toString());

    const response = await fetch(url.toString(), {
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`SerpAPI error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    return {
      success: true,
      results: {
        query: params.query,
        totalResults: data.search_information?.total_results || 0,
        searchTime: data.search_information?.time_taken || 0,
        results: (data.organic_results || []).map((r: any) => ({
          title: r.title,
          link: r.link,
          snippet: r.snippet,
          displayedLink: r.displayed_link
        }))
      },
      metadata: {
        provider: 'serpapi',
        timestamp: new Date().toISOString()
      }
    };
  }

  /**
   * Search using Google Custom Search API
   */
  private async searchGoogle(params: any): Promise<WebSearchResult> {
    if (!this.apiKey) {
      throw new Error('Google API key is required. Set GOOGLE_API_KEY environment variable.');
    }

    const cx = process.env.GOOGLE_SEARCH_ENGINE_ID;
    if (!cx) {
      throw new Error('Google Search Engine ID is required. Set GOOGLE_SEARCH_ENGINE_ID environment variable.');
    }

    const url = new URL(this.baseUrl);
    url.searchParams.append('key', this.apiKey);
    url.searchParams.append('cx', cx);
    url.searchParams.append('q', params.query);
    if (params.num) url.searchParams.append('num', params.num.toString());
    if (params.start) url.searchParams.append('start', params.start.toString());

    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error(`Google API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    return {
      success: true,
      results: {
        query: params.query,
        totalResults: data.searchInformation?.totalResults || 0,
        searchTime: data.searchInformation?.searchTime || 0,
        results: (data.items || []).map((r: any) => ({
          title: r.title,
          link: r.link,
          snippet: r.snippet,
          displayedLink: r.displayLink
        }))
      },
      metadata: {
        provider: 'google',
        timestamp: new Date().toISOString()
      }
    };
  }

  /**
   * Search using Bing Search API
   */
  private async searchBing(params: any): Promise<WebSearchResult> {
    if (!this.apiKey) {
      throw new Error('Bing API key is required. Set BING_API_KEY environment variable.');
    }

    const url = new URL(this.baseUrl);
    url.searchParams.append('q', params.query);
    if (params.num) url.searchParams.append('count', params.num.toString());
    if (params.start) url.searchParams.append('offset', params.start.toString());

    const response = await fetch(url.toString(), {
      headers: { 'Ocp-Apim-Subscription-Key': this.apiKey }
    });

    if (!response.ok) {
      throw new Error(`Bing API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    return {
      success: true,
      results: {
        query: params.query,
        totalResults: data.totalEstimatedMatches || 0,
        results: (data.webPages?.value || []).map((r: any) => ({
          title: r.name,
          link: r.url,
          snippet: r.snippet,
          displayedLink: r.displayUrl
        }))
      },
      metadata: {
        provider: 'bing',
        timestamp: new Date().toISOString()
      }
    };
  }

  /**
   * Search using DuckDuckGo (free, no API key required)
   */
  private async searchDuckDuckGo(params: any): Promise<WebSearchResult> {
    const url = new URL(this.baseUrl);
    url.searchParams.append('q', params.query);

    const response = await fetch(url.toString(), {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    });

    if (!response.ok) {
      throw new Error(`DuckDuckGo error: ${response.status} ${response.statusText}`);
    }

    const html = await response.text();
    const results = this.parseDuckDuckGoHTML(html);

    return {
      success: true,
      results: {
        query: params.query,
        totalResults: results.length,
        results: results.slice(0, params.num || 10)
      },
      metadata: {
        provider: 'duckduckgo',
        timestamp: new Date().toISOString()
      }
    };
  }

  /**
   * Parse DuckDuckGo HTML response
   */
  private parseDuckDuckGoHTML(html: string): Array<{
    title: string;
    link: string;
    snippet: string;
  }> {
    const results: Array<{ title: string; link: string; snippet: string }> = [];

    // Simple regex-based parsing for result extraction
    const resultRegex = /<a[^>]*class="result__a"[^>]*>(.*?)<\/a>.*?<a[^>]*class="result__url"[^>]*>(.*?)<\/a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)<\/a>/gis;
    let match;

    while ((match = resultRegex.exec(html)) !== null) {
      results.push({
        title: this.stripHTML(match[1]),
        link: this.stripHTML(match[2]),
        snippet: this.stripHTML(match[3])
      });
    }

    return results;
  }

  private stripHTML(str: string): string {
    return str.replace(/<[^>]*>/g, '').trim();
  }

  /**
   * Advanced search with filters
   */
  async advancedSearch(params: {
    query: string;
    site?: string;
    fileType?: string;
    dateRange?: { start: string; end: string };
    safe?: 'active' | 'off';
  }): Promise<WebSearchResult> {
    let query = params.query;

    if (params.site) {
      query += ` site:${params.site}`;
    }

    if (params.fileType) {
      query += ` filetype:${params.fileType}`;
    }

    return await this.search({ query, num: params.num });
  }

  /**
   * Search news articles
   */
  async searchNews(params: { query: string; num?: number }): Promise<WebSearchResult> {
    if (this.apiProvider === 'serpapi') {
      const url = new URL('https://serpapi.com/search');
      url.searchParams.append('q', params.query);
      url.searchParams.append('api_key', this.apiKey);
      url.searchParams.append('engine', 'google_news');
      if (params.num) url.searchParams.append('num', params.num.toString());

      const response = await fetch(url.toString());
      const data = await response.json();

      return {
        success: true,
        results: {
          query: params.query,
          results: (data.news_results || []).map((r: any) => ({
            title: r.title,
            link: r.link,
            snippet: r.snippet,
            source: r.source,
            date: r.date
          }))
        },
        metadata: { provider: 'serpapi', type: 'news' }
      };
    }

    // Fallback to regular search for other providers
    return await this.search({ query: `${params.query} news`, num: params.num });
  }

  /**
   * Search images
   */
  async searchImages(params: { query: string; num?: number }): Promise<WebSearchResult> {
    if (this.apiProvider === 'serpapi') {
      const url = new URL('https://serpapi.com/search');
      url.searchParams.append('q', params.query);
      url.searchParams.append('api_key', this.apiKey);
      url.searchParams.append('engine', 'google_images');
      if (params.num) url.searchParams.append('num', params.num.toString());

      const response = await fetch(url.toString());
      const data = await response.json();

      return {
        success: true,
        results: {
          query: params.query,
          results: (data.images_results || []).map((r: any) => ({
            title: r.title,
            link: r.link,
            thumbnail: r.thumbnail,
            source: r.source
          }))
        },
        metadata: { provider: 'serpapi', type: 'images' }
      };
    }

    throw new Error('Image search only supported with SerpAPI provider');
  }
}

export interface WebSearchParams {
  apiKey?: string;
  apiProvider?: 'serpapi' | 'google' | 'bing' | 'duckduckgo';
  timeout?: number;
  maxResults?: number;
}

export interface WebSearchResult {
  success: boolean;
  results?: any;
  error?: string;
  details?: any;
}
