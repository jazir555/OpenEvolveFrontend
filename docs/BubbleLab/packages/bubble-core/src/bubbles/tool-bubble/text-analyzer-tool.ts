import { ToolBubble } from '@bubblelab/bubble-core';
import { z } from 'zod';

/**
 * TextAnalyzerTool - Text analysis operations
 */
export class TextAnalyzerTool extends ToolBubble<TextAnalyzerParams, TextAnalyzerResult> {
  bubbleName = 'text-analyzer';
  type = 'tool';
  alias = 'text-analyzer';

  params = {
    timeout: z.number().int().positive().default(30000)
  };

  async execute(input: any): Promise<TextAnalyzerResult> {
    try {
      const result = await this.analyze(input);
      return { success: true, analysis: result };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async analyze(params: { text: string }): Promise<TextAnalyzerResult> {
    try {
      const words = params.text.split(/\s+/);
      const sentences = params.text.split(/[.!?]+/);
      const analysis = {
        wordCount: words.length,
        sentenceCount: sentences.length,
        characterCount: params.text.length,
        averageWordLength: words.join('').length / words.length,
        averageSentenceLength: words.length / sentences.length
      };
      return { success: true, analysis };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async extract(params: { text: string; entities: string[] }): Promise<TextAnalyzerResult> {
    try {
      const extracted = {};
      params.entities.forEach(entity => {
        const regex = new RegExp(`\\b(${entity})\\b`, 'gi');
        const matches = params.text.match(regex);
        extracted[entity] = matches ? matches.length : 0;
      });
      return { success: true, extracted };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  async sentiment(params: { text: string }): Promise<TextAnalyzerResult> {
    try {
      // Basic sentiment analysis
      const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful'];
      const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'poor'];
      const textLower = params.text.toLowerCase();

      let positiveCount = 0;
      let negativeCount = 0;

      positiveWords.forEach(word => {
        if (textLower.includes(word)) positiveCount++;
      });

      negativeWords.forEach(word => {
        if (textLower.includes(word)) negativeCount++;
      });

      const score = positiveCount - negativeCount;
      let sentiment = 'neutral';
      if (score > 0) sentiment = 'positive';
      if (score < 0) sentiment = 'negative';

      return { success: true, sentiment, score, confidence: Math.min(Math.abs(score) * 0.2, 1) };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

export interface TextAnalyzerParams {
  timeout?: number;
}

export interface TextAnalyzerResult {
  success: boolean;
  analysis?: any;
  extracted?: any;
  sentiment?: string;
  score?: number;
  confidence?: number;
  error?: string;
}
