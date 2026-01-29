/**
 * TEXT ANALYZER TOOL
 *
 * A tool bubble for comprehensive text analysis including sentiment analysis,
 * keyword extraction, readability scoring, and more.
 *
 * Features:
 * - Sentiment analysis (positive/negative/neutral)
 * - Keyword extraction
 * - Readability scoring
 * - Word frequency analysis
 * - Language detection
 * - Named entity recognition
 * - Text summarization
 */

import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';

/**
 * Text analyzer parameters schema
 */
const TextAnalyzerToolParamsSchema = z.object({
  // Input text
  text: z
    .string()
    .describe('Text to analyze'),

  // Analysis operations
  operations: z
    .array(z.enum([
      'sentiment',
      'keywords',
      'readability',
      'frequency',
      'language',
      'entities',
      'summary',
      'statistics'
    ]))
    .default(['sentiment', 'keywords', 'readability', 'statistics'])
    .describe('Analysis operations to perform'),

  // Options
  maxKeywords: z
    .number()
    .int()
    .min(1)
    .max(100)
    .default(10)
    .describe('Maximum number of keywords to extract'),

  minKeywordLength: z
    .number()
    .int()
    .min(1)
    .default(3)
    .describe('Minimum length for keywords'),

  summaryLength: z
    .number()
    .int()
    .min(1)
    .max(1000)
    .default(3)
    .describe('Number of sentences for summary'),

  language: z
    .string()
    .default('en')
    .describe('Language code for text (default: auto-detect)'),

  // Stop words
  removeStopWords: z
    .boolean()
    .default(true)
    .describe('Remove stop words from analysis'),

  customStopWords: z
    .array(z.string())
    .optional()
    .describe('Custom stop words to exclude'),

  // Credentials
  credentials: z
    .record(z.nativeEnum(CredentialType), z.string())
    .optional()
    .describe('Credentials for AI-powered analysis'),
});

/**
 * Text analyzer result schema
 */
const TextAnalyzerToolResultSchema = z.object({
  // Result
  success: z.boolean().describe('Whether the analysis was successful'),

  // Sentiment analysis
  sentiment: z
    .object({
      score: z.number().describe('Sentiment score (-1 to 1)'),
      label: z.string().describe('Sentiment label (positive/negative/neutral)'),
      confidence: z.number().describe('Confidence score (0 to 1)'),
    })
    .optional()
    .describe('Sentiment analysis results'),

  // Keywords
  keywords: z
    .array(z.object({
      word: z.string().describe('Keyword'),
      score: z.number().describe('Relevance score'),
      frequency: z.number().describe('Frequency in text'),
    }))
    .optional()
    .describe('Extracted keywords'),

  // Readability
  readability: z
    .object({
      fleschScore: z.number().describe('Flesch reading ease score'),
      gradeLevel: z.number().describe('Grade level'),
      label: z.string().describe('Readability label'),
    })
    .optional()
    .describe('Readability metrics'),

  // Word frequency
  frequency: z
    .array(z.object({
      word: z.string().describe('Word'),
      count: z.number().describe('Count'),
      percentage: z.number().describe('Percentage of total'),
    }))
    .optional()
    .describe('Word frequency analysis'),

  // Language detection
  language: z
    .object({
      code: z.string().describe('Language code (e.g., en, es, fr)'),
      name: z.string().describe('Language name'),
      confidence: z.number().describe('Confidence score'),
    })
    .optional()
    .describe('Detected language'),

  // Named entities
  entities: z
    .array(z.object({
      text: z.string().describe('Entity text'),
      type: z.string().describe('Entity type (PERSON, ORG, LOC, etc.)'),
      confidence: z.number().describe('Confidence score'),
    }))
    .optional()
    .describe('Named entities'),

  // Summary
  summary: z
    .string()
    .optional()
    .describe('Text summary'),

  // Statistics
  statistics: z
    .object({
      characterCount: z.number(),
      wordCount: z.number(),
      sentenceCount: z.number(),
      paragraphCount: z.number(),
      averageWordLength: z.number(),
      averageSentenceLength: z.number(),
      processingTime: z.number(),
    })
    .describe('Text statistics'),

  error: z.string().describe('Error message if analysis failed'),
});

// Type definitions
type TextAnalyzerToolParams = z.output<typeof TextAnalyzerToolParamsSchema>;
type TextAnalyzerToolResult = z.output<typeof TextAnalyzerToolResultSchema>;
type TextAnalyzerToolParamsInput = z.input<typeof TextAnalyzerToolParamsSchema>;

/**
 * Common English stop words
 */
const ENGLISH_STOP_WORDS = new Set([
  'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
  'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
  'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
  'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
  'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
  'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having',
  'can', 'could', 'should', 'would', 'may', 'might', 'must',
]);

/**
 * Positive and negative words for sentiment analysis
 */
const POSITIVE_WORDS = new Set([
  'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
  'love', 'like', 'happy', 'joy', 'pleased', 'satisfied', 'delighted',
  'best', 'better', 'awesome', 'perfect', 'outstanding', 'superb',
]);

const NEGATIVE_WORDS = new Set([
  'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
  'sad', 'disappointed', 'unsatisfied', 'worst', 'poor', 'inferior',
  'negative', 'failure', 'fail', 'problem', 'issue', 'wrong',
]);

/**
 * Text Analyzer Tool
 * Comprehensive text analysis with multiple metrics
 */
export class TextAnalyzerTool extends ToolBubble<
  TextAnalyzerToolParams,
  TextAnalyzerToolResult
> {
  /**
   * REQUIRED STATIC METADATA
   */
  static readonly type = 'tool' as const;
  static readonly bubbleName: BubbleName = 'text-analyzer-tool';
  static readonly schema = TextAnalyzerToolParamsSchema;
  static readonly resultSchema = TextAnalyzerToolResultSchema;
  static readonly shortDescription =
    'Analyze text for sentiment, keywords, readability, and more';
  static readonly longDescription = `
    A comprehensive text analysis tool providing multiple NLP capabilities.

    Features:
    - SENTIMENT: Analyze emotional tone (positive/negative/neutral)
    - KEYWORDS: Extract important keywords and phrases
    - READABILITY: Score text complexity and reading level
    - FREQUENCY: Analyze word frequency distribution
    - LANGUAGE: Detect text language
    - ENTITIES: Extract named entities (people, places, orgs)
    - SUMMARY: Generate text summary
    - STATISTICS: Basic text metrics (counts, averages)

    Sentiment Analysis:
    - Score from -1 (very negative) to +1 (very positive)
    - Label: positive, negative, or neutral
    - Confidence score based on word matches

    Keyword Extraction:
    - Extracts most relevant words
    - TF-IDF-like scoring
    - Configurable minimum word length
    - Stop word removal

    Readability Metrics:
    - Flesch Reading Ease score
    - Grade level estimation
    - Label (easy, moderate, difficult)

    Word Frequency:
    - Count word occurrences
    - Calculate percentages
    - Sort by frequency

    Use cases:
    - Content analysis
    - Social media monitoring
    - Customer feedback analysis
    - SEO optimization
    - Text quality assessment
    - Research and analysis

    Note: This is a rule-based implementation.
    For production use, consider using ML libraries like:
    - natural (Node.js NLP)
    - sentiment (sentiment analysis)
    - compromise (lightweight NLP)
  `;
  static readonly alias = 'analyze-text';

  constructor(
    params: TextAnalyzerToolParamsInput,
    context?: BubbleContext
  ) {
    super(params, context);
  }

  /**
   * Main action method - performs text analysis
   */
  async performAction(
    context?: BubbleContext
  ): Promise<TextAnalyzerToolResult> {
    void context; // Context available but not currently used
    const startTime = Date.now();

    try {
      console.log('[TextAnalyzerTool] Starting text analysis');

      const text = this.params.text.trim();

      if (!text) {
        throw new Error('Text is required for analysis');
      }

      const result: TextAnalyzerToolResult = {
        success: true,
        statistics: {
          characterCount: 0,
          wordCount: 0,
          sentenceCount: 0,
          paragraphCount: 0,
          averageWordLength: 0,
          averageSentenceLength: 0,
          processingTime: 0,
        },
        error: '',
      };

      // Perform requested operations
      for (const operation of this.params.operations) {
        switch (operation) {
          case 'sentiment':
            result.sentiment = this.analyzeSentiment(text);
            break;
          case 'keywords':
            result.keywords = this.extractKeywords(text);
            break;
          case 'readability':
            result.readability = this.analyzeReadability(text);
            break;
          case 'frequency':
            result.frequency = this.analyzeFrequency(text);
            break;
          case 'language':
            result.language = this.detectLanguage(text);
            break;
          case 'statistics':
            result.statistics = this.calculateStatistics(text);
            break;
          case 'summary':
            result.summary = this.generateSummary(text);
            break;
          case 'entities':
            result.entities = this.extractEntities(text);
            break;
        }
      }

      // Add processing time to statistics
      if (result.statistics) {
        result.statistics.processingTime = Date.now() - startTime;
      }

      console.log('[TextAnalyzerTool] Analysis completed');

      return result;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';

      console.error(`[TextAnalyzerTool] Analysis failed: ${errorMessage}`);

      return {
        success: false,
        statistics: {
          characterCount: 0,
          wordCount: 0,
          sentenceCount: 0,
          paragraphCount: 0,
          averageWordLength: 0,
          averageSentenceLength: 0,
          processingTime: Date.now() - startTime,
        },
        error: errorMessage,
      };
    }
  }

  /**
   * Analyze sentiment
   */
  private analyzeSentiment(text: string): {
    score: number;
    label: string;
    confidence: number;
  } {
    const words = this.tokenize(text);
    let positiveCount = 0;
    let negativeCount = 0;

    for (const word of words) {
      const lower = word.toLowerCase();
      if (POSITIVE_WORDS.has(lower)) {
        positiveCount++;
      } else if (NEGATIVE_WORDS.has(lower)) {
        negativeCount++;
      }
    }

    const total = positiveCount + negativeCount;
    const score = total > 0 ? (positiveCount - negativeCount) / total : 0;

    let label: string;
    if (score > 0.2) {
      label = 'positive';
    } else if (score < -0.2) {
      label = 'negative';
    } else {
      label = 'neutral';
    }

    const confidence = total > 0 ? Math.min(1, total / words.length * 5) : 0;

    console.log(`[TextAnalyzerTool] Sentiment: ${label} (${score.toFixed(2)})`);

    return { score, label, confidence };
  }

  /**
   * Extract keywords
   */
  private extractKeywords(text: string): Array<{
    word: string;
    score: number;
    frequency: number;
  }> {
    const words = this.tokenize(text);
    const frequency = new Map<string, number>();

    // Count word frequencies
    for (const word of words) {
      const lower = word.toLowerCase();

      // Skip stop words and short words
      if (lower.length < this.params.minKeywordLength) {
        continue;
      }

      if (this.params.removeStopWords && ENGLISH_STOP_WORDS.has(lower)) {
        continue;
      }

      if (this.params.customStopWords && this.params.customStopWords.includes(lower)) {
        continue;
      }

      frequency.set(lower, (frequency.get(lower) || 0) + 1);
    }

    // Calculate scores (simple TF-like scoring)
    const keywords = Array.from(frequency.entries()).map(([word, count]) => ({
      word,
      score: count / words.length,
      frequency: count,
    }));

    // Sort by score and return top N
    keywords.sort((a, b) => b.score - a.score);

    return keywords.slice(0, this.params.maxKeywords);
  }

  /**
   * Analyze readability
   */
  private analyzeReadability(text: string): {
    fleschScore: number;
    gradeLevel: number;
    label: string;
  } {
    const words = this.tokenize(text);
    const sentences = this.splitSentences(text);
    const syllables = this.countSyllables(text);

    const totalWords = words.length;
    const totalSentences = sentences.length;
    const totalSyllables = syllables;

    // Flesch Reading Ease score
    const fleschScore = 206.835 -
      1.015 * (totalWords / totalSentences) -
      84.6 * (totalSyllables / totalWords);

    // Grade level (approximate)
    const gradeLevel = 0.39 * (totalWords / totalSentences) +
      11.8 * (totalSyllables / totalWords) -
      15.59;

    let label: string;
    if (fleschScore >= 90) {
      label = 'Very Easy (5th grade)';
    } else if (fleschScore >= 80) {
      label = 'Easy (6th grade)';
    } else if (fleschScore >= 70) {
      label = 'Fairly Easy (7th grade)';
    } else if (fleschScore >= 60) {
      label = 'Standard (8th-9th grade)';
    } else if (fleschScore >= 50) {
      label = 'Fairly Difficult (10th-12th grade)';
    } else if (fleschScore >= 30) {
      label = 'Difficult (College)';
    } else {
      label = 'Very Difficult (Graduate school)';
    }

    console.log(`[TextAnalyzerTool] Readability: ${label} (${fleschScore.toFixed(1)})`);

    return {
      fleschScore: Math.max(0, Math.min(100, fleschScore)),
      gradeLevel: Math.max(0, gradeLevel),
      label,
    };
  }

  /**
   * Analyze word frequency
   */
  private analyzeFrequency(text: string): Array<{
    word: string;
    count: number;
    percentage: number;
  }> {
    const words = this.tokenize(text);
    const frequency = new Map<string, number>();

    for (const word of words) {
      const lower = word.toLowerCase();
      frequency.set(lower, (frequency.get(lower) || 0) + 1);
    }

    const result = Array.from(frequency.entries())
      .map(([word, count]) => ({
        word,
        count,
        percentage: (count / words.length) * 100,
      }))
      .sort((a, b) => b.count - a.count);

    return result.slice(0, 50); // Top 50 words
  }

  /**
   * Detect language
   */
  private detectLanguage(text: string): {
    code: string;
    name: string;
    confidence: number;
  } {
    // Simple language detection based on character patterns
    // In production, use a proper library like franc or langdetect

    const words = this.tokenize(text);
    let englishWords = 0;

    for (const word of words.slice(0, 100)) {
      if (ENGLISH_STOP_WORDS.has(word.toLowerCase())) {
        englishWords++;
      }
    }

    const englishRatio = englishWords / Math.min(words.length, 100);

    if (englishRatio > 0.3) {
      return {
        code: 'en',
        name: 'English',
        confidence: Math.min(1, englishRatio * 2),
      };
    }

    // Default to unknown
    return {
      code: 'unknown',
      name: 'Unknown',
      confidence: 0,
    };
  }

  /**
   * Generate summary
   */
  private generateSummary(text: string): string {
    const sentences = this.splitSentences(text);

    if (sentences.length <= this.params.summaryLength) {
      return text;
    }

    // Simple extractive summarization: take first N sentences
    const summarySentences = sentences.slice(0, this.params.summaryLength);

    return summarySentences.join(' ');
  }

  /**
   * Extract named entities
   */
  private extractEntities(text: string): Array<{
    text: string;
    type: string;
    confidence: number;
  }> {
    // Simple rule-based entity extraction
    const entities: Array<{ text: string; type: string; confidence: number }> = [];

    // Capitalized words (potential proper nouns)
    const capitalizedWords = text.match(/\b[A-Z][a-z]+\b/g) || [];

    for (const word of capitalizedWords) {
      if (!ENGLISH_STOP_WORDS.has(word.toLowerCase())) {
        entities.push({
          text: word,
          type: 'PROPER_NOUN',
          confidence: 0.5,
        });
      }
    }

    // Email addresses
    const emails = text.match(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g) || [];

    for (const email of emails) {
      entities.push({
        text: email,
        type: 'EMAIL',
        confidence: 0.95,
      });
    }

    // URLs
    const urls = text.match(/https?:\/\/[^\s]+/g) || [];

    for (const url of urls) {
      entities.push({
        text: url,
        type: 'URL',
        confidence: 0.95,
      });
    }

    return entities.slice(0, 50); // Limit to 50 entities
  }

  /**
   * Calculate text statistics
   */
  private calculateStatistics(text: string): {
    characterCount: number;
    wordCount: number;
    sentenceCount: number;
    paragraphCount: number;
    averageWordLength: number;
    averageSentenceLength: number;
    processingTime: number;
  } {
    const words = this.tokenize(text);
    const sentences = this.splitSentences(text);
    const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim());

    const totalCharacters = text.length;
    const totalCharactersNoSpaces = text.replace(/\s/g, '').length;
    const totalWords = words.length;
    const totalSentences = sentences.length;
    const totalParagraphs = paragraphs.length;

    const averageWordLength = totalWords > 0
      ? totalCharactersNoSpaces / totalWords
      : 0;

    const averageSentenceLength = totalSentences > 0
      ? totalWords / totalSentences
      : 0;

    return {
      characterCount: totalCharacters,
      wordCount: totalWords,
      sentenceCount: totalSentences,
      paragraphCount: totalParagraphs,
      averageWordLength,
      averageSentenceLength,
      processingTime: 0,
    };
  }

  /**
   * Tokenize text into words
   */
  private tokenize(text: string): string[] {
    return text.match(/\b[\w']+\b/g) || [];
  }

  /**
   * Split text into sentences
   */
  private splitSentences(text: string): string[] {
    return text
      .split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  /**
   * Count syllables
   */
  private countSyllables(text: string): number {
    const words = this.tokenize(text);
    let syllableCount = 0;

    for (const word of words) {
      syllableCount += this.countWordSyllables(word);
    }

    return syllableCount;
  }

  /**
   * Count syllables in a word
   */
  private countWordSyllables(word: string): number {
    word = word.toLowerCase();
    if (word.length <= 3) {
      return 1;
    }

    word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
    word = word.replace(/^y/, '');
    const syllables = word.match(/[aeiouy]{1,2}/g);

    return syllables ? syllables.length : 1;
  }
}
