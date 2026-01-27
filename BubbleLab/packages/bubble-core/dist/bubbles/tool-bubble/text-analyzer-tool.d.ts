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
declare const TextAnalyzerToolParamsSchema: z.ZodObject<{
    text: z.ZodString;
    operations: z.ZodDefault<z.ZodArray<z.ZodEnum<["sentiment", "keywords", "readability", "frequency", "language", "entities", "summary", "statistics"]>, "many">>;
    maxKeywords: z.ZodDefault<z.ZodNumber>;
    minKeywordLength: z.ZodDefault<z.ZodNumber>;
    summaryLength: z.ZodDefault<z.ZodNumber>;
    language: z.ZodDefault<z.ZodString>;
    removeStopWords: z.ZodDefault<z.ZodBoolean>;
    customStopWords: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    text: string;
    language: string;
    operations: ("summary" | "language" | "entities" | "keywords" | "readability" | "statistics" | "sentiment" | "frequency")[];
    maxKeywords: number;
    minKeywordLength: number;
    summaryLength: number;
    removeStopWords: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    customStopWords?: string[] | undefined;
}, {
    text: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    language?: string | undefined;
    operations?: ("summary" | "language" | "entities" | "keywords" | "readability" | "statistics" | "sentiment" | "frequency")[] | undefined;
    maxKeywords?: number | undefined;
    minKeywordLength?: number | undefined;
    summaryLength?: number | undefined;
    removeStopWords?: boolean | undefined;
    customStopWords?: string[] | undefined;
}>;
/**
 * Text analyzer result schema
 */
declare const TextAnalyzerToolResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    sentiment: z.ZodOptional<z.ZodObject<{
        score: z.ZodNumber;
        label: z.ZodString;
        confidence: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        label: string;
        confidence: number;
        score: number;
    }, {
        label: string;
        confidence: number;
        score: number;
    }>>;
    keywords: z.ZodOptional<z.ZodArray<z.ZodObject<{
        word: z.ZodString;
        score: z.ZodNumber;
        frequency: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        score: number;
        frequency: number;
        word: string;
    }, {
        score: number;
        frequency: number;
        word: string;
    }>, "many">>;
    readability: z.ZodOptional<z.ZodObject<{
        fleschScore: z.ZodNumber;
        gradeLevel: z.ZodNumber;
        label: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        label: string;
        fleschScore: number;
        gradeLevel: number;
    }, {
        label: string;
        fleschScore: number;
        gradeLevel: number;
    }>>;
    frequency: z.ZodOptional<z.ZodArray<z.ZodObject<{
        word: z.ZodString;
        count: z.ZodNumber;
        percentage: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        count: number;
        word: string;
        percentage: number;
    }, {
        count: number;
        word: string;
        percentage: number;
    }>, "many">>;
    language: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        name: z.ZodString;
        confidence: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        code: string;
        name: string;
        confidence: number;
    }, {
        code: string;
        name: string;
        confidence: number;
    }>>;
    entities: z.ZodOptional<z.ZodArray<z.ZodObject<{
        text: z.ZodString;
        type: z.ZodString;
        confidence: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        type: string;
        text: string;
        confidence: number;
    }, {
        type: string;
        text: string;
        confidence: number;
    }>, "many">>;
    summary: z.ZodOptional<z.ZodString>;
    statistics: z.ZodObject<{
        characterCount: z.ZodNumber;
        wordCount: z.ZodNumber;
        sentenceCount: z.ZodNumber;
        paragraphCount: z.ZodNumber;
        averageWordLength: z.ZodNumber;
        averageSentenceLength: z.ZodNumber;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        wordCount: number;
        processingTime: number;
        characterCount: number;
        sentenceCount: number;
        paragraphCount: number;
        averageWordLength: number;
        averageSentenceLength: number;
    }, {
        wordCount: number;
        processingTime: number;
        characterCount: number;
        sentenceCount: number;
        paragraphCount: number;
        averageWordLength: number;
        averageSentenceLength: number;
    }>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    statistics: {
        wordCount: number;
        processingTime: number;
        characterCount: number;
        sentenceCount: number;
        paragraphCount: number;
        averageWordLength: number;
        averageSentenceLength: number;
    };
    summary?: string | undefined;
    language?: {
        code: string;
        name: string;
        confidence: number;
    } | undefined;
    entities?: {
        type: string;
        text: string;
        confidence: number;
    }[] | undefined;
    keywords?: {
        score: number;
        frequency: number;
        word: string;
    }[] | undefined;
    readability?: {
        label: string;
        fleschScore: number;
        gradeLevel: number;
    } | undefined;
    sentiment?: {
        label: string;
        confidence: number;
        score: number;
    } | undefined;
    frequency?: {
        count: number;
        word: string;
        percentage: number;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    statistics: {
        wordCount: number;
        processingTime: number;
        characterCount: number;
        sentenceCount: number;
        paragraphCount: number;
        averageWordLength: number;
        averageSentenceLength: number;
    };
    summary?: string | undefined;
    language?: {
        code: string;
        name: string;
        confidence: number;
    } | undefined;
    entities?: {
        type: string;
        text: string;
        confidence: number;
    }[] | undefined;
    keywords?: {
        score: number;
        frequency: number;
        word: string;
    }[] | undefined;
    readability?: {
        label: string;
        fleschScore: number;
        gradeLevel: number;
    } | undefined;
    sentiment?: {
        label: string;
        confidence: number;
        score: number;
    } | undefined;
    frequency?: {
        count: number;
        word: string;
        percentage: number;
    }[] | undefined;
}>;
type TextAnalyzerToolParams = z.output<typeof TextAnalyzerToolParamsSchema>;
type TextAnalyzerToolResult = z.output<typeof TextAnalyzerToolResultSchema>;
type TextAnalyzerToolParamsInput = z.input<typeof TextAnalyzerToolParamsSchema>;
/**
 * Text Analyzer Tool
 * Comprehensive text analysis with multiple metrics
 */
export declare class TextAnalyzerTool extends ToolBubble<TextAnalyzerToolParams, TextAnalyzerToolResult> {
    /**
     * REQUIRED STATIC METADATA
     */
    static readonly type: "tool";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        text: z.ZodString;
        operations: z.ZodDefault<z.ZodArray<z.ZodEnum<["sentiment", "keywords", "readability", "frequency", "language", "entities", "summary", "statistics"]>, "many">>;
        maxKeywords: z.ZodDefault<z.ZodNumber>;
        minKeywordLength: z.ZodDefault<z.ZodNumber>;
        summaryLength: z.ZodDefault<z.ZodNumber>;
        language: z.ZodDefault<z.ZodString>;
        removeStopWords: z.ZodDefault<z.ZodBoolean>;
        customStopWords: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        text: string;
        language: string;
        operations: ("summary" | "language" | "entities" | "keywords" | "readability" | "statistics" | "sentiment" | "frequency")[];
        maxKeywords: number;
        minKeywordLength: number;
        summaryLength: number;
        removeStopWords: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        customStopWords?: string[] | undefined;
    }, {
        text: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        language?: string | undefined;
        operations?: ("summary" | "language" | "entities" | "keywords" | "readability" | "statistics" | "sentiment" | "frequency")[] | undefined;
        maxKeywords?: number | undefined;
        minKeywordLength?: number | undefined;
        summaryLength?: number | undefined;
        removeStopWords?: boolean | undefined;
        customStopWords?: string[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        sentiment: z.ZodOptional<z.ZodObject<{
            score: z.ZodNumber;
            label: z.ZodString;
            confidence: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            label: string;
            confidence: number;
            score: number;
        }, {
            label: string;
            confidence: number;
            score: number;
        }>>;
        keywords: z.ZodOptional<z.ZodArray<z.ZodObject<{
            word: z.ZodString;
            score: z.ZodNumber;
            frequency: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            score: number;
            frequency: number;
            word: string;
        }, {
            score: number;
            frequency: number;
            word: string;
        }>, "many">>;
        readability: z.ZodOptional<z.ZodObject<{
            fleschScore: z.ZodNumber;
            gradeLevel: z.ZodNumber;
            label: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            label: string;
            fleschScore: number;
            gradeLevel: number;
        }, {
            label: string;
            fleschScore: number;
            gradeLevel: number;
        }>>;
        frequency: z.ZodOptional<z.ZodArray<z.ZodObject<{
            word: z.ZodString;
            count: z.ZodNumber;
            percentage: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            count: number;
            word: string;
            percentage: number;
        }, {
            count: number;
            word: string;
            percentage: number;
        }>, "many">>;
        language: z.ZodOptional<z.ZodObject<{
            code: z.ZodString;
            name: z.ZodString;
            confidence: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            code: string;
            name: string;
            confidence: number;
        }, {
            code: string;
            name: string;
            confidence: number;
        }>>;
        entities: z.ZodOptional<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            type: z.ZodString;
            confidence: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            type: string;
            text: string;
            confidence: number;
        }, {
            type: string;
            text: string;
            confidence: number;
        }>, "many">>;
        summary: z.ZodOptional<z.ZodString>;
        statistics: z.ZodObject<{
            characterCount: z.ZodNumber;
            wordCount: z.ZodNumber;
            sentenceCount: z.ZodNumber;
            paragraphCount: z.ZodNumber;
            averageWordLength: z.ZodNumber;
            averageSentenceLength: z.ZodNumber;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            wordCount: number;
            processingTime: number;
            characterCount: number;
            sentenceCount: number;
            paragraphCount: number;
            averageWordLength: number;
            averageSentenceLength: number;
        }, {
            wordCount: number;
            processingTime: number;
            characterCount: number;
            sentenceCount: number;
            paragraphCount: number;
            averageWordLength: number;
            averageSentenceLength: number;
        }>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        statistics: {
            wordCount: number;
            processingTime: number;
            characterCount: number;
            sentenceCount: number;
            paragraphCount: number;
            averageWordLength: number;
            averageSentenceLength: number;
        };
        summary?: string | undefined;
        language?: {
            code: string;
            name: string;
            confidence: number;
        } | undefined;
        entities?: {
            type: string;
            text: string;
            confidence: number;
        }[] | undefined;
        keywords?: {
            score: number;
            frequency: number;
            word: string;
        }[] | undefined;
        readability?: {
            label: string;
            fleschScore: number;
            gradeLevel: number;
        } | undefined;
        sentiment?: {
            label: string;
            confidence: number;
            score: number;
        } | undefined;
        frequency?: {
            count: number;
            word: string;
            percentage: number;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        statistics: {
            wordCount: number;
            processingTime: number;
            characterCount: number;
            sentenceCount: number;
            paragraphCount: number;
            averageWordLength: number;
            averageSentenceLength: number;
        };
        summary?: string | undefined;
        language?: {
            code: string;
            name: string;
            confidence: number;
        } | undefined;
        entities?: {
            type: string;
            text: string;
            confidence: number;
        }[] | undefined;
        keywords?: {
            score: number;
            frequency: number;
            word: string;
        }[] | undefined;
        readability?: {
            label: string;
            fleschScore: number;
            gradeLevel: number;
        } | undefined;
        sentiment?: {
            label: string;
            confidence: number;
            score: number;
        } | undefined;
        frequency?: {
            count: number;
            word: string;
            percentage: number;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Analyze text for sentiment, keywords, readability, and more";
    static readonly longDescription = "\n    A comprehensive text analysis tool providing multiple NLP capabilities.\n\n    Features:\n    - SENTIMENT: Analyze emotional tone (positive/negative/neutral)\n    - KEYWORDS: Extract important keywords and phrases\n    - READABILITY: Score text complexity and reading level\n    - FREQUENCY: Analyze word frequency distribution\n    - LANGUAGE: Detect text language\n    - ENTITIES: Extract named entities (people, places, orgs)\n    - SUMMARY: Generate text summary\n    - STATISTICS: Basic text metrics (counts, averages)\n\n    Sentiment Analysis:\n    - Score from -1 (very negative) to +1 (very positive)\n    - Label: positive, negative, or neutral\n    - Confidence score based on word matches\n\n    Keyword Extraction:\n    - Extracts most relevant words\n    - TF-IDF-like scoring\n    - Configurable minimum word length\n    - Stop word removal\n\n    Readability Metrics:\n    - Flesch Reading Ease score\n    - Grade level estimation\n    - Label (easy, moderate, difficult)\n\n    Word Frequency:\n    - Count word occurrences\n    - Calculate percentages\n    - Sort by frequency\n\n    Use cases:\n    - Content analysis\n    - Social media monitoring\n    - Customer feedback analysis\n    - SEO optimization\n    - Text quality assessment\n    - Research and analysis\n\n    Note: This is a rule-based implementation.\n    For production use, consider using ML libraries like:\n    - natural (Node.js NLP)\n    - sentiment (sentiment analysis)\n    - compromise (lightweight NLP)\n  ";
    static readonly alias = "analyze-text";
    constructor(params: TextAnalyzerToolParamsInput, context?: BubbleContext);
    /**
     * Main action method - performs text analysis
     */
    performAction(context?: BubbleContext): Promise<TextAnalyzerToolResult>;
    /**
     * Analyze sentiment
     */
    private analyzeSentiment;
    /**
     * Extract keywords
     */
    private extractKeywords;
    /**
     * Analyze readability
     */
    private analyzeReadability;
    /**
     * Analyze word frequency
     */
    private analyzeFrequency;
    /**
     * Detect language
     */
    private detectLanguage;
    /**
     * Generate summary
     */
    private generateSummary;
    /**
     * Extract named entities
     */
    private extractEntities;
    /**
     * Calculate text statistics
     */
    private calculateStatistics;
    /**
     * Tokenize text into words
     */
    private tokenize;
    /**
     * Split text into sentences
     */
    private splitSentences;
    /**
     * Count syllables
     */
    private countSyllables;
    /**
     * Count syllables in a word
     */
    private countWordSyllables;
}
export {};
//# sourceMappingURL=text-analyzer-tool.d.ts.map