import { z } from 'zod';
export declare const YouTubeTranscriptScraperInputSchema: z.ZodObject<{
    videoUrl: z.ZodString;
}, "strip", z.ZodTypeAny, {
    videoUrl: string;
}, {
    videoUrl: string;
}>;
export declare const YouTubeTranscriptItemSchema: z.ZodObject<{
    start: z.ZodOptional<z.ZodString>;
    dur: z.ZodOptional<z.ZodString>;
    text: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    text?: string | undefined;
    start?: string | undefined;
    dur?: string | undefined;
}, {
    text?: string | undefined;
    start?: string | undefined;
    dur?: string | undefined;
}>;
export declare const YouTubeTranscriptResultSchema: z.ZodObject<{
    videoUrl: z.ZodOptional<z.ZodString>;
    data: z.ZodOptional<z.ZodArray<z.ZodObject<{
        start: z.ZodOptional<z.ZodString>;
        dur: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        text?: string | undefined;
        start?: string | undefined;
        dur?: string | undefined;
    }, {
        text?: string | undefined;
        start?: string | undefined;
        dur?: string | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    data?: {
        text?: string | undefined;
        start?: string | undefined;
        dur?: string | undefined;
    }[] | undefined;
    videoUrl?: string | undefined;
}, {
    data?: {
        text?: string | undefined;
        start?: string | undefined;
        dur?: string | undefined;
    }[] | undefined;
    videoUrl?: string | undefined;
}>;
//# sourceMappingURL=youtube-transcript-scraper.d.ts.map