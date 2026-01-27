import { z } from 'zod';
export declare const AvailableTools: z.ZodEnum<["web-search-tool", "web-scrape-tool", "web-crawl-tool", "web-extract-tool", "research-agent-tool", "reddit-scrape-tool", "instagram-tool", "list-bubbles-tool", "get-bubble-details-tool", "bubbleflow-validation-tool", "code-edit-tool", "chart-js-tool", "sql-query-tool"]>;
export type AvailableTool = z.infer<typeof AvailableTools>;
//# sourceMappingURL=available-tools.d.ts.map