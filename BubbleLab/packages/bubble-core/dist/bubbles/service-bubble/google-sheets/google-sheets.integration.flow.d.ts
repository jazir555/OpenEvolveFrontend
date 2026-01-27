import { BubbleFlow, type WebhookEvent } from '@bubblelab/bubble-core';
export interface Output {
    spreadsheetId: string;
    spreadsheetUrl: string;
    testResults: {
        operation: string;
        success: boolean;
        details?: string;
    }[];
}
/**
 * Payload for the Google Sheets Stress Test workflow.
 */
export interface SheetsStressTestPayload extends WebhookEvent {
    /**
     * The title for the test spreadsheet that will be created.
     * @canBeFile false
     */
    testTitle?: string;
}
export declare class GoogleSheetsStressTest extends BubbleFlow<'webhook/http'> {
    private createTestSpreadsheet;
    private createSheetWithSpaces;
    private writeRawDataWithNulls;
    private readFromRangeWithSpaces;
    private appendToRangeWithSpaces;
    private clearRangeWithSpaces;
    private deleteSheet;
    handle(payload: SheetsStressTestPayload): Promise<Output>;
}
//# sourceMappingURL=google-sheets.integration.flow.d.ts.map