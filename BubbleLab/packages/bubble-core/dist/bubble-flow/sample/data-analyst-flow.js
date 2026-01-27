import { BubbleFlow, SlackBubble, SlackDataAssistantWorkflow, } from '@bubblelab/bubble-core';
export class DataAnalystFlow extends BubbleFlow {
    constructor() {
        super('data-analyst-flow', 'AI-powered database analysis triggered by Slack mentions');
    }
    async handle(payload) {
        try {
            // Extract the question from the Slack message
            const userQuestion = payload.text.replace(/<@[^>]+>/g, '').trim();
            if (!userQuestion) {
                return {
                    success: false,
                    error: 'Please provide a question after mentioning me.',
                };
            }
            if (payload.monthlyLimitError) {
                // Send a message to the user with the monthly limit error message
                await new SlackBubble({
                    channel: payload.channel,
                    operation: 'send_message',
                    thread_ts: payload.thread_ts || payload.slack_event.event.ts,
                    text: payload.monthlyLimitError,
                }).action();
                return {
                    success: false,
                    error: payload.monthlyLimitError,
                };
            }
            const workflow = new SlackDataAssistantWorkflow({
                slackChannel: payload.channel,
                userQuestion: userQuestion,
                userName: payload.user,
                dataSourceType: 'postgresql',
                ignoreSSLErrors: true,
                aiModel: 'google/gemini-2.5-pro',
                temperature: 0.3,
                verbosity: '1',
                technicality: '1',
                includeQuery: true,
                includeExplanation: true,
                slackThreadTs: payload.thread_ts || payload.slack_event.event.ts,
            });
            const result = await workflow.action();
            return {
                success: result.success,
                directAnswer: result.data?.formattedResponse,
                insights: result.data?.queryResults?.map((result) => result.summary) ?? [],
                recommendations: result.data?.queryResults?.map((result) => result.summary) ?? [],
                slackMessageId: result.data?.slackMessageTs,
                error: result.error
                    ? result.error.substring(0, 100) + '...'
                    : undefined,
            };
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred',
            };
        }
    }
}
//# sourceMappingURL=data-analyst-flow.js.map