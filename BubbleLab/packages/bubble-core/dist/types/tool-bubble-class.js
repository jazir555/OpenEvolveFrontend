import { BaseBubble } from './base-bubble-class.js';
/**
 * Abstract base class for all tool bubbles that can be converted to LangGraph tools
 */
export class ToolBubble extends BaseBubble {
    type = 'tool';
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    // Static method - returns LangChain tool with credentials injected
    // Creates a LangGraph compatible tool with specific configurations that will
    // be passed in to the tool bubble
    static toolAgent(credentials, config, context) {
        // In static context, 'this' refers to the constructor/class
        const ToolClass = this;
        const { schema, bubbleName, shortDescription } = ToolClass;
        if (!schema || !bubbleName || !shortDescription) {
            throw new Error(`${ToolClass.name} must define static schema, bubbleName, and shortDescription`);
        }
        // Remove credentials from schema for agent use
        // Remove config from schema for agent use
        let agentSchema = schema;
        if (schema.shape?.credentials) {
            agentSchema = schema.omit({ credentials: true });
        }
        if (agentSchema.shape?.config) {
            agentSchema = agentSchema.omit({ config: true });
        }
        agentSchema = agentSchema.passthrough();
        return {
            name: bubbleName,
            description: shortDescription,
            schema: agentSchema,
            func: async (toolParams) => {
                // Create instance with credentials and config injected
                // Sometimes config should be dynamic and determined on each
                // tool invocation, rather than the start of agent run
                // In this case, we will replace the config (statically configured in the tool bubble)
                // with the runtime config
                const runtimeConfig = toolParams?.config;
                const enrichedParams = {
                    ...toolParams,
                    credentials,
                    config: runtimeConfig || config,
                };
                // 'this' in static context is the constructor
                const instance = new ToolClass(enrichedParams, context);
                // Use performAction directly to get raw result, not wrapped BubbleResult
                return instance.action();
            },
        };
    }
}
//# sourceMappingURL=tool-bubble-class.js.map