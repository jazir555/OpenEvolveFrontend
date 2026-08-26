   /*
    * Tests for SlackBubble
    *
    * These tests exercise the real Bubble base-class contract:
    * - The constructor never throws on invalid params; it captures a
    *   validationError and action() returns a controlled
    *   { success: false, error } result.
    * - The bubble exposes the expected static metadata.
    */

   import { describe, it, expect } from "vitest";
   import { SlackBubble } from "./slack.js";

   describe("SlackBubble", () => {
     it("does not throw on invalid params (captures validationError)", () => {
       expect(() => new SlackBubble({ notARealField: 1 } as any)).not.toThrow();
     });

     it("action() returns a controlled error for invalid params", async () => {
       const bubble = new SlackBubble({ notARealField: 1 } as any);
       const result = await bubble.action();
       expect(result.success).toBe(false);
       expect(typeof result.error).toBe("string");
       expect(result.error.length).toBeGreaterThan(0);
     });

     it("constructs with provided params without throwing", () => {
       const bubble = new SlackBubble({ operation: "send_message", channel: "C123", text: "hi" } as any);
       expect(bubble).toBeDefined();
     });

     it("exposes static metadata", () => {
       expect(SlackBubble.bubbleName).toBeDefined();
       expect(SlackBubble.service).toBeDefined();
       expect(SlackBubble.schema).toBeDefined();
     });
   });
