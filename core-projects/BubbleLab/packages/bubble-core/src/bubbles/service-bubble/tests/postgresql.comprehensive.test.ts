   /*
    * Tests for PostgreSQLBubble
    *
    * These tests exercise the real Bubble base-class contract:
    * - The constructor never throws on invalid params; it captures a
    *   validationError and action() returns a controlled
    *   { success: false, error } result.
    * - The bubble exposes the expected static metadata.
    */

   import { describe, it, expect } from "vitest";
   import { PostgreSQLBubble } from "../postgresql.js";

   describe("PostgreSQLBubble", () => {
     it("does not throw on invalid params (captures validationError)", () => {
       expect(() => new PostgreSQLBubble({ notARealField: 1 } as any)).not.toThrow();
     });

     it("action() returns a controlled error for invalid params", async () => {
       const bubble = new PostgreSQLBubble({ notARealField: 1 } as any);
       const result = await bubble.action();
       expect(result.success).toBe(false);
       expect(typeof result.error).toBe("string");
       expect(result.error.length).toBeGreaterThan(0);
     });

     it("constructs with provided params without throwing", () => {
       const bubble = new PostgreSQLBubble({ operation: "selectRows", table: "users" } as any);
       expect(bubble).toBeDefined();
     });

     it("exposes static metadata", () => {
       expect(PostgreSQLBubble.bubbleName).toBeDefined();
       expect(PostgreSQLBubble.service).toBeDefined();
       expect(PostgreSQLBubble.schema).toBeDefined();
     });
   });
